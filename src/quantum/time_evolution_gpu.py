r"""
GPU-Accelerated Quantum Time-Evolution Backend (PyTorch / CUDA).

This module is the GPU counterpart of ``src/quantum/time_evolution.py``.  It
reproduces *exactly* the same physics as the original PennyLane Trotter circuit
and the exact matrix-exponential propagator, but restructures the computation so
that it maps onto a GPU instead of a Python ``for`` loop.

--------------------------------------------------------------------------
Why the original PennyLane code cannot use a GPU efficiently
--------------------------------------------------------------------------
``PennyLaneTimeEvolution.simulate_trotter_trajectory`` calls
``circuit_at_step(step)`` once per time step, and each call *replays the entire
circuit from t_0*.  For ``n_steps = 500`` that is

    sum_{k=1}^{500} k  ~=  125,000 Trotter slices  x  ~16 Pauli rotations
                       ~=  2,000,000 gate applications,

each one dispatched through the Python/PennyLane gate queue.  The cost is
dominated by interpreter overhead on 4x4 matrices, so simply swapping
``lightning.qubit`` for ``lightning.gpu`` makes it *slower*: a 2-qubit state
vector is 64 bytes and every CUDA kernel launch costs ~5-10 us.

--------------------------------------------------------------------------
What this module does instead
--------------------------------------------------------------------------
Three structural changes turn the problem into GPU-shaped work:

1. **Closed-form Pauli rotations, fully vectorised.**
   Because every Pauli string satisfies ``P^2 = I``,

       exp(-i * theta/2 * P) = cos(theta/2) * I  -  i * sin(theta/2) * P

   so no per-gate matrix exponential is needed.  All
   ``(n_batch, n_steps, n_paulis)`` rotation angles are evaluated in a single
   fused tensor op.

2. **Batching over the impact-parameter / energy sweep.**
   ``sweep_impact_parameters`` used to spawn one OS process per impact
   parameter.  Here the whole sweep is one tensor with a leading batch axis, so
   25 (or 2500) trajectories are propagated by the same kernels.

3. **Parallel prefix scan over time.**
   The time-ordered product ``U_k U_{k-1} ... U_0`` is associative, so the
   *cumulative* products for **all** time steps are obtained with a
   Hillis-Steele scan in ``ceil(log2(n_steps))`` batched matmuls -- about
   9 GPU kernels for a 500-step run instead of 500 sequential Python
   iterations.  As a bonus, the scan's floating-point error grows like
   ``O(log n_steps)`` rather than ``O(n_steps)``, so single precision is
   comfortably accurate here.

--------------------------------------------------------------------------
Backends
--------------------------------------------------------------------------
``TorchTimeEvolution``
    The fast path.  Pure PyTorch, runs on CUDA (Linux **and** Windows), ROCm,
    or CPU.  This is what you want for production sweeps.

``PennyLaneGPUTimeEvolution``
    The literal "PennyLane on the GPU" path, kept for provenance and
    cross-validation.  It builds a real ``qml.PauliRot`` circuit and executes it
    on ``lightning.gpu`` (cuQuantum, Linux-only) if installed, otherwise on
    ``default.qubit`` with the ``torch`` interface and CUDA tensors.  It also
    fixes the quadratic replay by carrying the state forward with
    ``qml.StatePrep``, making it O(n_steps).

Both return populations with identical shape/semantics to the original
``simulate_exact_unitary_evolution``: ``(n_steps, 2 ** n_qubits)``.

--------------------------------------------------------------------------
Install
--------------------------------------------------------------------------
Windows / Linux, NVIDIA CUDA 12.x::

    pip install torch --index-url https://download.pytorch.org/whl/cu121

Optional cuQuantum path (Linux / WSL2 only)::

    pip install pennylane-lightning-gpu custatevec-cu12
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.quantum.mapper import MultiStatePauliMapper, build_pauli_string_matrix

# Physical conversion factors (Atomic units: hbar = m_e = e = 1).
# Kept identical to src/quantum/time_evolution.py so results are comparable.
EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_EV = 27.211386245988
TIME_AU_TO_FEMTOSECONDS = 0.024188843265857

__all__ = [
    "TorchBackend",
    "TorchTimeEvolution",
    "PennyLaneGPUTimeEvolution",
    "gpu_is_available",
    "cuda_arch_status",
    "describe_gpu_environment",
    "resolve_backend",
]


# ============================================================================
# Section 1: Backend discovery and device resolution
# ============================================================================

def _import_torch():
    """Imports torch lazily so the CPU-only code path never pays for it."""
    try:
        import torch  # noqa: F401
        return torch
    except Exception:  # pragma: no cover - depends on user environment
        return None


def cuda_arch_status() -> Tuple[bool, Optional[str]]:
    """
    Checks that the installed PyTorch actually has compiled kernels for the
    visible GPU's compute capability.

    This matters on new silicon: PyTorch will happily report
    ``torch.cuda.is_available() == True`` on an RTX 50-series (Blackwell,
    sm_120) card while the wheel only ships kernels up to sm_90.  Every CUDA op
    then fails at launch, or silently returns garbage.  Rather than let that
    reach the physics, the backend refuses the device and says why.

    Returns ``(ok, message)``; ``message`` is None when everything is fine.
    """
    torch = _import_torch()
    if torch is None:
        return False, "PyTorch is not installed."
    try:
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            return False, None  # no GPU at all; not an error worth explaining
    except Exception as exc:  # pragma: no cover
        return False, f"CUDA probe failed: {exc}"

    try:
        arch_list = list(torch.cuda.get_arch_list())
    except Exception:  # pragma: no cover - very old torch
        return True, None

    if not arch_list:
        return True, None

    major, minor = torch.cuda.get_device_capability(0)
    needed = f"sm_{major}{minor}"
    name = torch.cuda.get_device_properties(0).name

    if any(a.startswith(needed) for a in arch_list):
        return True, None

    built_cuda = getattr(torch.version, "cuda", "unknown")
    return False, (
        f"{name} has compute capability {needed}, but this PyTorch build "
        f"(torch {torch.__version__}, CUDA {built_cuda}) only ships kernels for "
        f"{', '.join(arch_list)}.\n"
        f"    Reinstall a Blackwell-capable build:\n"
        f"        pip uninstall -y torch\n"
        f"        pip install torch --index-url https://download.pytorch.org/whl/cu128"
    )


def gpu_is_available() -> bool:
    """
    True if a CUDA device is visible **and** the installed PyTorch has kernels
    compiled for its architecture.  See :func:`cuda_arch_status`.
    """
    ok, _ = cuda_arch_status()
    return ok


def describe_gpu_environment() -> Dict[str, Any]:
    """
    Returns a human-readable dictionary describing what acceleration is
    actually available.  Useful for a hackathon demo / README table.
    """
    info: Dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "compiled_architectures": [],
        "arch_compatible": False,
        "arch_problem": None,
        "device_count": 0,
        "devices": [],
        "pennylane_installed": False,
        "pennylane_version": None,
        "lightning_gpu_available": False,
    }

    torch = _import_torch()
    if torch is not None:
        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            try:
                info["compiled_architectures"] = list(torch.cuda.get_arch_list())
            except Exception:
                pass
            ok, problem = cuda_arch_status()
            info["arch_compatible"] = ok
            info["arch_problem"] = problem
            if info["cuda_available"]:
                info["device_count"] = torch.cuda.device_count()
                for i in range(info["device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    info["devices"].append(
                        {
                            "index": i,
                            "name": props.name,
                            "total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
                            "capability": f"{props.major}.{props.minor}",
                        }
                    )
        except Exception:  # pragma: no cover
            pass

    try:
        import pennylane as qml

        info["pennylane_installed"] = True
        info["pennylane_version"] = qml.__version__
        try:
            qml.device("lightning.gpu", wires=1)
            info["lightning_gpu_available"] = True
        except Exception:
            info["lightning_gpu_available"] = False
    except Exception:
        pass

    return info


@dataclass
class TorchBackend:
    """
    Resolved PyTorch execution context: module handle, device and dtypes.

    Parameters are resolved once and reused, so no per-call device queries hit
    the CUDA driver.
    """

    torch: Any
    device: Any
    complex_dtype: Any
    real_dtype: Any
    name: str

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def itemsize(self) -> int:
        """Bytes per complex element (8 for complex64, 16 for complex128)."""
        return 8 if self.complex_dtype == self.torch.complex64 else 16

    def free_memory_bytes(self) -> int:
        """Best-effort free-memory query, used to size batch chunks."""
        if self.is_cuda:
            try:
                free, _total = self.torch.cuda.mem_get_info(self.device)
                return int(free)
            except Exception:  # pragma: no cover
                return 2 * 1024 ** 3
        return 4 * 1024 ** 3  # conservative host-RAM budget

    def synchronize(self) -> None:
        if self.is_cuda:
            self.torch.cuda.synchronize(self.device)

    def summary(self) -> str:
        return (
            f"TorchBackend(device={self.name}, "
            f"complex_dtype={str(self.complex_dtype).split('.')[-1]})"
        )


def resolve_backend(
    device: str = "auto",
    precision: str = "auto",
    quiet: bool = False,
) -> Optional[TorchBackend]:
    """
    Resolves a PyTorch backend, or returns ``None`` if torch is unavailable.

    Parameters
    ----------
    device : str
        ``"auto"``  -> CUDA if visible, else CPU.
        ``"cuda"``  -> CUDA, raising if unavailable.
        ``"cuda:1"``-> a specific GPU.
        ``"cpu"``   -> force CPU (still benefits from batching + the scan).
    precision : str
        ``"auto"``   -> ``complex64`` on CUDA, ``complex128`` on CPU.
        ``"single"`` / ``"float32"`` -> ``complex64``.
        ``"double"`` / ``"float64"`` -> ``complex128``.

        Note: on GeForce/RTX consumer cards FP64 throughput is 1/32 to 1/64 of
        FP32.  ``complex64`` combined with the log-depth prefix scan keeps
        population errors below ~1e-6 for these problem sizes, which is well
        under the Trotter error itself.
    quiet : bool
        Suppress the informational device banner.
    """
    torch = _import_torch()
    if torch is None:
        if not quiet:
            print(
                "[!] PyTorch not installed - GPU backend unavailable. "
                "Install with:  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121"
            )
        return None

    request = (device or "auto").strip().lower()

    arch_ok, arch_problem = cuda_arch_status()

    if request == "auto":
        target = "cuda" if arch_ok else "cpu"
        if not arch_ok and arch_problem and not quiet:
            print(f"[!] GPU present but unusable, running on CPU tensors.\n    {arch_problem}")
    elif request.startswith("cuda"):
        if not arch_ok:
            if not quiet:
                reason = arch_problem or "no CUDA device is visible to PyTorch"
                print(f"[!] CUDA requested but unusable - falling back to CPU tensors.\n    {reason}")
            target = "cpu"
        else:
            target = request
    elif request in {"mps", "metal"}:
        # Apple Metal has incomplete complex-tensor support (no complex matmul
        # or linalg.eigh as of torch 2.x), so this path is intentionally CPU.
        if not quiet:
            print("[!] MPS lacks complex64 linear algebra - using CPU tensors instead.")
        target = "cpu"
    else:
        target = request

    torch_device = torch.device(target)

    prec = (precision or "auto").strip().lower()
    if prec == "auto":
        use_single = torch_device.type == "cuda"
    elif prec in {"single", "float32", "complex64", "fp32"}:
        use_single = True
    elif prec in {"double", "float64", "complex128", "fp64"}:
        use_single = False
    else:
        raise ValueError(f"Unknown precision {precision!r}; use auto/single/double.")

    complex_dtype = torch.complex64 if use_single else torch.complex128
    real_dtype = torch.float32 if use_single else torch.float64

    backend = TorchBackend(
        torch=torch,
        device=torch_device,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
        name=str(torch_device),
    )

    if not quiet:
        if backend.is_cuda:
            idx = torch_device.index or 0
            props = torch.cuda.get_device_properties(idx)
            print(
                f"[+] GPU backend active: {props.name} "
                f"({props.total_memory / 1024 ** 3:.1f} GB, sm_{props.major}{props.minor}), "
                f"dtype={str(complex_dtype).split('.')[-1]}"
            )
        else:
            print(
                f"[*] Torch backend on CPU (dtype={str(complex_dtype).split('.')[-1]}). "
                "Batching + prefix scan still apply."
            )

    return backend


# ============================================================================
# Section 2: The batched GPU propagator
# ============================================================================

class TorchTimeEvolution:
    r"""
    Batched, GPU-resident time-evolution engine.

    Physics is unchanged from ``src/quantum/time_evolution.py``:

      * ``method="trotter"`` reproduces the PennyLane circuit gate-for-gate,
        applying ``exp(-i c_m(t_k) P_m dt)`` in the same order as
        ``_apply_trotter_step_pennylane`` (identity-only terms are skipped
        exactly as ``qml.PauliRot`` skipping them; they are a global phase).
      * ``method="exact"`` reproduces ``simulate_exact_unitary_evolution``,
        diagonalising ``H(t_k)`` and propagating with the eigen-decomposition.

    What changed is *how* it is evaluated: every trajectory in the batch and
    every time step is built at once, and the time-ordered product is taken
    with a logarithmic-depth parallel scan.
    """

    def __init__(
        self,
        mapper: MultiStatePauliMapper,
        device: str = "auto",
        precision: str = "auto",
        scan_mode: str = "parallel",
        exact_method: str = "eigh",
        energy_to_hartree: float = 1.0,
        max_memory_fraction: float = 0.6,
        quiet: bool = False,
    ):
        # energy_to_hartree converts whatever units mapper.evaluate_hamiltonian_paulis_at_t
        # returns into Hartree before multiplying by dt in atomic units.
        #   1.0            -> the mapper already returns a.u. (current CPU convention)
        #   EV_TO_HARTREE  -> the mapper returns eV
        # Whatever you choose here MUST match what time_evolution.py does, or the
        # GPU and CPU paths will disagree by a factor of ~27.
        self.energy_to_hartree = float(energy_to_hartree)
        self.mapper = mapper
        self.n_qubits = mapper.n_qubits
        self.dim = 2 ** mapper.n_qubits
        self.scan_mode = scan_mode
        self.exact_method = exact_method
        self.max_memory_fraction = max_memory_fraction

        self.backend = resolve_backend(device=device, precision=precision, quiet=quiet)
        self.available = self.backend is not None
        self.init_error: Optional[str] = None if self.available else "PyTorch not installed"

        if self.available:
            try:
                self._build_static_tensors()
            except Exception as exc:  # pragma: no cover
                self.available = False
                self.init_error = str(exc)

    # -- static setup -------------------------------------------------------

    def _build_static_tensors(self) -> None:
        """
        Uploads the (small, constant) Pauli basis and the mapper's precomputed
        coefficient vectors to the device exactly once.
        """
        torch = self.backend.torch
        dev = self.backend.device
        cdt = self.backend.complex_dtype
        rdt = self.backend.real_dtype

        paulis: List[str] = list(self.mapper.active_paulis)
        self.pauli_strings = paulis
        self.n_paulis = len(paulis)

        # (n_paulis, dim, dim) stack of Pauli-string matrices.
        stacked = np.stack([build_pauli_string_matrix(p) for p in paulis], axis=0)
        self.P = torch.as_tensor(stacked, dtype=cdt, device=dev)

        # Identity-only strings ("II", "III", ...) contribute a global phase
        # only.  The PennyLane reference skips them, so the Trotter path skips
        # them too and stays bit-comparable.  The exact path keeps them (a
        # global phase does not change |psi|^2 either way).
        # Kept as a plain Python list: testing it inside the gate loop must not
        # force a device synchronisation.
        self.identity_flags: List[bool] = [all(ch == "I" for ch in p) for p in paulis]
        self.n_active_paulis = sum(1 for flag in self.identity_flags if not flag)

        self.eye = torch.eye(self.dim, dtype=cdt, device=dev)

        # Precomputed Pauli-coefficient vectors from the mapper:
        #   c_m(t) = h0_m - (mux_m * Ex + muy_m * Ey + muz_m * Ez)
        self.vec_h0 = torch.as_tensor(
            np.asarray(self.mapper.vec_h0, dtype=float), dtype=rdt, device=dev
        )
        self.mu_stack = torch.as_tensor(
            np.stack(
                [
                    np.asarray(self.mapper.vec_mux, dtype=float),
                    np.asarray(self.mapper.vec_muy, dtype=float),
                    np.asarray(self.mapper.vec_muz, dtype=float),
                ],
                axis=0,
            ),
            dtype=rdt,
            device=dev,
        )  # (3, n_paulis)

    # -- coefficient evaluation --------------------------------------------

    def _coefficients(self, e_fields, dipole_coupling_scale: float = 1.0):
        """
        Vectorised equivalent of ``mapper.evaluate_hamiltonian_paulis_at_t``
        over a whole (batch, time) grid.

        e_fields : (..., 3) tensor of field vectors, in atomic units.
        returns  : (..., n_paulis) tensor of real coefficients, in eV.
        """
        return self.vec_h0 - dipole_coupling_scale * self.backend.torch.einsum(
            "...a,am->...m", e_fields, self.mu_stack
        )

    # -- step unitaries -----------------------------------------------------

    def _trotter_step_unitaries(self, coeffs, dt_scaled: float):
        r"""
        Builds one Trotter-slice unitary per (batch, time) entry:

            U_k = R_{M-1} ... R_1 R_0,
            R_m = exp(-i * c_m * dt * P_m)
                = cos(c_m * dt) * I  -  i * sin(c_m * dt) * P_m

        The right-to-left product order matches PennyLane's gate queue, where
        the *first* applied gate ends up rightmost in the matrix product.

        coeffs : (B, K, n_paulis) real
        returns: (B, K, dim, dim) complex
        """
        torch = self.backend.torch
        cdt = self.backend.complex_dtype

        # theta_m = 2 * c_m * dt  ->  half-angle is c_m * dt (see qml.PauliRot).
        half = coeffs * dt_scaled                       # (B, K, M)
        cos_h = torch.cos(half).to(cdt)[..., None, None]
        sin_h = torch.sin(half).to(cdt)[..., None, None]

        batch_shape = coeffs.shape[:-1]
        U = self.eye.expand(*batch_shape, self.dim, self.dim).clone()

        for m in range(self.n_paulis):
            if self.identity_flags[m]:
                continue  # global phase; skipped by the PennyLane reference too
            # R_m for every (batch, time) at once: (B, K, dim, dim)
            R_m = cos_h[..., m, :, :] * self.eye - 1j * sin_h[..., m, :, :] * self.P[m]
            U = R_m @ U  # left-multiply == applied after everything so far

        return U

    def _exact_step_unitaries(self, coeffs, dt_scaled: float):
        r"""
        Builds the exact slice propagator ``exp(-i H(t_k) dt)`` for every
        (batch, time) entry, mirroring ``simulate_exact_unitary_evolution``.

        coeffs : (B, K, n_paulis) real
        returns: (B, K, dim, dim) complex
        """
        torch = self.backend.torch
        cdt = self.backend.complex_dtype

        # H = sum_m c_m P_m   ->  (B, K, dim, dim)
        H = torch.einsum("...m,mij->...ij", coeffs.to(cdt), self.P)
        # Guard against round-off asymmetry before the Hermitian solver.
        H = 0.5 * (H + H.conj().transpose(-1, -2))

        method = self.exact_method.lower()
        if method == "expm":
            return torch.linalg.matrix_exp(-1j * dt_scaled * H)

        # Default: eigen-decomposition, identical to the NumPy reference.
        evals, evecs = torch.linalg.eigh(H)                      # evals real
        phases = torch.exp(-1j * evals.to(cdt) * dt_scaled)      # (B, K, dim)
        return (evecs * phases[..., None, :]) @ evecs.conj().transpose(-1, -2)

    # -- time-ordered product ----------------------------------------------

    def _prefix_products(self, U):
        r"""
        Cumulative time-ordered products ``S_k = U_k U_{k-1} ... U_0`` for every
        k, via a Hillis-Steele inclusive scan.

        Matrix multiplication is associative but not commutative, so the scan
        must always place the later-time factor on the left; that is what
        ``S[k] @ S[k - offset]`` does.

        Depth is ``ceil(log2(K))`` batched matmuls instead of K sequential ones.

        U : (B, K, dim, dim)
        returns: (B, K, dim, dim)
        """
        torch = self.backend.torch
        K = U.shape[1]
        if K == 0:
            return U

        if self.scan_mode == "sequential":
            out = torch.empty_like(U)
            acc = U[:, 0]
            out[:, 0] = acc
            for k in range(1, K):
                acc = U[:, k] @ acc
                out[:, k] = acc
            return out

        S = U
        offset = 1
        while offset < K:
            head = S[:, :offset]
            tail = S[:, offset:] @ S[:, : K - offset]
            S = torch.cat((head, tail), dim=1)
            offset *= 2
        return S

    # -- memory-aware chunking ---------------------------------------------

    def _chunk_size(self, n_batch: int, n_slices: int) -> int:
        """
        Largest batch chunk that keeps the transient tensors inside the memory
        budget.  Peak live set is roughly 4 tensors of (chunk, K, dim, dim):
        the step unitaries, the scan's concatenated copy, and two temporaries.
        """
        bytes_per_traj = n_slices * self.dim * self.dim * self.backend.itemsize
        budget = int(self.backend.free_memory_bytes() * self.max_memory_fraction)
        per_traj = max(1, 4 * bytes_per_traj)
        chunk = max(1, budget // per_traj)
        return int(min(n_batch, chunk))

    # -- public API ---------------------------------------------------------

    def simulate_batch(
        self,
        t_grid_au: np.ndarray,
        e_fields_batch: np.ndarray,
        method: str = "trotter",
        dipole_coupling_scale: float = 1.0,
        return_numpy: bool = True,
    ) -> np.ndarray:
        r"""
        Propagates a whole batch of collision trajectories at once.

        Parameters
        ----------
        t_grid_au : (n_steps,) array
            Shared time grid, atomic units, uniformly spaced.
        e_fields_batch : (n_batch, n_steps, 3) or (n_steps, 3) array
            Electric-field vectors per trajectory, atomic units.
        method : {"trotter", "exact"}
            ``"trotter"`` matches the PennyLane circuit; ``"exact"`` matches the
            matrix-exponential reference propagator.
        dipole_coupling_scale : float
            Passed through to the Hamiltonian, as in the mapper.

        Returns
        -------
        populations : (n_batch, n_steps, 2 ** n_qubits) array
            ``P_i(t) = |<i|psi(t)>|^2``.  If a single trajectory was supplied,
            the leading axis is dropped and the shape is ``(n_steps, dim)``,
            matching ``simulate_exact_unitary_evolution``.
        """
        if not self.available:
            raise RuntimeError(f"Torch backend unavailable: {self.init_error}")

        torch = self.backend.torch
        dev = self.backend.device
        rdt = self.backend.real_dtype

        fields = np.asarray(e_fields_batch, dtype=float)
        squeeze_batch = fields.ndim == 2
        if squeeze_batch:
            fields = fields[None, ...]
        if fields.ndim != 3 or fields.shape[-1] != 3:
            raise ValueError(
                f"e_fields_batch must have shape (n_batch, n_steps, 3); got {fields.shape}"
            )

        t_grid = np.asarray(t_grid_au, dtype=float)
        n_steps = int(t_grid.size)
        if fields.shape[1] != n_steps:
            raise ValueError(
                f"e_fields_batch time axis ({fields.shape[1]}) != len(t_grid_au) ({n_steps})"
            )

        n_batch = fields.shape[0]
        dim = self.dim
        populations = np.zeros((n_batch, n_steps, dim), dtype=float)

        # Initial state |0...0>: population 1 in the ground manifold state.
        populations[:, 0, 0] = 1.0
        if n_steps < 2:
            return populations[0] if squeeze_batch else populations

        # Uniform grid assumption, identical to the original implementation.
        dt_au = float(t_grid[1] - t_grid[0])
        dt_scaled = dt_au * self.energy_to_hartree

        # Only the first n_steps-1 fields drive a propagation slice: field k
        # carries the state from step k to step k+1 (same indexing as the
        # NumPy reference and as circuit_at_step()).
        n_slices = n_steps - 1
        method_key = method.lower()
        if method_key not in {"trotter", "exact"}:
            raise ValueError(f"method must be 'trotter' or 'exact'; got {method!r}")

        chunk = self._chunk_size(n_batch, n_slices)

        with torch.no_grad():
            for start in range(0, n_batch, chunk):
                stop = min(start + chunk, n_batch)
                e_chunk = torch.as_tensor(
                    fields[start:stop, :n_slices, :], dtype=rdt, device=dev
                )

                coeffs = self._coefficients(e_chunk, dipole_coupling_scale)

                if method_key == "trotter":
                    U = self._trotter_step_unitaries(coeffs, dt_scaled)
                else:
                    U = self._exact_step_unitaries(coeffs, dt_scaled)

                S = self._prefix_products(U)  # (b, n_slices, dim, dim)

                # psi_0 = e_0, so psi(t_{k+1}) is column 0 of S_k.
                psi = S[..., :, 0]                        # (b, n_slices, dim)
                probs = (psi.real ** 2 + psi.imag ** 2)   # |amplitude|^2

                # Renormalise, mirroring the reference propagator's per-step
                # normalisation guard against accumulated round-off.
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)

                populations[start:stop, 1:, :] = probs.to(torch.float64).cpu().numpy()

                del e_chunk, coeffs, U, S, psi, probs

            if self.backend.is_cuda:
                torch.cuda.empty_cache()

        return populations[0] if squeeze_batch else populations

    def simulate_trotter_trajectory(
        self,
        t_grid_au: np.ndarray,
        e_fields_au: np.ndarray,
    ) -> np.ndarray:
        """
        Drop-in replacement for ``PennyLaneTimeEvolution.simulate_trotter_trajectory``.
        Returns ``(n_steps, 2 ** n_qubits)`` populations.
        """
        return self.simulate_batch(t_grid_au, e_fields_au, method="trotter")

    def simulate_exact_trajectory(
        self,
        t_grid_au: np.ndarray,
        e_fields_au: np.ndarray,
    ) -> np.ndarray:
        """
        Drop-in replacement for ``simulate_exact_unitary_evolution``, on device.
        """
        return self.simulate_batch(t_grid_au, e_fields_au, method="exact")

    def summary(self) -> str:
        if not self.available:
            return f"TorchTimeEvolution(unavailable: {self.init_error})"
        return (
            f"TorchTimeEvolution({self.backend.summary()}, "
            f"n_qubits={self.n_qubits}, dim={self.dim}, "
            f"n_pauli_terms={self.n_paulis} ({self.n_active_paulis} non-identity), "
            f"scan={self.scan_mode}, "
            f"exact_method={self.exact_method})"
        )


# ============================================================================
# Section 3: The literal PennyLane-on-GPU path (validation / provenance)
# ============================================================================

class PennyLaneGPUTimeEvolution:
    r"""
    Executes the *actual* PennyLane Trotter circuit on a GPU-backed device.

    Device selection order:
      1. ``lightning.gpu``  - cuQuantum custatevec.  Linux / WSL2 only, and the
         right choice once ``n_qubits`` is large (>~ 18).
      2. ``default.qubit``  - with ``interface="torch"`` and CUDA tensors.
         Works on Windows.
      3. ``lightning.qubit`` / ``default.qubit`` on CPU as a final fallback.

    Unlike the original implementation this is **O(n_steps)**, not O(n_steps^2):
    the state is carried forward between slices with ``qml.StatePrep`` instead of
    replaying the circuit from t_0 at every step.

    For 2-3 qubit manifolds this will still be slower than
    :class:`TorchTimeEvolution` -- kernel-launch overhead dominates a 4x4 state
    vector.  Use it to cross-check the fast path, and as the scaling story for
    larger manifolds.
    """

    def __init__(
        self,
        mapper: MultiStatePauliMapper,
        prefer: str = "auto",
        shots: Optional[int] = None,
        energy_to_hartree: float = 1.0,
        quiet: bool = False,
    ):
        self.energy_to_hartree = float(energy_to_hartree)
        self.mapper = mapper
        self.n_qubits = mapper.n_qubits
        self.dim = 2 ** mapper.n_qubits
        self.shots = shots
        self.qml = None
        self.dev = None
        self.device_name = None
        self.interface = None
        self.torch_device = None
        self.available = False
        self.init_error: Optional[str] = None

        try:
            import pennylane as qml
        except Exception as exc:
            self.init_error = f"PennyLane not installed ({exc})"
            return

        self.qml = qml
        torch = _import_torch()
        cuda_ok = gpu_is_available()

        candidates: List[Tuple[str, Optional[str]]] = []
        pref = (prefer or "auto").strip().lower()

        if pref in {"auto", "lightning.gpu", "cuquantum"}:
            candidates.append(("lightning.gpu", None))
        if pref in {"auto", "torch", "default.qubit"} and torch is not None and cuda_ok:
            candidates.append(("default.qubit", "torch"))
        if pref == "auto":
            candidates.append(("lightning.qubit", None))
            candidates.append(("default.qubit", None))

        errors = []
        for dev_name, interface in candidates:
            try:
                dev = qml.device(dev_name, wires=self.n_qubits, shots=shots)
                self.dev = dev
                self.device_name = dev_name
                self.interface = interface
                self.available = True
                break
            except Exception as exc:
                errors.append(f"{dev_name}: {exc}")

        if not self.available:
            self.init_error = "; ".join(errors) or "no PennyLane device could be created"
            return

        if self.interface == "torch":
            self.torch_device = torch.device("cuda")

        if not quiet:
            suffix = " (CUDA tensors)" if self.interface == "torch" else ""
            print(f"[+] PennyLane device: {self.device_name}{suffix}, wires={self.n_qubits}")

    # ----------------------------------------------------------------------

    def _apply_trotter_step(self, coeffs: Sequence[float], pauli_strings: Sequence[str], dt_scaled: float) -> None:
        r"""
        One Trotter slice, ``prod_m exp(-i c_m P_m dt)``, as real PennyLane gates.
        Identical to ``_apply_trotter_step_pennylane`` in the CPU module.
        """
        qml = self.qml
        for c, p_str in zip(coeffs, pauli_strings):
            c = float(c)
            if abs(c) < 1e-8:
                continue
            if all(ch == "I" for ch in p_str):
                continue  # global phase only
            theta = 2.0 * c * dt_scaled

            active_chars = []
            active_wires = []
            for wire_idx, ch in enumerate(p_str):
                if ch != "I":
                    active_chars.append(ch)
                    active_wires.append(wire_idx)

            qml.PauliRot(theta, "".join(active_chars), wires=active_wires)

    def simulate_trotter_trajectory(
        self,
        t_grid_au: np.ndarray,
        e_fields_au: np.ndarray,
    ) -> np.ndarray:
        """
        Incremental (O(n_steps)) Trotter propagation on the selected device.

        Returns ``(n_steps, 2 ** n_qubits)`` populations.
        """
        if not self.available:
            raise RuntimeError(f"PennyLane GPU backend unavailable: {self.init_error}")

        qml = self.qml
        t_grid = np.asarray(t_grid_au, dtype=float)
        fields = np.asarray(e_fields_au, dtype=float)
        n_steps = int(t_grid.size)
        dim = self.dim

        populations = np.zeros((n_steps, dim), dtype=float)
        populations[0, 0] = 1.0
        if n_steps < 2:
            return populations

        dt_au = float(t_grid[1] - t_grid[0])
        dt_scaled = dt_au * self.energy_to_hartree

        qnode_kwargs: Dict[str, Any] = {"diff_method": None}
        if self.interface is not None:
            qnode_kwargs["interface"] = self.interface

        wires = list(range(self.n_qubits))
        state_prep = getattr(qml, "StatePrep", None) or getattr(qml, "QubitStateVector")

        # The per-slice Pauli data is passed through a closure rather than as a
        # QNode argument: PennyLane's interface layer tries to tensor-ify every
        # positional argument, and a list of Pauli strings is not tensor-able.
        slice_data: Dict[str, Any] = {"coeffs": None, "paulis": None}

        @qml.qnode(self.dev, **qnode_kwargs)
        def step_circuit(state_in):
            state_prep(state_in, wires=wires)
            self._apply_trotter_step(slice_data["coeffs"], slice_data["paulis"], dt_scaled)
            return qml.state()

        # Initial |0...0> state, placed on the GPU when using the torch interface.
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        if self.interface == "torch":
            torch = _import_torch()
            state = torch.as_tensor(
                state, dtype=torch.complex128, device=self.torch_device
            )

        for k in range(n_steps - 1):
            coeffs_k, paulis_k = self.mapper.evaluate_hamiltonian_paulis_at_t(fields[k])
            slice_data["coeffs"] = np.asarray(coeffs_k, dtype=float)
            slice_data["paulis"] = paulis_k
            state = step_circuit(state)

            if self.interface == "torch":
                probs = (state.real ** 2 + state.imag ** 2).detach().cpu().numpy()
            else:
                probs = np.abs(np.asarray(state)) ** 2

            total = probs.sum()
            populations[k + 1, :] = probs / total if total > 1e-30 else probs

        return populations

    def summary(self) -> str:
        if not self.available:
            return f"PennyLaneGPUTimeEvolution(unavailable: {self.init_error})"
        return (
            f"PennyLaneGPUTimeEvolution(device={self.device_name}, "
            f"interface={self.interface or 'autograd'}, wires={self.n_qubits})"
        )


# ============================================================================
# Section 4: Benchmark helper
# ============================================================================

def benchmark_backends(
    mapper: MultiStatePauliMapper,
    t_grid_au: np.ndarray,
    e_fields_batch: np.ndarray,
    include_pennylane: bool = False,
) -> Dict[str, Any]:
    """
    Times the available backends on the same workload and reports throughput.

    ``e_fields_batch`` should be ``(n_batch, n_steps, 3)``.  Returns a dict of
    ``{backend_name: {"seconds": float, "trajectories_per_second": float}}``.
    """
    import time

    fields = np.asarray(e_fields_batch, dtype=float)
    if fields.ndim == 2:
        fields = fields[None, ...]
    n_batch = fields.shape[0]

    results: Dict[str, Any] = {}

    def _time(label: str, fn) -> None:
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        results[label] = {
            "seconds": elapsed,
            "trajectories_per_second": n_batch / elapsed if elapsed > 0 else float("inf"),
            "final_ground_state_population": float(np.asarray(out).reshape(n_batch, -1, 2 ** mapper.n_qubits)[0, -1, 0]),
        }

    cpu_engine = TorchTimeEvolution(mapper, device="cpu", precision="double", quiet=True)
    if cpu_engine.available:
        # Warm-up excluded from the timing.
        cpu_engine.simulate_batch(t_grid_au[:8], fields[:1, :8], method="trotter")
        _time("torch-cpu", lambda: cpu_engine.simulate_batch(t_grid_au, fields, method="trotter"))

    if gpu_is_available():
        gpu_engine = TorchTimeEvolution(mapper, device="cuda", quiet=True)
        if gpu_engine.available:
            gpu_engine.simulate_batch(t_grid_au[:8], fields[:1, :8], method="trotter")
            gpu_engine.backend.synchronize()
            _time("torch-cuda", lambda: gpu_engine.simulate_batch(t_grid_au, fields, method="trotter"))

    if include_pennylane:
        pl_engine = PennyLaneGPUTimeEvolution(mapper, quiet=True)
        if pl_engine.available:
            _time(
                f"pennylane-{pl_engine.device_name}",
                lambda: np.stack(
                    [pl_engine.simulate_trotter_trajectory(t_grid_au, fields[i]) for i in range(n_batch)]
                ),
            )

    return results


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="GPU time-evolution backend diagnostics")
    parser.add_argument("--species", type=str, default="Be")
    parser.add_argument("--batch", type=int, default=32, help="Number of trajectories to benchmark")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--pennylane", action="store_true", help="Also time the PennyLane circuit path")
    args = parser.parse_args()

    print("=== GPU environment ===")
    print(json.dumps(describe_gpu_environment(), indent=2))

    path = f"data/raw_pyscf/manifold_{args.species}.json"
    if not os.path.exists(path):
        print(f"\n[!] {path} not found - run pyscf_runner.py first to benchmark.")
        raise SystemExit(0)

    from src.collision import electric_field, generate_trajectory, load_manifold

    manifold = load_manifold(path)
    mapper = MultiStatePauliMapper(manifold)
    charge = manifold.get("charge", 0)

    t_grid = np.linspace(-15.0, 15.0, args.steps)
    b_values = np.linspace(0.5, 10.0, args.batch)
    traj_type = "coulomb" if charge > 0 else "straight"
    fields = np.stack(
        [
            electric_field(
                generate_trajectory(
                    time=t_grid,
                    impact_parameter_bohr=float(b),
                    incident_energy_ev=50.0,
                    trajectory_type=traj_type,
                    ion_charge=charge,
                ).position
            )
            for b in b_values
        ]
    )

    print(f"\n=== Benchmark: {args.species}, {args.batch} trajectories x {args.steps} steps ===")
    for name, stats in benchmark_backends(
        mapper, t_grid, fields, include_pennylane=args.pennylane
    ).items():
        print(
            f"  {name:<28} {stats['seconds']:8.3f} s   "
            f"{stats['trajectories_per_second']:10.1f} traj/s   "
            f"P0(final)={stats['final_ground_state_population']:.6f}"
        )
