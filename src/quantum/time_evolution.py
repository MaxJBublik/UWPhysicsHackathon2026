r"""
Quantum Time-Evolution Circuit and Multi-State Population Dynamics Engine.

Task 2.2: Simulates the time-dependent unitary evolution of the atomic/ionic
system under an electron-impact collision pulse:
    i \frac{\partial}{\partial t} |\psi(t)\rangle = H(t) |\psi(t)\rangle
    where H(t) = \sum_k c_k(t) \hat{P}_k is mapped in the n-qubit Pauli basis.

Key Features:
  1. Trotterized Quantum Circuit (PennyLane):
     - Discretizes time into slices \Delta t.
     - Implements unitary step U(t_k+\Delta t, t_k) \approx \prod_m \exp(-i c_m(t_k) \hat{P}_m \Delta t)
       via parameterized Pauli rotations (qml.PauliRot).
  2. GPU-Accelerated Backend (PyTorch / CUDA):
     - src/quantum/time_evolution_gpu.py reproduces the same Trotter product and
       the same exact propagator as batched tensor algebra, with a
       logarithmic-depth parallel scan over the time axis.
     - Selected automatically when a CUDA device and PyTorch are present.
  3. Multi-State Population Tracking:
     - Measures probability distributions P_i(t) = |\langle i | \psi(t) \rangle|^2
       at every time step.
  4. High-Precision Reference Solver:
     - Exact matrix exponential unitary propagation for benchmarking Trotter error.
  5. Batch Parameter Sweeps:
     - Sweeps across impact parameters b and incident energies E_inc,
       exporting data to data/processed_circuits/populations_{species}.json.
     - On a batching backend every impact parameter at one energy is propagated
       in a single call; without torch it falls back to the process pool.

UNITS
-----
``HAMILTONIAN_ENERGY_TO_HARTREE`` is the single knob that decides how the
mapper's Pauli coefficients are interpreted before being multiplied by dt in
atomic units.  It must agree with what ``MultiStatePauliMapper`` actually
produces:

    1.0             mapper returns Hartree / atomic units   <-- current setting
    EV_TO_HARTREE   mapper returns eV

As shipped, ``mapper.py`` builds ``H0_matrix`` from ``excitation_energies_ev``
(eV) while the dipole terms are already atomic units, so the two halves of
H(t) = H0 - mu.E do not share a unit system.  Fix that in ``mapper.py`` -
convert H0 to a.u. at construction - and leave this constant at 1.0.  Do not
"fix" it by changing this constant: that only moves the error from one term to
the other.
"""

# VERSION STAMP: gpu-routed-v2. If a copy of this file is missing this line,
# it is stale -- re-sync it before debugging anything else.
_MODULE_VERSION = "gpu-routed-v2"

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import concurrent.futures
from functools import partial

from src.collision import (
    load_manifold,
    generate_trajectory,
    electric_field,
)
from src.collision.trajectories import incident_velocity, TrajectoryResult

from src.quantum.mapper import (
    MultiStatePauliMapper,
    get_n_qubits,
    build_pauli_string_matrix,
    reconstruct_matrix_from_paulis,
)

# GPU backend.  Importing this module does NOT import torch; torch is only
# touched when a backend is actually requested.
try:
    from src.quantum.time_evolution_gpu import (
        TorchTimeEvolution,
        PennyLaneGPUTimeEvolution,
        gpu_is_available,
        cuda_arch_status,
        describe_gpu_environment,
    )
    _GPU_MODULE_AVAILABLE = True
    _GPU_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - defensive
    TorchTimeEvolution = None
    PennyLaneGPUTimeEvolution = None
    describe_gpu_environment = None
    cuda_arch_status = None
    _GPU_MODULE_AVAILABLE = False
    _GPU_IMPORT_ERROR = str(_exc)

    def gpu_is_available() -> bool:
        return False


# Physical conversion factors (Atomic units: \hbar = m_e = e = 1)
EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_EV = 27.211386245988
TIME_AU_TO_FEMTOSECONDS = 0.024188843265857

# See the UNITS note in the module docstring before touching this.
HAMILTONIAN_ENERGY_TO_HARTREE = 1.0

# Collision-window and field-regularisation defaults, shared by every backend
# so the CPU and GPU paths always see byte-identical inputs.
DEFAULT_SOFTENING_BOHR = 0.2
MIN_HALF_WINDOW_AU = 15.0
WINDOW_LENGTH_BOHR = 30.0


# ============================================================================
# Section 1: Collision Geometry, Time Grid, and Field Helpers
# ============================================================================

def collision_time_grid(
    incident_energy_ev: float,
    n_time_steps: int = 500,
    min_half_window_au: float = MIN_HALF_WINDOW_AU,
    window_length_bohr: float = WINDOW_LENGTH_BOHR,
) -> np.ndarray:
    r"""
    Energy-adaptive symmetric time grid, in atomic units.

    A fast projectile crosses the interaction region sooner, so the window is
    scaled by the asymptotic speed: ``t_boundary = max(15, 30 / v)``.  Every
    backend must use this function, otherwise the GPU and CPU paths silently
    integrate over different windows.

    Note that this makes ``dt`` energy-dependent, which is why the (E, b) sweep
    batches over b at fixed E rather than over the whole grid at once.
    """
    v_au = incident_velocity(incident_energy_ev)
    t_boundary = max(float(min_half_window_au), float(window_length_bohr) / v_au)
    return np.linspace(-t_boundary, t_boundary, int(n_time_steps))


def trajectory_type_for_charge(charge: float) -> str:
    """Coulomb-hyperbolic orbit for charged targets, straight line for neutrals."""
    return "coulomb" if charge > 0 else "straight"


def compute_collision_electric_field(
    t_grid_au: np.ndarray,
    impact_parameter_bohr: float,
    incident_energy_ev: float,
    trajectory_type: str = "straight",
    ion_charge: float = 0.0,
    softening_bohr: float = DEFAULT_SOFTENING_BOHR,
) -> np.ndarray:
    r"""
    Electric field E(t) seen by the target as the projectile electron flies past.

    Thin wrapper over ``generate_trajectory`` + ``electric_field`` so there is
    exactly one place that defines the field, and one entry point for callers
    (``app.py`` and ``tests/test_time_evolution.py`` both import this).

    Returns
    -------
    e_fields : np.ndarray of shape (len(t_grid_au), 3)
    """
    trajectory = generate_trajectory(
        time=np.asarray(t_grid_au, dtype=float),
        impact_parameter_bohr=float(impact_parameter_bohr),
        incident_energy_ev=float(incident_energy_ev),
        trajectory_type=trajectory_type,
        ion_charge=float(ion_charge),
    )
    return electric_field(trajectory.position, softening_bohr=softening_bohr)


def build_field_batch(
    t_grid_au: np.ndarray,
    impact_parameters_bohr: np.ndarray,
    incident_energy_ev: float,
    trajectory_type: str = "straight",
    ion_charge: float = 0.0,
    softening_bohr: float = DEFAULT_SOFTENING_BOHR,
) -> np.ndarray:
    r"""
    Stacks the fields for a whole impact-parameter sweep at ONE energy into an
    array of shape ``(n_b, n_steps, 3)``, ready for a single batched GPU pass.
    """
    return np.stack(
        [
            compute_collision_electric_field(
                t_grid_au,
                impact_parameter_bohr=float(b),
                incident_energy_ev=incident_energy_ev,
                trajectory_type=trajectory_type,
                ion_charge=ion_charge,
                softening_bohr=softening_bohr,
            )
            for b in np.asarray(impact_parameters_bohr, dtype=float)
        ],
        axis=0,
    )


# ============================================================================
# Section 2: PennyLane Trotterized Quantum Circuit Engine (CPU)
# ============================================================================

class PennyLaneTimeEvolution:
    """
    Implements multi-step Trotterized quantum time evolution using PennyLane.

    Note on the O(n_steps) rewrite
    ------------------------------
    The original implementation called ``circuit_at_step(step)`` once per time
    step, and every call replayed the circuit from t_0.  That is quadratic: a
    500-step run queued ~125,000 Trotter slices (~2 million gate applications).
    The state is now carried forward between slices with ``qml.StatePrep``, so a
    500-step run queues 500 slices - identical physics, ~250x fewer gates.
    """

    def __init__(
        self,
        mapper: MultiStatePauliMapper,
        shots: Optional[int] = None,
        device_name: Optional[str] = None,
    ):
        self.mapper = mapper
        self.n_qubits = mapper.n_qubits
        self.shots = shots
        self.has_pennylane = False
        self.init_error = None
        self.device_name = None

        try:
            import pennylane as qml
            self.qml = qml
            self.has_pennylane = True

            candidates = [device_name] if device_name else []
            candidates += ["lightning.qubit", "default.qubit"]

            errors = []
            for name in candidates:
                if name is None:
                    continue
                try:
                    self.dev = qml.device(name, wires=self.n_qubits, shots=shots)
                    self.device_name = name
                    break
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
            else:
                raise RuntimeError("; ".join(errors))

        except Exception as e:
            self.qml = None
            self.dev = None
            self.has_pennylane = False
            self.init_error = str(e)

    def _apply_trotter_step_pennylane(self, coeffs: np.ndarray, pauli_strings: List[str], dt_h: float):
        r"""
        Applies a single Trotter slice: \prod_k \exp(-i * c_k * \hat{P}_k * \Delta t)
        using PennyLane parameterized Pauli rotation gates (qml.PauliRot).

        ``dt_h`` is dt already expressed in the same energy units as ``coeffs``
        (see HAMILTONIAN_ENERGY_TO_HARTREE).
        """
        qml = self.qml

        for c, p_str in zip(coeffs, pauli_strings):
            if abs(c) < 1e-8:
                continue
            # PennyLane qml.PauliRot applies exp(-i * theta / 2 * P), so theta = 2 * c * dt
            theta = 2.0 * float(c) * dt_h

            # Check if all Identity
            if all(char == "I" for char in p_str):
                # Global phase shift (does not affect state probabilities)
                continue

            # Extract non-identity wires and pauli characters
            active_chars = []
            active_wires = []
            for wire_idx, char in enumerate(p_str):
                if char != "I":
                    active_chars.append(char)
                    active_wires.append(wire_idx)

            active_word = "".join(active_chars)
            qml.PauliRot(theta, active_word, wires=active_wires)

    def _state_prep_op(self):
        """``qml.StatePrep`` on modern PennyLane, ``QubitStateVector`` before 0.35."""
        qml = self.qml
        return getattr(qml, "StatePrep", None) or getattr(qml, "QubitStateVector")

    def simulate_trotter_trajectory(
        self,
        t_grid_au: np.ndarray,
        e_fields_au: np.ndarray,
    ) -> np.ndarray:
        """
        Runs the full Trotterized time-evolution circuit from t_0 to t_final.
        Measures the state probability vector at each time step.

        Returns
        -------
        populations : np.ndarray of shape (len(t_grid), 2^n_qubits)
        """
        if not self.has_pennylane:
            print(f"[WARNING] PennyLane unavailable ({self.init_error}). Reverting to exact matrix exponential simulation.")
            return simulate_exact_unitary_evolution(self.mapper, t_grid_au, e_fields_au)

        try:
            qml = self.qml
            n_steps = len(t_grid_au)
            dt_au = t_grid_au[1] - t_grid_au[0] if n_steps > 1 else 0.1
            dt_h = dt_au * HAMILTONIAN_ENERGY_TO_HARTREE
            dim = 2 ** self.n_qubits

            populations = np.zeros((n_steps, dim), dtype=float)
            populations[0, 0] = 1.0
            if n_steps < 2:
                return populations

            state_prep = self._state_prep_op()
            wires = list(range(self.n_qubits))

            # Per-slice Pauli data travels through a closure rather than as a
            # QNode argument: PennyLane tries to tensor-ify positional args, and
            # a list of Pauli strings is not tensor-able.
            slice_data = {"coeffs": None, "paulis": None}

            @qml.qnode(self.dev, diff_method=None)
            def trotter_slice(state_in):
                state_prep(state_in, wires=wires)
                self._apply_trotter_step_pennylane(
                    slice_data["coeffs"], slice_data["paulis"], dt_h
                )
                return qml.state()

            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0 + 0.0j

            for k in range(n_steps - 1):
                coeffs_k, paulis_k = self.mapper.evaluate_hamiltonian_paulis_at_t(e_fields_au[k])
                slice_data["coeffs"] = coeffs_k
                slice_data["paulis"] = paulis_k
                state = np.asarray(trotter_slice(state), dtype=complex)

                probs = np.abs(state) ** 2
                total = probs.sum()
                populations[k + 1, :] = probs / total if total > 1e-30 else probs

            return populations

        except Exception as e:
            print(f"[WARNING] PennyLane circuit execution failed with error ({e}). Reverting to exact matrix exponential simulation.")
            return simulate_exact_unitary_evolution(self.mapper, t_grid_au, e_fields_au)


# ============================================================================
# Section 3: Exact Unitary Reference Propagator (Matrix Exponential)
# ============================================================================

def simulate_exact_unitary_evolution(
    mapper: MultiStatePauliMapper,
    t_grid_au: np.ndarray,
    e_fields_au: np.ndarray,
) -> np.ndarray:
    r"""
    Exact matrix exponential propagation for benchmarking and fast simulation:
        |\psi(t + \Delta t)\rangle = \exp(-i * H(t) * \Delta t) |\psi(t)\rangle

    Returns
    -------
    populations : np.ndarray of shape (len(t_grid), 2^n_qubits)
    """
    n_steps = len(t_grid_au)
    dt_au = t_grid_au[1] - t_grid_au[0] if n_steps > 1 else 0.1
    dt_h = dt_au * HAMILTONIAN_ENERGY_TO_HARTREE
    dim = 2 ** mapper.n_qubits

    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 + 0.0j

    populations = np.zeros((n_steps, dim), dtype=float)
    populations[0, :] = np.abs(psi) ** 2

    for k in range(n_steps - 1):
        coeffs_k, paulis_k = mapper.evaluate_hamiltonian_paulis_at_t(e_fields_au[k])
        H_k = reconstruct_matrix_from_paulis(coeffs_k.tolist(), paulis_k)

        # exp(-i * H * dt) = V * diag(exp(-i * lambda * dt)) * V^\dagger
        eigenvalues, eigenvectors = np.linalg.eigh(H_k)
        phase_factors = np.exp(-1j * eigenvalues * dt_h)
        U_step = eigenvectors @ np.diag(phase_factors) @ eigenvectors.conj().T

        psi = U_step @ psi

        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi = psi / norm

        populations[k + 1, :] = np.abs(psi) ** 2

    return populations


# ============================================================================
# Section 4: High-Level Population Simulation Runner & Batch Sweeper
# ============================================================================

def _parallel_worker(args: tuple):
    """Standalone worker to prevent pickling unpicklable module attributes."""
    json_path, b, energy, use_trotter, n_time_steps = args

    # This pool only runs when no batching (torch) backend exists at all, so the
    # child is pinned to the PennyLane / NumPy path.  Several processes fighting
    # over one GPU context would be slower than the batched call it replaces.
    local_sim = CollisionDynamicsSimulator(
        json_path,
        backend="pennylane" if use_trotter else "exact",
        quiet=True,
    )

    return local_sim.run_single_collision(
        impact_parameter_bohr=b,
        incident_energy_ev=energy,
        n_time_steps=n_time_steps,
        use_trotter=use_trotter,
    )


class CollisionDynamicsSimulator:
    """
    End-to-end simulation runner for electron-impact collisional excitation.

    Parameters
    ----------
    manifold_json_path : str
        Path to data/raw_pyscf/manifold_{species}.json
    use_pennylane_if_available : bool
        Kept for backwards compatibility; ignored when ``backend`` is explicit.
    unphysical_penalty_ev : float
        Diagonal energy penalty on padded (unphysical) qubit states.
    backend : {"auto", "gpu", "torch", "pennylane", "exact", "cpu"}
        "auto" prefers the GPU, then PennyLane, then the NumPy propagator.
    device : str
        Torch device for the GPU backend: "auto", "cuda", "cuda:1", "cpu".
    precision : str
        "auto" (complex64 on CUDA, complex128 on CPU), "single", or "double".
    """

    def __init__(
        self,
        manifold_json_path: str,
        use_pennylane_if_available: bool = True,
        unphysical_penalty_ev: float = 100.0,
        backend: str = "auto",
        device: str = "auto",
        precision: str = "auto",
        softening_bohr: float = DEFAULT_SOFTENING_BOHR,
        quiet: bool = False,
    ):
        self.manifold_json_path = manifold_json_path
        self.backend_request = (backend or "auto").strip().lower()
        self.softening_bohr = softening_bohr
        self.quiet = quiet

        self.manifold_data = load_manifold(manifold_json_path)

        self.species = self.manifold_data["species"]
        self.charge = self.manifold_data.get("charge", 0)
        self.mapper = MultiStatePauliMapper(
            self.manifold_data, unphysical_penalty_ev=unphysical_penalty_ev
        )

        self.gpu_engine = None
        self.cpu_batch_engine = None
        self.pennylane_engine = None

        want_gpu = self.backend_request in {"auto", "gpu", "torch", "cuda"}
        want_pennylane = (
            self.backend_request in {"auto", "pennylane"} and use_pennylane_if_available
        )

        # -- GPU / PyTorch backend -----------------------------------------
        if want_gpu and _GPU_MODULE_AVAILABLE:
            engine = TorchTimeEvolution(
                self.mapper,
                device=device,
                precision=precision,
                energy_to_hartree=HAMILTONIAN_ENERGY_TO_HARTREE,
                quiet=quiet,
            )
            if engine.available:
                # In "auto" mode only claim the GPU path when there really is a
                # GPU; a CPU torch loop has no edge over PennyLane for a single
                # trajectory (it does for batched sweeps, handled separately).
                if self.backend_request != "auto" or engine.backend.is_cuda:
                    self.gpu_engine = engine
                else:
                    self.cpu_batch_engine = engine
            elif self.backend_request in {"gpu", "torch", "cuda"}:
                raise RuntimeError(
                    f"backend='{backend}' requested but the PyTorch backend is "
                    f"unavailable: {engine.init_error}"
                )
        elif want_gpu and self.backend_request in {"gpu", "torch", "cuda"}:
            raise RuntimeError(f"GPU backend module failed to import: {_GPU_IMPORT_ERROR}")

        # A CPU torch engine still helps batched sweeps even without CUDA.
        if (
            self.gpu_engine is None
            and self.cpu_batch_engine is None
            and _GPU_MODULE_AVAILABLE
            and self.backend_request in {"auto", "cpu"}
        ):
            engine = TorchTimeEvolution(
                self.mapper,
                device="cpu",
                precision=precision,
                energy_to_hartree=HAMILTONIAN_ENERGY_TO_HARTREE,
                quiet=True,
            )
            self.cpu_batch_engine = engine if engine.available else None

        # -- PennyLane backend ---------------------------------------------
        if self.gpu_engine is None and want_pennylane:
            self.pennylane_engine = PennyLaneTimeEvolution(self.mapper)

        self.active_backend = self._resolve_active_backend()
        if not quiet:
            print(f"[*] CollisionDynamicsSimulator backend: {self.active_backend}")

    # ------------------------------------------------------------------

    def _resolve_active_backend(self) -> str:
        if self.gpu_engine is not None:
            return f"torch:{self.gpu_engine.backend.name}"
        if self.pennylane_engine is not None and self.pennylane_engine.has_pennylane:
            return f"pennylane:{self.pennylane_engine.device_name}"
        if self.cpu_batch_engine is not None:
            return "torch:cpu"
        return "numpy:exact"

    def _batch_engine(self):
        """The engine that can propagate many trajectories in one call, if any."""
        return self.gpu_engine or self.cpu_batch_engine

    def _trajectory_type(self) -> str:
        return trajectory_type_for_charge(self.charge)

    # ------------------------------------------------------------------

    def run_single_collision(
        self,
        impact_parameter_bohr: float,
        incident_energy_ev: float,
        n_time_steps: int = 500,
        use_trotter: bool = True,
    ) -> Dict[str, Any]:
        """
        Simulates a single collision event at a given impact parameter and energy.

        The time window is chosen by :func:`collision_time_grid`, so it adapts to
        the projectile speed rather than being fixed at +/-15 a.u.
        """
        t_grid = collision_time_grid(incident_energy_ev, n_time_steps)

        e_fields = compute_collision_electric_field(
            t_grid,
            impact_parameter_bohr=impact_parameter_bohr,
            incident_energy_ev=incident_energy_ev,
            trajectory_type=self._trajectory_type(),
            ion_charge=self.charge,
            softening_bohr=self.softening_bohr,
        )

        method = "trotter" if use_trotter else "exact"

        if self.gpu_engine is not None:
            populations_all = self.gpu_engine.simulate_batch(t_grid, e_fields, method=method)
            backend_used = f"torch:{self.gpu_engine.backend.name}:{method}"
        elif use_trotter and self.pennylane_engine is not None and self.pennylane_engine.has_pennylane:
            populations_all = self.pennylane_engine.simulate_trotter_trajectory(t_grid, e_fields)
            backend_used = f"pennylane:{self.pennylane_engine.device_name}:trotter"
        elif self.cpu_batch_engine is not None:
            populations_all = self.cpu_batch_engine.simulate_batch(t_grid, e_fields, method=method)
            backend_used = f"torch:cpu:{method}"
        else:
            populations_all = simulate_exact_unitary_evolution(self.mapper, t_grid, e_fields)
            backend_used = "numpy:exact"

        return self._package_result(
            populations_all, t_grid, impact_parameter_bohr, incident_energy_ev, backend_used
        )

    def _package_result(
        self,
        populations_all: np.ndarray,
        t_grid: np.ndarray,
        impact_parameter_bohr: float,
        incident_energy_ev: float,
        backend_used: str,
    ) -> Dict[str, Any]:
        n_manifold = self.mapper.n_states
        state_pop_dict = {
            f"state_{s}": populations_all[:, s].tolist() for s in range(n_manifold)
        }
        total_pop_t = np.sum(populations_all[:, :n_manifold], axis=1)

        return {
            "species": self.species,
            "n_states": n_manifold,
            "incident_energy_ev": float(incident_energy_ev),
            "impact_parameter_bohr": float(impact_parameter_bohr),
            "time_grid_au": t_grid.tolist(),
            "time_grid_fs": (t_grid * TIME_AU_TO_FEMTOSECONDS).tolist(),
            "populations": state_pop_dict,
            "final_populations": [float(populations_all[-1, i]) for i in range(n_manifold)],
            "unitarity_preserved": bool(np.allclose(total_pop_t, 1.0, atol=1e-3)),
            "backend": backend_used,
        }

    # ------------------------------------------------------------------

    def sweep_impact_parameters(
        self,
        incident_energy_ev: float,
        b_min_bohr: float = 0.5,
        b_max_bohr: float = 10.0,
        n_b_points: int = 25,
        output_dir: str = "data/processed_circuits",
        use_trotter: bool = True,
        n_time_steps: int = 500,
        write_json: bool = True,
    ) -> Dict[str, Any]:
        """
        Sweeps the impact parameter at fixed incident energy.

        Every impact parameter shares one time grid (the energy is fixed), so on
        a batching backend the whole sweep is a single call.  Without torch this
        falls back to the original process pool.
        """
        b_grid = np.linspace(b_min_bohr, b_max_bohr, n_b_points)
        final_probs = {f"state_{i}": [] for i in range(self.mapper.n_states)}

        engine = self._batch_engine()
        method = "trotter" if use_trotter else "exact"

        if engine is not None:
            t_grid = collision_time_grid(incident_energy_ev, n_time_steps)
            fields = build_field_batch(
                t_grid,
                b_grid,
                incident_energy_ev=incident_energy_ev,
                trajectory_type=self._trajectory_type(),
                ion_charge=self.charge,
                softening_bohr=self.softening_bohr,
            )
            if not self.quiet:
                print(
                    f"[*] Batched sweep: {n_b_points} trajectories x {n_time_steps} steps "
                    f"in one pass on {engine.backend.name}..."
                )
            populations = engine.simulate_batch(t_grid, fields, method=method)  # (B, T, dim)
            backend_used = f"torch:{engine.backend.name}:{method}"

            for i in range(self.mapper.n_states):
                final_probs[f"state_{i}"] = [float(p) for p in populations[:, -1, i]]

            unitarity = bool(
                np.allclose(
                    np.sum(populations[:, :, : self.mapper.n_states], axis=-1), 1.0, atol=1e-3
                )
            )
        else:
            worker_args = [
                (self.manifold_json_path, b, incident_energy_ev, use_trotter, n_time_steps)
                for b in b_grid
            ]
            if not self.quiet:
                print(f"[*] Running {n_b_points} parallel collision simulations (process pool)...")

            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(_parallel_worker, worker_args))

            for res in results:
                for i in range(self.mapper.n_states):
                    final_probs[f"state_{i}"].append(res["final_populations"][i])

            backend_used = "processpool:" + (results[0]["backend"] if results else "unknown")
            unitarity = all(bool(r["unitarity_preserved"]) for r in results)

        sweep_data = {
            "species": self.species,
            "charge": self.charge,
            "n_states": self.mapper.n_states,
            "incident_energy_ev": float(incident_energy_ev),
            "impact_parameters_bohr": b_grid.tolist(),
            "excitation_probabilities_vs_b": final_probs,
            "backend": backend_used,
            "unitarity_preserved": unitarity,
        }

        if write_json:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir, f"populations_{self.species}_E{int(incident_energy_ev)}eV.json"
            )
            with open(out_path, "w") as f:
                json.dump(sweep_data, f, indent=2)
            print(f"[+] Saved impact parameter sweep to: {out_path}")

        return sweep_data

    # ------------------------------------------------------------------

    def sweep_energies_and_impact_parameters(
        self,
        energies_ev: Union[List[float], np.ndarray],
        b_min_bohr: float = 0.5,
        b_max_bohr: float = 10.0,
        n_b_points: int = 25,
        output_dir: str = "data/processed_circuits",
        use_trotter: bool = True,
        n_time_steps: int = 500,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Two-dimensional (E_inc, b) sweep.

        The time window is energy-dependent (see :func:`collision_time_grid`), so
        dt differs between energies and the whole grid cannot go into one tensor.
        Each energy is therefore one batched call over its impact parameters -
        still O(n_energies) kernel launches instead of O(n_energies x n_b).
        """
        return {
            float(e): self.sweep_impact_parameters(
                incident_energy_ev=float(e),
                b_min_bohr=b_min_bohr,
                b_max_bohr=b_max_bohr,
                n_b_points=n_b_points,
                output_dir=output_dir,
                use_trotter=use_trotter,
                n_time_steps=n_time_steps,
            )
            for e in np.asarray(energies_ev, dtype=float)
        }


# ============================================================================
# Section 5: CLI Entrypoint for Testing and Batch Runs
# ============================================================================

def run_all_species_simulation(
    energy_ev: float = 50.0,
    backend: str = "auto",
    device: str = "auto",
    precision: str = "auto",
):
    """Runs collision dynamics simulation for all 3 species."""
    species_list = ["Be", "C2+", "Fe22+"]
    for sp in species_list:
        path = f"data/raw_pyscf/manifold_{sp}.json"
        if os.path.exists(path):
            print(f"\n================ Running Collision Dynamics for {sp} ================")
            sim = CollisionDynamicsSimulator(
                path, backend=backend, device=device, precision=precision
            )
            res = sim.run_single_collision(
                impact_parameter_bohr=2.0, incident_energy_ev=energy_ev, use_trotter=True
            )
            print(f"[*] Final Populations for {sp} (b=2.0 a0, E={energy_ev} eV) via {res['backend']}:")
            for idx, p in enumerate(res["final_populations"]):
                print(f"    State {idx}: {p * 100:.2f}%")
            sim.sweep_impact_parameters(incident_energy_ev=energy_ev)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Time-Evolution & Population Dynamics")
    parser.add_argument("--species", type=str, default="Be", choices=["Be", "C2+", "Fe22+", "all"])
    parser.add_argument("--energy", type=float, default=50.0, help="Incident electron kinetic energy in eV")
    parser.add_argument("--impact-b", type=float, default=2.0, help="Impact parameter in Bohr (a_0)")
    # Trotter is the default; --no-trotter selects the exact propagator.
    # (The previous `--use-trotter` with action="store_false" inverted its own
    #  meaning: passing the flag turned Trotter OFF.)
    parser.add_argument("--no-trotter", action="store_true", help="Use the exact propagator instead of the Trotter circuit")
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "gpu", "torch", "pennylane", "exact", "cpu"],
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device: auto | cuda | cuda:0 | cpu")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "single", "double"])
    parser.add_argument("--gpu-info", action="store_true", help="Print the detected GPU environment and exit")

    args = parser.parse_args()
    use_trotter = not args.no_trotter

    if args.gpu_info:
        if describe_gpu_environment is None:
            print(f"[!] GPU module unavailable: {_GPU_IMPORT_ERROR}")
        else:
            print(json.dumps(describe_gpu_environment(), indent=2))
        raise SystemExit(0)

    if args.species == "all":
        run_all_species_simulation(
            energy_ev=args.energy, backend=args.backend,
            device=args.device, precision=args.precision,
        )
    else:
        json_path = f"data/raw_pyscf/manifold_{args.species}.json"
        if os.path.exists(json_path):
            sim = CollisionDynamicsSimulator(
                json_path, backend=args.backend, device=args.device, precision=args.precision
            )
            res = sim.run_single_collision(
                impact_parameter_bohr=args.impact_b,
                incident_energy_ev=args.energy,
                use_trotter=use_trotter,
            )
            print(f"\n[*] Single collision simulation results for {args.species} ({res['backend']}):")
            for idx, p in enumerate(res["final_populations"]):
                print(f"    State {idx}: {p * 100:.2f}%")
            print(f"[*] Unitarity preserved: {res['unitarity_preserved']}")
            sim.sweep_impact_parameters(incident_energy_ev=args.energy, use_trotter=use_trotter)
        else:
            print(f"[!] JSON file not found: {json_path}. Run pyscf_runner.py first.")