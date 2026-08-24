r"""
Localize a CPU-vs-GPU population mismatch.

The point of this script is to answer ONE question: which stage diverges?

It builds an independent, dependency-free NumPy Trotter reference -- the same
gate sequence PennyLane would queue, multiplied out by hand in float64 -- and
compares it against every other path.  That reference is the arbiter:

    * GPU == numpy_trotter, but PennyLane differs
          -> the GPU is fine; the PennyLane path is the odd one out.
             Most likely it silently fell back to the exact propagator
             (look for a [WARNING] line), or it is skipping |c| < 1e-8 terms.

    * GPU != numpy_trotter
          -> the GPU tensor path is wrong. The per-stage output below says
             whether it is the step unitary, the prefix scan, or precision.

    * everything agrees except torch:cuda
          -> the GPU is computing garbage. On an RTX 50-series card this is
             almost always missing sm_120 kernels; check the arch line printed
             at the top.

Usage
-----
    python scripts/diagnose_mismatch.py
    python scripts/diagnose_mismatch.py --species Fe22+ --steps 200 --energy 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.collision import load_manifold  # noqa: E402
from src.quantum.mapper import (  # noqa: E402
    MultiStatePauliMapper,
    build_pauli_string_matrix,
    reconstruct_matrix_from_paulis,
)
from src.quantum.time_evolution import (  # noqa: E402
    HAMILTONIAN_ENERGY_TO_HARTREE,
    PennyLaneTimeEvolution,
    collision_time_grid,
    compute_collision_electric_field,
    simulate_exact_unitary_evolution,
    trajectory_type_for_charge,
)
from src.quantum.time_evolution_gpu import (  # noqa: E402
    TorchTimeEvolution,
    cuda_arch_status,
    gpu_is_available,
)


# ---------------------------------------------------------------------------
# The arbiter: a hand-rolled NumPy Trotter product, no torch, no PennyLane.
# ---------------------------------------------------------------------------

def numpy_trotter(
    mapper: MultiStatePauliMapper,
    t_grid_au: np.ndarray,
    e_fields_au: np.ndarray,
    energy_to_hartree: float = HAMILTONIAN_ENERGY_TO_HARTREE,
    skip_tiny: bool = False,
) -> np.ndarray:
    r"""
    U_k = R_{M-1} ... R_1 R_0,  R_m = cos(c_m dt) I - i sin(c_m dt) P_m

    Ascending m is applied first-to-last, i.e. left-multiplied, matching the
    order PennyLane queues gates in.  ``skip_tiny`` mimics the PennyLane path's
    ``if abs(c) < 1e-8: continue``.
    """
    paulis = list(mapper.active_paulis)
    P = [build_pauli_string_matrix(p) for p in paulis]
    is_identity = [all(ch == "I" for ch in p) for p in paulis]

    dim = 2 ** mapper.n_qubits
    eye = np.eye(dim, dtype=complex)
    n_steps = len(t_grid_au)
    dt_h = (t_grid_au[1] - t_grid_au[0]) * energy_to_hartree

    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    pops = np.zeros((n_steps, dim), dtype=float)
    pops[0, 0] = 1.0

    for k in range(n_steps - 1):
        coeffs, _ = mapper.evaluate_hamiltonian_paulis_at_t(e_fields_au[k])
        U = eye.copy()
        for m in range(len(paulis)):
            if is_identity[m]:
                continue
            c = float(coeffs[m])
            if skip_tiny and abs(c) < 1e-8:
                continue
            half = c * dt_h
            U = (np.cos(half) * eye - 1j * np.sin(half) * P[m]) @ U
        psi = U @ psi
        probs = np.abs(psi) ** 2
        pops[k + 1] = probs / probs.sum()

    return pops


def first_divergence(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    """Index of the first time step where two population arrays differ by > tol."""
    d = np.max(np.abs(a - b), axis=-1)
    bad = np.nonzero(d > tol)[0]
    return int(bad[0]) if bad.size else -1


def main() -> int:
    p = argparse.ArgumentParser(description="Localize a CPU/GPU mismatch")
    p.add_argument("--species", type=str, default="Be")
    p.add_argument("--energy", type=float, default=50.0)
    p.add_argument("--impact-b", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    path = f"data/raw_pyscf/manifold_{args.species}.json"
    if not os.path.exists(path):
        print(f"[!] {path} not found.")
        return 1

    manifold = load_manifold(path)
    mapper = MultiStatePauliMapper(manifold)
    charge = manifold.get("charge", 0)
    dim = 2 ** mapper.n_qubits

    t_grid = collision_time_grid(args.energy, args.steps)
    fields = compute_collision_electric_field(
        t_grid,
        impact_parameter_bohr=args.impact_b,
        incident_energy_ev=args.energy,
        trajectory_type=trajectory_type_for_charge(charge),
        ion_charge=charge,
    )

    print("=" * 78)
    print(f"{mapper.species}:  N={mapper.n_states}  n_qubits={mapper.n_qubits}  dim={dim}")
    print(f"  Pauli terms      : {len(mapper.active_paulis)}  -> {', '.join(mapper.active_paulis)}")
    print(f"  E_inc            : {args.energy} eV      b = {args.impact_b} a0")
    print(f"  time window      : [{t_grid[0]:.3f}, {t_grid[-1]:.3f}] a.u., "
          f"{args.steps} steps, dt = {t_grid[1] - t_grid[0]:.6f}")
    print(f"  energy_to_hartree: {HAMILTONIAN_ENERGY_TO_HARTREE}")
    print(f"  |c| range        : {np.abs(mapper.vec_h0).max():.6f} (H0), "
          f"max|field| = {np.abs(fields).max():.6f} a.u.")
    arch_ok, arch_problem = cuda_arch_status()
    print(f"  CUDA usable      : {arch_ok}")
    if arch_problem:
        print("    " + arch_problem.replace("\n", "\n    "))
    print("=" * 78)

    # ---------------------------------------------------------------- refs
    results = {}
    results["numpy_trotter"] = numpy_trotter(mapper, t_grid, fields)
    results["numpy_trotter_skiptiny"] = numpy_trotter(mapper, t_grid, fields, skip_tiny=True)
    results["numpy_exact"] = simulate_exact_unitary_evolution(mapper, t_grid, fields)

    # ---------------------------------------------------------------- torch
    cpu_engine = TorchTimeEvolution(
        mapper, device="cpu", precision="double",
        energy_to_hartree=HAMILTONIAN_ENERGY_TO_HARTREE, quiet=True,
    )
    if cpu_engine.available:
        results["torch_cpu_f64_trotter"] = cpu_engine.simulate_batch(t_grid, fields, method="trotter")
        results["torch_cpu_f64_exact"] = cpu_engine.simulate_batch(t_grid, fields, method="exact")

        seq = TorchTimeEvolution(
            mapper, device="cpu", precision="double", scan_mode="sequential",
            energy_to_hartree=HAMILTONIAN_ENERGY_TO_HARTREE, quiet=True,
        )
        results["torch_cpu_f64_seqscan"] = seq.simulate_batch(t_grid, fields, method="trotter")

    if gpu_is_available():
        for prec, tag in (("double", "f64"), ("single", "f32")):
            eng = TorchTimeEvolution(
                mapper, device="cuda", precision=prec,
                energy_to_hartree=HAMILTONIAN_ENERGY_TO_HARTREE, quiet=True,
            )
            if eng.available and eng.backend.is_cuda:
                results[f"torch_cuda_{tag}_trotter"] = eng.simulate_batch(
                    t_grid, fields, method="trotter"
                )

    # ---------------------------------------------------------------- pennylane
    pl = PennyLaneTimeEvolution(mapper)
    if pl.has_pennylane:
        results[f"pennylane_{pl.device_name}"] = pl.simulate_trotter_trajectory(t_grid, fields)
    else:
        print(f"[ ** ] PennyLane unavailable: {pl.init_error}")

    # ---------------------------------------------------------------- report
    ref = results["numpy_trotter"]
    print("\n--- Everything vs the NumPy Trotter arbiter ---")
    print(f"{'path':<30} {'max|dP|':>12}  {'1st bad step':>12}  {'P0(final)':>10}")
    for name, pops in results.items():
        d = float(np.max(np.abs(pops - ref)))
        idx = first_divergence(pops, ref, args.tol)
        print(f"{name:<30} {d:12.3e}  {idx:>12}  {pops[-1, 0]:10.6f}")

    print("\n--- Final populations ---")
    header = "  ".join(f"P{i}" .rjust(9) for i in range(mapper.n_states))
    print(f"{'path':<30} {header}")
    for name, pops in results.items():
        row = "  ".join(f"{pops[-1, i]:9.6f}" for i in range(mapper.n_states))
        print(f"{name:<30} {row}")

    # ---------------------------------------------------------------- verdict
    print("\n--- Reading it ---")
    trot_vs_exact = float(np.max(np.abs(results["numpy_trotter"] - results["numpy_exact"])))
    print(f"  Genuine Trotter error (numpy_trotter vs numpy_exact): {trot_vs_exact:.3e}")
    print("  Any path within that of the arbiter is fine. A path matching")
    print("  numpy_exact instead of numpy_trotter has fallen back to the exact")
    print("  propagator -- for PennyLane that means the circuit raised and was")
    print("  swallowed by the try/except; scroll up for its [WARNING] line.")

    if "torch_cpu_f64_trotter" in results and "torch_cuda_f64_trotter" in results:
        d = float(np.max(np.abs(
            results["torch_cuda_f64_trotter"] - results["torch_cpu_f64_trotter"]
        )))
        print(f"\n  torch cuda(f64) vs torch cpu(f64): {d:.3e}")
        if d > 1e-9:
            print("  Same code, same dtype, different device -> the GPU itself is")
            print("  computing incorrectly. Check the CUDA-usable line above.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
