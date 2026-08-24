r"""
Self-check for the GPU time-evolution backend.

Run this once after installing torch.  It answers three questions:

  1. Does the GPU backend reproduce the NumPy reference propagator?
     (exact-vs-exact should agree to ~1e-6; trotter-vs-exact differs by the
     genuine Trotter error, which is reported, not asserted away.)
  2. Does the GPU Trotter product reproduce the actual PennyLane circuit?
  3. Is one batched call identical to propagating each trajectory separately?

Usage
-----
    python scripts/verify_gpu_backend.py
    python scripts/verify_gpu_backend.py --species Fe22+ --steps 500 --batch 12
    python scripts/verify_gpu_backend.py --bench        # also time the backends
"""

from __future__ import annotations

_MODULE_VERSION = "gpu-routed-v2"

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collision import load_manifold  # noqa: E402
from src.quantum.mapper import MultiStatePauliMapper  # noqa: E402
from src.quantum.time_evolution import (  # noqa: E402
    PennyLaneTimeEvolution,
    build_field_batch,
    compute_collision_electric_field,
    simulate_exact_unitary_evolution,
)
from src.quantum.time_evolution_gpu import (  # noqa: E402
    PennyLaneGPUTimeEvolution,
    TorchTimeEvolution,
    benchmark_backends,
    describe_gpu_environment,
    gpu_is_available,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[ ** ]"

_failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _failures
    tag = PASS if condition else FAIL
    if not condition:
        _failures += 1
    print(f"  {tag} {label}" + (f"   {detail}" if detail else ""))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the GPU time-evolution backend")
    parser.add_argument("--species", type=str, default="Be")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--impact-b", type=float, default=2.0)
    parser.add_argument("--energy", type=float, default=50.0)
    parser.add_argument("--bench", action="store_true", help="Also run a timing comparison")
    parser.add_argument("--tol", type=float, default=2e-5, help="Tolerance for exact-vs-exact agreement")
    args = parser.parse_args()

    print("=" * 78)
    print("GPU ENVIRONMENT")
    print("=" * 78)
    env = describe_gpu_environment()
    print(json.dumps(env, indent=2))

    if not env["torch_installed"]:
        print(
            "\n[!] PyTorch is not installed. Install it, then re-run:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
        return 1
    if not env["cuda_available"]:
        print(
            "\n[!] No CUDA device visible. The checks below will still run on CPU "
            "tensors and validate correctness, but there will be no speedup."
        )

    path = f"data/raw_pyscf/manifold_{args.species}.json"
    if not os.path.exists(path):
        print(f"\n[!] {path} not found. Run the PySCF stage first.")
        return 1

    manifold = load_manifold(path)
    mapper = MultiStatePauliMapper(manifold)
    charge = manifold.get("charge", 0)
    traj_type = "coulomb" if charge > 0 else "straight"
    dim = 2 ** mapper.n_qubits

    print("\n" + "=" * 78)
    print(f"MANIFOLD: {mapper.species}   N={mapper.n_states} states -> "
          f"{mapper.n_qubits} qubits (dim {dim}), {len(mapper.active_paulis)} Pauli terms")
    print("=" * 78)

    t_grid = np.linspace(-15.0, 15.0, args.steps)
    e_fields = compute_collision_electric_field(
        t_grid,
        impact_parameter_bohr=args.impact_b,
        incident_energy_ev=args.energy,
        trajectory_type=traj_type,
        ion_charge=charge,
    )

    device = "cuda" if gpu_is_available() else "cpu"
    engine = TorchTimeEvolution(mapper, device=device, precision="double")
    if not engine.available:
        print(f"[!] Torch engine unavailable: {engine.init_error}")
        return 1
    print(f"{INFO} {engine.summary()}")

    # ---------------------------------------------------------------- 1
    print("\n--- 1. Exact propagator: GPU vs NumPy reference ---")
    ref_exact = simulate_exact_unitary_evolution(mapper, t_grid, e_fields)
    gpu_exact = engine.simulate_batch(t_grid, e_fields, method="exact")

    d = max_abs_diff(ref_exact, gpu_exact)
    check("populations agree at every time step", d < args.tol, f"max|dP| = {d:.3e}")
    check(
        "unitarity preserved (sum_i P_i == 1)",
        bool(np.allclose(gpu_exact.sum(axis=-1), 1.0, atol=1e-9)),
        f"max deviation = {float(np.max(np.abs(gpu_exact.sum(axis=-1) - 1.0))):.3e}",
    )
    padded = gpu_exact[:, mapper.n_states:]
    if padded.size == 0:
        # N is already a power of 2 (e.g. Fe22+/Be/C2+ with N=4 -> dim=4): there
        # are no padded qubit states at all, so there is nothing to check.
        print(f"  {INFO} no padded states to check (n_states == dim == {mapper.n_states})")
    else:
        check(
            "no population leaks into padded unphysical states",
            float(np.max(padded)) < 1e-2,
            f"max padded population = {float(np.max(padded)):.3e}",
        )

    # ---------------------------------------------------------------- 2
    print("\n--- 2. Trotter product: GPU vs PennyLane circuit ---")
    gpu_trotter = engine.simulate_batch(t_grid, e_fields, method="trotter")

    trotter_error = max_abs_diff(gpu_trotter, gpu_exact)
    print(f"{INFO} Trotter-vs-exact error (physics, not a bug): max|dP| = {trotter_error:.3e}")

    pl_cpu = PennyLaneTimeEvolution(mapper)
    if pl_cpu.has_pennylane:
        pl_pops = pl_cpu.simulate_trotter_trajectory(t_grid, e_fields)
        d = max_abs_diff(pl_pops, gpu_trotter)
        check(
            f"GPU Trotter == PennyLane {pl_cpu.device_name} circuit",
            d < 1e-4,
            f"max|dP| = {d:.3e}",
        )
    else:
        print(f"{INFO} PennyLane unavailable ({pl_cpu.init_error}) - circuit cross-check skipped.")

    pl_gpu = PennyLaneGPUTimeEvolution(mapper, quiet=True)
    if pl_gpu.available:
        print(f"{INFO} {pl_gpu.summary()}")
        pl_gpu_pops = pl_gpu.simulate_trotter_trajectory(t_grid, e_fields)
        d = max_abs_diff(pl_gpu_pops, gpu_trotter)
        check("GPU Trotter == PennyLane GPU-device circuit", d < 1e-4, f"max|dP| = {d:.3e}")
    else:
        print(f"{INFO} PennyLane GPU device unavailable ({pl_gpu.init_error}) - skipped.")

    # ---------------------------------------------------------------- 3
    print("\n--- 3. Batched sweep == serial propagation ---")
    b_grid = np.linspace(0.5, 10.0, args.batch)
    fields = build_field_batch(
        t_grid, b_grid, incident_energy_ev=args.energy,
        trajectory_type=traj_type, ion_charge=charge,
    )
    batched = engine.simulate_batch(t_grid, fields, method="trotter")
    serial = np.stack(
        [engine.simulate_batch(t_grid, fields[i], method="trotter") for i in range(args.batch)]
    )
    d = max_abs_diff(batched, serial)
    check(f"{args.batch} trajectories batched vs one-at-a-time", d < 1e-9, f"max|dP| = {d:.3e}")

    # ---------------------------------------------------------------- 4
    print("\n--- 4. Parallel prefix scan == sequential product ---")
    seq_engine = TorchTimeEvolution(
        mapper, device=device, precision="double", scan_mode="sequential", quiet=True
    )
    seq = seq_engine.simulate_batch(t_grid, fields, method="trotter")
    d = max_abs_diff(batched, seq)
    check("log-depth scan reproduces the sequential time-ordered product",
          d < 1e-8, f"max|dP| = {d:.3e}")

    # ---------------------------------------------------------------- 5
    print("\n--- 5. Single precision (complex64) accuracy on the GPU ---")
    single = TorchTimeEvolution(mapper, device=device, precision="single", quiet=True)
    if single.available:
        s_pops = single.simulate_batch(t_grid, fields, method="trotter")
        d = max_abs_diff(s_pops, batched)
        check(
            "complex64 within 1e-4 of complex128 (well under Trotter error)",
            d < 1e-4,
            f"max|dP| = {d:.3e}",
        )

    # ---------------------------------------------------------------- bench
    if args.bench:
        print("\n" + "=" * 78)
        print(f"BENCHMARK: {args.batch} trajectories x {args.steps} steps")
        print("=" * 78)
        for name, stats in benchmark_backends(
            mapper, t_grid, fields, include_pennylane=True
        ).items():
            print(
                f"  {name:<30} {stats['seconds']:8.3f} s   "
                f"{stats['trajectories_per_second']:9.1f} traj/s"
            )

    print("\n" + "=" * 78)
    if _failures == 0:
        print("ALL CHECKS PASSED - the GPU backend matches the CPU reference.")
    else:
        print(f"{_failures} CHECK(S) FAILED - see [FAIL] lines above.")
    print("=" * 78)
    return 0 if _failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())