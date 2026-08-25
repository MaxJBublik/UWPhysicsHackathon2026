r"""
Run the collision dynamics on the GPU and write the results. That's it.

    python scripts/run_gpu.py

Everything (all species x all energies x all impact parameters) is propagated
in batched GPU passes and written to data/processed_circuits/.

Options:
    --species Be C2+ Fe22+      which species (default: all three)
    --energies 15 30 50 ...     incident energies in eV
    --b-points 25               impact parameters per energy
    --steps 500                 time steps per trajectory
    --device cuda               cuda | cuda:1 | cpu
    --precision single          single | double
    --exact                     use the exact propagator instead of Trotter
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# The data/ paths in this project are relative to the repo root, so anchor the
# process there no matter which directory the script was launched from.
os.chdir(REPO_ROOT)

from src.quantum.time_evolution import CollisionDynamicsSimulator  # noqa: E402
from src.quantum.time_evolution_gpu import (  # noqa: E402
    cuda_arch_status,
    describe_gpu_environment,
    gpu_is_available,
)

DEFAULT_ENERGIES = [i for i in range(1, 1001, 10)]


def main() -> int:
    p = argparse.ArgumentParser(description="Run collision dynamics on the GPU")
    p.add_argument("--species", nargs="+", default=["Be", "C2+", "Fe22+"])
    p.add_argument("--energies", nargs="+", type=float, default=DEFAULT_ENERGIES)
    p.add_argument("--b-min", type=float, default=0.5)
    p.add_argument("--b-max", type=float, default=10.0)
    p.add_argument("--b-points", type=int, default=25)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="single", choices=["single", "double", "auto"])
    p.add_argument("--exact", action="store_true", help="Exact propagator instead of Trotter")
    p.add_argument("--out", type=str, default="data/processed_circuits")
    p.add_argument(
        "--allow-cpu", action="store_true",
        help="Run on CPU tensors if the GPU is unusable instead of stopping",
    )
    args = p.parse_args()

    print(f"[*] Repo root: {REPO_ROOT}")

    env = describe_gpu_environment()
    if not env["torch_installed"]:
        print(
            "[!] PyTorch is not installed.\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
        return 1

    if gpu_is_available():
        gpu = env["devices"][0]
        print(
            f"[+] GPU: {gpu['name']} ({gpu['total_memory_gb']} GB, "
            f"sm_{gpu['capability'].replace('.', '')}) - kernels available"
        )
    else:
        _, problem = cuda_arch_status()
        print("\n" + "=" * 76)
        print("[!] The GPU cannot be used with this PyTorch build.")
        if problem:
            print("    " + problem.replace("\n", "\n    "))
        else:
            print("    No CUDA device is visible to PyTorch.")
        print("=" * 76)
        if not args.allow_cpu:
            print(
                "\nStopping rather than quietly producing CPU numbers you would "
                "report as GPU results.\nRe-run with --allow-cpu to compute them on "
                "CPU tensors anyway (same physics, no speedup)."
            )
            return 2
        print("\n[*] --allow-cpu set: continuing on CPU tensors.\n")
        args.device = "cpu"

    use_trotter = not args.exact
    n_traj = len(args.energies) * args.b_points
    grand_total = 0.0

    for species in args.species:
        path = f"data/raw_pyscf/manifold_{species}.json"
        if not os.path.exists(path):
            print(f"[!] {path} not found - skipping {species}.")
            continue

        print(f"\n=== {species} : {n_traj} trajectories x {args.steps} steps ===")
        sim = CollisionDynamicsSimulator(
            path,
            backend="gpu",
            device=args.device,
            precision=args.precision,
        )

        t0 = time.perf_counter()
        sweeps = sim.sweep_energies_and_impact_parameters(
            energies_ev=args.energies,
            b_min_bohr=args.b_min,
            b_max_bohr=args.b_max,
            n_b_points=args.b_points,
            n_time_steps=args.steps,
            output_dir=args.out,
            use_trotter=use_trotter,
        )
        elapsed = time.perf_counter() - t0
        grand_total += elapsed

        print(f"[+] {species}: {n_traj} trajectories in {elapsed:.2f} s "
              f"({n_traj / elapsed:.0f} traj/s)")

        # Compact summary: excitation probability at the closest impact parameter.
        for energy in sorted(sweeps):
            data = sweeps[energy]
            probs = data["excitation_probabilities_vs_b"]
            n_states = data["n_states"]
            p_closest = [probs[f"state_{i}"][0] for i in range(n_states)]
            excited = 1.0 - p_closest[0]
            unitary = "ok" if data["unitarity_preserved"] else "VIOLATED"
            print(
                f"    E={energy:6.1f} eV  b={args.b_min:.2f} a0  "
                f"P_excited={excited * 100:6.2f}%   "
                + "  ".join(f"P{i}={p * 100:5.2f}%" for i, p in enumerate(p_closest))
                + f"   [unitarity {unitary}]"
            )

    print(f"\n[+] Done. {grand_total:.2f} s total. JSON written to {args.out}/")
    print("    Next:  python main.py --energy 50    # cross-sections + scaling laws")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
