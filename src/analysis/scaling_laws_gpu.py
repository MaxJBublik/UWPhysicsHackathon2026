"""
CUDA-accelerated isoelectronic scaling analysis.

Run from the repository root with::

    python src\\analysis\\scaling_laws_gpu.py

The electronic-structure summary and plotting code are shared with
``scaling_laws.py``. Collision sweeps are forced through the PyTorch CUDA
backend and are performed in batches over impact parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.analysis.cross_sections import (  # noqa: E402
    calculate_cross_sections_and_branching_ratios,
    save_processed_circuit_cross_sections,
)
from src.analysis.scaling_laws import (  # noqa: E402
    IsoelectronicScalingAnalyzer,
    generate_scaling_plots,
)
from src.quantum.time_evolution import CollisionDynamicsSimulator  # noqa: E402
from src.quantum.time_evolution_gpu import (  # noqa: E402
    cuda_arch_status,
    describe_gpu_environment,
    gpu_is_available,
)

DEFAULT_ENERGIES = [float(i) for i in range(1, 501, 5)]


class GpuIsoelectronicScalingAnalyzer(IsoelectronicScalingAnalyzer):
    """Scaling analyzer whose collision sweeps run on the CUDA backend."""

    def compute_energy_dependent_branching_ratios(
        self,
        energy_grid_ev: Optional[List[float]] = None,
        output_dir: str | Path = "data/scaling_analysis",
        b_min_bohr: float = 0.5,
        b_max_bohr: float = 10.0,
        n_b_points: int = 25,
        n_time_steps: int = 300,
        precision: str = "single",
        device: str = "cuda",
    ) -> Dict[str, Any]:
        if energy_grid_ev is None:
            energy_grid_ev = DEFAULT_ENERGIES

        output_path = Path(output_dir)
        processed_path = PROJECT_ROOT / "data" / "processed_circuits"
        cross_sections_path = PROJECT_ROOT / "data" / "cross_sections"
        output_path.mkdir(parents=True, exist_ok=True)
        processed_path.mkdir(parents=True, exist_ok=True)
        cross_sections_path.mkdir(parents=True, exist_ok=True)

        results_by_species: Dict[str, Any] = {}
        for species in self.species_keys:
            manifold_path = self.manifold_paths.get(species)
            if manifold_path is None or not manifold_path.exists():
                continue

            simulator = CollisionDynamicsSimulator(
                str(manifold_path),
                backend="gpu",
                device=device,
                precision=precision,
            )
            records = []
            for incident_energy in energy_grid_ev:
                sweep = simulator.sweep_impact_parameters(
                    incident_energy_ev=float(incident_energy),
                    b_min_bohr=b_min_bohr,
                    b_max_bohr=b_max_bohr,
                    n_b_points=n_b_points,
                    n_time_steps=n_time_steps,
                    output_dir=str(processed_path),
                )
                cross_section_records = calculate_cross_sections_and_branching_ratios(sweep)
                cross_section_file = (
                    cross_sections_path
                    / f"cross_sections_{species}_E{int(incident_energy)}eV.json"
                )
                save_processed_circuit_cross_sections(sweep, cross_section_file)

                state_one = next(
                    (record for record in cross_section_records if record["state_index"] == 1),
                    cross_section_records[0],
                )
                records.append(
                    {
                        "incident_energy_ev": float(incident_energy),
                        "total_excitation_sigma_mb": state_one["total_excitation_sigma_mb"],
                        "state_cross_sections_mb": {
                            record["state_label"]: record["sigma_mb"]
                            for record in cross_section_records
                        },
                        "branching_ratios": {
                            record["state_label"]: record["branching_ratio"]
                            for record in cross_section_records
                        },
                        "backend": sweep["backend"],
                    }
                )

            results_by_species[species] = {
                "species": species,
                "atomic_number": self.manifolds[species]["atomic_number"],
                "charge": self.manifolds[species].get("charge", 0),
                "energy_grid_ev": energy_grid_ev,
                "energy_sweep_results": records,
            }

        payload = {
            "series": "Beryllium Isoelectronic Sequence",
            "energy_grid_ev": energy_grid_ev,
            "species_data": results_by_species,
        }
        result_file = output_path / "energy_dependent_branching_ratios_gpu.json"
        with result_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        print(f"[+] Saved CUDA scaling results to: {result_file}")
        return payload


def generate_cross_section_plot(
    output_dir: str | Path = "data/scaling_analysis",
    result_filename: str = "energy_dependent_branching_ratios_gpu.json",
) -> Optional[Path]:
    """Plot state-resolved cross sections as a function of incident energy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not installed; skipping cross-section plot.")
        return None

    result_file = Path(output_dir) / result_filename
    if not result_file.exists():
        print(f"[!] Result file not found: {result_file}")
        return None

    with result_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    species_data = data.get("species_data", {})
    if not species_data:
        print("[!] No species data found; skipping cross-section plot.")
        return None

    figure, axes = plt.subplots(
        len(species_data),
        1,
        figsize=(10, 4 * len(species_data)),
        sharex=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab10")

    for axis, (species, species_data_item) in zip(axes[:, 0], species_data.items()):
        records = species_data_item.get("energy_sweep_results", [])
        energies = [record["incident_energy_ev"] for record in records]
        state_labels = sorted(
            {
                label
                for record in records
                for label in record.get("state_cross_sections_mb", {})
            },
            key=lambda label: int(label.rsplit("_", 1)[-1]),
        )

        for state_index, state_label in enumerate(state_labels):
            cross_sections = [
                record.get("state_cross_sections_mb", {}).get(state_label, 0.0)
                for record in records
            ]
            axis.plot(
                energies,
                cross_sections,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors(state_index % 10),
                label=state_label,
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel("Cross-section (Mb)")
        axis.set_title(
            f"{species} (Z={species_data_item['atomic_number']}, "
            f"q={species_data_item['charge']})"
        )
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.legend(ncol=4, fontsize=9)

    axes[-1, 0].set_xlabel("Incident electron energy (eV)")
    figure.suptitle("State-resolved cross sections vs incident energy", fontsize=15)
    figure.tight_layout()
    plot_path = Path(output_dir) / "cross_sections_vs_energy_gpu.png"
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[+] Saved cross-section plot to: {plot_path}")
    return plot_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scaling-law collision sweeps on CUDA")
    parser.add_argument("--energies", nargs="+", type=float, default=DEFAULT_ENERGIES)
    parser.add_argument("--b-points", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--b-min", type=float, default=0.5)
    parser.add_argument("--b-max", type=float, default=10.0)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1")
    parser.add_argument("--precision", choices=["single", "double", "auto"], default="single")
    parser.add_argument("--out", default="data/scaling_analysis")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    environment = describe_gpu_environment()
    if not gpu_is_available():
        _, problem = cuda_arch_status()
        print("[!] CUDA is unavailable or incompatible with the installed PyTorch build.")
        if problem:
            print(problem)
        if not args.allow_cpu:
            print("    Re-run with --allow-cpu to permit CPU tensors.")
            return 2
        args.device = "cpu"

    print(f"[*] Device request: {args.device}; precision: {args.precision}")
    if environment["devices"]:
        print(f"[*] GPU: {environment['devices'][0]['name']}")

    analyzer = GpuIsoelectronicScalingAnalyzer()
    analyzer.save_scaling_summary(args.out)
    analyzer.compute_energy_dependent_branching_ratios(
        energy_grid_ev=args.energies,
        output_dir=args.out,
        b_min_bohr=args.b_min,
        b_max_bohr=args.b_max,
        n_b_points=args.b_points,
        n_time_steps=args.steps,
        precision=args.precision,
        device=args.device,
    )
    generate_scaling_plots(
        args.out,
        result_filename="energy_dependent_branching_ratios_gpu.json",
    )
    generate_cross_section_plot(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
