"""Calculate and plot cross sections for every processed circuit JSON file.

This script reads all files matching ``populations_*.json`` from
``data/processed_circuits``. For each file it uses the calculation implemented
in ``src.analysis.cross_sections``, writes the resulting state records to
``data/cross_sections``, and plots cross section versus incident energy.

Run from the repository root:

    python scripts\\calculate_all_cross_sections.py

Optional example for a different integration tolerance:

    python scripts\\calculate_all_cross_sections.py --tolerance 0.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.analysis.cross_sections import (  # noqa: E402
    calculate_cross_sections_and_branching_ratios,
    save_processed_circuit_cross_sections,
)

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "processed_circuits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cross_sections"
DEFAULT_PLOT_DIR = REPO_ROOT / "data" / "scaling_analysis"
SUPPORTED_SPECIES = ("Be", "C2+", "Fe22+")
ELEMENT_NAMES = {"Be": "Be", "C2+": "C", "Fe22+": "Fe"}


CrossSectionSeries = DefaultDict[str, DefaultDict[str, List[Tuple[float, float]]]]


def calculate_all_files(
    input_dir: Path,
    output_dir: Path,
    tolerance: float,
    selected_species: Tuple[str, ...] = SUPPORTED_SPECIES,
) -> Tuple[CrossSectionSeries, CrossSectionSeries, int, int]:
    """Calculate cross sections for every processed-circuit JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    series_mb: CrossSectionSeries = defaultdict(lambda: defaultdict(list))
    series_cm2: CrossSectionSeries = defaultdict(lambda: defaultdict(list))
    processed_count = 0
    skipped_count = 0

    input_files = sorted(input_dir.glob("populations_*.json"))
    if not input_files:
        raise FileNotFoundError(f"No populations_*.json files found in {input_dir}")

    for input_file in input_files:
        try:
            with input_file.open("r", encoding="utf-8") as file:
                processed_data = json.load(file)
            species = str(processed_data.get("species", ""))
            if species not in selected_species:
                continue
            if species not in SUPPORTED_SPECIES:
                raise ValueError(
                    f"unsupported species {species!r}; expected one of {SUPPORTED_SPECIES}"
                )

            records = calculate_cross_sections_and_branching_ratios(
                processed_data,
                integration_tolerance=tolerance,
            )
            if not records:
                raise ValueError("calculation returned no state records")

            calculated_species = str(records[0]["species"])
            if calculated_species != species:
                raise ValueError(
                    f"species mismatch: input={species!r}, calculated={calculated_species!r}"
                )
            energy = float(records[0]["incident_energy_ev"])
            output_file = output_dir / f"cross_sections_{species}_E{int(energy)}eV.json"
            save_processed_circuit_cross_sections(
                processed_data,
                output_file,
                integration_tolerance=tolerance,
            )

            for record in records:
                state = str(record["state_label"])
                series_mb[species][state].append(
                    (energy, float(record["sigma_mb"]))
                )
                series_cm2[species][state].append(
                    (energy, float(record["sigma_cm2"]))
                )
            processed_count += 1
            print(f"[+] {input_file.name} -> {output_file.name}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as error:
            skipped_count += 1
            print(f"[!] Skipping {input_file.name}: {error}")

    for states in series_mb.values():
        for values in states.values():
            values.sort(key=lambda pair: pair[0])
    for states in series_cm2.values():
        for values in states.values():
            values.sort(key=lambda pair: pair[0])

    return series_mb, series_cm2, processed_count, skipped_count


def plot_total_cross_sections(
    series: CrossSectionSeries,
    plot_path: Path,
    unit: str = "Mb",
) -> None:
    """Plot total excitation cross section versus energy for each species."""
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")

    for species_index, (species, states) in enumerate(sorted(series.items())):
        totals: DefaultDict[float, float] = defaultdict(float)
        for values in states.values():
            for energy, cross_section in values:
                totals[energy] += cross_section
        points = sorted(totals.items())
        if not points:
            continue
        energies, cross_sections = zip(*points)
        axis.plot(
            energies,
            cross_sections,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors(species_index % 10),
            label=species,
        )

    # axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Incident electron energy (eV)")
    axis.set_ylabel(f"Total excitation cross section ({unit})")
    axis.set_title(f"Total cross section vs incident energy ({unit})")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[+] Saved total cross-section plot to: {plot_path}")


def plot_state_cross_sections(
    series: CrossSectionSeries,
    plot_path: Path,
    unit: str = "Mb",
) -> None:
    """Plot every excited-state cross section in one subplot per species."""
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(10, max(4, 3.5 * len(series))),
        sharex=True,
        squeeze=False,
    )
    colors = plt.get_cmap("tab10")

    for axis, (species, states) in zip(axes[:, 0], sorted(series.items())):
        for state_index, (state, values) in enumerate(sorted(states.items())):
            energies, cross_sections = zip(*values)
            axis.plot(
                energies,
                cross_sections,
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                color=colors(state_index % 10),
                label=state,
            )
        # axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(f"Cross section ({unit})")
        axis.set_title(species)
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.legend(ncol=4, fontsize=8)

    axes[-1, 0].set_xlabel("Incident electron energy (eV)")
    figure.suptitle(
        f"State-resolved cross sections vs incident energy ({unit})", fontsize=15
    )
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[+] Saved state-resolved plot to: {plot_path}")


def plot_p_manifold_cross_sections(
    series: CrossSectionSeries,
    plot_path: Path,
    unit: str = "Mb",
) -> None:
    """Plot the average p-orbital cross section for states 1, 2, and 3."""
    figure, axis = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")

    for species_index, (species, states) in enumerate(sorted(series.items())):
        p_totals: DefaultDict[float, float] = defaultdict(float)
        for state in ("state_1", "state_2", "state_3"):
            for energy, cross_section in states.get(state, []):
                p_totals[energy] += cross_section

        points = sorted(p_totals.items())
        if not points:
            continue
        energies, cross_sections = zip(*points)
        average_cross_sections = [cross_section / 3.0 for cross_section in cross_sections]
        axis.plot(
            energies,
            average_cross_sections,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=colors(species_index % 10),
            label=species,
        )

    axis.set_yscale("log")
    axis.set_xlabel("Incident electron energy (eV)")
    axis.set_ylabel(f"Average p-orbital cross section ({unit})")
    axis.set_title(f"Average p-orbital (px + py + pz) vs energy ({unit})")
    axis.grid(True, which="both", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[+] Saved p-manifold plot to: {plot_path}")


def plot_each_species_cross_sections(
    series: CrossSectionSeries,
    output_dir: Path,
    unit: str = "Mb",
) -> None:
    """Save one average p-orbital cross-section plot for each species."""
    for species, states in sorted(series.items()):
        if not states:
            continue

        figure, axis = plt.subplots(figsize=(10, 6))
        p_totals: DefaultDict[float, float] = defaultdict(float)
        p_state_count: DefaultDict[float, int] = defaultdict(int)
        for state in ("state_1", "state_2", "state_3"):
            for energy, cross_section in states.get(state, []):
                p_totals[energy] += cross_section
                p_state_count[energy] += 1
        p_points = sorted(
            (energy, p_totals[energy] / p_state_count[energy])
            for energy in p_totals
            if p_state_count[energy] == 3
        )
        if p_points:
            energies, averages = zip(*p_points)
            axis.plot(
                energies,
                averages,
                color="black",
                linestyle="--",
                linewidth=2.2,
                label="average p orbital (px, py, pz)",
            )

        element = ELEMENT_NAMES.get(species, species)
        axis.set_yscale("log")
        axis.set_xlabel("Incident electron energy (eV)")
        axis.set_ylabel(f"Cross section ({unit})")
        axis.set_title(f"{element} ({species}) average p-orbital cross section vs energy ({unit})")
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.legend(ncol=3, fontsize=9)
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename_unit = "cm2" if unit.startswith("cm") else unit
        plot_path = output_dir / f"{element}_average_p_cross_section_vs_energy_{filename_unit}.png"
        figure.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        print(f"[+] Saved {element} cross-section plot to: {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate cross sections from all processed circuit JSON files"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--species",
        nargs="+",
        choices=SUPPORTED_SPECIES,
        default=list(SUPPORTED_SPECIES),
        help="Species to process (default: Be C2+ Fe22+)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[!] Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if args.tolerance <= 0:
        print("[!] --tolerance must be positive", file=sys.stderr)
        return 1

    try:
        series_mb, series_cm2, processed_count, skipped_count = calculate_all_files(
            args.input_dir,
            args.output_dir,
            args.tolerance,
            tuple(args.species),
        )
    except FileNotFoundError as error:
        print(f"[!] {error}", file=sys.stderr)
        return 1

    if not series_mb:
        print("[!] No valid processed-circuit files were calculated.", file=sys.stderr)
        return 1

    plot_total_cross_sections(
        series_mb,
        args.plot_dir / "total_cross_sections_vs_energy.png",
        unit="Mb",
    )
    plot_state_cross_sections(
        series_mb,
        args.plot_dir / "state_cross_sections_vs_energy.png",
        unit="Mb",
    )
    plot_total_cross_sections(
        series_cm2,
        args.plot_dir / "total_cross_sections_vs_energy_cm2.png",
        unit="cm²",
    )
    plot_state_cross_sections(
        series_cm2,
        args.plot_dir / "state_cross_sections_vs_energy_cm2.png",
        unit="cm²",
    )
    plot_p_manifold_cross_sections(
        series_mb,
        args.plot_dir / "p_manifold_cross_sections_vs_energy.png",
        unit="Mb",
    )
    plot_p_manifold_cross_sections(
        series_cm2,
        args.plot_dir / "p_manifold_cross_sections_vs_energy_cm2.png",
        unit="cm²",
    )
    plot_each_species_cross_sections(series_mb, args.plot_dir, unit="Mb")
    plot_each_species_cross_sections(series_cm2, args.plot_dir, unit="cm²")
    print(f"[+] Calculated {processed_count} files; skipped {skipped_count} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
