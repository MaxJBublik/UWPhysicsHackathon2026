"""Plot cross sections versus incident energy from cross-section JSON files.

The input files are produced by ``src.analysis.cross_sections`` and are
expected under ``data/cross_sections``. Example:

    python scripts/plot_cross_sections_vs_energy.py
    python scripts/plot_cross_sections_vs_energy.py --species Be C2+ Fe22+
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "cross_sections"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "scaling_analysis" / "cross_sections_vs_energy.png"
FILE_PATTERN = re.compile(r"^cross_sections_(?P<species>.+)_E(?P<energy>[-+0-9.eE]+)eV\.json$")


def load_cross_sections(input_dir: Path) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Read per-energy cross-section JSON files into species/state series."""
    series: DefaultDict[str, DefaultDict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in sorted(input_dir.glob("cross_sections_*.json")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                records = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            print(f"[!] Skipping {path.name}: {error}")
            continue

        if not isinstance(records, list) or not records:
            print(f"[!] Skipping {path.name}: expected a non-empty JSON list")
            continue

        energy = float(records[0]["incident_energy_ev"])
        species = str(records[0].get("species", match.group("species")))
        for record in records:
            state = str(record["state_label"])
            series[species][state].append((energy, float(record["sigma_mb"])))

    for state_series in series.values():
        for values in state_series.values():
            values.sort(key=lambda item: item[0])

    return {species: dict(states) for species, states in series.items()}


def plot_cross_sections(
    series: Dict[str, Dict[str, List[Tuple[float, float]]]],
    output_path: Path,
    selected_species: List[str] | None = None,
) -> None:
    """Create one subplot per atomic species with one curve per excited state."""
    if selected_species:
        series = {
            species: states
            for species, states in series.items()
            if species in selected_species
        }
    if not series:
        raise ValueError("No matching cross-section data was found.")

    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(10, 4 * len(series)),
        squeeze=False,
        sharex=True,
    )
    colors = plt.get_cmap("tab10")

    for axis, (species, states) in zip(axes[:, 0], sorted(series.items())):
        for state_index, (state, values) in enumerate(sorted(states.items())):
            energies, cross_sections = zip(*values)
            axis.plot(
                energies,
                cross_sections,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors(state_index % 10),
                label=state,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel("Cross section (Mb)")
        axis.set_title(species)
        axis.grid(True, which="both", linestyle="--", alpha=0.35)
        axis.legend(ncol=4, fontsize=9)

    axes[-1, 0].set_xlabel("Incident electron energy (eV)")
    figure.suptitle("Cross sections vs incident energy", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"[+] Saved plot to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot cross sections versus energy from cross-section JSON files"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--species", nargs="+", help="Species to plot, e.g. Be C2+ Fe22+")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"[!] Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    try:
        series = load_cross_sections(args.input_dir)
        plot_cross_sections(series, args.output, args.species)
    except (KeyError, TypeError, ValueError) as error:
        print(f"[!] Could not create plot: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
