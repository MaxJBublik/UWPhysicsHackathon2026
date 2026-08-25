"""Plot excitation probability vs. impact parameter from processed-circuit runs.

Each ``data/processed_circuits/populations_<species>_E<energy>eV.json`` file
holds excitation probabilities for states 0-3 as a function of impact
parameter, at one incident energy. This module groups those files by species
and plots states 1-3 against impact parameter, with one curve per energy.

Run from the repository root with:
    python -m src.analysis.impact_parameter_plots
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed_circuits"
OUTPUT_DIR = DATA_DIR / "plots"
FILENAME_PATTERN = re.compile(r"populations_(?P<species>.+)_E(?P<energy>[\d.]+)eV\.json")
PLOTTED_STATES = (1, 2, 3)
PROBABILITY_FLOOR = 1e-30


def discover_runs(data_dir: Path = DATA_DIR) -> dict[str, dict[float, Path]]:
    """Map each species to its {incident energy (eV): file path} runs."""
    runs: dict[str, dict[float, Path]] = {}
    for path in sorted(data_dir.glob("populations_*.json")):
        match = FILENAME_PATTERN.match(path.name)
        if match is None:
            continue
        species = match.group("species")
        energy = float(match.group("energy"))
        runs.setdefault(species, {})[energy] = path
    if not runs:
        raise ValueError(f"No populations_*.json runs found in {data_dir}.")
    return runs


def load_run(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load impact parameters and per-state probability arrays from one run."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    impact_parameters = np.asarray(payload["impact_parameters_bohr"])
    probabilities = {
        int(key.removeprefix("state_")): np.asarray(values)
        for key, values in payload["excitation_probabilities_vs_b"].items()
    }
    return impact_parameters, probabilities


def plot_species(species: str, energy_to_path: dict[float, Path], output_dir: Path) -> Path:
    """Plot states 1-3 excitation probability vs. impact parameter for one species."""
    energies = sorted(energy_to_path)
    norm = LogNorm(vmin=min(energies), vmax=max(energies))
    colormap = plt.colormaps["viridis"]

    with plt.rc_context({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10}):
        # States 1-3 span very different probability magnitudes (state 3
        # dominates by 15+ orders of magnitude), so each panel gets its own
        # y-scale rather than sharing one that would flatten the others.
        fig, axes = plt.subplots(1, len(PLOTTED_STATES), figsize=(6 * len(PLOTTED_STATES), 6),
                                 sharex=True, sharey=False)
        for energy in energies:
            impact_parameters, probabilities = load_run(energy_to_path[energy])
            color = colormap(norm(energy))
            for axis, state in zip(axes, PLOTTED_STATES):
                values = np.clip(probabilities[state], PROBABILITY_FLOOR, None)
                axis.plot(impact_parameters, values, "-", color=color, linewidth=2)

        for axis, state in zip(axes, PLOTTED_STATES):
            axis.set_yscale("log")
            axis.set_xlabel("Impact parameter (Bohr)")
            axis.set_ylabel("Excitation probability")
            axis.set_title(f"State {state}")
            axis.grid(True, which="both", alpha=0.28)

        mappable = ScalarMappable(norm=norm, cmap=colormap)
        colorbar = fig.colorbar(mappable, ax=axes, pad=0.02)
        colorbar.set_label("Incident electron energy (eV)")

        fig.suptitle(f"{species} excitation probability vs. impact parameter")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"excitation_probability_vs_b_{species}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_plots(output_dir: Path = OUTPUT_DIR) -> tuple[Path, ...]:
    """Create one impact-parameter plot per species."""
    runs = discover_runs()
    return tuple(plot_species(species, energy_to_path, output_dir)
                 for species, energy_to_path in sorted(runs.items()))


if __name__ == "__main__":
    for path in build_plots():
        print(f"Wrote {path}")
