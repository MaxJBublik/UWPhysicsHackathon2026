"""Compare simulated C III cross sections with the 2s² ¹S–2s2p ¹P fit.

Run from the repository root with:
    python -m src.analysis.carbon_validation_plot
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    # Supports ``python -m src.analysis.carbon_validation_plot``.
    from .carbon_comparison import TRANSITIONS, cross_section_cm2
except ImportError:
    # Supports direct execution: ``python src/analysis/carbon_validation_plot.py``.
    from carbon_comparison import TRANSITIONS, cross_section_cm2


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "carbon_validation_graphs"
SIMULATION_DATA_PATH = OUTPUT_DIR / "cross_sections_by_state_cm2.csv"
REFERENCE_LABEL = "2s² ¹S–2s2p ¹P"


def load_simulated_cross_sections(
    data_path: Path = SIMULATION_DATA_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the C²⁺ state-1–3 average cross sections in cm²."""
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"species", "energy", "state_1_3_average"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{data_path} must include {sorted(required_columns)}.")
        points = [
            (float(row["energy"]), float(row["state_1_3_average"]))
            for row in reader if row["species"] == "C2+"
        ]
    if not points:
        raise ValueError(f"No C2+ simulation data found in {data_path}.")
    energy, cross_section = zip(*sorted(points))
    return np.asarray(energy), np.asarray(cross_section)


def reference_transition():
    """Return the selected tabulated C III reference transition."""
    return next(transition for transition in TRANSITIONS if transition.label == REFERENCE_LABEL)


def plot_cross_section(output_dir: Path = OUTPUT_DIR) -> Path:
    """Plot the C²⁺ simulation, reference fit, and percent difference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    energy, simulated_cross_section = load_simulated_cross_sections()
    transition = reference_transition()
    reference_cross_section = cross_section_cm2(energy, transition)
    valid = reference_cross_section > 0
    percent_difference = 100 * (
        simulated_cross_section[valid] - reference_cross_section[valid]
    ) / reference_cross_section[valid]

    with plt.rc_context({"font.size": 20, "axes.titlesize": 20,
                         "axes.labelsize": 20, "xtick.labelsize": 20,
                         "ytick.labelsize": 20, "legend.fontsize": 20}):
        fig, axis = plt.subplots(figsize=(12, 7.5))
        axis.plot(energy[valid], reference_cross_section[valid], "o-", color="#2878b5",
                  linewidth=2.5, markersize=5, label=f"CIII Validation Data")
        axis.plot(energy, simulated_cross_section, "s--", color="#d1495b",
                  linewidth=2.5, markersize=5, label="Simulated Cross Section")
        difference_axis = axis.twinx()
        difference_axis.plot(energy[valid], percent_difference, "d-", color="#3c9d5d",
                             linewidth=2, markersize=5, label="Percent difference")
        difference_axis.axhline(0, color="#3c9d5d", linewidth=1, alpha=0.4)
        axis.set_yscale("log")
        axis.set_xlabel("Incident electron energy (eV)")
        axis.set_ylabel("Excitation cross section (cm²)")
        difference_axis.set_ylabel("Percent difference (%)")
        axis.set_title("C III 1s2.2s² ¹S–1s2.2s.2p ¹P excitation cross section")
        axis.grid(True, which="both", alpha=0.28)
        lines, labels = axis.get_legend_handles_labels()
        difference_lines, difference_labels = difference_axis.get_legend_handles_labels()
        axis.legend(lines + difference_lines, labels + difference_labels, loc="best")
        fig.tight_layout()
    output_path = output_dir / "carbon_cross_section_validation.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Wrote {plot_cross_section()}")
