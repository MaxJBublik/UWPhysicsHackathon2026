"""Plot the supplied Be excitation cross-section validation data.

Run from the repository root with:
    python -m src.analysis.be_validation_plot
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "Be_validation_graphs"
CROSS_SECTION_DIR = OUTPUT_DIR / "cross_sections"

# Supplied validation values: incident energy in eV and cross section in cm².
ENERGY_EV = np.array(
    [6.0, 7.0, 8.2, 9.06, 10.0, 11.5, 15.0, 20.0, 30.0, 40.0, 50.0,
     70.0, 100.0, 200.0, 500.0, 1000.0]
)
CROSS_SECTION_CM2 = np.array(
    [1.41763e-16, 3.05530e-16, 4.89467e-16, 5.93754e-16, 6.69153e-16,
     6.92005e-16, 7.45055e-16, 7.82101e-16, 7.68030e-16, 7.35624e-16,
     6.91400e-16, 6.04070e-16, 5.03294e-16, 3.28519e-16, 1.67481e-16,
     9.57481e-17]
)


def load_average_state_cross_sections(
    cross_section_dir: Path = CROSS_SECTION_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Average the state 1–3 cross sections in each extracted Be JSON file."""
    points: list[tuple[float, float]] = []
    for path in cross_section_dir.glob("cross_sections_Be_E*eV.json"):
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        states = {record["state_index"]: record["sigma_cm2"] for record in records}
        required_states = {1, 2, 3}
        if not required_states.issubset(states):
            raise ValueError(f"{path} is missing one or more of states 1, 2, and 3.")
        points.append((records[0]["incident_energy_ev"],
                       np.mean([states[index] for index in sorted(required_states)])))
    if not points:
        raise FileNotFoundError(f"No extracted Be cross-section files found in {cross_section_dir}")
    points.sort()
    energies, averages = zip(*points)
    return np.asarray(energies), np.asarray(averages)


def plot_cross_section(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create a semi-log Be cross-section plot (linear energy, log cross section)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    average_energy_ev, average_cross_section_cm2 = load_average_state_cross_sections()
    with plt.rc_context({"font.size": 20, "axes.titlesize": 20,
                         "axes.labelsize": 20, "xtick.labelsize": 20,
                         "ytick.labelsize": 20, "legend.fontsize": 20}):
        fig, axis = plt.subplots(figsize=(12, 7.5))
        axis.plot(ENERGY_EV, CROSS_SECTION_CM2, "o-", color="#2878b5",
                  linewidth=2.5, markersize=7, label="Supplied Be validation data")
        axis.plot(average_energy_ev, average_cross_section_cm2, "s--",
                  color="#d1495b", linewidth=2.5, markersize=7,
                  label="Mean of simulated states 1–3")
        axis.set_yscale("log")
        axis.set_xlabel("Incident electron energy (eV)")
        axis.set_ylabel("Excitation cross section (cm²)")
        axis.set_title("Beryllium excitation cross section")
        axis.grid(True, which="both", alpha=0.28)
        axis.legend()
        fig.tight_layout()
    output_path = output_dir / "be_cross_section_validation.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_data(output_dir: Path = OUTPUT_DIR) -> Path:
    """Store the supplied values next to the generated plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "be_cross_section_validation.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_ev", "cross_section_cm2"])
        writer.writerows(zip(ENERGY_EV, CROSS_SECTION_CM2))
    return output_path


def write_averaged_data(output_dir: Path = OUTPUT_DIR) -> Path:
    """Store the per-file mean of the extracted state 1–3 cross sections."""
    output_dir.mkdir(parents=True, exist_ok=True)
    energies, averages = load_average_state_cross_sections()
    output_path = output_dir / "be_state_1_to_3_average_cross_sections.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_ev", "mean_sigma_cm2_states_1_to_3"])
        writer.writerows(zip(energies, averages))
    return output_path


def build_plot() -> tuple[Path, Path, Path]:
    """Write the data table and generated validation plot."""
    return write_data(), write_averaged_data(), plot_cross_section()


if __name__ == "__main__":
    for path in build_plot():
        print(f"Wrote {path}")
