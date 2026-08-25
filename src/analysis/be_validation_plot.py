"""Plot the supplied Be excitation cross-section validation data.

Run from the repository root with:
    python -m src.analysis.be_validation_plot
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "Be_validation_graphs"
SIMULATION_DATA_PATH = OUTPUT_DIR / "beryllium_cross_sections.csv"

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


def load_average_p_cross_sections(
    data_path: Path = SIMULATION_DATA_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the supplied average p-state cross sections in cm²."""
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"incident_energy_ev", "average_p_sigma_cm2"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{data_path} must include {sorted(required_columns)}.")
        points = [
            (float(row["incident_energy_ev"]), float(row["average_p_sigma_cm2"]))
            for row in reader
        ]
    if not points:
        raise ValueError(f"No simulation cross-section data found in {data_path}.")
    energies, cross_sections = zip(*sorted(points))
    return np.asarray(energies), np.asarray(cross_sections)


def plot_cross_section(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create a semi-log Be cross-section plot (linear energy, log cross section)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    average_energy_ev, average_cross_section_cm2 = load_average_p_cross_sections()
    percent_difference = 100 * (
        np.interp(ENERGY_EV, average_energy_ev, average_cross_section_cm2*3)
        - CROSS_SECTION_CM2
    ) / CROSS_SECTION_CM2
    with plt.rc_context({"font.size": 20, "axes.titlesize": 20,
                         "axes.labelsize": 20, "xtick.labelsize": 20,
                         "ytick.labelsize": 20, "legend.fontsize": 20}):
        fig, axis = plt.subplots(figsize=(12, 7.5))
        axis.plot(ENERGY_EV, CROSS_SECTION_CM2, "o-", color="#2878b5",
                  linewidth=2.5, markersize=7, label="Be validation data")
        axis.plot(average_energy_ev, average_cross_section_cm2*3, "s--",
                  color="#d1495b", linewidth=2.5, markersize=7,
                  label="Simulated cross section")
        difference_axis = axis.twinx()
        difference_axis.plot(ENERGY_EV, percent_difference, "d-",
                             color="#3c9d5d", linewidth=2, markersize=6,
                             label="Percent difference")
        difference_axis.axhline(0, color="#3c9d5d", linewidth=1, alpha=0.4)
        difference_axis.set_ylim(0, 100)
        axis.set_yscale("log")
        axis.set_xlabel("Incident electron energy (eV)")
        axis.set_ylabel("Excitation cross section (cm²)")
        difference_axis.set_ylabel("Percent difference (%)")
        axis.set_title("BeI excitation cross section 1s2.2s2 (1S) → 1s2.2s2p (1P°)")
        axis.grid(True, which="both", alpha=0.28)
        lines, labels = axis.get_legend_handles_labels()
        difference_lines, difference_labels = difference_axis.get_legend_handles_labels()
        axis.legend(lines + difference_lines, labels + difference_labels)
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


def build_plot() -> tuple[Path, Path]:
    """Write the data table and generated validation plot."""
    return write_data(), plot_cross_section()


if __name__ == "__main__":
    for path in build_plot():
        print(f"Wrote {path}")
