"""Create Be and C III validation plots from the all-elements cross-section table.

The input table stores Be, Fe22+, and C2+ as three repeated column blocks.
This module appends the Be and C2+ sums over states 1–3, then plots each
derived simulation series against its appropriate validation reference.

Run from the repository root with:
    python -m src.analysis.all_element_validation
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .carbon_comparison import TRANSITIONS, cross_section_cm2
except ImportError:
    from carbon_comparison import TRANSITIONS, cross_section_cm2


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "All Elemental Data"
ALL_ELEMENTS_PATH = DATA_DIR / "cross_sections_by_state_cm2_all_elements.csv"
CARBON_REFERENCE_LABEL = "2s² ¹S–2s2p ¹P"
BE_SUM_COLUMN = "be_state_1_to_3_sum_cm2"
CARBON_SUM_COLUMN = "carbon_state_1_to_3_sum_cm2"

# Supplied Be validation values: incident energy in eV and cross section in
# cm². Formerly stored in data/Be_validation_graphs/be_cross_section_validation.csv.
BE_VALIDATION_ENERGY_EV = np.array(
    [6.0, 7.0, 8.2, 9.06, 10.0, 11.5, 15.0, 20.0, 30.0, 40.0, 50.0,
     70.0, 100.0, 200.0, 500.0, 1000.0]
)
BE_VALIDATION_CROSS_SECTION_CM2 = np.array(
    [1.41763e-16, 3.05530e-16, 4.89467e-16, 5.93754e-16, 6.69153e-16,
     6.92005e-16, 7.45055e-16, 7.82101e-16, 7.68030e-16, 7.35624e-16,
     6.91400e-16, 6.04070e-16, 5.03294e-16, 3.28519e-16, 1.67481e-16,
     9.57481e-17]
)


def _read_rows(data_path: Path) -> tuple[list[str], list[list[str]]]:
    with data_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"No data found in {data_path}.")
    return rows[0], rows[1:]


def add_state_sum_columns(data_path: Path = ALL_ELEMENTS_PATH) -> None:
    """Append Be and C2+ state-1–3 sum columns to the wide source table."""
    header, rows = _read_rows(data_path)
    if BE_SUM_COLUMN in header or CARBON_SUM_COLUMN in header:
        if {BE_SUM_COLUMN, CARBON_SUM_COLUMN}.issubset(header):
            return
        raise ValueError(f"{data_path} has only one of the required derived columns.")
    if len(header) != 30:
        raise ValueError(f"Expected 30 source columns in {data_path}, found {len(header)}.")

    augmented_rows: list[list[str]] = []
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"Row {row_number} in {data_path} has {len(row)} columns.")
        if row[0] != "Be" or row[20] != "C2+":
            raise ValueError(f"Unexpected species ordering in row {row_number} of {data_path}.")
        be_sum = sum(float(row[index]) for index in (2, 3, 4))
        carbon_sum = sum(float(row[index]) for index in (22, 23, 24))
        augmented_rows.append([*row, f"{be_sum:.16e}", f"{carbon_sum:.16e}"])

    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*header, BE_SUM_COLUMN, CARBON_SUM_COLUMN])
        writer.writerows(augmented_rows)


def load_state_sums(
    data_path: Path = ALL_ELEMENTS_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load energy and state-1–3 sum series for Be and C2+.

    The source table repeats ``species``/``energy`` headers across its three
    blocks, so ``csv.DictReader`` cannot address them by name (duplicate keys
    collapse to the last occurrence). Read positionally instead, matching how
    ``add_state_sum_columns`` locates the Be and C2+ blocks.
    """
    add_state_sum_columns(data_path)
    header, rows = _read_rows(data_path)
    be_sum_index = header.index(BE_SUM_COLUMN)
    carbon_sum_index = header.index(CARBON_SUM_COLUMN)
    be_energy = np.asarray([float(row[1]) for row in rows])
    # The final (C2+) repeated block supplies its own energy column.
    carbon_energy = np.asarray([float(row[21]) for row in rows])
    be_sum = np.asarray([float(row[be_sum_index]) for row in rows])
    carbon_sum = np.asarray([float(row[carbon_sum_index]) for row in rows])
    return be_energy, be_sum, carbon_energy, carbon_sum


def load_be_validation() -> tuple[np.ndarray, np.ndarray]:
    """Return the supplied Be validation points in cm²."""
    return BE_VALIDATION_ENERGY_EV, BE_VALIDATION_CROSS_SECTION_CM2


def carbon_reference_transition():
    return next(item for item in TRANSITIONS if item.label == CARBON_REFERENCE_LABEL)


def _plot_comparison(
    *, energy: np.ndarray, simulation: np.ndarray, reference_energy: np.ndarray,
    reference: np.ndarray, title: str, reference_label: str, simulation_label: str,
    output_path: Path,
) -> Path:
    """Plot a cross-section comparison and simulated-minus-reference percentage."""
    simulated_at_reference = np.interp(reference_energy, energy, simulation)
    percent_difference = 100 * (simulated_at_reference - reference) / reference
    with plt.rc_context({"font.size": 20, "axes.titlesize": 20,
                         "axes.labelsize": 20, "xtick.labelsize": 20,
                         "ytick.labelsize": 20, "legend.fontsize": 20}):
        fig, axis = plt.subplots(figsize=(12, 7.5))
        axis.plot(reference_energy, reference, "o-", color="#2878b5", linewidth=2.5,
                  markersize=7, label=reference_label)
        axis.plot(energy, simulation, "s--", color="#d1495b", linewidth=2.5,
                  markersize=6, label=simulation_label)
        difference_axis = axis.twinx()
        difference_axis.plot(reference_energy, percent_difference, "d-", color="#3c9d5d",
                             linewidth=2, markersize=6, label="Percent difference")
        difference_axis.axhline(0, color="#3c9d5d", linewidth=1, alpha=0.4)
        difference_axis.set_ylim(-100, 100)
        axis.set_yscale("log")
        axis.set_xlabel("Incident electron energy (eV)")
        axis.set_ylabel("Excitation cross section (cm²)")
        difference_axis.set_ylabel("Percent difference (%)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.28)
        lines, labels = axis.get_legend_handles_labels()
        difference_lines, difference_labels = difference_axis.get_legend_handles_labels()
        axis.legend(lines + difference_lines, labels + difference_labels)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    return output_path


def build_plots(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Augment the input CSV and create the separate Be and carbon plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    be_energy, be_sum, carbon_energy, carbon_sum = load_state_sums()
    validation_energy, validation_cross_section = load_be_validation()
    be_plot = _plot_comparison(
        energy=be_energy, simulation=be_sum, reference_energy=validation_energy,
        reference=validation_cross_section, title="BeI excitation cross section",
        reference_label="Be validation data", simulation_label="Be states 1–3 sum",
        output_path=output_dir / "beryllium_state_1_to_3_validation.png",
    )
    transition = carbon_reference_transition()
    carbon_reference = cross_section_cm2(carbon_energy, transition)
    valid = carbon_reference > 0
    carbon_plot = _plot_comparison(
        energy=carbon_energy, simulation=carbon_sum,
        reference_energy=carbon_energy[valid], reference=carbon_reference[valid],
        title="CIII excitation cross section",
        reference_label="C III reference fit", simulation_label="C²⁺ states 1–3 sum",
        output_path=output_dir / "carbon_state_1_to_3_validation.png",
    )
    return be_plot, carbon_plot


if __name__ == "__main__":
    for path in build_plots():
        print(f"Wrote {path}")
