"""Create validation graphs for tabulated C III electron-impact transitions.

The collision-strength fit is ``Omega_if(X) = A + B/X + C/X**2 + D/X**3
+ E*ln(X)``, where ``X`` is the dimensionless reduced incident energy.  The
coefficients and thresholds below are transcribed from the supplied table.

Run from the repository root: ``python -m src.analysis.carbon_comparison``.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "carbon_validation_graphs"
REFERENCE_PATH = ROOT / "data" / "reference" / "nist_reference.json"
MANIFOLD_PATH = ROOT / "data" / "raw_pyscf" / "manifold_C2+.json"


@dataclass(frozen=True)
class Transition:
    """One C III transition and its Eq. (6) collision-strength fit."""

    label: str
    threshold_ev: float
    A: float
    B: float
    C: float
    D: float
    E: float
    rms: float | None = None


# ``None`` means that no rms value was reported in the supplied table.
TRANSITIONS = (
    Transition("2s² ¹S–2s2p ³P", 6.5, -4.024e-2, 2.914, -4.344, 2.121, 0.0),
    Transition("2s² ¹S–2s2p ¹P", 12.7, 6.261e-1, 3.044, 5.384e-1, 0.0, 4.364, 0.04),
    Transition("2s² ¹S–2p² ³P", 17.2, -5.732e-3, 8.467e-2, -1.676e-1, 1.307e-1, 0.0, 0.01),
    Transition("2s² ¹S–2p² ¹D", 18.2, 2.915e-1, 1.655e-1, -3.445e-1, 3.720e-1, 0.0, 0.03),
    Transition("2s² ¹S–2p² ¹S", 22.9, 2.092e-2, 2.289e-1, -4.268e-1, 2.137e-1, 0.0, 0.01),
    Transition("2s² ¹S–2s3s ³S", 29.5, 3.910e-3, 2.740e-1, 1.120, 1.110, 0.0, 0.36),
    Transition("2s² ¹S–2s3s ¹S", 30.6, 4.260e-1, 4.350e-1, -2.280, 1.760, 0.0, 0.0004),
    Transition("2s² ¹S–2s3p ¹P", 32.1, -1.140e-1, -5.230e-1, 8.110e-1, 0.0, 3.730e-1, 0.0005),
    Transition("2s² ¹S–2s3p ³P", 32.2, -2.190e-2, 3.660e-1, -1.030, 8.240e-1, 0.0, 0.20),
    Transition("2s² ¹S–2s3d ³D", 33.5, 4.190e-2, 6.490e-2, 1.100e-2, 1.050e-1, 0.0, 0.0005),
    Transition("2s² ¹S–2s3d ¹D", 34.3, 7.330e-1, 5.400e-2, -2.110, 1.490, 0.0, 0.0005),
    Transition("2s2p ³P–2p² ³P", 10.6, 9.842, -8.660, 2.184e1, 0.0, 1.631),
    Transition("2s2p ³P–2p² ¹D", 11.7, -1.822e-1, 4.490, -4.889, 2.056, 0.0, 0.06),
    Transition("2s2p ³P–2p² ¹S", 16.5, 3.280e-4, 1.676e-1, 2.778e-1, -4.524e-1, 0.0, 0.02),
    Transition("2s2p ³P–2s3s ¹S", 24.1, -1.200e-2, 1.310, -4.030, 3.590, 0.0, 0.88),
    Transition("2s2p ³P–2s3p ¹P", 25.6, -1.310e-2, 1.580, -4.800, 4.270, 0.0, 0.0004),
    Transition("2s2p ³P–2s3p ³P", 25.7, 1.980, 1.490e1, -4.330e1, 3.170e1, 0.0, 0.0004),
    Transition("2s2p ³P–2s3d ³D", 27.0, -2.900e-1, 2.090, 7.110, 0.0, 1.010e1, 0.80),
    Transition("2s2p ³P–2s3s ¹D", 27.8, 9.840e-3, 5.080e-1, -7.040e-1, 9.550e-1, 0.0, 0.0004),
    Transition("2s2p ¹P–2p² ³P", 4.3, -8.624e-3, 4.141, -7.089, 3.824, 0.0),
    Transition("2s2p ¹P–2p² ¹D", 4.3, 3.762, 9.351, -3.004, 0.0, 7.320),
    Transition("2s2p ¹P–2p² ¹S", 10.1, 3.032, -2.375, 2.675, 0.0, 2.991),
    Transition("2p² ³P–2p² ¹D", 1.01, 1.925, 1.440e1, -3.560e1, 2.727e1, 0.0),
    Transition("2p² ³P–2p² ¹S", 5.8, 4.319e-2, 1.004, -1.037, 3.630e-1, 0.0),
    Transition("2p² ¹D–2p² ¹S", 4.79, 8.948e-1, -8.607e-1, 1.007, -4.136e-1, 0.0),
)

TERM_STATISTICAL_WEIGHTS = {
    "¹S": 1, "¹P": 3, "¹D": 5,
    "³S": 3, "³P": 9, "³D": 15,
}


def omega_if(X: float | np.ndarray, A: float, B: float, C: float, D: float,
             E: float) -> float | np.ndarray:
    """Evaluate the tabulated Eq. (6) collision-strength fit."""
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0):
        raise ValueError("Reduced energy X must be positive.")
    return A + B / X + C / X**2 + D / X**3 + E * np.log(X)


def initial_statistical_weight(label: str) -> int:
    """Return ω_i = (2S + 1)(2L + 1) for the initial term in ``label``."""
    initial_state = label.split("–", maxsplit=1)[0]
    match = re.search(r"[¹³][SPD]$", initial_state)
    if match is None:
        raise ValueError(f"Could not identify initial term in {label!r}")
    return TERM_STATISTICAL_WEIGHTS[match.group()]


def cross_section_cm2(incident_energy_ev: float | np.ndarray,
                      transition: Transition) -> float | np.ndarray:
    """Evaluate Q_if in cm² from Eq. (3) of the supplied reference.

    ``incident_energy_ev`` must be at or above the transition threshold. Below
    threshold the excitation cross section is zero.
    """
    energy = np.asarray(incident_energy_ev, dtype=float)
    if np.any(energy <= 0):
        raise ValueError("Incident electron energy must be positive.")
    reduced_energy = energy / transition.threshold_ev
    collision_strength = omega_if(reduced_energy, transition.A, transition.B,
                                  transition.C, transition.D, transition.E)
    sigma = 1.1969e-15 * collision_strength / (
        initial_statistical_weight(transition.label) * energy)
    return np.where(energy >= transition.threshold_ev, sigma, 0.0)


def plot_cross_sections(transitions: tuple[Transition, ...], output_dir: Path) -> Path:
    """Plot Eq. (3) cross-section fits in cm² versus incident energy in eV."""
    energy_values = np.geomspace(1.0, 4_000.0, 1_000)
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.colormaps["tab20"](np.linspace(0, 1, len(transitions)))
    for transition, color in zip(transitions, colors):
        sigma = cross_section_cm2(energy_values, transition)
        # A logarithmic plot cannot display non-positive extrapolated values.
        sigma = np.where(sigma > 0, sigma, np.nan)
        ax.plot(energy_values, sigma, color=color, linewidth=1.4,
                label=f"{transition.label} ({transition.threshold_ev:g} eV)")
    ax.set_yscale("log")
    ax.set_xlabel("Incident electron energy, Eₑ (eV)")
    ax.set_ylabel("Excitation cross section, Q_if (cm²)")
    ax.set_title("C III excitation cross-section fits from the supplied transition table")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    fig.tight_layout()
    path = output_dir / "cross_section_fits_cm2.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_transition_summary(transitions: tuple[Transition, ...], output_dir: Path) -> Path:
    """Show the threshold and reported fit rms for each transition."""
    ordered = sorted(transitions, key=lambda item: item.threshold_ev)
    labels = [item.label for item in ordered]
    thresholds = [item.threshold_ev for item in ordered]
    rms = [np.nan if item.rms is None else item.rms for item in ordered]
    positions = np.arange(len(ordered))
    fig, (threshold_axis, rms_axis) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    threshold_axis.barh(positions, thresholds, color="#2878b5")
    threshold_axis.set_xlabel("Transition threshold V_if (eV)")
    threshold_axis.set_ylabel("Transition")
    threshold_axis.set_yticks(positions, labels, fontsize=8)
    threshold_axis.grid(axis="x", alpha=0.25)
    rms_axis.scatter(rms, positions, color="#d1495b", s=35, zorder=3)
    rms_axis.set_xscale("log")
    rms_axis.set_xlabel("Reported fit rms (log scale; missing values omitted)")
    rms_axis.grid(axis="x", which="both", alpha=0.25)
    fig.suptitle("C III transition thresholds and Eq. (6) fit quality", y=0.98)
    fig.tight_layout()
    path = output_dir / "transition_thresholds_and_rms.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ground_state_validation(output_dir: Path) -> Path:
    """Compare ground-state table thresholds with local NIST and CASCI levels."""
    with REFERENCE_PATH.open(encoding="utf-8") as handle:
        reference = json.load(handle)["species"]["C2+"]["levels"]
    with MANIFOLD_PATH.open(encoding="utf-8") as handle:
        calculated = json.load(handle)["excitation_energies_ev"]
    nist_triplet = np.average(
        [reference[f"2s2p_3P_J{j}"]["energy_ev"] for j in (0, 1, 2)],
        weights=[1, 3, 5],
    )
    rows = (
        ("2s2p ³P", 6.5, nist_triplet, calculated[1]),
        ("2s2p ¹P", 12.7, reference["2s2p_1P_J1"]["energy_ev"], calculated[3]),
    )
    labels = [row[0] for row in rows]
    x_positions = np.arange(len(rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_positions - width, [row[1] for row in rows], width, label="Supplied table", color="#2878b5")
    ax.bar(x_positions, [row[2] for row in rows], width, label="NIST ASD", color="#5b8c5a")
    ax.bar(x_positions + width, [row[3] for row in rows], width, label="Local RHF/CASCI", color="#d98c3f")
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Excitation energy from 2s² ¹S (eV)")
    ax.set_title("C III ground-state transition validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, 0.01, "³P NIST value is the statistical J-average (1:3:5).", transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    path = output_dir / "ground_state_energy_validation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_transition_table(transitions: tuple[Transition, ...], output_dir: Path) -> Path:
    """Save the transcribed values in a machine-readable companion CSV."""
    path = output_dir / "carbon_transition_fits.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [*asdict(transitions[0]), "initial_statistical_weight"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({**asdict(transition), "initial_statistical_weight": initial_statistical_weight(transition.label)} for transition in transitions)
    return path


def build_graphs(output_dir: Path = OUTPUT_DIR) -> tuple[Path, ...]:
    """Generate all artifacts in ``data/carbon_validation_graphs``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        write_transition_table(TRANSITIONS, output_dir),
        plot_cross_sections(TRANSITIONS, output_dir),
        plot_transition_summary(TRANSITIONS, output_dir),
        plot_ground_state_validation(output_dir),
    )


if __name__ == "__main__":
    for output_path in build_graphs():
        print(f"Wrote {output_path}")
