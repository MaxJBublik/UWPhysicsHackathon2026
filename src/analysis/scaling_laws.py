r"""Isoelectronic Z-scaling synthesis for the Be sequence (Task 3.2).

Compares neutral Be (Z=4), C2+ (Z=6), and Fe22+ (Z=26) — all with four
electrons — to quantify how collisional-excitation physics changes with
nuclear charge Z:

1. **Energy-gap scaling**: excitation energy of the dominant dipole-allowed
   transition, fitted as ``dE ~ Z**alpha`` (expected alpha > 0).
2. **Transition-dipole scaling**: dominant dipole magnitude ``|mu_0j|``,
   fitted as ``mu ~ Z**beta`` (expected beta < 0, roughly 1/Z).
3. **Coulomb focusing**: cross-section enhancement ratios
   ``sigma_ion / sigma_Be`` at incident energies shared by all species.

Inputs (produced by Tracks A and B/C-3.1):
    data/raw_pyscf/manifold_{species}.json
    data/cross_sections/cross_sections_{species}_E{E}eV.json  (globbed)

Outputs:
    data/scaling/scaling_summary.json
    data/scaling/deltaE_vs_Z.png
    data/scaling/dipole_vs_Z.png
    data/scaling/sigma_ratio.png

Run with:  python -m src.analysis.scaling_laws
"""
from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPECIES_LIST: tuple[str, ...] = ("Be", "C2+", "Fe22+")
DIPOLE_NOISE_FLOOR: float = 1.0e-8
MANIFOLD_DIR = Path("data/raw_pyscf")
SIGMA_DIR = Path("data/cross_sections")
OUTPUT_DIR = Path("data/scaling")


# ============================================================================
# Section 1: Data loading
# ============================================================================

def load_manifold(species: str, data_dir: Path = Path(".")) -> Dict[str, Any]:
    """Read one electronic-structure manifold JSON produced by Track A."""
    path = data_dir / MANIFOLD_DIR / f"manifold_{species}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Manifold file not found: {path}. "
            "Run src/electronic_structure/pyscf_runner.py first."
        )
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    dipole_xyz = np.stack(
        [np.asarray(raw[f"dipole_matrix_{axis}"], dtype=float)
         for axis in ("x", "y", "z")]
    )
    energies = np.asarray(
        raw.get("excitation_energies_ev", raw["energies_ev"]), dtype=float
    )
    return {
        "species": raw["species"],
        "Z": int(raw["atomic_number"]),
        "charge": int(raw["charge"]),
        "n_states": int(raw["n_states"]),
        "energies_ev": energies,
        "dipole_xyz": dipole_xyz,
    }


def dominant_dipole_magnitudes(dipole_xyz: np.ndarray,
                               n_states: int) -> np.ndarray:
    """``|mu_0j| = sqrt(mu_x^2 + mu_y^2 + mu_z^2)`` from the ground state.

    Values below :data:`DIPOLE_NOISE_FLOOR` are zeroed — the PySCF manifolds
    carry ~1e-15 numerical noise that must not win an argmax.
    """
    mu = np.sqrt(np.sum(np.abs(dipole_xyz[:, 0, :n_states]) ** 2, axis=0))
    mu[mu < DIPOLE_NOISE_FLOOR] = 0.0
    return mu


def load_cross_sections(species: str,
                        data_dir: Path = Path(".")) -> pd.DataFrame:
    """Glob all ``cross_sections_{species}_E*eV.json`` files.

    Globbing (rather than hardcoding 50 eV) means new incident energies
    produced by Track B are picked up automatically.
    """
    pattern = str(data_dir / SIGMA_DIR / f"cross_sections_{species}_E*eV.json")
    frames: List[pd.DataFrame] = []
    for filename in sorted(glob.glob(pattern)):
        with open(filename, encoding="utf-8") as handle:
            records = json.load(handle)
        frames.append(pd.DataFrame(records))
    if not frames:
        warnings.warn(f"No cross-section files matched {pattern}")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_species_summary(species: str,
                         data_dir: Path = Path(".")) -> Dict[str, Any]:
    """Combine manifold + cross-section data for one species."""
    manifold = load_manifold(species, data_dir)
    mu = dominant_dipole_magnitudes(manifold["dipole_xyz"],
                                    manifold["n_states"])
    # Dominant channel: the excited state with the largest |mu_0j|.  Chosen by
    # dipole strength, not index — degenerate states can reorder per species.
    j_dom = int(np.argmax(mu[1:]) + 1)
    return {
        **manifold,
        "mu_per_state": mu,
        "dominant_state": j_dom,
        "delta_E_dominant_ev": float(manifold["energies_ev"][j_dom]),
        "mu_dominant": float(mu[j_dom]),
        "sigma_table": load_cross_sections(species, data_dir),
    }


def build_scaling_table(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One tidy row per (species, excited state, incident energy)."""
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        for record in s["sigma_table"].to_dict("records"):
            j = int(record["state_index"])
            rows.append({
                "species": s["species"],
                "Z": s["Z"],
                "charge": s["charge"],
                "state_index": j,
                "delta_E_ev": float(s["energies_ev"][j]),
                "mu_au": float(s["mu_per_state"][j]),
                "sigma_au": float(record["sigma_au"]),
                "E_inc_ev": float(record["incident_energy_ev"]),
                "converged": bool(record.get("integration_converged", True)),
                "is_dominant": j == s["dominant_state"],
            })
    return pd.DataFrame(rows)


# ============================================================================
# Section 2: Power-law fitting
# ============================================================================

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit ``y = A * x**alpha`` by linear regression in log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points for a power-law fit")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fit requires strictly positive x and y")
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    log_pred = intercept + slope * np.log(x)
    ss_res = float(np.sum((np.log(y) - log_pred) ** 2))
    ss_tot = float(np.sum((np.log(y) - np.mean(np.log(y))) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"alpha": float(slope),
            "prefactor": float(np.exp(intercept)),
            "r_squared": float(r_squared)}


# ============================================================================
# Section 3: Scaling analyses
# ============================================================================

def _dominant_points(table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """One (Z, value) point per species for the dominant channel."""
    sub = (table[table["is_dominant"]]
           .drop_duplicates(subset="species")
           .sort_values("Z"))
    return sub[["species", "Z", value_column]].reset_index(drop=True)


def analyze_energy_scaling(table: pd.DataFrame) -> Dict[str, Any]:
    """Step 2: fit dE(Z) ~ Z**alpha for the dominant transition."""
    points = _dominant_points(table, "delta_E_ev")
    fit = fit_power_law(points["Z"].to_numpy(),
                        points["delta_E_ev"].to_numpy())
    if fit["alpha"] <= 0:
        warnings.warn("Unexpected energy scaling (alpha <= 0) — "
                      "check state ordering in the manifolds")
    return {"points": points.to_dict("records"), "fit": fit}


def analyze_dipole_scaling(table: pd.DataFrame) -> Dict[str, Any]:
    """Step 3: fit mu(Z) ~ Z**alpha (expected alpha < 0, roughly -1)."""
    points = _dominant_points(table, "mu_au")
    fit = fit_power_law(points["Z"].to_numpy(), points["mu_au"].to_numpy())
    if fit["alpha"] >= 0:
        warnings.warn("Unexpected dipole scaling (alpha >= 0)")
    return {"points": points.to_dict("records"), "fit": fit}


def _total_sigma(group: pd.DataFrame) -> tuple[float, bool]:
    """Total excitation cross section, summing only converged channels.

    Returns (sigma_total_au, all_channels_converged).  Using the total rather
    than a single state makes the focusing ratio robust to which excited
    state dominates in each species (state ordering varies, and the
    dipole-dominant state can carry a negligible sigma in the circuit data).
    """
    converged = group[group["converged"]]
    return float(converged["sigma_au"].sum()), bool(group["converged"].all())


def analyze_coulomb_focusing(table: pd.DataFrame,
                             reference: str = "Be") -> List[Dict[str, Any]]:
    """Step 4: total sigma_ion / sigma_reference at shared incident energies."""
    energies_per_species = {
        sp: set(grp["E_inc_ev"]) for sp, grp in table.groupby("species")
    }
    if reference not in energies_per_species:
        raise ValueError(f"reference species {reference!r} has no sigma data")
    common_energies = set.intersection(*energies_per_species.values())
    if not common_energies:
        warnings.warn("No incident energy is shared by all species; "
                      "cannot compute focusing ratios")
    results: List[Dict[str, Any]] = []
    for energy in sorted(common_energies):
        at_e = table[table["E_inc_ev"] == energy]
        sigma_ref, ref_ok = _total_sigma(at_e[at_e["species"] == reference])
        for species, group in at_e[at_e["species"] != reference].groupby(
                "species", sort=False):
            sigma_sp, sp_ok = _total_sigma(group)
            reliable = bool(ref_ok and sp_ok and sigma_ref > 0)
            ratio = (sigma_sp / sigma_ref if sigma_ref > 0 else float("nan"))
            results.append({
                "species": species,
                "Z": int(group["Z"].iloc[0]),
                "E_inc_ev": float(energy),
                "sigma_total_au": sigma_sp,
                "sigma_reference_total_au": sigma_ref,
                "ratio": float(ratio),
                "reliable": reliable,
            })
    return results


# ============================================================================
# Section 4: Plotting
# ============================================================================

def _loglog_scaling_plot(points: List[Dict[str, Any]], fit: Dict[str, float],
                         value_key: str, ylabel: str, symbol: str,
                         output_path: Path) -> Path:
    z = np.array([p["Z"] for p in points], dtype=float)
    values = np.array([p[value_key] for p in points], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(z, values, "o", markersize=9, color="tab:blue", zorder=3)
    z_fine = np.geomspace(z.min() * 0.8, z.max() * 1.2, 100)
    ax.loglog(z_fine, fit["prefactor"] * z_fine ** fit["alpha"], "--",
              color="tab:red",
              label=(f"{symbol} $\\propto Z^{{{fit['alpha']:.2f}}}$ "
                     f"($R^2$={fit['r_squared']:.3f})"))
    for point in points:
        ax.annotate(point["species"], (point["Z"], point[value_key]),
                    textcoords="offset points", xytext=(8, 5))
    ax.set_xlabel("Nuclear charge $Z$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Isoelectronic scaling: {ylabel}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_energy_scaling(analysis: Mapping[str, Any],
                        output_dir: Path) -> Path:
    return _loglog_scaling_plot(
        analysis["points"], analysis["fit"], "delta_E_ev",
        "Excitation energy $\\Delta E$ (eV)", "$\\Delta E$",
        output_dir / "deltaE_vs_Z.png")


def plot_dipole_scaling(analysis: Mapping[str, Any],
                        output_dir: Path) -> Path:
    return _loglog_scaling_plot(
        analysis["points"], analysis["fit"], "mu_au",
        "Transition dipole $|\\mu_{0j}|$ (a.u.)", "$\\mu$",
        output_dir / "dipole_vs_Z.png")


def plot_sigma_ratios(focusing: List[Dict[str, Any]],
                      output_dir: Path,
                      reference: str = "Be") -> Path:
    output_path = output_dir / "sigma_ratio.png"
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    if focusing:
        labels = [f"{r['species']}\n@{r['E_inc_ev']:.0f} eV" for r in focusing]
        ratios = [r["ratio"] for r in focusing]
        colors = ["tab:green" if r["reliable"] else "lightgray"
                  for r in focusing]
        bars = ax.bar(labels, ratios, color=colors)
        for bar, record in zip(bars, focusing):
            if not record["reliable"]:
                bar.set_hatch("//")
        if any(not r["reliable"] for r in focusing):
            ax.set_title("Hatched bars: integration not converged",
                         fontsize=9, loc="right", color="gray")
    else:
        ax.text(0.5, 0.5, "No common incident energy across species",
                ha="center", va="center", transform=ax.transAxes)
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":",
               label="no enhancement")
    ax.set_yscale("log")
    ax.set_ylabel(f"$\\sigma / \\sigma_{{{reference}}}$ (dominant channel)")
    ax.legend()
    fig.suptitle("Coulomb focusing: ion vs neutral cross sections")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ============================================================================
# Section 5: Export & entry point
# ============================================================================

def export_summary(table: pd.DataFrame, energy_fit: Mapping[str, Any],
                   dipole_fit: Mapping[str, Any],
                   focusing: List[Dict[str, Any]],
                   output_dir: Path) -> Path:
    """Write the machine-readable contract consumed by the Streamlit app."""
    caveats = []
    energies = sorted(set(table["E_inc_ev"]))
    if len({e for e in energies}) <= 1:
        caveats.append("Focusing ratios rest on a single common incident "
                       "energy; rerun Track B sweeps for sigma(E) curves.")
    n_bad = int((~table["converged"]).sum())
    if n_bad:
        caveats.append(f"{n_bad} cross-section rows have non-converged "
                       "impact-parameter integration and are unreliable.")
    payload = {
        "species": list(SPECIES_LIST),
        "per_state_table": table.to_dict("records"),
        "energy_scaling": dict(energy_fit),
        "dipole_scaling": dict(dipole_fit),
        "coulomb_focusing": focusing,
        "caveats": caveats,
    }
    output_path = output_dir / "scaling_summary.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def run_full_analysis(data_dir: Path = Path("."),
                      output_dir: Path = OUTPUT_DIR,
                      reference: str = "Be") -> Dict[str, Any]:
    """End-to-end Task 3.2 pipeline; returns everything it computed."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [load_species_summary(s, data_dir) for s in SPECIES_LIST]
    table = build_scaling_table(summaries)
    energy_fit = analyze_energy_scaling(table)
    dipole_fit = analyze_dipole_scaling(table)
    focusing = analyze_coulomb_focusing(table, reference=reference)

    plot_energy_scaling(energy_fit, output_dir)
    plot_dipole_scaling(dipole_fit, output_dir)
    plot_sigma_ratios(focusing, output_dir, reference=reference)
    summary_path = export_summary(table, energy_fit, dipole_fit,
                                  focusing, output_dir)

    print(f"[+] dE(Z)  ~ Z^{energy_fit['fit']['alpha']:+.2f} "
          f"(R^2 = {energy_fit['fit']['r_squared']:.3f})")
    print(f"[+] mu(Z)  ~ Z^{dipole_fit['fit']['alpha']:+.2f} "
          f"(R^2 = {dipole_fit['fit']['r_squared']:.3f})")
    for record in focusing:
        marker = "" if record["reliable"] else "  [NOT CONVERGED]"
        print(f"[+] sigma({record['species']})/sigma({reference}) "
              f"@ {record['E_inc_ev']:.0f} eV = "
              f"{record['ratio']:.3g}{marker}")
    print(f"[+] Summary written to {summary_path}")
    return {"table": table, "energy_scaling": energy_fit,
            "dipole_scaling": dipole_fit, "coulomb_focusing": focusing}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 3.2: isoelectronic Z-scaling synthesis")
    parser.add_argument("--data-dir", type=Path, default=Path("."),
                        help="repo root containing data/ (default: cwd)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="where plots and scaling_summary.json go")
    parser.add_argument("--reference-species", default="Be",
                        help="neutral reference for focusing ratios")
    args = parser.parse_args()
    run_full_analysis(args.data_dir, args.output_dir,
                      reference=args.reference_species)


if __name__ == "__main__":
    main()
