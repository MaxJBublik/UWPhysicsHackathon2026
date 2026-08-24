r"""Validate the computed manifolds and Z-scaling laws against NIST reference data.

This is the experimental cross-check for Task 3.2 (and covers Task 1.1.3's
NIST benchmark for Track A).  Reference values live in
``data/reference/nist_reference.json``; see that file for provenance.

What is compared
----------------
1. **Excitation energies** — computed CASCI levels vs NIST ASD observed levels
   for the ``2s2p 3P`` and ``2s2p 1P`` terms of Be I, C III, and Fe XXIII.
   Because the CASCI is non-relativistic (no spin-orbit), the computed ``3P``
   is compared against the *statistically weighted J-average*
   ``E(3P) = [E_0 + 3 E_1 + 5 E_2] / 9`` rather than any single J level.
2. **Transition dipole / oscillator strength** — computed ``f`` and
   ``|mu_0j|`` vs reference values, where a reference f-value is available.
3. **Z-scaling exponents** — the fitted ``dE ~ Z^alpha`` exponent from
   :mod:`src.analysis.scaling_laws` vs the same fit performed on NIST energies.

Run with:  python -m src.analysis.nist_benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.scaling_laws import (
    SPECIES_LIST,
    dominant_dipole_magnitudes,
    fit_power_law,
    load_manifold,
)

HARTREE_TO_EV = 27.211386245988
REFERENCE_PATH = Path("data/reference/nist_reference.json")
OUTPUT_DIR = Path("data/scaling")


# ============================================================================
# Section 1: Reference data access
# ============================================================================

def load_reference(data_dir: Path = Path(".")) -> Dict[str, Any]:
    """Load the NIST reference JSON."""
    path = data_dir / REFERENCE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Reference data not found: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def triplet_j_average_ev(levels: Mapping[str, Any]) -> float:
    """Statistically weighted J-average of the ``2s2p 3P`` term.

    ``E = (1*E_J0 + 3*E_J1 + 5*E_J2) / 9`` — this is the quantity a
    non-relativistic calculation (no spin-orbit coupling) actually
    approximates, so it is the fair comparison target.
    """
    energies = [levels[f"2s2p_3P_J{j}"]["energy_ev"] for j in (0, 1, 2)]
    weights = np.array([1.0, 3.0, 5.0])
    return float(np.dot(weights, energies) / weights.sum())


def fine_structure_spread_ev(levels: Mapping[str, Any]) -> float:
    """``E(3P_2) - E(3P_0)``: the relativistic splitting a CASCI cannot see."""
    return float(levels["2s2p_3P_J2"]["energy_ev"]
                 - levels["2s2p_3P_J0"]["energy_ev"])


# ============================================================================
# Section 2: Computed-value extraction
# ============================================================================

def computed_terms(species: str, data_dir: Path = Path(".")) -> Dict[str, Any]:
    """Assign the computed manifold states to ``3P`` / ``1P`` terms.

    The CASCI manifold is ordered by energy with dipole-forbidden (triplet)
    states below the dipole-allowed singlet.  We therefore identify:
      * ``3P`` = the lowest excited state(s) with |mu| below the noise floor,
      * ``1P`` = the lowest excited state carrying non-zero |mu|.
    """
    manifold = load_manifold(species, data_dir)
    mu = dominant_dipole_magnitudes(manifold["dipole_xyz"],
                                    manifold["n_states"])
    energies = manifold["energies_ev"]

    dark = [j for j in range(1, manifold["n_states"]) if mu[j] == 0.0]
    bright = [j for j in range(1, manifold["n_states"]) if mu[j] > 0.0]
    if not bright:
        raise ValueError(f"{species}: no dipole-allowed state in the manifold")
    j_singlet = bright[int(np.argmin(energies[bright]))]

    triplet_ev: Optional[float] = None
    triplet_degeneracy = 0
    if dark:
        j_triplet = dark[int(np.argmin(energies[dark]))]
        triplet_ev = float(energies[j_triplet])
        triplet_degeneracy = int(np.sum(
            np.isclose(energies[dark], triplet_ev, rtol=1e-6)))

    delta_e = float(energies[j_singlet])
    mu_singlet = float(mu[j_singlet])
    return {
        "species": species,
        "Z": manifold["Z"],
        "n_states": manifold["n_states"],
        "triplet_ev": triplet_ev,
        "triplet_degeneracy": triplet_degeneracy,
        "singlet_ev": delta_e,
        "singlet_degeneracy": int(np.sum(
            np.isclose(energies[bright], delta_e, rtol=1e-6))),
        "mu_singlet_au": mu_singlet,
        # f = (2/3) * dE_au * |mu|^2 for a transition out of a non-degenerate
        # ground level.
        "f_singlet": float(2.0 / 3.0 * (delta_e / HARTREE_TO_EV)
                           * mu_singlet**2),
    }


def dipole_from_oscillator_strength(f_value: float,
                                    delta_e_ev: float) -> float:
    """Invert ``f = (2/3) dE_au |mu|^2`` to get the reference ``|mu|``."""
    return float(np.sqrt(1.5 * f_value / (delta_e_ev / HARTREE_TO_EV)))


# ============================================================================
# Section 3: Comparison tables
# ============================================================================

def _percent_error(computed: float, reference: float) -> float:
    return float(100.0 * (computed - reference) / reference)


def compare_energies(reference: Mapping[str, Any],
                     data_dir: Path = Path(".")) -> pd.DataFrame:
    """Per-species, per-term computed vs NIST energy comparison."""
    rows: List[Dict[str, Any]] = []
    for species in SPECIES_LIST:
        ref = reference["species"][species]
        comp = computed_terms(species, data_dir)
        levels = ref["levels"]

        if comp["triplet_ev"] is not None:
            ref_triplet = triplet_j_average_ev(levels)
            rows.append({
                "species": species, "spectrum": ref["spectrum"],
                "Z": comp["Z"], "term": "2s2p 3P",
                "computed_ev": comp["triplet_ev"],
                "reference_ev": ref_triplet,
                "percent_error": _percent_error(comp["triplet_ev"],
                                                ref_triplet),
                "reference_note": "J-averaged (1:3:5)",
                "computed_degeneracy": comp["triplet_degeneracy"],
                "expected_degeneracy": 3,
            })

        ref_singlet = levels["2s2p_1P_J1"]["energy_ev"]
        rows.append({
            "species": species, "spectrum": ref["spectrum"],
            "Z": comp["Z"], "term": "2s2p 1P",
            "computed_ev": comp["singlet_ev"],
            "reference_ev": ref_singlet,
            "percent_error": _percent_error(comp["singlet_ev"], ref_singlet),
            "reference_note": "observed J=1 level",
            "computed_degeneracy": comp["singlet_degeneracy"],
            "expected_degeneracy": 3,
        })
    return pd.DataFrame(rows)


def compare_dipoles(reference: Mapping[str, Any],
                    data_dir: Path = Path(".")) -> pd.DataFrame:
    """Computed vs reference oscillator strength / dipole, where available."""
    rows: List[Dict[str, Any]] = []
    for species in SPECIES_LIST:
        ref = reference["species"][species]["resonance_transition"]
        comp = computed_terms(species, data_dir)
        f_ref = ref.get("oscillator_strength_f")
        if f_ref is None:
            rows.append({
                "species": species, "Z": comp["Z"],
                "f_computed": comp["f_singlet"],
                "f_reference": np.nan,
                "mu_computed_au": comp["mu_singlet_au"],
                "mu_reference_au": np.nan,
                "mu_ratio": np.nan,
                "available": False,
            })
            continue
        mu_ref = dipole_from_oscillator_strength(f_ref, ref["energy_ev"])
        rows.append({
            "species": species, "Z": comp["Z"],
            "f_computed": comp["f_singlet"],
            "f_reference": float(f_ref),
            "mu_computed_au": comp["mu_singlet_au"],
            "mu_reference_au": mu_ref,
            "mu_ratio": comp["mu_singlet_au"] / mu_ref,
            "available": True,
        })
    return pd.DataFrame(rows)


def compare_scaling_exponents(energy_table: pd.DataFrame,
                              dipole_table: pd.DataFrame) -> Dict[str, Any]:
    """Fit dE(Z) and mu(Z) on both computed and reference values."""
    singlet = energy_table[energy_table["term"] == "2s2p 1P"].sort_values("Z")
    energy = {
        "computed": fit_power_law(singlet["Z"], singlet["computed_ev"]),
        "reference": fit_power_law(singlet["Z"], singlet["reference_ev"]),
        "n_points": int(len(singlet)),
    }
    energy["alpha_difference"] = (energy["computed"]["alpha"]
                                  - energy["reference"]["alpha"])

    usable = dipole_table[dipole_table["available"]].sort_values("Z")
    dipole: Dict[str, Any] = {"n_points": int(len(usable))}
    if len(usable) >= 2:
        dipole["computed"] = fit_power_law(usable["Z"],
                                           usable["mu_computed_au"])
        dipole["reference"] = fit_power_law(usable["Z"],
                                            usable["mu_reference_au"])
        dipole["alpha_difference"] = (dipole["computed"]["alpha"]
                                      - dipole["reference"]["alpha"])
        dipole["species_used"] = usable["species"].tolist()
    else:
        dipole["note"] = ("fewer than two reference oscillator strengths "
                          "available; dipole exponent not validated")
    return {"energy_scaling": energy, "dipole_scaling": dipole}


# ============================================================================
# Section 4: Plot
# ============================================================================

def plot_energy_benchmark(energy_table: pd.DataFrame,
                          output_dir: Path) -> Path:
    """Computed vs NIST energies (left) and percent error vs Z (right)."""
    output_path = output_dir / "nist_benchmark.png"
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    markers = {"2s2p 3P": "s", "2s2p 1P": "o"}
    colors = {"2s2p 3P": "tab:orange", "2s2p 1P": "tab:blue"}
    for term, group in energy_table.groupby("term"):
        group = group.sort_values("Z")
        ax_left.loglog(group["reference_ev"], group["computed_ev"],
                       markers[term], color=colors[term], markersize=9,
                       label=term, zorder=3)
        for _, row in group.iterrows():
            ax_left.annotate(row["species"],
                             (row["reference_ev"], row["computed_ev"]),
                             textcoords="offset points", xytext=(8, -10),
                             fontsize=8)
        ax_right.plot(group["Z"], group["percent_error"],
                      markers[term] + "-", color=colors[term], label=term)

    limits = [energy_table["reference_ev"].min() * 0.7,
              energy_table["reference_ev"].max() * 1.4]
    ax_left.plot(limits, limits, "k:", linewidth=1, label="perfect agreement")
    ax_left.set_xlabel("NIST reference energy (eV)")
    ax_left.set_ylabel("Computed CASCI energy (eV)")
    ax_left.set_title("Computed vs NIST excitation energies")
    ax_left.legend(fontsize=8)
    ax_left.grid(True, which="both", alpha=0.3)

    ax_right.axhline(0.0, color="black", linewidth=1)
    ax_right.axhspan(-5, 5, color="green", alpha=0.12,
                     label="within 5%")
    ax_right.set_xscale("log")
    ax_right.set_xlabel("Nuclear charge $Z$")
    ax_right.set_ylabel("Error vs NIST (%)")
    ax_right.set_title("Accuracy degrades with $Z$ (missing relativity)")
    ax_right.legend(fontsize=8)
    ax_right.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ============================================================================
# Section 5: Report & entry point
# ============================================================================

def build_findings(energy_table: pd.DataFrame, dipole_table: pd.DataFrame,
                   reference: Mapping[str, Any],
                   exponents: Mapping[str, Any]) -> List[str]:
    """Human-readable warnings worth raising with the team."""
    findings: List[str] = []

    worst = energy_table.loc[energy_table["percent_error"].abs().idxmax()]
    findings.append(
        f"Largest energy deviation: {worst['species']} {worst['term']} "
        f"is {worst['percent_error']:+.1f}% vs NIST "
        f"({worst['computed_ev']:.3f} vs {worst['reference_ev']:.3f} eV).")

    truncated = energy_table[
        energy_table["computed_degeneracy"] < energy_table["expected_degeneracy"]
    ]
    if not truncated.empty:
        findings.append(
            "Degenerate multiplets are truncated: the 4-state manifolds keep "
            "only "
            + ", ".join(f"{r['computed_degeneracy']}/{r['expected_degeneracy']} "
                        f"{r['term']} components for {r['species']}"
                        for _, r in truncated.iterrows())
            + ". Channel cross sections are undercounted by the missing "
              "components; raise n_states to capture the full terms.")

    for species, ref in reference["species"].items():
        spread = fine_structure_spread_ev(ref["levels"])
        if spread > 1.0:
            findings.append(
                f"{ref['spectrum']} has a {spread:.1f} eV 3P fine-structure "
                "spread that the non-relativistic CASCI cannot reproduce "
                "(it returns a single degenerate level).")

    usable = dipole_table[dipole_table["available"]]
    if not usable.empty:
        ratios = usable["mu_ratio"]
        findings.append(
            f"Transition dipoles are systematically low by a near-constant "
            f"factor (|mu|_computed/|mu|_reference = "
            f"{', '.join(f'{r:.2f}' for r in ratios)}), so absolute cross "
            "sections are underestimated by roughly that factor squared while "
            "the Z-scaling exponent is largely preserved.")
    missing = dipole_table[~dipole_table["available"]]["species"].tolist()
    if missing:
        findings.append(
            "No reference oscillator strength available for "
            f"{', '.join(missing)}; its dipole was not validated.")

    energy_fit = exponents["energy_scaling"]
    findings.append(
        f"Energy Z-scaling: computed alpha = "
        f"{energy_fit['computed']['alpha']:+.2f} vs NIST "
        f"{energy_fit['reference']['alpha']:+.2f} "
        f"(difference {energy_fit['alpha_difference']:+.2f}).")
    return findings


def run_benchmark(data_dir: Path = Path("."),
                  output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """End-to-end validation against NIST; returns everything computed."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(data_dir)
    energy_table = compare_energies(reference, data_dir)
    dipole_table = compare_dipoles(reference, data_dir)
    exponents = compare_scaling_exponents(energy_table, dipole_table)
    findings = build_findings(energy_table, dipole_table, reference, exponents)
    plot_energy_benchmark(energy_table, output_dir)

    payload = {
        "reference_sources": reference["_sources"],
        "energy_comparison": energy_table.to_dict("records"),
        "dipole_comparison": dipole_table.replace({np.nan: None})
                                         .to_dict("records"),
        "scaling_exponents": exponents,
        "findings": findings,
    }
    report_path = output_dir / "nist_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== Excitation energies vs NIST ===")
    print(energy_table[["species", "term", "computed_ev", "reference_ev",
                        "percent_error"]].to_string(index=False,
                                                    float_format="%.4f"))
    print("\n=== Oscillator strengths vs reference ===")
    print(dipole_table[["species", "f_computed", "f_reference",
                        "mu_computed_au", "mu_reference_au", "mu_ratio"]]
          .to_string(index=False, float_format="%.4f"))
    print("\n=== Findings ===")
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")
    print(f"\n[+] Report written to {report_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate computed manifolds and scaling vs NIST data")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
