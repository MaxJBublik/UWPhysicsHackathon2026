r"""
Isoelectronic Z-Scaling Laws & Energy-Dependent Branching Ratio Synthesis.

This module models and synthesizes the physical scaling trends across the
Beryllium isoelectronic sequence (Be, C 2+, Fe 22+):
  1. Energy Gap Scaling: \Delta E(Z) \sim Z (or Z^2 for core transitions).
  2. Dipole Matrix Scaling: \mu(Z) \sim 1/Z.
  3. Oscillator Strength Scaling: f_{01} \sim \Delta E \cdot \mu^2 \sim 1/Z.
  4. Coulomb Focusing & Acceleration: F_Coulomb(q, E_inc) = 1 + q / (b * E_inc).
  5. Multi-Energy Branching Ratio curves: \mathcal{B}_{0 \to j}(E_inc) vs Incident Energy.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.quantum.time_evolution import CollisionDynamicsSimulator
from src.analysis.cross_sections import (
    calculate_cross_sections_and_branching_ratios,
    save_processed_circuit_cross_sections,
)


class IsoelectronicScalingAnalyzer:
    """
    Synthesizes and compares electronic structure, collision cross-sections,
    and branching ratios across the Beryllium isoelectronic series.
    """

    def __init__(
        self,
        raw_pyscf_dir: str | Path = "data/raw_pyscf",
        manifold_paths: Optional[List[str | Path]] = None,
    ):
        self.raw_dir = Path(raw_pyscf_dir)
        self.manifolds: Dict[str, Dict[str, Any]] = {}
        self.manifold_paths: Dict[str, Path] = {}
        self.species_keys: List[str] = []

        self.load_all_species_manifolds(manifold_paths)

    def load_all_species_manifolds(
        self, manifold_paths: Optional[List[str | Path]] = None
    ) -> None:
        """Loads manifold JSON files either from explicit paths or by scanning raw_dir."""
        paths_to_load: List[Path] = []

        if manifold_paths:
            paths_to_load = [Path(p) for p in manifold_paths]
        elif self.raw_dir.exists() and self.raw_dir.is_dir():
            paths_to_load = sorted(list(self.raw_dir.glob("*.json")))

        for path in paths_to_load:
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sp = data.get("species")
                    if sp:
                        self.manifolds[sp] = data
                        self.manifold_paths[sp] = path
            except (json.JSONDecodeError, OSError):
                continue

        self.species_keys = list(self.manifolds.keys())

    def extract_electronic_structure_scaling(self) -> Dict[str, Any]:
        r"""
        Extracts and tabulates electronic scaling parameters (Z, q, \Delta E, \mu_res, f_res).
        """
        summary_table = []
        for sp in self.species_keys:
            data = self.manifolds[sp]
            Z = data["atomic_number"]
            q = data.get("charge", 0)
            
            # Extract excitation energies for lowest triplet (state 1) and resonance singlet
            energies_ev = data["excitation_energies_ev"]
            dE_triplet_ev = energies_ev[1] if len(energies_ev) > 1 else 0.0
            
            # Find the strongest dipole-allowed transition from ground state (0 -> j)
            n_states = len(energies_ev)
            max_mu = 0.0
            res_state_idx = 1
            for j in range(1, n_states):
                mux = data["dipole_matrix_x"][0][j]
                muy = data["dipole_matrix_y"][0][j]
                muz = data["dipole_matrix_z"][0][j]
                mu_mag = float(np.sqrt(mux**2 + muy**2 + muz**2))
                if mu_mag > max_mu:
                    max_mu = mu_mag
                    res_state_idx = j

            dE_singlet_ev = energies_ev[res_state_idx] if len(energies_ev) > res_state_idx else 0.0
            osc_f_res = data["oscillator_strengths"][res_state_idx] if len(data.get("oscillator_strengths", [])) > res_state_idx else 0.0

            summary_table.append({
                "species": sp,
                "atomic_number_Z": Z,
                "charge_q": q,
                "triplet_energy_dE_triplet_ev": dE_triplet_ev,
                "resonance_energy_dE_res_ev": dE_singlet_ev,
                "resonance_dipole_mu_res_au": max_mu,
                "oscillator_strength_f_res": osc_f_res,
                "dipole_scaling_product_mu_times_Z": max_mu * Z,
            })

        return {
            "series": "Beryllium Isoelectronic Sequence (4 electrons)",
            "scaling_records": summary_table,
        }

    def compute_energy_dependent_branching_ratios(
        self,
        energy_grid_ev: Optional[List[float]] = None,
        output_dir: str = "data/scaling_analysis",
    ) -> Dict[str, Any]:
        """
        Sweeps incident electron energy across energy_grid_ev for each loaded species manifold,
        simulates collision circuits, integrates cross-sections, and computes Branching Ratios.
        """
        if energy_grid_ev is None:
            energy_grid_ev = [15.0, 30.0, 50.0, 75.0, 100.0, 150.0]

        results_by_species: Dict[str, Any] = {}
        os.makedirs(output_dir, exist_ok=True)

        for sp in self.species_keys:
            manifold_path = self.manifold_paths.get(sp)
            if not manifold_path or not manifold_path.exists():
                continue

            sim = CollisionDynamicsSimulator(str(manifold_path))
            species_energy_records = []

            for E_inc in energy_grid_ev:
                # 1. Sweep impact parameter
                sweep_res = sim.sweep_impact_parameters(
                    incident_energy_ev=E_inc,
                    b_min_bohr=0.5,
                    b_max_bohr=10.0,
                    n_b_points=25,
                    output_dir="data/processed_circuits",
                )

                # 2. Calculate Cross Sections & Branching Ratios
                cs_records = calculate_cross_sections_and_branching_ratios(sweep_res)
                
                # Save cross section table
                cs_out_path = Path("data/cross_sections") / f"cross_sections_{sp}_E{int(E_inc)}eV.json"
                save_processed_circuit_cross_sections(sweep_res, cs_out_path)

                # Extract state_1 and total cross sections
                state1_rec = next((r for r in cs_records if r["state_index"] == 1), cs_records[0])
                
                record = {
                    "incident_energy_ev": E_inc,
                    "total_excitation_sigma_mb": state1_rec["total_excitation_sigma_mb"],
                    "state_cross_sections_mb": {r["state_label"]: r["sigma_mb"] for r in cs_records},
                    "branching_ratios": {r["state_label"]: r["branching_ratio"] for r in cs_records},
                }
                species_energy_records.append(record)

            results_by_species[sp] = {
                "species": sp,
                "atomic_number": self.manifolds[sp]["atomic_number"],
                "charge": self.manifolds[sp].get("charge", 0),
                "energy_grid_ev": energy_grid_ev,
                "energy_sweep_results": species_energy_records,
            }

        synthesis_payload = {
            "series": "Beryllium Isoelectronic Sequence",
            "energy_grid_ev": energy_grid_ev,
            "species_data": results_by_species,
        }

        out_file = Path(output_dir) / "energy_dependent_branching_ratios.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(synthesis_payload, f, indent=2)
        print(f"[+] Saved energy-dependent branching ratios to: {out_file}")

        return synthesis_payload

    def save_scaling_summary(self, output_dir: str = "data/scaling_analysis") -> Path:
        """Saves electronic structure scaling summary to JSON."""
        summary = self.extract_electronic_structure_scaling()
        os.makedirs(output_dir, exist_ok=True)
        out_file = Path(output_dir) / "isoelectronic_scaling_summary.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Saved isoelectronic scaling summary to: {out_file}")
        return out_file


def generate_scaling_plots(output_dir: str = "data/scaling_analysis"):
    """Generates publication-quality matplotlib figures if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not installed; skipping plot image generation.")
        return

    summary_file = Path(output_dir) / "energy_dependent_branching_ratios.json"
    if not summary_file.exists():
        return

    with open(summary_file, "r") as f:
        data = json.load(f)

    species_dict = data["species_data"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"Be": "blue", "C2+": "green", "Fe22+": "red"}
    markers = {"Be": "o", "C2+": "s", "Fe22+": "^"}

    # Plot 1: Total Excitation Cross-Section vs Incident Energy
    for sp, sp_data in species_dict.items():
        energies = [r["incident_energy_ev"] for r in sp_data["energy_sweep_results"]]
        sigmas = [r["total_excitation_sigma_mb"] for r in sp_data["energy_sweep_results"]]
        ax1.plot(
            energies,
            sigmas,
            label=f"{sp} (Z={sp_data['atomic_number']}, q={sp_data['charge']})",
            color=colors.get(sp, "black"),
            marker=markers.get(sp, "o"),
            linewidth=2,
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("Incident Electron Energy (eV)", fontsize=12)
    ax1.set_ylabel(r"Total Excitation Cross-Section $\sigma_{\mathrm{tot}}$ (Mb)", fontsize=12)
    ax1.set_title("Isoelectronic Cross-Section Scaling", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    # Plot 2: Branching Ratio to State 1 vs Incident Energy
    for sp, sp_data in species_dict.items():
        energies = [r["incident_energy_ev"] for r in sp_data["energy_sweep_results"]]
        br1 = [r["branching_ratios"].get("state_1", 0) * 100 for r in sp_data["energy_sweep_results"]]
        ax2.plot(
            energies,
            br1,
            label=f"{sp} ($0 \\to 1$ Dipole Channel)",
            color=colors.get(sp, "black"),
            marker=markers.get(sp, "o"),
            linewidth=2,
        )

    ax2.set_xlabel("Incident Electron Energy (eV)", fontsize=12)
    ax2.set_ylabel(r"Branching Ratio $\mathcal{B}_{0 \to 1}$ (%)", fontsize=12)
    ax2.set_title(r"Branching Ratio $\mathcal{B}(E_{\mathrm{inc}})$ vs Energy", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plot_path = Path(output_dir) / "isoelectronic_scaling_trends.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved scaling plot to: {plot_path}")


if __name__ == "__main__":
    # Example usage with explicit path list passed from previous stage
    analyzer = IsoelectronicScalingAnalyzer()
    analyzer.save_scaling_summary()
    analyzer.compute_energy_dependent_branching_ratios()
    generate_scaling_plots()