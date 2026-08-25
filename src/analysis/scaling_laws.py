r"""
Isoelectronic Z-Scaling Laws & Multi-State Energy-Dependent Branching Ratio Synthesis.
Supports arbitrary N-state electronic manifolds (N >= 4).
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
    Synthesizes electronic structure, collision cross-sections, and branching
    ratios across the Beryllium isoelectronic series for arbitrary N-state manifolds.
    """

    def __init__(self, raw_pyscf_dir: str = "data/raw_pyscf"):
        self.raw_dir = Path(raw_pyscf_dir)
        self.species_keys = ["Be", "C2+", "Fe22+"]
        self.manifolds: Dict[str, Dict[str, Any]] = {}
        self.load_all_species_manifolds()

    def load_all_species_manifolds(self):
        """Loads available manifold JSON files for Be, C2+, Fe22+."""
        for sp in self.species_keys:
            path = self.raw_dir / f"manifold_{sp}.json"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.manifolds[sp] = json.load(f)

    def extract_electronic_structure_scaling(self) -> Dict[str, Any]:
        r"""
        Extracts electronic scaling parameters across all available states in the manifold.
        """
        summary_table = []
        for sp, data in self.manifolds.items():
            Z = data.get("atomic_number", data.get("Z", 4))
            q = data.get("charge", 0)

            # Safely extract or compute excitation energies in eV
            if "excitation_energies_ev" in data:
                energies_ev = data["excitation_energies_ev"]
            else:
                energies_au = np.array(data["energies_au"], dtype=float)
                energies_ev = (energies_au - energies_au[0]) * 27.211386

            dE_triplet_ev = energies_ev[1] if len(energies_ev) > 1 else 0.0

            # Scan all N excited states for the dominant dipole-allowed resonance state
            n_states = len(energies_ev)
            max_mu = 0.0
            res_state_idx = 1
            for j in range(1, n_states):
                mux = complex(data["dipole_matrix_x"][0][j])
                muy = complex(data["dipole_matrix_y"][0][j])
                muz = complex(data["dipole_matrix_z"][0][j])
                mu_mag = float(np.sqrt(abs(mux)**2 + abs(muy)**2 + abs(muz)**2))
                if mu_mag > max_mu:
                    max_mu = mu_mag
                    res_state_idx = j

            dE_singlet_ev = energies_ev[res_state_idx] if len(energies_ev) > res_state_idx else 0.0
            dE_singlet_au = dE_singlet_ev / 27.211386
            f_res = (2.0 / 3.0) * dE_singlet_au * (max_mu ** 2)
            mu_times_Z = max_mu * Z

            summary_table.append({
                "species": sp,
                "atomic_number_Z": Z,
                "charge_q": q,
                "num_states": n_states,
                "resonance_state_index": res_state_idx,
                "triplet_energy_dE_triplet_ev": dE_triplet_ev,
                "resonance_energy_dE_res_ev": dE_singlet_ev,
                "resonance_dipole_mu_res_au": max_mu,
                "oscillator_strength_f_res": f_res,
                "dipole_scaling_product_mu_times_Z": mu_times_Z,
            })

        return {
            "series": "Beryllium Isoelectronic Sequence",
            "scaling_records": summary_table,
        }

    def compute_energy_dependent_branching_ratios(
        self,
        energy_grid_ev: Optional[List[float]] = None,
        output_dir: str = "data/scaling_analysis",
    ) -> Dict[str, Any]:
        """
        Sweeps incident electron energy across energy_grid_ev for each species,
        simulates collision circuits, integrates cross-sections, and computes
        Branching Ratios across all N states as a function of incident energy.
        """
        if energy_grid_ev is None:
            energy_grid_ev = [15.0, 30.0, 50.0, 75.0, 100.0, 150.0]

        results_by_species: Dict[str, Any] = {}
        os.makedirs(output_dir, exist_ok=True)

        for sp in self.species_keys:
            manifold_path = self.raw_dir / f"manifold_{sp}.json"
            if not manifold_path.exists():
                continue

            sim = CollisionDynamicsSimulator(str(manifold_path))
            species_energy_records = []

            for E_inc in energy_grid_ev:
                sweep_res = sim.sweep_impact_parameters(
                    incident_energy_ev=E_inc,
                    b_min_bohr=0.5,
                    b_max_bohr=10.0,
                    n_b_points=25,
                    output_dir="data/processed_circuits",
                )

                cs_records = calculate_cross_sections_and_branching_ratios(sweep_res)
                
                cs_out_path = Path("data/cross_sections") / f"cross_sections_{sp}_E{int(E_inc)}eV.json"
                save_processed_circuit_cross_sections(sweep_res, cs_out_path)

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
    """Generates publication-quality matplotlib figures supporting N-state manifolds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not installed; skipping plot image generation.")
        return

    summary_file = Path(output_dir) / "energy_dependent_branching_ratios.json"
    if not summary_file.exists():
        print(f"[!] Summary file missing at {summary_file}. Run compute_energy_dependent_branching_ratios() first.")
        return

    with open(summary_file, "r") as f:
        data = json.load(f)

    species_dict = data["species_data"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {"Be": "blue", "C2+": "green", "Fe22+": "red"}
    markers = {"Be": "o", "C2+": "s", "Fe22+": "^"}
    line_styles = ["-", "--", "-.", ":"]

    # --- Plot 1: Total Excitation Cross-Section vs Incident Energy ---
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

    # --- Plot 2: Dynamically Plot All Active State Branching Ratios (N-State General) ---
    for sp, sp_data in species_dict.items():
        energies = [r["incident_energy_ev"] for r in sp_data["energy_sweep_results"]]
        
        # Discover all state keys across the sweep
        all_state_keys = set()
        for r in sp_data["energy_sweep_results"]:
            all_state_keys.update(r["branching_ratios"].keys())
            
        sorted_state_keys = sorted(all_state_keys, key=lambda k: int(k.split("_")[1]))
        base_color = colors.get(sp, "black")
        sp_marker = markers.get(sp, "o")

        # Plot each active excitation channel (BR > 0.1% at any energy point)
        for s_key in sorted_state_keys:
            state_num = int(s_key.split("_")[1])
            br_vals = [r["branching_ratios"].get(s_key, 0.0) * 100.0 for r in sp_data["energy_sweep_results"]]
            
            if max(br_vals) > 0.1:  # Filter out spin-forbidden 0% channels
                ls = line_styles[(state_num - 1) % len(line_styles)]
                ax2.plot(
                    energies,
                    br_vals,
                    label=f"{sp} ($0 \\to {state_num}$ Channel)",
                    color=base_color,
                    marker=sp_marker,
                    linestyle=ls,
                    linewidth=2,
                    alpha=0.85,
                )

    ax2.set_xlabel("Incident Electron Energy (eV)", fontsize=12)
    ax2.set_ylabel(r"Branching Ratio $\mathcal{B}_{0 \to j}$ (%)", fontsize=12)
    ax2.set_title(r"Branching Ratio Spectrum $\mathcal{B}(E_{\mathrm{inc}})$ ($N$-State General)", fontsize=13, fontweight="bold")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9, loc="best", ncol=max(1, len(species_dict) // 2))

    plt.tight_layout()
    plot_path = Path(output_dir) / "isoelectronic_scaling_trends.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved updated N-state scaling plot to: {plot_path}")


if __name__ == "__main__":
    analyzer = IsoelectronicScalingAnalyzer()
    analyzer.save_scaling_summary()
    analyzer.compute_energy_dependent_branching_ratios()
    generate_scaling_plots()