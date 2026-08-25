"""
Master Pipeline Runner: Quantum Mechanical Collisional Excitation of Plasma Neutrals and Ions.

Executes the complete end-to-end workflow:
  1. Electronic Structure Manifolds (PySCF)
  2. Quantum Hamiltonian Mapping (PennyLane Pauli strings)
  3. Quantum Time Evolution & Multi-State Population Dynamics
  4. Collision Cross-Sections & Branching Ratios
  5. Isoelectronic Z-Scaling Synthesis & Summary Report
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.electronic_structure.pyscf_runner import run_all_species
from src.quantum.time_evolution import run_all_species_simulation
from src.analysis.cross_sections import (
    calculate_cross_sections_and_branching_ratios,
    save_processed_circuit_cross_sections,
)
from src.analysis.scaling_laws import IsoelectronicScalingAnalyzer, generate_scaling_plots


def run_full_pipeline(energy_ev: float = 50.0):
    print("================================================================================")
    print("⚛️  UWPhysics2026: Quantum Collisional Excitation End-to-End Pipeline")
    print("================================================================================")

    # 1. PySCF Electronic Structure
    print("\n[Step 1/4] Checking / Generating PySCF Electronic Structure Manifolds...")
    manifold_dir = Path("data/new_raw_pyscf")
    species_list = ["Be", "C2+", "Fe22+"]

    found_species = set()
    if manifold_dir.exists():
        for file_path in manifold_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "species" in data:
                        found_species.add(data["species"])
            except (json.JSONDecodeError, OSError):
                continue  # Skip unreadable or corrupted JSON files

    all_exist = set(species_list).issubset(found_species)
    if not all_exist:
        missing = set(species_list) - found_species
        print(f"[*] Missing manifold(s) for {missing}. Running electronic structure calculations...")
        run_all_species(n_states=4, output_dir=str(manifold_dir), input_dir="data/new_raw_pyscf")
    else:
        print(f"[+] All PySCF manifold files verified in {manifold_dir}/ (Found species: {found_species})")
    # 2. Quantum Time Evolution & Population Tracking
    print(f"\n[Step 2/4] Simulating Quantum Time Evolution & Population Dynamics (E={energy_ev} eV)...")
    run_all_species_simulation(energy_ev=energy_ev, input_dir="data/new_raw_pyscf")

    # 3. Integrate Cross-Sections & Branching Ratios
    print("\n[Step 3/4] Integrating Cross-Sections & Calculating Branching Ratios...")
    processed_dir = Path("data/processed_circuits")
    output_dir = Path("data/cross_sections")
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = sorted(processed_dir.glob("populations_*.json"))
    for p_file in processed_files:
        out_file = output_dir / p_file.name.replace("populations_", "cross_sections_")
        save_processed_circuit_cross_sections(p_file, out_file)

    print(f"[+] Computed and saved cross-sections for {len(processed_files)} sweep files.")

    # 4. Isoelectronic Z-Scaling Synthesis
    print("\n[Step 4/4] Synthesizing Isoelectronic Z-Scaling Laws & Branching Ratio Spectra...")
    analyzer = IsoelectronicScalingAnalyzer()
    analyzer.save_scaling_summary()
    analyzer.compute_energy_dependent_branching_ratios(energy_grid_ev=[15.0, 30.0, 50.0, 75.0, 100.0, 150.0])
    generate_scaling_plots()

    # Final Summary Output
    print("\n================================================================================")
    print("🏁 Pipeline Execution Complete! Summary Table:")
    print("================================================================================")
    summary = analyzer.extract_electronic_structure_scaling()
    print(f"{'Species':<8} | {'Z':<4} | {'Charge':<6} | {'ΔE(³P) (eV)':<11} | {'ΔE(¹P) (eV)':<11} | {'Res. Dipole μ (a₀)':<18} | {'Osc. Str. f':<12} | {'μ × Z (a₀)':<10}")
    print("-" * 95)
    for r in summary["scaling_records"]:
        print(f"{r['species']:<8} | {r['atomic_number_Z']:<4} | {r['charge_q']:<6} | {r['triplet_energy_dE_triplet_ev']:<11.2f} | {r['resonance_energy_dE_res_ev']:<11.2f} | {r['resonance_dipole_mu_res_au']:<18.4f} | {r['oscillator_strength_f_res']:<12.4f} | {r['dipole_scaling_product_mu_times_Z']:<10.2f}")

    print("\n[+] To view the interactive Web GUI, run:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Full End-to-End Pipeline")
    parser.add_argument("--energy", type=float, default=50.0, help="Incident electron energy in eV")
    args = parser.parse_args()
    run_full_pipeline(energy_ev=args.energy)