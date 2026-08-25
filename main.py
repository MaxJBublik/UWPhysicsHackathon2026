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


def run_full_pipeline(energy_ev: float = 50.0, manifold_dir: str | Path = "data/raw_pyscf", force_classic: bool = False):
    print("================================================================================")
    print("⚛️  UWPhysics2026: Quantum Collisional Excitation End-to-End Pipeline")
    print("================================================================================")

    # 1. PySCF Electronic Structure
    print("\n[Step 1/4] Checking / Generating PySCF Electronic Structure Manifolds...")
    if isinstance(manifold_dir, str):
        manifold_dir = Path(manifold_dir)

    # Delegate validation or generation directly to run_all_species
    if manifold_dir.exists() and any(manifold_dir.glob("*.json")):
        print(f"[*] Validating existing manifolds in {manifold_dir}/...")
        paths = run_all_species(input_dir=manifold_dir)
    else:
        print(f"[*] Missing manifolds in {manifold_dir}/. Running electronic structure calculations...")
        paths = run_all_species(n_states=4, output_dir=str(manifold_dir))
    print(f"[+] Electronic structure manifolds ready. Found {len(paths)} species.")
    # 2. Quantum Time Evolution & Population Tracking
    print(f"\n[Step 2/4] Simulating Quantum Time Evolution & Population Dynamics (E={energy_ev} eV)...")
    run_all_species_simulation(energy_ev=energy_ev, input_paths=paths, force_classic=force_classic)

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
    analyzer = IsoelectronicScalingAnalyzer(
        raw_pyscf_dir=manifold_dir,
        manifold_paths=paths
    )
    analyzer.save_scaling_summary()
    analyzer.compute_energy_dependent_branching_ratios(
        energy_grid_ev=[6.0, 8.0, 10.0, 12.0, 15.0, 30.0, 50.0, 75.0, 100.0, 150.0]
    )
    generate_scaling_plots()

    # Final Summary Output
    print("\n================================================================================")
    print("🏁 Pipeline Execution Complete! Summary Table:")
    print("================================================================================")
    
    summary = analyzer.extract_electronic_structure_scaling()
    records = summary.get("scaling_records", [])
    print(records)
    if not records:
        print("[!] No scaling records available to display.")
    else:
        header = f"{'Species':<16} | {'Z':>3} | {'Charge':>6} | {'ΔE(³P) (eV)':>11} | {'ΔE(¹P) (eV)':>11} | {'Res. Dipole μ (a₀)':>18} | {'Osc. Str. f':>12} | {'μ × Z (a₀)':>10}"
        print(header)
        print("-" * len(header))
        for r in records:
            # Extract scalar float if oscillator strength is passed as a list/array
            osc_raw = r["oscillator_strength_f_res"]
            if isinstance(osc_raw, (list, tuple)):
                pos_vals = [float(v) for v in osc_raw if v > 0]
                osc_val = max(pos_vals) if pos_vals else float(osc_raw[0])
            else:
                osc_val = float(osc_raw)

            print(
                f"{r['species']:<16} | "
                f"{r['atomic_number_Z']:>3} | "
                f"{r['charge_q']:>6} | "
                f"{r['triplet_energy_dE_triplet_ev']:>11.2f} | "
                f"{r['resonance_energy_dE_res_ev']:>11.2f} | "
                f"{r['resonance_dipole_mu_res_au']:>18.4f} | "
                f"{osc_val:>12.4f} | "
                f"{r['dipole_scaling_product_mu_times_Z']:>10.2f}"
            )

    print("\n[+] To view the interactive Web GUI, run:")
    print("    streamlit run app.py\n")

def cleanup_directories(target_dirs=("data/cross_sections", "data/processed_circuits")):
    """Removes all existing output files from specified directories before starting a new run."""
    for dir_path in target_dirs:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            for file in p.glob("*"):
                if file.is_file():
                    file.unlink()
            print(f"[*] Cleared previous artifacts in: {dir_path}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Full End-to-End Pipeline")
    parser.add_argument("--energy", type=float, default=50.0, help="Incident electron energy in eV")
    parser.add_argument("--manifold-dir", type=str, default="data/raw_pyscf", help="Directory for PySCF manifolds")
    parser.add_argument("--force-classic", action="store_true", help="Force use of classic (non-PennyLane) simulation")
    args = parser.parse_args()

    # Clean previous run output files before running pipeline
    cleanup_directories()

    run_full_pipeline(energy_ev=args.energy, manifold_dir=args.manifold_dir, force_classic=args.force_classic)