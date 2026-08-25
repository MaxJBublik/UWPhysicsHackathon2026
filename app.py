"""
Streamlit Web Dashboard: Quantum Mechanical Collisional Excitation of Plasma Neutrals and Ions.

Interactive GUI to simulate electron-impact collision dynamics dynamically parsed
from available PySCF electronic structure manifolds and cross-section datasets.
"""

import os
import json
from pathlib import Path
import numpy as np

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from src.quantum.mapper import MultiStatePauliMapper
from src.quantum.time_evolution import (
    CollisionDynamicsSimulator,
    compute_collision_electric_field,
    simulate_exact_unitary_evolution,
)
from src.collision.trajectories import generate_trajectory
from src.collision.interaction import electric_field
from src.analysis.cross_sections import calculate_cross_sections_and_branching_ratios
from src.analysis.scaling_laws import IsoelectronicScalingAnalyzer


def discover_manifold_files(manifold_dir: str = "data/raw_pyscf") -> dict:
    """Dynamically scans the manifold directory for valid JSON electronic structure files."""
    p = Path(manifold_dir)
    if not p.exists() or not p.is_dir():
        return {}

    discovered = {}
    for file_path in sorted(p.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "excitation_energies_ev" in data:
                    stem = file_path.stem.replace("manifold_", "")
                    discovered[stem] = {
                        "path": file_path,
                        "data": data,
                        "species": data.get("species", stem),
                        "charge": data.get("charge", 0),
                        "atomic_number": data.get("atomic_number", "N/A"),
                    }
        except (json.JSONDecodeError, OSError):
            continue

    return discovered


def find_matching_cross_section_file(stem: str, energy_ev: float, cs_dir: str = "data/cross_sections") -> Path | None:
    """Finds precomputed cross-section JSON file matching stem and incident energy."""
    p = Path(cs_dir)
    if not p.exists():
        return None

    energy_str = f"E{int(energy_ev)}eV"
    for file_path in p.glob("*.json"):
        filename = file_path.name
        if energy_str in filename and stem in filename:
            return file_path
    return None


def run_dashboard():
    if not HAS_STREAMLIT:
        print("[!] Streamlit / Matplotlib is not installed in the active environment.")
        print("    To launch the web dashboard, install streamlit: pip install streamlit")
        return

    st.set_page_config(
        page_title="Quantum Plasma Collisions | UWPhysics2026",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚛️ Quantum Mechanical Collisional Excitation of Plasma Neutrals & Ions")
    st.markdown(
        """
        **Hybrid Quantum-Classical Modeling Framework** coupling **PySCF Multi-Root Electronic Structure**
        with **PennyLane Multi-Level Time-Evolution Circuits**.
        """
    )

    # ---------------- Dynamic File Discovery ----------------
    manifolds = discover_manifold_files()

    if not manifolds:
        st.error(
            "❌ No electronic structure manifold files found in `data/raw_pyscf/`. "
            "Please run `pyscf_runner.py` or `main.py` first to generate manifold data."
        )
        return

    # ---------------- Sidebar Controls ----------------
    st.sidebar.header("🎯 Simulation Controls")

    selected_key = st.sidebar.selectbox(
        "Target Manifold / Calculation",
        options=list(manifolds.keys()),
        index=0,
        format_func=lambda k: (
            f"{k} ({manifolds[k]['species']}, Z={manifolds[k]['atomic_number']}, q=+{manifolds[k]['charge']})"
            if manifolds[k]["charge"] > 0
            else f"{k} ({manifolds[k]['species']}, Z={manifolds[k]['atomic_number']}, neutral)"
        ),
    )

    selected_item = manifolds[selected_key]
    manifold_path = selected_item["path"]
    manifold_data = selected_item["data"]

    incident_energy_ev = st.sidebar.slider(
        "Incident Electron Energy E_inc (eV)",
        min_value=5.0,
        max_value=150.0,
        value=50.0,
        step=5.0,
    )

    impact_b = st.sidebar.slider(
        "Impact Parameter b (Bohr radii, a_0)",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.25,
    )

    t_max = st.sidebar.slider("Collision Half-Window T_max (a.u.)", 5.0, 25.0, 12.0, 1.0)
    n_time_steps = st.sidebar.slider("Time Discretization Slices", 50, 300, 120, 10)

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌊 Quantum Population Dynamics",
        "🔬 PySCF Electronic Structure",
        "📊 Cross Sections & Branching Ratios",
        "⚡ Isoelectronic Z-Scaling",
    ])

    # ================= TAB 1: Dynamics =================
    with tab1:
        st.subheader(f"Collision Time-Evolution for {selected_key} (E={incident_energy_ev} eV, b={impact_b} a₀)")

        t_grid = np.linspace(-t_max, t_max, n_time_steps)
        charge = manifold_data.get("charge", 0)
        traj_type = "straight_line" if charge == 0 else "coulomb_hyperbolic"

        traj = generate_trajectory(
            t_grid, impact_b, incident_energy_ev, trajectory_type=traj_type, ion_charge=charge
        )
        e_fields = electric_field(traj.position, softening_bohr=0.1)

        mapper = MultiStatePauliMapper(manifold_data)
        pops = simulate_exact_unitary_evolution(mapper, t_grid, e_fields)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Collision Trajectory & Field Pulse")
            fig_traj, ax_traj = plt.subplots(figsize=(6, 4))
            ax_traj.plot(traj.position[:, 2], traj.position[:, 0], "b-", label=f"Trajectory ({traj_type})")
            ax_traj.plot(0, 0, "ro", markersize=10, label="Target Core")
            ax_traj.set_xlabel("z Position (Bohr radii)", fontsize=10)
            ax_traj.set_ylabel("x Position (Bohr radii)", fontsize=10)
            ax_traj.set_title("Incident Electron Orbit", fontsize=11, fontweight="bold")
            ax_traj.grid(True, linestyle="--", alpha=0.5)
            ax_traj.legend()
            st.pyplot(fig_traj)
            plt.close()

        with col2:
            st.markdown("#### 2. Multi-State Population Dynamics P_i(t)")
            fig_pop, ax_pop = plt.subplots(figsize=(6, 4))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            for idx in range(mapper.n_states):
                energy_val = manifold_data["excitation_energies_ev"][idx]
                label_txt = f"State {idx} ({energy_val:.2f} eV)"
                ax_pop.plot(t_grid, pops[:, idx], label=label_txt, color=colors[idx % len(colors)], linewidth=2)

            ax_pop.set_xlabel("Time (a.u.)", fontsize=10)
            ax_pop.set_ylabel("State Probability |c_j(t)|²", fontsize=10)
            ax_pop.set_title("Quantum Population Tracking", fontsize=11, fontweight="bold")
            ax_pop.grid(True, linestyle="--", alpha=0.5)
            ax_pop.legend(fontsize=9)
            st.pyplot(fig_pop)
            plt.close()

        st.markdown("#### 🏁 Post-Collision State Distribution")
        m_cols = st.columns(mapper.n_states)
        for idx in range(mapper.n_states):
            p_final = pops[-1, idx]
            m_cols[idx].metric(
                label=f"State {idx} ({manifold_data['excitation_energies_ev'][idx]:.2f} eV)",
                value=f"{p_final * 100:.3f}%",
            )

        total_pop = np.sum(pops[-1, :mapper.n_states])
        st.success(f"✅ Unitarity Verification: Total probability $\\sum P_i = {total_pop:.6f}$")

    # ================= TAB 2: Electronic Structure =================
    with tab2:
        st.subheader(f"PySCF Electronic Structure Manifold: {selected_key}")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Energy Levels Breakdown")
            levels_ev = manifold_data["excitation_energies_ev"]

            fig_lvl, ax_lvl = plt.subplots(figsize=(6, 4))
            for i, ev in enumerate(levels_ev):
                ax_lvl.hlines(ev, 0.2, 0.8, colors="#1f77b4", linewidth=3)
                ax_lvl.text(0.5, ev + 0.1, f"State {i}: {ev:.2f} eV", horizontalalignment="center", fontsize=9)

            ax_lvl.set_xlim(0.0, 1.0)
            ax_lvl.set_xticks([0.5])
            ax_lvl.set_xticklabels([selected_key])
            ax_lvl.set_ylabel("Excitation Energy (eV)", fontsize=10)
            ax_lvl.set_title(f"Energy Spectrum ({len(levels_ev)} States)", fontsize=11, fontweight="bold")
            ax_lvl.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_lvl)
            plt.close()

        with c2:
            st.markdown("#### Transition Dipole Matrix & Pauli Decomposition")
            st.write(f"**Qubit Encoding:** $n = \\lceil \\log_2 {mapper.n_states} \\rceil = {mapper.n_qubits}$ qubits")
            st.write(f"**Active Pauli Strings ({len(mapper.active_paulis)}):** `{', '.join(mapper.active_paulis)}`")
            st.write("**Dipole Matrix Elements $\\mu_z$ (atomic units):**")
            st.dataframe(np.array(manifold_data["dipole_matrix_z"]))

    # ================= TAB 3: Cross Sections & Branching Ratios =================
    with tab3:
        st.subheader(f"Excitation Cross-Sections & Branching Ratios: {selected_key}")
        cs_file = find_matching_cross_section_file(selected_key, incident_energy_ev)

        if cs_file and cs_file.exists():
            st.success(f"📁 Loaded precomputed results from `{cs_file.name}`")
            with open(cs_file, "r", encoding="utf-8") as f:
                cs_data = json.load(f)
            st.json(cs_data)
        else:
            st.info(f"⚡ Computing live cross-sections for {selected_key} at E={incident_energy_ev} eV...")
            sim = CollisionDynamicsSimulator(str(manifold_path))
            sweep = sim.sweep_impact_parameters(incident_energy_ev, b_min_bohr=0.5, b_max_bohr=8.0, n_b_points=20)
            recs = calculate_cross_sections_and_branching_ratios(sweep)
            st.dataframe(recs)

    # ================= TAB 4: Scaling Laws =================
    with tab4:
        st.subheader("⚡ Multi-Variant Scaling Synthesis")

        summary_file = Path("data/scaling_analysis/energy_dependent_branching_ratios.json")
        scaling_img = Path("data/scaling_analysis/isoelectronic_scaling_trends.png")

        if scaling_img.exists():
            st.image(str(scaling_img), caption="Cross Section & Branching Ratio Scaling Trends")

        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            st.markdown("#### Synthesized Multi-Species Analysis Data")
            st.json(summary_data)
        else:
            st.info("Generating dynamic scaling summary...")
            analyzer = IsoelectronicScalingAnalyzer()
            analyzer.load_all_species_manifolds()
            summary_payload = analyzer.compute_energy_dependent_branching_ratios()
            st.json(summary_payload)


if __name__ == "__main__":
    run_dashboard()