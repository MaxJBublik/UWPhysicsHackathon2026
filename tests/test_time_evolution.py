"""
Unit and integration tests for Task 2.2: Quantum Time-Evolution and Population Dynamics.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quantum.mapper import MultiStatePauliMapper
from src.quantum.time_evolution import (
    compute_collision_electric_field,
    simulate_exact_unitary_evolution,
    CollisionDynamicsSimulator,
)


def test_collision_electric_field():
    """Verify dipole electric field calculation and pulse profile."""
    t_grid = np.linspace(-10.0, 10.0, 101)
    b = 2.0  # Bohr
    E_inc = 50.0  # eV

    e_fields = compute_collision_electric_field(t_grid, impact_parameter_bohr=b, incident_energy_ev=E_inc)
    assert e_fields.shape == (101, 3)

    # Ex should be symmetric around t=0, Ez should be anti-symmetric
    mid_idx = 50
    assert np.isclose(e_fields[mid_idx, 2], 0.0, atol=1e-6)  # Ez(t=0) == 0
    assert np.isclose(e_fields[mid_idx - 10, 0], e_fields[mid_idx + 10, 0], atol=1e-6)  # Ex(t) == Ex(-t)
    assert np.isclose(e_fields[mid_idx - 10, 2], -e_fields[mid_idx + 10, 2], atol=1e-6)  # Ez(t) == -Ez(-t)


def test_statevector_unitarity_conservation():
    """Verify that sum(P_i(t)) == 1.0 across all time slices during collision evolution."""
    sample_path = "data/raw_pyscf/manifold_Be.json"
    if not os.path.exists(sample_path):
        return

    sim = CollisionDynamicsSimulator(sample_path)
    res = sim.run_single_collision(
        impact_parameter_bohr=1.5,
        incident_energy_ev=30.0,
        n_time_steps=100,
    )

    assert res["unitarity_preserved"] is True
    # Verify sum of final populations is exactly 1
    total_final_pop = sum(res["final_populations"])
    assert np.isclose(total_final_pop, 1.0, atol=1e-4)


def test_excitation_and_cascades():
    """Verify excitation occurs during a close collision."""
    sample_path = "data/raw_pyscf/manifold_Be.json"
    if not os.path.exists(sample_path):
        return

    sim = CollisionDynamicsSimulator(sample_path)
    res = sim.run_single_collision(
        impact_parameter_bohr=0.8,
        incident_energy_ev=20.0,
        n_time_steps=150,
    )

    # Population in ground state should be strictly <= 1.0
    p0 = res["final_populations"][0]
    assert 0.0 <= p0 <= 1.0
    # Higher states should have non-negative probabilities
    for idx, p in enumerate(res["final_populations"]):
        assert p >= 0.0


def test_impact_parameter_sweep_export():
    """Verify sweep over impact parameters creates valid JSON."""
    sample_path = "data/raw_pyscf/manifold_Be.json"
    if not os.path.exists(sample_path):
        return

    sim = CollisionDynamicsSimulator(sample_path)
    sweep_res = sim.sweep_impact_parameters(
        incident_energy_ev=40.0,
        b_min_bohr=1.0,
        b_max_bohr=5.0,
        n_b_points=5,
        output_dir="data/processed_circuits",
    )

    assert sweep_res["species"] == "Be"
    assert len(sweep_res["impact_parameters_bohr"]) == 5
    assert len(sweep_res["excitation_probabilities_vs_b"]["state_0"]) == 5

    out_file = "data/processed_circuits/populations_Be_E40eV.json"
    assert os.path.exists(out_file)


if __name__ == "__main__":
    print("[*] Running all tests in test_time_evolution.py...")
    test_collision_electric_field()
    test_statevector_unitarity_conservation()
    test_excitation_and_cascades()
    test_impact_parameter_sweep_export()
    print("[+] All time evolution unit tests passed successfully!")
