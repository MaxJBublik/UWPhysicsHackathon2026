import json

import numpy as np
import pytest

from src.collision.interaction import collision_hamiltonian, electric_field, hamiltonian
from src.collision.trajectories import (HARTREE_TO_EV,
    coulomb_hyperbolic_trajectory, incident_velocity, straight_line_trajectory)
from src.electronic_structure.pyscf_runner import MultiRootElectronicStructure


def test_straight_line_closest_approach_and_asymptote():
    time = np.array([-40.0, 0.0, 40.0])
    path = straight_line_trajectory(time, 2.5, 50.0)
    assert path.position.shape == (3, 3)
    assert path.radius[1] == pytest.approx(2.5)
    assert path.velocity[:, 2] == pytest.approx(incident_velocity(50.0))
    assert path.position[0, 2] == pytest.approx(-path.position[-1, 2])


def test_coulomb_hyperbola_conserves_energy_and_angular_momentum():
    time = np.linspace(-100.0, 100.0, 501)
    charge, impact, energy_ev = 2.0, 3.0, 80.0
    path = coulomb_hyperbolic_trajectory(time, impact, energy_ev, charge)
    energy = 0.5 * np.sum(path.velocity**2, axis=1) - charge / path.radius
    angular = np.linalg.norm(np.cross(path.position, path.velocity), axis=1)
    assert energy == pytest.approx(energy_ev / HARTREE_TO_EV, rel=2e-11)
    assert angular == pytest.approx(impact * incident_velocity(energy_ev), rel=2e-11)
    assert np.argmin(path.radius) == len(time) // 2


def test_coulomb_orbit_reaches_incident_speed_asymptotically():
    path = coulomb_hyperbolic_trajectory([-1e5, 1e5], 2.0, 100.0, 22.0)
    assert np.linalg.norm(path.velocity, axis=1) == pytest.approx(
        incident_velocity(100.0), rel=2e-4)


def test_electric_field_direction_and_singularity_guard():
    assert electric_field([2.0, 0.0, 0.0]) == pytest.approx([-0.25, 0.0, 0.0])
    with pytest.raises(ValueError, match="singular"):
        electric_field([0.0, 0.0, 0.0])
    assert np.all(np.isfinite(electric_field([0, 0, 0], softening_bohr=0.1)))


def test_hamiltonian_is_hermitian_and_has_expected_coupling():
    dipoles = np.zeros((3, 2, 2))
    dipoles[0, 0, 1] = dipoles[0, 1, 0] = 0.5
    matrix = hamiltonian([0.0, 0.2], dipoles, [2.0, 0.0, 0.0])
    assert np.allclose(matrix, [[0.0, 0.125], [0.125, 0.2]])
    assert np.allclose(matrix, matrix.conj().T)


def test_time_dependent_hamiltonian_shape_and_far_field_limit():
    energies = np.array([0.0, 0.1, 0.3])
    dipoles = np.zeros((3, 3, 3))
    dipoles[2, 0, 1] = dipoles[2, 1, 0] = 1.0
    matrices, path = collision_hamiltonian(
        [-1000.0, 0.0, 1000.0], energies, dipoles, 2.0, 30.0,
        return_trajectory=True)
    assert matrices.shape == (3, 3, 3)
    assert path.position.shape == (3, 3)
    assert np.allclose(matrices, matrices.conj().transpose(0, 2, 1))
    assert np.allclose(matrices[0], np.diag(energies), atol=1e-6)
    assert np.allclose(matrices[-1], np.diag(energies), atol=1e-6)


def test_mock_manifold_is_complete_and_json_serializable(tmp_path):
    calculation = MultiRootElectronicStructure("Be", n_states=7)
    calculation.results = calculation._generate_mock_manifold()
    with open(calculation.save_json(tmp_path), encoding="utf-8") as stream:
        saved = json.load(stream)
    assert saved["n_states"] == len(saved["energies_au"]) == 7
    assert np.shape(saved["dipole_matrix_z"]) == (7, 7)
    assert saved["energies_au"][0] == 0.0


@pytest.mark.parametrize("bad_energy", [0.0, -1.0, np.nan])
def test_invalid_incident_energy_rejected(bad_energy):
    with pytest.raises(ValueError):
        straight_line_trajectory([0.0], 1.0, bad_energy)
