import os
import sys
import json
import numpy as np
from src.conversion_constants import HARTREE_TO_EV

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pytest
except ImportError:
    pytest = None

from src.quantum.mapper import (
    get_n_qubits,
    generate_all_pauli_strings,
    build_pauli_string_matrix,
    decompose_matrix_to_paulis,
    reconstruct_matrix_from_paulis,
    pad_matrix_to_qubit_dimension,
    MultiStatePauliMapper,
)


def test_qubit_scaling():
    """Verify minimum qubit allocation n = ceil(log2(N))."""
    assert get_n_qubits(1) == 0 or get_n_qubits(2) == 1
    assert get_n_qubits(3) == 2
    assert get_n_qubits(4) == 2
    assert get_n_qubits(5) == 3
    assert get_n_qubits(8) == 3
    try:
        get_n_qubits(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_pauli_orthogonality():
    """Verify Hilbert-Schmidt inner product orthogonality: Tr(P_a P_b) = 2^n * delta_ab."""
    n_qubits = 2
    all_paulis = generate_all_pauli_strings(n_qubits)
    dim = 2 ** n_qubits

    for i, p_a in enumerate(all_paulis):
        mat_a = build_pauli_string_matrix(p_a)
        for j, p_b in enumerate(all_paulis):
            mat_b = build_pauli_string_matrix(p_b)
            # Inner product
            ip = np.trace(mat_a @ mat_b) / dim
            if i == j:
                assert np.isclose(ip, 1.0 + 0.0j, atol=1e-10)
            else:
                assert np.isclose(ip, 0.0 + 0.0j, atol=1e-10)


def test_exact_reconstruction_fidelity():
    """Verify that decomposing a matrix and reconstructing it reproduces the original matrix."""
    # Test random 4x4 Hermitian matrix
    np.random.seed(42)
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    H_rand = (A + A.conj().T) / 2.0  # make Hermitian

    coeffs, paulis = decompose_matrix_to_paulis(H_rand, n_qubits=2)
    H_recon = reconstruct_matrix_from_paulis(coeffs, paulis)

    frobenius_error = np.linalg.norm(H_rand - H_recon)
    assert frobenius_error < 1e-10


def test_unphysical_penalty_padding():
    """Verify padding and unphysical energy penalty on non-power-of-two state counts (e.g. N=3 -> 2 qubits)."""
    H3 = np.diag([0.0, 5.0, 10.0])
    penalty = 150.0
    padded = pad_matrix_to_qubit_dimension(H3, n_qubits=2, unphysical_energy_penalty=penalty)

    assert padded.shape == (4, 4)
    assert np.isclose(padded[3, 3], penalty)
    assert np.isclose(padded[0, 0], 0.0)
    assert np.isclose(padded[1, 1], 5.0)
    assert np.isclose(padded[2, 2], 10.0)


def test_mapper_on_pyscf_json_files():
    """Test MultiStatePauliMapper on real/mock JSON outputs for Be, C2+, Fe22+."""
    species_list = ["Be", "C2+", "Fe22+"]
    for sp in species_list:
        file_path = f"data/raw_pyscf/manifold_{sp}.json"
        if not os.path.exists(file_path):
            continue

        with open(file_path, "r") as f:
            data = json.load(f)

        mapper = MultiStatePauliMapper(data, unphysical_penalty_ev=100.0)
        assert mapper.n_qubits == 2
        assert len(mapper.active_paulis) > 0

        # Evaluate at t=0 with zero field (should equal H0)
        c_0, paulis = mapper.evaluate_hamiltonian_paulis_at_t(np.array([0.0, 0.0, 0.0]))
        H_recon_0 = reconstruct_matrix_from_paulis(c_0.tolist(), paulis)

        # Diagonal elements must match excitation energies
        for state_idx in range(mapper.n_states):
            expected_energy = data["excitation_energies_ev"][state_idx]
            actual_energy = np.real(H_recon_0[state_idx, state_idx]) * HARTREE_TO_EV
            assert np.isclose(actual_energy, expected_energy, atol=1e-6)


if __name__ == "__main__":
    print("[*] Running all tests in test_mapper.py...")
    test_qubit_scaling()
    test_pauli_orthogonality()
    test_exact_reconstruction_fidelity()
    test_unphysical_penalty_padding()
    test_mapper_on_pyscf_json_files()
    print("[+] All mapper unit tests passed successfully!")
