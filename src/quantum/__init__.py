"""
PennyLane quantum circuit simulation and Hamiltonian mapper module.
"""

from src.quantum.mapper import (
    MultiStatePauliMapper,
    decompose_matrix_to_paulis,
    reconstruct_matrix_from_paulis,
    get_n_qubits,
    generate_all_pauli_strings,
    build_pauli_string_matrix,
    pad_matrix_to_qubit_dimension,
)

__all__ = [
    "MultiStatePauliMapper",
    "decompose_matrix_to_paulis",
    "reconstruct_matrix_from_paulis",
    "get_n_qubits",
    "generate_all_pauli_strings",
    "build_pauli_string_matrix",
    "pad_matrix_to_qubit_dimension",
]
