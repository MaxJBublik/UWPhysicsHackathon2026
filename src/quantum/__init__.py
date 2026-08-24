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
from src.quantum.time_evolution import (
    PennyLaneTimeEvolution,
    simulate_exact_unitary_evolution,
    CollisionDynamicsSimulator,
    compute_collision_electric_field,
)

__all__ = [
    "MultiStatePauliMapper",
    "decompose_matrix_to_paulis",
    "reconstruct_matrix_from_paulis",
    "get_n_qubits",
    "generate_all_pauli_strings",
    "build_pauli_string_matrix",
    "pad_matrix_to_qubit_dimension",
    "PennyLaneTimeEvolution",
    "simulate_exact_unitary_evolution",
    "CollisionDynamicsSimulator",
    "compute_collision_electric_field",
]
