r"""
Quantum Hamiltonian Mapper for Multi-State Atomic/Ionic Manifolds.

Task 2.1: Maps an N-state manifold (E_0, E_1, ..., E_{N-1}) and transition
dipole couplings into an n-qubit Pauli operator basis:
    H(t) = \sum_k c_k(t) \hat{P}_k, where \hat{P}_k \in {I, X, Y, Z}^{\otimes n}.

Why N <= 8 is optimal for this hackathon:
  1. Qubit Scaling: n = ceil(log2(N)).
     - N <= 4 requires n = 2 qubits (up to 4^2 = 16 Pauli terms).
     - N <= 8 requires n = 3 qubits (up to 4^3 = 64 Pauli terms).
     - N > 8 requires n >= 4 qubits (up to 4^4 = 256 Pauli terms), leading
       to quadratic/exponential Trotter circuit gate depth growth.
  2. Physical Completeness:
     - For 4-electron systems (Be isoelectronic sequence), the lowest 4 to 8
       roots (1s^2 2s^2, 1s^2 2s2p, 1s^2 2p^2) capture >95% of non-adiabatic
       impact excitation cross sections without entering ionization continuum.
"""

import json
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

# Single-qubit Pauli matrices in standard computational basis {|0>, |1>}
PAULI_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

PAULI_DICT = {
    "I": PAULI_I,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
}


# ============================================================================
# Section 1: Helper Functions for Pauli Matrix Construction and Algebra
# ============================================================================

def get_n_qubits(n_states: int) -> int:
    """
    Computes the minimum number of qubits n required to encode N states:
    n = ceil(log2(N)).
    """
    if n_states < 1:
        raise ValueError(f"Number of states must be >= 1, got {n_states}")
    return int(np.ceil(np.log2(n_states)))


def build_pauli_string_matrix(pauli_str: str) -> np.ndarray:
    r"""
    Constructs the 2^n x 2^n matrix representation of an n-qubit Pauli string
    via Kronecker (tensor) products.
    Example: 'XZ' -> X \otimes Z
    """
    mat = np.array([[1.0]], dtype=complex)
    for char in pauli_str.upper():
        if char not in PAULI_DICT:
            raise ValueError(f"Invalid Pauli character '{char}'. Allowed: I, X, Y, Z")
        mat = np.kron(mat, PAULI_DICT[char])
    return mat


def generate_all_pauli_strings(n_qubits: int) -> List[str]:
    """
    Generates all 4^n possible Pauli string permutations of length n.
    Example for n=2: ['II', 'IX', 'IY', 'IZ', 'XI', ..., 'ZZ']
    """
    chars = ["I", "X", "Y", "Z"]
    return ["".join(p) for p in itertools.product(chars, repeat=n_qubits)]


# ============================================================================
# Section 2: Hilbert Space Embedding and Padding
# ============================================================================

def pad_matrix_to_qubit_dimension(matrix: np.ndarray, n_qubits: int, unphysical_energy_penalty: float = 0.0) -> np.ndarray:
    """
    Embeds an (N x N) manifold matrix into the full (2^n x 2^n) qubit Hilbert space.

    Parameters:
    -----------
    matrix : np.ndarray
        N x N matrix (e.g. Hamiltonian or transition dipole).
    n_qubits : int
        Target qubit count (dimension 2^n).
    unphysical_energy_penalty : float
        Diagonal energy penalty added to unused states (from index N to 2^n - 1)
        to prevent unphysical leakage during quantum evolution.

    Returns:
    --------
    padded_matrix : np.ndarray of shape (2^n, 2^n)
    """
    dim_target = 2 ** n_qubits
    dim_source = matrix.shape[0]

    if dim_source > dim_target:
        raise ValueError(
            f"Source matrix dimension ({dim_source}) exceeds qubit Hilbert space ({dim_target}) for n={n_qubits}."
        )

    padded = np.zeros((dim_target, dim_target), dtype=complex)
    padded[:dim_source, :dim_source] = matrix

    # Apply energy penalty to unused orthogonal states if requested
    if unphysical_energy_penalty != 0.0 and dim_source < dim_target:
        for idx in range(dim_source, dim_target):
            padded[idx, idx] = unphysical_energy_penalty

    return padded


# ============================================================================
# Section 3: Core Pauli Decomposition via Orthogonal Projection
# ============================================================================

def decompose_matrix_to_paulis(
    matrix: np.ndarray,
    n_qubits: Optional[int] = None,
    tolerance: float = 1e-8,
    unphysical_penalty: float = 0.0
) -> Tuple[List[float], List[str]]:
    r"""
    Decomposes an N x N Hermitian matrix into a linear combination of n-qubit Pauli strings:
        H = \sum_k c_k \hat{P}_k

    Uses the Hilbert-Schmidt inner product orthogonality:
        c_k = (1 / 2^n) * Tr( H * \hat{P}_k )

    Parameters:
    -----------
    matrix : np.ndarray
        N x N real-symmetric or Hermitian matrix.
    n_qubits : int, optional
        Number of qubits. If None, defaults to ceil(log2(N)).
    tolerance : float
        Threshold below which coefficients are truncated to zero for sparsity.
    unphysical_penalty : float
        Energy offset for unused Hilbert space basis states.

    Returns:
    --------
    coefficients : List[float]
        Real coefficients c_k for non-zero Pauli terms.
    pauli_strings : List[str]
        Corresponding Pauli string representations (e.g., ['II', 'IZ', 'XX', 'YY']).
    """
    n_states = matrix.shape[0]
    if n_qubits is None:
        n_qubits = get_n_qubits(n_states)

    padded = pad_matrix_to_qubit_dimension(matrix, n_qubits, unphysical_energy_penalty=unphysical_penalty)
    dim = 2 ** n_qubits
    normalization = 1.0 / dim

    all_strings = generate_all_pauli_strings(n_qubits)
    coefficients: List[float] = []
    pauli_strings: List[str] = []

    for p_str in all_strings:
        p_mat = build_pauli_string_matrix(p_str)
        # Compute projection: Tr(H * P) / 2^n
        coeff_complex = normalization * np.trace(padded @ p_mat)

        # For Hermitian matrices, coefficients are strictly real
        coeff_real = float(np.real(coeff_complex))

        if abs(coeff_real) >= tolerance:
            coefficients.append(coeff_real)
            pauli_strings.append(p_str)

    return coefficients, pauli_strings


def reconstruct_matrix_from_paulis(coefficients: List[float], pauli_strings: List[str]) -> np.ndarray:
    r"""
    Reconstructs the full 2^n x 2^n matrix from Pauli string coefficients:
        H_reconstructed = \sum_k c_k \hat{P}_k
    """
    if len(coefficients) != len(pauli_strings):
        raise ValueError("Coefficients and pauli_strings must have identical lengths.")
    if len(coefficients) == 0:
        return np.zeros((1, 1), dtype=complex)

    dim = 2 ** len(pauli_strings[0])
    recon = np.zeros((dim, dim), dtype=complex)

    for c, p_str in zip(coefficients, pauli_strings):
        recon += c * build_pauli_string_matrix(p_str)

    return recon


# ============================================================================
# Section 4: High-Performance Pre-Decomposed Collision Hamiltonian Mapper
# ============================================================================

class MultiStatePauliMapper:
    """
    High-level mapper that converts PySCF electronic structure manifold data
    into Pauli representation, and provides ultra-fast evaluation of H(t)
    during time-dependent collision pulses.
    """

    def __init__(self, manifold_data: Dict[str, Any], unphysical_penalty_ev: float = 100.0):
        """
        Initializes the mapper using a PySCF manifold JSON dictionary.

        Parameters:
        -----------
        manifold_data : Dict[str, Any]
            Dictionary loaded from data/raw_pyscf/manifold_{species}.json
        unphysical_penalty_ev : float
            Energy penalty (in eV) placed on unused qubit states.
        """
        self.species = manifold_data["species"]
        self.n_states = manifold_data["n_states"]
        self.n_qubits = get_n_qubits(self.n_states)
        self.unphysical_penalty_ev = unphysical_penalty_ev

        # 1. Unperturbed diagonal atomic Hamiltonian H_0 = diag(E_0, ..., E_{N-1})
        energies_ev = np.array(manifold_data["excitation_energies_ev"], dtype=float)
        self.H0_matrix = np.diag(energies_ev)

        # 2. Transition dipole matrices in atomic units (e * a_0)
        self.mux_matrix = np.array(manifold_data["dipole_matrix_x"], dtype=float)
        self.muy_matrix = np.array(manifold_data["dipole_matrix_y"], dtype=float)
        self.muz_matrix = np.array(manifold_data["dipole_matrix_z"], dtype=float)

        # 3. Pre-decompose static H0 and dipole operators into Pauli basis
        # This allows computing H(t) = H0 - mu . E(t) via linear combination of coefficients!
        self._precompute_pauli_basis()

    def _precompute_pauli_basis(self):
        """
        Pre-computes Pauli representations for H0, mux, muy, muz.
        """
        self.h0_coeffs, self.h0_paulis = decompose_matrix_to_paulis(
            self.H0_matrix, n_qubits=self.n_qubits, unphysical_penalty=self.unphysical_penalty_ev
        )
        self.mux_coeffs, self.mux_paulis = decompose_matrix_to_paulis(self.mux_matrix, n_qubits=self.n_qubits)
        self.muy_coeffs, self.muy_paulis = decompose_matrix_to_paulis(self.muy_matrix, n_qubits=self.n_qubits)
        self.muz_coeffs, self.muz_paulis = decompose_matrix_to_paulis(self.muz_matrix, n_qubits=self.n_qubits)

        # Union of all active Pauli strings across H0 and dipoles
        all_active_paulis = set(self.h0_paulis) | set(self.mux_paulis) | set(self.muy_paulis) | set(self.muz_paulis)
        self.active_paulis = sorted(list(all_active_paulis))

        # Build coefficient lookup vectors indexed by active_paulis
        self.vec_h0 = np.array([self._get_coeff(p, self.h0_coeffs, self.h0_paulis) for p in self.active_paulis])
        self.vec_mux = np.array([self._get_coeff(p, self.mux_coeffs, self.mux_paulis) for p in self.active_paulis])
        self.vec_muy = np.array([self._get_coeff(p, self.muy_coeffs, self.muy_paulis) for p in self.active_paulis])
        self.vec_muz = np.array([self._get_coeff(p, self.muz_coeffs, self.muz_paulis) for p in self.active_paulis])

    @staticmethod
    def _get_coeff(pauli_str: str, coeffs: List[float], paulis: List[str]) -> float:
        if pauli_str in paulis:
            return coeffs[paulis.index(pauli_str)]
        return 0.0

    def evaluate_hamiltonian_paulis_at_t(
        self,
        e_field_vector: np.ndarray,
        dipole_coupling_scale: float = 1.0,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Instantly computes the time-dependent Pauli coefficients c_k(t) under an external
        electric field pulse E(t) = (Ex, Ey, Ez):
            H(t) = H_0 - (mux * Ex + muy * Ey + muz * Ez) * scale

        Returns:
        --------
        coefficients : np.ndarray
            Vector of real coefficients corresponding to self.active_paulis.
        pauli_strings : List[str]
            List of Pauli strings.
        """
        ex, ey, ez = e_field_vector
        coeffs_t = (
            self.vec_h0
            - (self.vec_mux * ex + self.vec_muy * ey + self.vec_muz * ez) * dipole_coupling_scale
        )
        return coeffs_t, self.active_paulis

    def to_pennylane_hamiltonian(self, coefficients: np.ndarray, pauli_strings: List[str]):
        """
        Converts the decomposed Pauli strings into a native PennyLane Hamiltonian / LinearCombination.
        Safe to call with or without PennyLane installed (returns tuple fallback if uninstalled).
        """
        try:
            import pennylane as qml

            obs_list = []
            for p_str in pauli_strings:
                op_terms = []
                for wire_idx, char in enumerate(p_str):
                    if char == "I":
                        op_terms.append(qml.Identity(wire_idx))
                    elif char == "X":
                        op_terms.append(qml.PauliX(wire_idx))
                    elif char == "Y":
                        op_terms.append(qml.PauliY(wire_idx))
                    elif char == "Z":
                        op_terms.append(qml.PauliZ(wire_idx))

                # Combine single-qubit operators into a tensor product
                if len(op_terms) == 1:
                    obs_list.append(op_terms[0])
                else:
                    obs = op_terms[0]
                    for term in op_terms[1:]:
                        obs = obs @ term
                    obs_list.append(obs)

            return qml.Hamiltonian(coefficients.tolist(), obs_list)
        except ImportError:
            # Fallback if pennylane is not in the active python environment
            return coefficients, pauli_strings

    def summary(self) -> str:
        """Returns a human-readable summary of the mapping."""
        lines = [
            f"=== MultiStatePauliMapper Summary ===",
            f"Species: {self.species}",
            f"Manifold States (N): {self.n_states}",
            f"Required Qubits (n): {self.n_qubits} (Hilbert dimension = {2**self.n_qubits})",
            f"Total Active Pauli Terms: {len(self.active_paulis)}",
            f"Active Pauli Strings: {', '.join(self.active_paulis)}",
            f"Unphysical Energy Penalty: {self.unphysical_penalty_ev} eV",
        ]
        return "\n".join(lines)


# ============================================================================
# Section 5: CLI Utility to Inspect and Test Mappings
# ============================================================================

def load_and_map_species(json_path: str) -> MultiStatePauliMapper:
    """Loads a PySCF manifold JSON and initializes its Pauli mapper."""
    with open(json_path, "r") as f:
        data = json.load(f)
    mapper = MultiStatePauliMapper(data)
    print(mapper.summary())
    return mapper


if __name__ == "__main__":
    import os

    sample_path = "data/raw_pyscf/manifold_Be.json"
    if os.path.exists(sample_path):
        print(f"[*] Testing Pauli Mapper on {sample_path}...")
        mapper = load_and_map_species(sample_path)

        # Test evaluation under mock electric pulse
        mock_e_field = np.array([0.0, 0.0, 0.15])  # 0.15 a.u. field along z-axis
        c_t, paulis = mapper.evaluate_hamiltonian_paulis_at_t(mock_e_field)
        print("\n[*] Time-dependent Pauli terms under test field E_z = 0.15 a.u.:")
        for c, p in zip(c_t, paulis):
            if abs(c) > 1e-4:
                print(f"    {p}: {c:+.6f}")
    else:
        print(f"[!] File not found: {sample_path}. Run pyscf_runner.py first.")
