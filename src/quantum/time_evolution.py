r"""
Quantum Time-Evolution Circuit and Multi-State Population Dynamics Engine.

Task 2.2: Simulates the time-dependent unitary evolution of the atomic/ionic
system under an electron-impact collision pulse:
    i \frac{\partial}{\partial t} |\psi(t)\rangle = H(t) |\psi(t)\rangle
    where H(t) = \sum_k c_k(t) \hat{P}_k is mapped in the n-qubit Pauli basis.

Key Features:
  1. Trotterized Quantum Circuit (PennyLane):
     - Discretizes time into slices \Delta t.
     - Implements unitary step U(t_k+\Delta t, t_k) \approx \prod_m \exp(-i c_m(t_k) \hat{P}_m \Delta t)
       via parameterized Pauli rotations (qml.PauliRot).
  2. Multi-State Population Tracking:
     - Measures probability distributions P_i(t) = |\langle i | \psi(t) \rangle|^2
       at every time step.
  3. High-Precision Reference Solver:
     - Exact matrix exponential unitary propagation for benchmarking Trotter error.
  4. Batch Parameter Sweeps:
     - Sweeps across impact parameters b and incident energies E_inc,
       exporting data to data/processed_circuits/populations_{species}.json.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from src.quantum.mapper import (
    MultiStatePauliMapper,
    get_n_qubits,
    build_pauli_string_matrix,
    reconstruct_matrix_from_paulis,
)

# Physical conversion factors (Atomic units: \hbar = m_e = e = 1)
EV_TO_HARTREE = 1.0 / 27.211386245988
HARTREE_TO_EV = 27.211386245988
TIME_AU_TO_FEMTOSECONDS = 0.024188843265857


# ============================================================================
# Section 1: Semiclassical Collision Electric Field Generator (Helper)
# ============================================================================

def compute_collision_electric_field(
    t_grid_au: np.ndarray,
    impact_parameter_bohr: float,
    incident_energy_ev: float,
    ion_charge: int = 0,
) -> np.ndarray:
    r"""
    Computes the time-dependent electric field vector \vec{\mathcal{E}}(t) in atomic units
    produced at the atom's origin by a passing electron with impact parameter b
    and incident kinetic energy E_inc:
        v = \sqrt{2 * E_inc} (in a.u.)
        \vec{R}(t) = (b, 0, v * t)
        \vec{\mathcal{E}}(t) = -e * \vec{R}(t) / |\vec{R}(t)|^3

    Parameters:
    -----------
    t_grid_au : np.ndarray
        Array of time points in atomic units (a.u.).
    impact_parameter_bohr : float
        Impact parameter b in Bohr radii (a_0).
    incident_energy_ev : float
        Incident electron kinetic energy in eV.
    ion_charge : int
        Target charge state (used for Coulomb acceleration correction).

    Returns:
    --------
    e_fields : np.ndarray of shape (len(t_grid_au), 3)
        Electric field vector (Ex, Ey, Ez) at each time step in a.u.
    """
    # Electron velocity in atomic units: v = \sqrt{2 * E_kin (Hartree)}
    e_kin_au = max(incident_energy_ev * EV_TO_HARTREE, 1e-4)
    v_electron = np.sqrt(2.0 * e_kin_au)

    # Coulomb acceleration / focusing correction for charged ions
    # At distance r ~ b, effective kinetic energy is enhanced: E_eff ~ E_inc + q / b
    if ion_charge > 0:
        coulomb_enhancement = 1.0 + (float(ion_charge) / (max(impact_parameter_bohr, 0.5) * e_kin_au))
        v_electron *= np.sqrt(coulomb_enhancement)

    b = max(impact_parameter_bohr, 0.1)
    n_pts = len(t_grid_au)
    e_fields = np.zeros((n_pts, 3), dtype=float)

    for idx, t in enumerate(t_grid_au):
        # Straight-line trajectory along z with offset b along x
        rx = b
        ry = 0.0
        rz = v_electron * t
        r_mag = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        r_cubed = r_mag ** 3

        # Electric field on the atomic target: \vec{\mathcal{E}} = - \vec{R} / R^3
        e_fields[idx, 0] = -rx / r_cubed
        e_fields[idx, 1] = -ry / r_cubed
        e_fields[idx, 2] = -rz / r_cubed

    return e_fields


# ============================================================================
# Section 2: PennyLane Trotterized Quantum Circuit Engine
# ============================================================================

class PennyLaneTimeEvolution:
    """
    Implements multi-step Trotterized quantum time evolution using PennyLane.
    """

    def __init__(self, mapper: MultiStatePauliMapper, shots: Optional[int] = None):
        self.mapper = mapper
        self.n_qubits = mapper.n_qubits
        self.shots = shots
        self.has_pennylane = False

        try:
            import pennylane as qml
            self.qml = qml
            self.has_pennylane = True
            # Setup PennyLane device (default.qubit)
            self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        except ImportError:
            self.qml = None
            self.dev = None

    def _apply_trotter_step_pennylane(self, coeffs: np.ndarray, pauli_strings: List[str], dt_au: float):
        r"""
        Applies a single Trotter slice: \prod_k \exp(-i * c_k * \hat{P}_k * \Delta t)
        using PennyLane parameterized Pauli rotation gates (qml.PauliRot).
        """
        qml = self.qml
        # In a.u., Hamiltonian energy in eV must be converted to a.u. for time propagation
        dt_scaled = dt_au * EV_TO_HARTREE

        for c, p_str in zip(coeffs, pauli_strings):
            if abs(c) < 1e-8:
                continue
            # PennyLane qml.PauliRot applies exp(-i * theta / 2 * P), so theta = 2 * c * dt
            theta = 2.0 * float(c) * dt_scaled
            
            # Check if all Identity
            if all(char == "I" for char in p_str):
                # Global phase shift (does not affect state probabilities, but preserves phase)
                continue
            
            # Extract non-identity wires and pauli characters
            active_chars = []
            active_wires = []
            for wire_idx, char in enumerate(p_str):
                if char != "I":
                    active_chars.append(char)
                    active_wires.append(wire_idx)
            
            active_word = "".join(active_chars)
            qml.PauliRot(theta, active_word, wires=active_wires)

    def simulate_trotter_trajectory(
        self,
        t_grid_au: np.ndarray,
        e_fields_au: np.ndarray,
    ) -> np.ndarray:
        """
        Runs the full Trotterized time-evolution circuit from t_0 to t_final.
        Measures state probability vector at each time step.

        Parameters:
        -----------
        t_grid_au : np.ndarray
            Discretized time grid in atomic units.
        e_fields_au : np.ndarray
            Electric field vector (Ex, Ey, Ez) at each time step.

        Returns:
        --------
        populations : np.ndarray of shape (len(t_grid), 2^n_qubits)
            State populations |c_j(t)|^2 across all computational basis states.
        """
        if not self.has_pennylane:
            # If PennyLane is not installed, use the exact reference statevector propagator
            return simulate_exact_unitary_evolution(self.mapper, t_grid_au, e_fields_au)

        qml = self.qml
        n_steps = len(t_grid_au)
        dt_au = t_grid_au[1] - t_grid_au[0] if n_steps > 1 else 0.1
        dim = 2 ** self.n_qubits

        populations = np.zeros((n_steps, dim), dtype=float)

        # Build progressive Trotter circuits for each time step
        # Note: In PennyLane, we construct a QNode that steps up to time index k
        @qml.qnode(self.dev)
        def circuit_at_step(step_index: int):
            # 1. State preparation: initialize in ground state |0...0> (E_0)
            # Default qubit state is |00...0>
            
            # 2. Sequential Trotter evolution up to step_index
            for k in range(step_index):
                coeffs_k, paulis_k = self.mapper.evaluate_hamiltonian_paulis_at_t(e_fields_au[k])
                self._apply_trotter_step_pennylane(coeffs_k, paulis_k, dt_au)
            
            # 3. Measurement: state probabilities across computational basis
            return qml.probs(wires=range(self.n_qubits))

        # Initial state at t = t_grid[0]
        populations[0, 0] = 1.0

        for step in range(1, n_steps):
            probs = circuit_at_step(step)
            populations[step, :] = np.array(probs)

        return populations


# ============================================================================
# Section 3: Exact Unitary Reference Propagator (Matrix Exponential)
# ============================================================================

def simulate_exact_unitary_evolution(
    mapper: MultiStatePauliMapper,
    t_grid_au: np.ndarray,
    e_fields_au: np.ndarray,
) -> np.ndarray:
    r"""
    Exact matrix exponential propagation for benchmarking and fast simulation:
        |\psi(t + \Delta t)\rangle = \exp(-i * H(t) * \Delta t) |\psi(t)\rangle

    Parameters:
    -----------
    mapper : MultiStatePauliMapper
        Mapper containing atomic manifold and dipole parameters.
    t_grid_au : np.ndarray
        Array of time points in atomic units.
    e_fields_au : np.ndarray
        Array of shape (len(t_grid), 3) containing electric field vectors.

    Returns:
    --------
    populations : np.ndarray of shape (len(t_grid), 2^n_qubits)
        State probabilities P_i(t) = |\langle i | \psi(t) \rangle|^2.
    """
    n_steps = len(t_grid_au)
    dt_au = t_grid_au[1] - t_grid_au[0] if n_steps > 1 else 0.1
    dt_scaled = dt_au * EV_TO_HARTREE
    dim = 2 ** mapper.n_qubits

    # Statevector initialized to ground state |0...0>
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0 + 0.0j

    populations = np.zeros((n_steps, dim), dtype=float)
    populations[0, :] = np.abs(psi) ** 2

    for k in range(n_steps - 1):
        # 1. Evaluate Hamiltonian at current time slice in eV
        coeffs_k, paulis_k = mapper.evaluate_hamiltonian_paulis_at_t(e_fields_au[k])
        H_k_ev = reconstruct_matrix_from_paulis(coeffs_k.tolist(), paulis_k)

        # 2. Diagonalize H_k for stable, exact unitary matrix exponentiation:
        # exp(-i * H * dt) = V * diag(exp(-i * lambda * dt)) * V^\dagger
        eigenvalues, eigenvectors = np.linalg.eigh(H_k_ev)
        
        # Convert eigenvalues to a.u. for time propagation
        phase_factors = np.exp(-1j * eigenvalues * dt_scaled)
        U_step = eigenvectors @ np.diag(phase_factors) @ eigenvectors.conj().T

        # 3. Propagate statevector
        psi = U_step @ psi

        # Normalize statevector to guarantee strict numerical unitarity
        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi = psi / norm

        # 4. Measure state populations
        populations[k + 1, :] = np.abs(psi) ** 2

    return populations


# ============================================================================
# Section 4: High-Level Population Simulation Runner & Batch Sweeper
# ============================================================================

class CollisionDynamicsSimulator:
    """
    End-to-end simulation runner for electron-impact collisional excitation.
    """

    def __init__(
        self,
        manifold_json_path: str,
        use_pennylane_if_available: bool = True,
        unphysical_penalty_ev: float = 100.0,
    ):
        with open(manifold_json_path, "r") as f:
            self.manifold_data = json.load(f)

        self.species = self.manifold_data["species"]
        self.charge = self.manifold_data.get("charge", 0)
        self.mapper = MultiStatePauliMapper(
            self.manifold_data, unphysical_penalty_ev=unphysical_penalty_ev
        )
        self.pennylane_engine = (
            PennyLaneTimeEvolution(self.mapper) if use_pennylane_if_available else None
        )

    def run_single_collision(
        self,
        impact_parameter_bohr: float,
        incident_energy_ev: float,
        t_min_au: float = -15.0,
        t_max_au: float = 15.0,
        n_time_steps: int = 2000,
        use_trotter: bool = False,
    ) -> Dict[str, Any]:
        """
        Simulates a single collision event with specified impact parameter and energy.

        Returns:
        --------
        result : Dict[str, Any]
            Dictionary containing time grid, electric fields, and state populations.
        """
        t_grid = np.linspace(t_min_au, t_max_au, n_time_steps)
        e_fields = compute_collision_electric_field(
            t_grid_au=t_grid,
            impact_parameter_bohr=impact_parameter_bohr,
            incident_energy_ev=incident_energy_ev,
            ion_charge=self.charge,
        )

        if use_trotter and self.pennylane_engine is not None and self.pennylane_engine.has_pennylane:
            populations_all = self.pennylane_engine.simulate_trotter_trajectory(t_grid, e_fields)
        else:
            populations_all = simulate_exact_unitary_evolution(self.mapper, t_grid, e_fields)

        # Extract only physical manifold states (0 to N-1)
        n_manifold = self.mapper.n_states
        state_pop_dict = {}
        for s_idx in range(n_manifold):
            state_pop_dict[f"state_{s_idx}"] = populations_all[:, s_idx].tolist()

        # Check total population conservation
        total_pop_t = np.sum(populations_all[:, :n_manifold], axis=1)

        return {
            "species": self.species,
            "n_states": n_manifold,
            "incident_energy_ev": float(incident_energy_ev),
            "impact_parameter_bohr": float(impact_parameter_bohr),
            "time_grid_au": t_grid.tolist(),
            "time_grid_fs": (t_grid * TIME_AU_TO_FEMTOSECONDS).tolist(),
            "populations": state_pop_dict,
            "final_populations": [float(populations_all[-1, i]) for i in range(n_manifold)],
            "unitarity_preserved": bool(np.allclose(total_pop_t, 1.0, atol=1e-3)),
        }

    def sweep_impact_parameters(
        self,
        incident_energy_ev: float,
        b_min_bohr: float = 0.5,
        b_max_bohr: float = 10.0,
        n_b_points: int = 25,
        output_dir: str = "data/processed_circuits",
    ) -> Dict[str, Any]:
        """
        Sweeps impact parameter b over [b_min, b_max] to compute excitation probabilities P_{0->j}(b).
        Saves results to JSON for downstream cross-section integration (Track C).
        """
        b_grid = np.linspace(b_min_bohr, b_max_bohr, n_b_points)
        final_probs = {f"state_{i}": [] for i in range(self.mapper.n_states)}

        for b in b_grid:
            res = self.run_single_collision(
                impact_parameter_bohr=b,
                incident_energy_ev=incident_energy_ev,
                n_time_steps=2000,
            )
            for i in range(self.mapper.n_states):
                final_probs[f"state_{i}"].append(res["final_populations"][i])

        sweep_data = {
            "species": self.species,
            "charge": self.charge,
            "n_states": self.mapper.n_states,
            "incident_energy_ev": float(incident_energy_ev),
            "impact_parameters_bohr": b_grid.tolist(),
            "excitation_probabilities_vs_b": final_probs,
        }

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"populations_{self.species}_E{int(incident_energy_ev)}eV.json")
        with open(out_path, "w") as f:
            json.dump(sweep_data, f, indent=2)
        print(f"[+] Saved impact parameter sweep to: {out_path}")

        return sweep_data


# ============================================================================
# Section 5: CLI Entrypoint for Testing and Batch Runs
# ============================================================================

def run_all_species_simulation(energy_ev: float = 50.0):
    """Runs collision dynamics simulation for all 3 species."""
    species_list = ["Be", "C2+", "Fe22+"]
    for sp in species_list:
        path = f"data/raw_pyscf/manifold_{sp}.json"
        if os.path.exists(path):
            print(f"\n================ Running Collision Dynamics for {sp} ================")
            sim = CollisionDynamicsSimulator(path)
            res = sim.run_single_collision(impact_parameter_bohr=2.0, incident_energy_ev=energy_ev)
            print(f"[*] Final Populations for {sp} (b=2.0 a0, E={energy_ev} eV):")
            for idx, p in enumerate(res["final_populations"]):
                print(f"    State {idx}: {p * 100:.2f}%")
            # Run impact parameter sweep
            sim.sweep_impact_parameters(incident_energy_ev=energy_ev)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Time-Evolution & Population Dynamics")
    parser.add_argument("--species", type=str, default="Be", choices=["Be", "C2+", "Fe22+", "all"])
    parser.add_argument("--energy", type=float, default=50.0, help="Incident electron kinetic energy in eV")
    parser.add_argument("--impact-b", type=float, default=2.0, help="Impact parameter in Bohr (a_0)")
    parser.add_argument("--use-trotter", action="store_true", help="Use PennyLane Trotter circuit")

    args = parser.parse_args()

    if args.species == "all":
        run_all_species_simulation(energy_ev=args.energy)
    else:
        json_path = f"data/raw_pyscf/manifold_{args.species}.json"
        if os.path.exists(json_path):
            sim = CollisionDynamicsSimulator(json_path)
            res = sim.run_single_collision(
                impact_parameter_bohr=args.impact_b,
                incident_energy_ev=args.energy,
                use_trotter=args.use_trotter,
            )
            print(f"\n[*] Single collision simulation results for {args.species}:")
            for idx, p in enumerate(res["final_populations"]):
                print(f"    State {idx}: {p * 100:.2f}%")
            print(f"[*] Unitarity preserved: {res['unitarity_preserved']}")
            sim.sweep_impact_parameters(incident_energy_ev=args.energy)
        else:
            print(f"[!] JSON file not found: {json_path}. Run pyscf_runner.py first.")
