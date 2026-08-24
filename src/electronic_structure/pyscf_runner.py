r"""
PySCF Electronic Structure Runner for Multi-Root Atomic Manifolds.

Computes multi-root ground and excited states (E_0, E_1, ..., E_N) and
the full transition dipole matrix elements (\mu_x, \mu_y, \mu_z) for
the Be isoelectronic series (Be, C 2+, Fe 22+).
"""

import os
import json
import argparse
from typing import Dict, Any, Optional, List
import numpy as np

from src.electronic_structure.species_configs import get_species_config, SPECIES_CONFIGS

# Physical constants
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


class MultiRootElectronicStructure:
    """
    Solves multi-root electronic structure for Be-like species using PySCF.
    Extracts manifold energies and transition dipole matrices.
    """

    def __init__(self, species_key: str, basis: Optional[str] = None, n_states: int = 4):
        self.species_config = get_species_config(species_key)
        self.species_key = species_key
        self.basis = basis or self.species_config["recommended_basis"]
        self.n_states = n_states
        self.results: Optional[Dict[str, Any]] = None

    def run_calculation(self) -> Dict[str, Any]:
        """
        Executes PySCF SCF + Full CI / CASCI multi-root solver.
        """
        try:
            from pyscf import gto, scf, fci, mcscf, ao2mo
        except ImportError:
            print("[Warning] PySCF is not installed in the active environment.")
            print("[Info] Generating analytical/semi-empirical mock manifold for development...")
            return self._generate_mock_manifold()

        symbol = self.species_config["symbol"]
        charge = self.species_config["charge"]
        spin = self.species_config["spin"]
        n_elec = self.species_config["num_electrons"]

        print(f"[*] Building PySCF Molecule: {symbol} (Z={self.species_config['atomic_number']}, Charge={charge}+, Basis={self.basis})")
        mol = gto.M(
            atom=f"{symbol} 0.0 0.0 0.0",
            basis=self.basis,
            charge=charge,
            spin=spin,
            unit="Bohr",
            verbose=3,
        )

        # 1. Mean-field calculation (RHF)
        print("[*] Running Restricted Hartree-Fock (RHF)...")
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            print("[!] RHF did not converge cleanly; proceeding with current density.")

        # 2. Setup Multi-Root Active Space / FCI
        norb = mol.nao_nr()
        n_cas = min(norb, self.species_config["active_orbitals"])
        nelecas = self.species_config["active_electrons"]

        print(f"[*] Solving Multi-Root CI for {self.n_states} roots (norb={norb}, nelec={n_elec})...")
        cisolver = fci.FCISolver(mol)
        cisolver.nroots = self.n_states
        cisolver.conv_tol = 1e-9

        # 1e and 2e integrals in MO basis
        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        g2e = ao2mo.kernel(mol, mf.mo_coeff)

        # Solve for ground and excited states
        energies_tot, civecs = cisolver.kernel(h1e, g2e, norb, (n_elec // 2, n_elec // 2), ecore=mf.energy_nuc())

        if self.n_states == 1:
            energies_tot = [energies_tot]
            civecs = [civecs]

        energies_au = np.array(energies_tot)
        energies_ev = (energies_au - energies_au[0]) * HARTREE_TO_EV

        # 3. Compute Transition Dipole Moments: <psi_i | r_alpha | psi_j>
        print("[*] Computing Transition Dipole Moments and 1-TDMs...")
        dipole_integrals_ao = mol.intor("int1e_r")  # shape: (3, nao, nao) in a.u.
        dipole_integrals_mo = np.einsum("pi,xpq,qj->xij", mf.mo_coeff, dipole_integrals_ao, mf.mo_coeff)

        dipole_matrix_x = np.zeros((self.n_states, self.n_states))
        dipole_matrix_y = np.zeros((self.n_states, self.n_states))
        dipole_matrix_z = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            for j in range(self.n_states):
                # Compute 1-particle transition reduced density matrix
                dm1 = cisolver.trans_rdm1(civecs[i], civecs[j], norb, (n_elec // 2, n_elec // 2))
                dipole_matrix_x[i, j] = np.einsum("pq,pq->", dipole_integrals_mo[0], dm1)
                dipole_matrix_y[i, j] = np.einsum("pq,pq->", dipole_integrals_mo[1], dm1)
                dipole_matrix_z[i, j] = np.einsum("pq,pq->", dipole_integrals_mo[2], dm1)

        # 4. Oscillator strengths for transitions from ground state (i=0 -> j)
        oscillator_strengths = []
        for j in range(self.n_states):
            if j == 0:
                oscillator_strengths.append(0.0)
            else:
                dE_au = energies_au[j] - energies_au[0]
                mu_sq = (
                    dipole_matrix_x[0, j] ** 2
                    + dipole_matrix_y[0, j] ** 2
                    + dipole_matrix_z[0, j] ** 2
                )
                f_0j = (2.0 / 3.0) * dE_au * mu_sq
                oscillator_strengths.append(float(f_0j))

        self.results = {
            "species": self.species_key,
            "atomic_number": self.species_config["atomic_number"],
            "charge": self.species_config["charge"],
            "n_states": self.n_states,
            "basis": self.basis,
            "energies_au": energies_au.tolist(),
            "energies_ev": energies_ev.tolist(),
            "excitation_energies_ev": energies_ev.tolist(),
            "dipole_matrix_x": dipole_matrix_x.tolist(),
            "dipole_matrix_y": dipole_matrix_y.tolist(),
            "dipole_matrix_z": dipole_matrix_z.tolist(),
            "oscillator_strengths": oscillator_strengths,
            "is_mock": False,
        }
        return self.results

    def _generate_mock_manifold(self) -> Dict[str, Any]:
        """
        Generates physically scaled mock electronic structure data for development
        when PySCF is not locally installed.
        Scalings:
          Delta E ~ Z (valence) to Z^2 (core)
          Dipole mu ~ 1 / Z
        """
        Z = self.species_config["atomic_number"]
        q = self.species_config["charge"]
        
        # Base excitation energies in eV (scaled by Z/4)
        z_factor = (Z / 4.0)
        base_ev = np.array([0.0, 5.28 * z_factor, 7.40 * z_factor, 10.8 * z_factor, 14.5 * (z_factor ** 1.2)])[:self.n_states]
        energies_ev = base_ev
        energies_au = energies_ev / HARTREE_TO_EV

        # Dipole matrix elements (scale as 1/Z)
        mu_scale = 1.8 / z_factor
        dipole_z = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states - 1):
            dipole_z[i, i + 1] = mu_scale * np.sqrt(i + 1) / (i + 2)
            dipole_z[i + 1, i] = dipole_z[i, i + 1]

        dipole_x = np.zeros((self.n_states, self.n_states))
        dipole_y = np.zeros((self.n_states, self.n_states))

        oscillator_strengths = [0.0]
        for j in range(1, self.n_states):
            dE_au = energies_au[j] - energies_au[0]
            mu_sq = dipole_z[0, j] ** 2
            f_0j = (2.0 / 3.0) * dE_au * mu_sq
            oscillator_strengths.append(float(f_0j))

        self.results = {
            "species": self.species_key,
            "atomic_number": Z,
            "charge": q,
            "n_states": self.n_states,
            "basis": self.basis,
            "energies_au": energies_au.tolist(),
            "energies_ev": energies_ev.tolist(),
            "excitation_energies_ev": energies_ev.tolist(),
            "dipole_matrix_x": dipole_x.tolist(),
            "dipole_matrix_y": dipole_y.tolist(),
            "dipole_matrix_z": dipole_z.tolist(),
            "oscillator_strengths": oscillator_strengths,
            "is_mock": True,
        }
        return self.results

    def save_json(self, output_dir: str = "data/raw_pyscf") -> str:
        """Saves calculation results to a JSON file."""
        if self.results is None:
            self.run_calculation()
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"manifold_{self.species_key}.json")
        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[+] Saved electronic structure manifold to: {file_path}")
        return file_path


def run_all_species(n_states: int = 4, output_dir: str = "data/raw_pyscf") -> List[str]:
    """Runs calculation for all 3 species in the isoelectronic sequence."""
    saved_files = []
    for key in SPECIES_CONFIGS.keys():
        print(f"\n==================== Running {key} ====================")
        calc = MultiRootElectronicStructure(species_key=key, n_states=n_states)
        calc.run_calculation()
        saved_files.append(calc.save_json(output_dir=output_dir))
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySCF Multi-Root Electronic Structure Generator")
    parser.add_argument(
        "--species",
        type=str,
        default="Be",
        choices=["Be", "C2+", "Fe22+", "all"],
        help="Target species to compute (or 'all' for full isoelectronic sequence)",
    )
    parser.add_argument("--n-states", type=int, default=4, help="Number of multi-root states (E_0 to E_{N-1})")
    parser.add_argument("--basis", type=str, default=None, help="Quantum chemistry basis set")
    parser.add_argument("--outdir", type=str, default="data/raw_pyscf", help="Output directory for JSON results")

    args = parser.parse_args()

    if args.species == "all":
        run_all_species(n_states=args.n_states, output_dir=args.outdir)
    else:
        calc = MultiRootElectronicStructure(species_key=args.species, basis=args.basis, n_states=args.n_states)
        calc.run_calculation()
        calc.save_json(output_dir=args.outdir)
