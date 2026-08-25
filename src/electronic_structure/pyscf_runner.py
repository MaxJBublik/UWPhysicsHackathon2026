r"""Multi-root active-space electronic-structure calculations with PySCF.

Values written to JSON are in atomic units unless a field ends in ``_ev``.
``energies_au`` is the excitation manifold; absolute energies are preserved in
``total_energies_au`` for ab-initio calculations.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.electronic_structure.species_configs import SPECIES_CONFIGS, get_species_config

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903

REQUIRED_MANIFOLD_KEYS = {
    "species",
    "n_states",
    "energies_au",
    "energies_ev",
    "dipole_matrix_x",
    "dipole_matrix_y",
    "dipole_matrix_z",
    "charge",
}


def validate_manifold_file(file_path: Path) -> Dict[str, Any]:
    """Assert that a JSON manifold file contains all required items for downstream calculations."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as err:
        raise ValueError(f"Failed to parse manifold file '{file_path}': {err}") from err

    missing_keys = REQUIRED_MANIFOLD_KEYS - set(data.keys())
    if missing_keys:
        raise AssertionError(
            f"Manifold file '{file_path.name}' is missing required downstream fields: {sorted(missing_keys)}"
        )

    return data


class MultiRootElectronicStructure:
    """Compute a small low-energy manifold for a Be-like atom or ion."""

    def __init__(self, species_key: str, basis: Optional[str] = None,
                 n_states: int = 4, *, allow_mock: bool = True,
                 verbose: int = 0) -> None:
        if not isinstance(n_states, int) or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        self.species_config = get_species_config(species_key)
        self.species_key = species_key
        self.basis = basis or self.species_config["recommended_basis"]
        self.n_states = n_states
        self.allow_mock = allow_mock
        self.verbose = verbose
        self.results: Optional[Dict[str, Any]] = None

    def run_calculation(self) -> Dict[str, Any]:
        """Run RHF followed by multi-root active-space FCI (CASCI)."""
        try:
            from pyscf import gto, mcscf, scf
        except ImportError:
            if not self.allow_mock:
                raise RuntimeError(
                    "PySCF is required; install pyscf>=2.5 for an ab-initio manifold"
                ) from None
            self.results = self._generate_mock_manifold()
            return self.results

        cfg = self.species_config
        mol = gto.M(atom=[[cfg["symbol"], (0.0, 0.0, 0.0)]], basis=self.basis,
                    charge=cfg["charge"], spin=cfg["spin"], unit="Bohr",
                    symmetry=False, verbose=self.verbose)
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-10
        mf.max_cycle = 200
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.conv_tol = 1e-10
            mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"RHF failed to converge for {self.species_key}")

        ncas = min(int(cfg["active_orbitals"]), mol.nao_nr())
        nelecas = int(cfg["active_electrons"])
        if nelecas > mol.nelectron or nelecas > 2 * ncas:
            raise ValueError("invalid active-space electron/orbital configuration")
        cas = mcscf.CASCI(mf, ncas, nelecas)
        cas.fcisolver.nroots = self.n_states
        cas.fcisolver.conv_tol = 1e-9
        output = cas.kernel()
        total_energies = np.atleast_1d(np.asarray(output[0], dtype=float))
        ci_vectors = [output[2]] if total_energies.size == 1 else list(output[2])
        if total_energies.size != self.n_states:
            raise RuntimeError(
                f"PySCF returned {total_energies.size} roots, expected {self.n_states}"
            )

        active_mo = cas.mo_coeff[:, cas.ncore:cas.ncore + ncas]
        r_ao = mol.intor("int1e_r", comp=3)
        r_cas = np.einsum("pi,xpq,qj->xij", active_mo, r_ao, active_mo,
                          optimize=True)
        dipoles = np.zeros((3, self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            for j in range(i, self.n_states):
                dm1 = cas.fcisolver.trans_rdm1(ci_vectors[i], ci_vectors[j],
                                                ncas, cas.nelecas)
                value = np.einsum("xpq,qp->x", r_cas, dm1,
                                  optimize=True).real
                dipoles[:, i, j] = value
                dipoles[:, j, i] = value

        excitation_au = total_energies - total_energies[0]
        self.results = self._pack_results(
            excitation_au, dipoles,
            self._oscillator_strengths(excitation_au, dipoles),
            is_mock=False, total_energies_au=total_energies,
            method="RHF/CASCI",
            active_space={"orbitals": ncas, "electrons": nelecas})
        return self.results

    @staticmethod
    def _oscillator_strengths(energies_au: np.ndarray,
                              dipoles: np.ndarray) -> List[float]:
        values = (2.0 / 3.0) * energies_au * np.sum(dipoles[:, 0, :] ** 2,
                                                     axis=0)
        values[0] = 0.0
        return values.astype(float).tolist()

    def _pack_results(self, energies_au: np.ndarray, dipoles: np.ndarray,
                      oscillator_strengths: List[float], *, is_mock: bool,
                      **metadata: Any) -> Dict[str, Any]:
        return {
            "species": self.species_key,
            "atomic_number": self.species_config["atomic_number"],
            "charge": self.species_config["charge"],
            "n_states": self.n_states,
            "basis": self.basis,
            "energies_au": energies_au.tolist(),
            "energies_ev": (energies_au * HARTREE_TO_EV).tolist(),
            "excitation_energies_ev": (energies_au * HARTREE_TO_EV).tolist(),
            "dipole_matrix_x": dipoles[0].tolist(),
            "dipole_matrix_y": dipoles[1].tolist(),
            "dipole_matrix_z": dipoles[2].tolist(),
            "oscillator_strengths": oscillator_strengths,
            "is_mock": is_mock,
            **{key: value.tolist() if isinstance(value, np.ndarray) else value
               for key, value in metadata.items()},
        }

    def _generate_mock_manifold(self) -> Dict[str, Any]:
        """Generate a deterministic, correctly shaped development manifold."""
        z_factor = self.species_config["atomic_number"] / 4.0
        index = np.arange(self.n_states, dtype=float)
        energies_ev = np.where(index == 0, 0.0,
                               5.28 * z_factor * index ** 0.72)
        energies_au = energies_ev / HARTREE_TO_EV
        dipoles = np.zeros((3, self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states - 1):
            value = (1.8 / z_factor) * np.sqrt(i + 1) / (i + 2)
            dipoles[2, i, i + 1] = dipoles[2, i + 1, i] = value
        return self._pack_results(
            energies_au, dipoles,
            self._oscillator_strengths(energies_au, dipoles), is_mock=True,
            method="scaled-development-manifold")

    def save_json(
        self, 
        output_dir: str | Path = "data/raw_pyscf", 
        filename: Optional[str] = None
    ) -> str:
        """Calculate if necessary and save the manifold JSON with overwrite protection."""
        if self.results is None:
            self.run_calculation()
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        
        if filename:
            path = directory / filename
        else:
            base_name = f"manifold_{self.species_key}"
            path = directory / f"{base_name}.json"
            counter = 1
            while path.exists():
                path = directory / f"{base_name}_{counter}.json"
                counter += 1

        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.results, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return str(path)


def run_all_species(n_states: int = 4,
                    input_dir: Optional[str | Path] = None,
                    output_dir: str | Path = "data/raw_pyscf", *,
                    basis: Optional[str] = None,
                    allow_mock: bool = True) -> List[str]:
    """Validate input JSON files if input_dir is specified, otherwise generate objects via PySCF."""
    if input_dir is not None:
        
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")

        json_files = sorted(list(input_path.glob("*.json")))
        if not json_files:
            warnings.warn(f"Input directory '{input_dir}' is empty or contains no JSON files.", UserWarning)
            return []

        existing_paths = []
        for file_path in json_files:

            #validation of file contents
            validate_manifold_file(file_path)
            existing_paths.append(str(file_path))
        
        return existing_paths
    paths = []
    for species in SPECIES_CONFIGS:
        calculation = MultiRootElectronicStructure(
            species, basis=basis, n_states=n_states, allow_mock=allow_mock)
        calculation.run_calculation()
        paths.append(calculation.save_json(output_dir))
    return paths


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="Be", choices=[*SPECIES_CONFIGS, "all"])
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--basis")
    parser.add_argument("--indir", help="Directory containing pre-existing JSON manifold files.")
    parser.add_argument("--outdir", default="data/raw_pyscf")
    parser.add_argument("--require-pyscf", action="store_true")
    args = parser.parse_args()
    if args.species == "all":
        run_all_species(args.n_states, input_dir=args.indir, output_dir=args.outdir,
                        basis=args.basis, allow_mock=not args.require_pyscf)
    else:
        if args.indir is not None:
            target_path = Path(args.indir) / f"manifold_{args.species}.json"
            if not target_path.is_file():
                target_path = Path(args.indir) / f"{args.species}.json"
            validate_manifold_file(target_path)
        else:
            calculation = MultiRootElectronicStructure(
                args.species, basis=args.basis, n_states=args.n_states,
                allow_mock=not args.require_pyscf)
            calculation.run_calculation()
            calculation.save_json(args.outdir)


if __name__ == "__main__":
    _main()