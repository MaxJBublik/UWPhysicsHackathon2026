"""Build a symmetry-resolved Be singlet manifold in the standard JSON schema."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf

HARTREE_TO_EV = 27.211386245988


def solve_sector(mol, mf, sector: str, nroots: int):
    mc = mcscf.CASCI(mf, mol.nao_nr(), mol.nelectron)
    solver = fci.direct_spin0_symm.FCI(mol)
    solver.wfnsym = sector
    solver.nroots = nroots
    solver.conv_tol = 1e-10
    mc.fcisolver = solver
    output = mc.kernel()
    energies = np.atleast_1d(np.asarray(output[0], dtype=float))
    vectors = [output[2]] if energies.size == 1 else list(output[2])
    return mc, energies, vectors


def calculate(output_path: str | Path, basis: str = "cc-pvtz") -> Path:
    mol = gto.M(atom="Be 0 0 0", basis=basis, charge=0, spin=0,
                symmetry="D2h", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Be RHF did not converge")

    solved = {}
    for sector, nroots in {"Ag": 4, "B1g": 1, "B2g": 1, "B3g": 1,
                           "B1u": 1, "B2u": 1, "B3u": 1}.items():
        solved[sector] = solve_sector(mol, mf, sector, nroots)

    # 1S, three 1P components, then five 1D components.
    selection = [
        ("Ag", 0, "1S"),
        ("B1u", 0, "1P"), ("B2u", 0, "1P"), ("B3u", 0, "1P"),
        ("Ag", 1, "1D"), ("Ag", 2, "1D"),
        ("B1g", 0, "1D"), ("B2g", 0, "1D"), ("B3g", 0, "1D"),
    ]
    total = np.array([solved[s][1][r] for s, r, _ in selection])
    vectors = [solved[s][2][r] for s, r, _ in selection]
    excitation = total - total[0]

    ncas = mol.nao_nr()
    mo = mf.mo_coeff
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao_dipole = mol.intor("int1e_r", comp=3)
    mo_dipole = np.einsum("pi,xpq,qj->xij", mo, ao_dipole, mo,
                          optimize=True)
    dipoles = np.zeros((3, len(selection), len(selection)))
    for i in range(len(selection)):
        for j in range(i, len(selection)):
            tdm1 = fci.direct_spin1.trans_rdm1(vectors[i], vectors[j],
                                               ncas, mol.nelectron)
            value = -np.einsum("xpq,qp->x", mo_dipole, tdm1,
                               optimize=True).real
            dipoles[:, i, j] = dipoles[:, j, i] = value

    delta_e = excitation[None, :] - excitation[:, None]
    oscillator = (2.0 / 3.0) * delta_e * np.sum(dipoles**2, axis=0)
    result = {
        "species": "Be",
        "atomic_number": 4,
        "charge": 0,
        "spin_2s": 0,
        "spin_multiplicity": 1,
        "n_states": len(selection),
        "basis": basis,
        "energies_au": excitation.tolist(),
        "energies_ev": (excitation * HARTREE_TO_EV).tolist(),
        "excitation_energies_ev": (excitation * HARTREE_TO_EV).tolist(),
        "dipole_matrix_x": dipoles[0].tolist(),
        "dipole_matrix_y": dipoles[1].tolist(),
        "dipole_matrix_z": dipoles[2].tolist(),
        "oscillator_strengths": oscillator[0].tolist(),
        "oscillator_strength_matrix": oscillator.tolist(),
        "is_mock": False,
        "total_energies_au": total.tolist(),
        "method": (f"RHF/symmetry-resolved CASCI({mol.nelectron},{ncas}), "
                   "singlet full-basis FCI"),
        "active_space": {"orbitals": ncas, "electrons": mol.nelectron},
        "spatial_symmetry": "D2h subgroup of atomic SO(3)",
        "state_components": [
            {"index": i, "term": term, "irrep": sector,
             "sector_root": root}
            for i, (sector, root, term) in enumerate(selection)
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path
