"""
Neutral beryllium, ground and excited singlet states, end to end.

  RHF -> state-averaged CASSCF -> transition dipoles -> oscillator strengths
       -> NIST energy correction

Writes two files. The raw one is the untouched ab initio record. The corrected
one carries measured excitation energies and is the file to feed a dynamics
model.

Run:  python be_master.py
"""

import json
import numpy as np
from pyscf import gto, scf, mcscf, fci

# ============================================================== configuration

SPECIES = "Be"
CHARGE  = 0
SPIN    = 0            # 2S value, 0 is the closed-shell singlet
BASIS   = "cc-pvdz"

# Active space. NELECAS electrons in NCAS orbitals.
# (4, 5) is 1s + 2s + 2p_x,y,z holding all four electrons. Dropping to NCAS = 4
# removes one 2p and breaks the threefold degeneracy of the 1P state.
NELECAS, NCAS = 4, 5

# Number of singlet roots. Only certain values are physically sensible, since
# each term brings a fixed number of degenerate components with it.
#    1  ->  1S (2s^2) on its own
#    4  ->  plus 1P (2s2p), three components
#    9  ->  plus 1D (2p^2), five components
#   10  ->  plus the upper 1S (2p^2), which is autoionizing in reality
# Any other value slices through a degenerate set and skews the averaged
# orbitals, which visibly splits states that should be identical.
N_STATES = 9

# NIST Atomic Spectra Database, Be I, in cm^-1. One entry per degenerate block,
# in ascending energy order, and the list must cover every block the
# calculation produces. Kramida & Martin, J. Phys. Chem. Ref. Data 26, 1185
# (1997), as tabulated in the NIST Handbook of Basic Atomic Spectroscopic Data.
NIST_BLOCKS = [
    {"term": "2s2 1S",  "energy_cm": 0.00,     "degeneracy": 1},
    {"term": "2s2p 1P", "energy_cm": 42565.35, "degeneracy": 3},
    {"term": "2p2 1D",  "energy_cm": 56882.43, "degeneracy": 5},
]

# Optional second correction. Energy rescaling alone leaves the resonance-line
# oscillator strength too low, because the computed transition dipole is also in
# error and the two errors were partly cancelling. Setting this True scales the
# ground-to-1P dipole block so the total f matches the measured value.
#
# VERIFY F_EXP_RESONANCE against NIST ASD before trusting it. The value below is
# approximate and is a placeholder, not a citation.
SCALE_RESONANCE_DIPOLE = False
F_EXP_RESONANCE = 1.34

RAW_OUT  = "data/new_raw_pyscf/be_casscf_raw.json"
NIST_OUT = "data/new_raw_pyscf/be_casscf_nist.json"

VERBOSE = 0           # bump to 4 for the full PySCF log

CM2AU      = 4.5563352529e-6
HARTREE2EV = 27.211386245988
DEGEN_TOL_EV = 1e-3

# ================================================================ calculation


def run_casscf():
    """RHF followed by state-averaged CASSCF. Returns the mc object."""
    mol = gto.M(
        atom     = f"{SPECIES} 0 0 0",
        basis    = BASIS,
        charge   = CHARGE,
        spin     = SPIN,
        symmetry = False,   # off so the averaged solver can mix irreps freely
        verbose  = VERBOSE,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.fcisolver = fci.direct_spin0.FCI(mol)      # spin-adapted, singlets only
    mc.fcisolver.nroots = N_STATES
    mc = mc.state_average_([1.0 / N_STATES] * N_STATES)
    mc.kernel()
    return mol, mf, mc


def dipole_matrix(mol, mc):
    """Full N x N dipole matrix over the state-averaged CASSCF states.

    Off-diagonal entries are transition dipoles. Diagonal entries are permanent
    dipoles and include the core-electron and nuclear contributions, so for an
    atom at the origin they come out at ~1e-15. That is a free sanity check.
    """
    ncore   = mc.ncore
    mo_core = mc.mo_coeff[:, :ncore]
    mo_cas  = mc.mo_coeff[:, ncore:ncore + NCAS]

    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor("int1e_r", comp=3)

    dip_cas  = np.einsum("pi,xpq,qj->xij", mo_cas, ao_dip, mo_cas)
    core_dm  = 2.0 * mo_core @ mo_core.T
    core_dip = np.einsum("xpq,qp->x", ao_dip, core_dm)
    nucl_dip = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())

    dip = np.zeros((3, N_STATES, N_STATES))
    for i in range(N_STATES):
        for j in range(N_STATES):
            tdm1 = fci.direct_spin1.trans_rdm1(mc.ci[i], mc.ci[j],
                                               NCAS, NELECAS)
            d = -np.einsum("xij,ji->x", dip_cas, tdm1)   # electron charge is -1
            if i == j:
                d = d - core_dip + nucl_dip
            dip[:, i, j] = d
    return dip


def oscillator_matrix(energies_au, dip):
    """f_ij = (2/3)(E_j - E_i)|<i|r|j>|^2, atomic units throughout.

    The upper triangle is absorption. The lower triangle carries a minus sign,
    which is the usual emission convention.
    """
    dE = energies_au[None, :] - energies_au[:, None]
    return (2.0 / 3.0) * dE * np.sum(dip ** 2, axis=0)


# ================================================================= correction


def group_degenerate(energies_ev, tol=DEGEN_TOL_EV):
    """Split an ascending energy list into blocks of near-equal values."""
    blocks, current = [], [0]
    for k in range(1, len(energies_ev)):
        if abs(energies_ev[k] - energies_ev[current[0]]) < tol:
            current.append(k)
        else:
            blocks.append(current)
            current = [k]
    blocks.append(current)
    return blocks


def apply_nist(calc_ev, calc_total_au, dip):
    """Swap computed excitation energies for measured ones.

    The ground state keeps its ab initio absolute energy as the anchor. Every
    excited block is placed at that value plus the measured gap. Blocks are
    matched to NIST entries by degeneracy rather than by hardcoded index, so a
    symmetry break or a changed state count raises an error instead of silently
    mismatching levels.
    """
    blocks = group_degenerate(calc_ev)

    if len(blocks) != len(NIST_BLOCKS):
        raise SystemExit(
            f"Found {len(blocks)} degenerate blocks but have "
            f"{len(NIST_BLOCKS)} NIST entries. Block sizes were "
            f"{[len(b) for b in blocks]}."
        )
    for blk, ref in zip(blocks, NIST_BLOCKS):
        if len(blk) != ref["degeneracy"]:
            raise SystemExit(
                f"Block for {ref['term']} has {len(blk)} states, expected "
                f"{ref['degeneracy']}. The calculation may have broken symmetry."
            )

    n = len(calc_ev)
    nist_au = np.zeros(n)
    term_of = [""] * n
    for blk, ref in zip(blocks, NIST_BLOCKS):
        for k in blk:
            nist_au[k] = ref["energy_cm"] * CM2AU
            term_of[k] = ref["term"]

    dip = dip.copy()
    scale = 1.0
    if SCALE_RESONANCE_DIPOLE:
        ground_blk, first_blk = blocks[0], blocks[1]
        dE_res  = nist_au[first_blk[0]] - nist_au[ground_blk[0]]
        mu2_res = np.sum(dip[:, ground_blk[0], first_blk] ** 2)
        scale = float(np.sqrt(F_EXP_RESONANCE / ((2.0 / 3.0) * dE_res * mu2_res)))
        for i in ground_blk:
            for j in first_blk:
                dip[:, i, j] *= scale
                dip[:, j, i] *= scale

    total_au = calc_total_au[0] + nist_au
    return blocks, nist_au, total_au, term_of, dip, scale


# ====================================================================== output


def package(energies_au, total_au, dip, **extra):
    f = oscillator_matrix(energies_au, dip)
    out = {
        "species": SPECIES,
        "atomic_number": 4,
        "charge": CHARGE,
        "n_states": N_STATES,
        "basis": BASIS,
        "energies_au": energies_au.tolist(),
        "energies_ev": (energies_au * HARTREE2EV).tolist(),
        "excitation_energies_ev": (energies_au * HARTREE2EV).tolist(),
        "dipole_matrix_x": dip[0].tolist(),
        "dipole_matrix_y": dip[1].tolist(),
        "dipole_matrix_z": dip[2].tolist(),
        "oscillator_strengths": f.tolist(),
        "is_mock": False,
        "total_energies_au": total_au.tolist(),
        "method": f"RHF/SA-CASSCF({NELECAS},{NCAS})",
        "active_space": {"orbitals": NCAS, "electrons": NELECAS},
    }
    out.update(extra)
    return out, f


def main():
    mol, mf, mc = run_casscf()

    calc_total = np.array(mc.e_states)
    calc_au    = calc_total - calc_total[0]
    calc_ev    = calc_au * HARTREE2EV
    dip        = dipole_matrix(mol, mc)

    raw, f_raw = package(
        calc_au, calc_total, dip,
        energy_source=f"computed, RHF/SA-CASSCF({NELECAS},{NCAS})/{BASIS}",
        dipole_source=f"RHF/SA-CASSCF({NELECAS},{NCAS})/{BASIS}",
    )
    json.dump(raw, open(RAW_OUT, "w"), indent=2)

    blocks, nist_au, nist_total, terms, dip_c, scale = apply_nist(
        calc_ev, calc_total, dip
    )
    corrected, f_nist = package(
        nist_au, nist_total, dip_c,
        term_symbols=terms,
        energy_source="NIST ASD (Kramida & Martin 1997), Be I",
        dipole_source=f"RHF/SA-CASSCF({NELECAS},{NCAS})/{BASIS}",
        dipole_scaling_applied=(scale if SCALE_RESONANCE_DIPOLE else None),
        computed_energies_ev=calc_ev.tolist(),
    )
    json.dump(corrected, open(NIST_OUT, "w"), indent=2)

    # ------------------------------------------------------------- report
    nist_ev = nist_au * HARTREE2EV
    print(f"RHF        {mf.e_tot:.6f} Ha")
    print(f"SA-CASSCF  {calc_total[0]:.6f} Ha   "
          f"method RHF/SA-CASSCF({NELECAS},{NCAS})/{BASIS}\n")
    print(f"{'idx':>5} {'term':>10} {'calc (eV)':>11} "
          f"{'NIST (eV)':>11} {'shift':>9}")
    for blk, ref in zip(blocks, NIST_BLOCKS):
        k = blk[0]
        label = str(blk[0]) if len(blk) == 1 else f"{blk[0]}-{blk[-1]}"
        print(f"{label:>5} {ref['term']:>10} {calc_ev[k]:11.4f} "
              f"{nist_ev[k]:11.4f} {nist_ev[k] - calc_ev[k]:9.4f}")

    first = blocks[1]
    print(f"\nresonance f   raw {f_raw[0, first].sum():.4f}"
          f"   corrected {f_nist[0, first].sum():.4f}")
    if SCALE_RESONANCE_DIPOLE:
        print(f"resonance dipole block scaled by {scale:.4f}")
    print(f"\nwritten to {RAW_OUT} and {NIST_OUT}")


if __name__ == "__main__":
    main()