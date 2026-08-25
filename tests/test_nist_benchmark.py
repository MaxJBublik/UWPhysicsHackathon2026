"""Unit tests for the NIST validation module (src/analysis/nist_benchmark.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis import nist_benchmark as nb

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (
    (REPO_ROOT / "data" / "raw_pyscf" / "manifold_Be.json").exists()
    and (REPO_ROOT / "data" / "reference" / "nist_reference.json").exists()
)
needs_data = pytest.mark.skipif(not HAS_DATA, reason="reference data missing")


def test_triplet_j_average_uses_1_3_5_weights():
    levels = {
        "2s2p_3P_J0": {"energy_ev": 1.0},
        "2s2p_3P_J1": {"energy_ev": 2.0},
        "2s2p_3P_J2": {"energy_ev": 3.0},
    }
    # (1*1 + 3*2 + 5*3) / 9 = 22/9
    assert nb.triplet_j_average_ev(levels) == pytest.approx(22.0 / 9.0)
    assert nb.fine_structure_spread_ev(levels) == pytest.approx(2.0)


def test_dipole_oscillator_strength_roundtrip():
    delta_e_ev, mu = 5.27744322, 3.2193
    f_value = 2.0 / 3.0 * (delta_e_ev / nb.HARTREE_TO_EV) * mu**2
    assert nb.dipole_from_oscillator_strength(
        f_value, delta_e_ev) == pytest.approx(mu, rel=1e-9)


@needs_data
def test_reference_energies_are_monotonic_in_Z():
    reference = nb.load_reference(REPO_ROOT)
    singlets = [reference["species"][s]["levels"]["2s2p_1P_J1"]["energy_ev"]
                for s in ("Be", "C2+", "Fe22+")]
    assert singlets == sorted(singlets)


@needs_data
def test_computed_terms_separate_dark_and_bright_states():
    for species in nb.SPECIES_LIST:
        terms = nb.computed_terms(species, REPO_ROOT)
        assert terms["mu_singlet_au"] > 0
        # the dipole-forbidden triplet must lie below the allowed singlet
        assert terms["triplet_ev"] is not None
        assert terms["triplet_ev"] < terms["singlet_ev"]


@needs_data
def test_low_Z_energies_agree_with_nist_within_10_percent():
    reference = nb.load_reference(REPO_ROOT)
    table = nb.compare_energies(reference, REPO_ROOT)
    low_z = table[table["Z"] <= 6]
    assert (low_z["percent_error"].abs() < 10.0).all()


@needs_data
def test_benchmark_end_to_end(tmp_path):
    payload = nb.run_benchmark(REPO_ROOT, tmp_path)
    assert (tmp_path / "nist_benchmark.json").exists()
    assert (tmp_path / "nist_benchmark.png").exists()
    assert payload["findings"]
    energy_fit = payload["scaling_exponents"]["energy_scaling"]
    # both computed and reference energies must grow with Z
    assert energy_fit["computed"]["alpha"] > 0
    assert energy_fit["reference"]["alpha"] > 0
    assert abs(energy_fit["alpha_difference"]) < 0.5
