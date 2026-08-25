"""Unit tests for Task 3.2 (src/analysis/scaling_laws.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.analysis import scaling_laws as sl

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (REPO_ROOT / "data" / "raw_pyscf" / "manifold_Be.json").exists()

needs_data = pytest.mark.skipif(not HAS_DATA,
                                reason="pipeline data not generated yet")


def test_fit_power_law_recovers_known_exponent():
    x = np.array([4.0, 6.0, 26.0])
    y = 2.0 * x**1.5
    fit = sl.fit_power_law(x, y)
    assert fit["alpha"] == pytest.approx(1.5, abs=1e-10)
    assert fit["prefactor"] == pytest.approx(2.0, rel=1e-10)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-10)


def test_fit_power_law_rejects_bad_input():
    with pytest.raises(ValueError):
        sl.fit_power_law([4.0], [1.0])
    with pytest.raises(ValueError):
        sl.fit_power_law([4.0, 6.0], [1.0, 0.0])


def test_dominant_dipole_zeroes_numerical_noise():
    dipoles = np.zeros((3, 4, 4))
    dipoles[0, 0, 3] = 1.0        # real coupling to state 3
    dipoles[1, 0, 1] = 1e-14      # PySCF noise on state 1
    mu = sl.dominant_dipole_magnitudes(dipoles, 4)
    assert mu[1] == 0.0
    assert mu[3] == pytest.approx(1.0)


@needs_data
def test_loader_returns_all_species():
    summaries = [sl.load_species_summary(s, REPO_ROOT)
                 for s in sl.SPECIES_LIST]
    assert [s["Z"] for s in summaries] == [4, 6, 26]
    for s in summaries:
        assert s["n_states"] >= 2
        assert s["delta_E_dominant_ev"] > 0
        assert s["mu_dominant"] > 0


@needs_data
def test_scaling_directions_on_real_data():
    summaries = [sl.load_species_summary(s, REPO_ROOT)
                 for s in sl.SPECIES_LIST]
    table = sl.build_scaling_table(summaries)
    assert sl.analyze_energy_scaling(table)["fit"]["alpha"] > 0
    assert sl.analyze_dipole_scaling(table)["fit"]["alpha"] < 0


@needs_data
def test_end_to_end(tmp_path):
    results = sl.run_full_analysis(REPO_ROOT, tmp_path)
    assert (tmp_path / "scaling_summary.json").exists()
    for name in ("deltaE_vs_Z.png", "dipole_vs_Z.png", "sigma_ratio.png"):
        assert (tmp_path / name).exists()
    for record in results["coulomb_focusing"]:
        assert record["ratio"] >= 0 or np.isnan(record["ratio"])
