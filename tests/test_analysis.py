"""
Unit and integration tests for Task 3: Cross Sections, Branching Ratios, and Scaling Laws.
"""

import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.cross_sections import (
    _trapezoid_quadrature,
    _simpson_quadrature,
    calculate_cross_sections_and_branching_ratios,
    A0_SQ_TO_CM2,
    A0_SQ_TO_MB,
)
from src.analysis.scaling_laws import IsoelectronicScalingAnalyzer


def test_numerical_quadrature_accuracy():
    """Verify trapezoid and Simpson integration on exact polynomial f(x) = x^2 over [0, 2] -> 8/3."""
    x = np.linspace(0.0, 2.0, 101)
    y = x ** 2
    exact = 8.0 / 3.0

    simp = _simpson_quadrature(y, x)
    trap = _trapezoid_quadrature(y, x)

    assert np.isclose(simp, exact, atol=1e-5)
    assert np.isclose(trap, exact, atol=1e-3)


def test_cross_sections_and_branching_ratio_normalization():
    """Verify that branching ratios sum to 100% across all excited channels."""
    sample_processed = {
        "species": "Be",
        "incident_energy_ev": 50.0,
        "impact_parameters_bohr": [0.5, 1.0, 2.0, 4.0, 8.0],
        "excitation_probabilities_vs_b": {
            "state_0": [0.9, 0.95, 0.99, 0.999, 1.0],
            "state_1": [0.09, 0.045, 0.009, 0.0009, 0.0],
            "state_2": [0.01, 0.005, 0.001, 0.0001, 0.0],
        },
    }

    records = calculate_cross_sections_and_branching_ratios(sample_processed)
    assert len(records) == 2  # state_1 and state_2

    # Verify cross section units
    for r in records:
        assert r["sigma_cm2"] == r["sigma_au"] * A0_SQ_TO_CM2
        assert r["sigma_mb"] == r["sigma_au"] * A0_SQ_TO_MB
        assert 0.0 <= r["branching_ratio"] <= 1.0

    # Branching ratios must sum to 1.0
    total_br = sum(r["branching_ratio"] for r in records)
    assert np.isclose(total_br, 1.0, atol=1e-6)


def test_isoelectronic_scaling_analyzer():
    """Verify that analyzer extracts scaling parameters for available species."""
    analyzer = IsoelectronicScalingAnalyzer(raw_pyscf_dir="data/raw_pyscf")
    summary = analyzer.extract_electronic_structure_scaling()
    records = summary["scaling_records"]
    assert len(records) >= 1

    # Verify atomic number ordering
    for r in records:
        assert r["atomic_number_Z"] in [4, 6, 26]
        assert r["excitation_energy_dE1_ev"] > 0.0


if __name__ == "__main__":
    print("[*] Running all tests in test_analysis.py...")
    test_numerical_quadrature_accuracy()
    test_cross_sections_and_branching_ratio_normalization()
    test_isoelectronic_scaling_analyzer()
    print("[+] All analysis unit tests passed successfully!")
