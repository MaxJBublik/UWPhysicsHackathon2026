"""Analytic checks for the cross-section quadrature."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.cross_sections import _integrate_weighted_probability


def test_constant_probability_cross_section_matches_analytic_result() -> None:
    b_grid = np.linspace(1.0, 5.0, 9)
    probability = 0.25
    constant_probabilities = np.full_like(b_grid, probability, dtype=float)

    sigma_simpson, sigma_trapezoid, rel_error = _integrate_weighted_probability(
        b_grid=b_grid,
        probabilities=constant_probabilities,
    )

    sigma_expected = np.pi * probability * (b_grid[-1] ** 2 - b_grid[0] ** 2)

    assert np.isclose(float(sigma_simpson), sigma_expected, rtol=1e-12, atol=1e-12)
    assert np.isclose(float(sigma_trapezoid), sigma_expected, rtol=1e-12, atol=1e-12)
    assert float(rel_error) < 1e-12


if __name__ == "__main__":
    test_constant_probability_cross_section_matches_analytic_result()
    print("constant-probability check passed")