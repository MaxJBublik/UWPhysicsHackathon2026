"""Tests that state_0 is integrated like the other states."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.cross_sections import calculate_processed_circuit_cross_sections


def test_state_zero_cross_section_is_integrated() -> None:
    processed_data = {
        "impact_parameters_bohr": [1.0, 2.0, 3.0],
        "incident_energy_ev": 40.0,
        "excitation_probabilities_vs_b": {
            "state_0": [1.0, 1.0, 1.0],
            "state_1": [0.0, 0.0, 0.0],
        },
    }

    frame = calculate_processed_circuit_cross_sections(processed_data)

    expected_sigma_au = 2.0 * np.pi * 4.0

    assert np.isclose(frame.loc[0, "sigma_au"], expected_sigma_au)
    assert np.isclose(frame.loc[0, "sigma_trapezoid_au"], expected_sigma_au)
    assert frame.loc[0, "sigma_au"] > 0.0
    assert frame.loc[0, "sigma_cm2"] > 0.0
    assert frame.loc[0, "sigma_mb"] > 0.0
    assert frame.loc[0, "integration_relative_error"] >= 0.0
