"""Tests for pairwise branching-ratio columns."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.cross_sections import calculate_branching_ratios


def test_pairwise_branching_ratios_cover_all_state_pairs() -> None:
    frame = pd.DataFrame(
        {
            "state_index": [0, 1, 2, 3],
            "state_label": ["state_0", "state_1", "state_2", "state_3"],
            "sigma_au": [1.0, 2.0, 4.0, 8.0],
            "incident_energy_ev": [40.0, 40.0, 40.0, 40.0],
        }
    )

    result = calculate_branching_ratios(frame)

    expected_columns = {
        "branching_ratio_1_0",
        "branching_ratio_2_0",
        "branching_ratio_2_1",
        "branching_ratio_3_0",
        "branching_ratio_3_1",
        "branching_ratio_3_2",
    }
    assert expected_columns.issubset(result.columns)

    assert np.isclose(result.loc[1, "branching_ratio_1_0"], 2.0)
    assert np.isclose(result.loc[2, "branching_ratio_2_1"], 2.0)
    assert np.isclose(result.loc[2, "branching_ratio_2_0"], 4.0)
    assert np.isclose(result.loc[3, "branching_ratio_3_2"], 2.0)
    assert np.isclose(result.loc[3, "branching_ratio_3_1"], 4.0)
    assert np.isclose(result.loc[3, "branching_ratio_3_0"], 8.0)
