r"""Cross-section utilities for processed impact-parameter sweeps.

This module reads the probability sweeps stored under
``data/processed_circuits/`` and integrates them with

.. math::

   \sigma_{0\to j} = 2\pi \int P_{0\to j}(b)\, b\, db.

Cross-sections are returned in atomic units (``a_0^2``), cm^2, and Megabarns
(Mb), where ``1 Mb = 10^{-18} cm^2``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.integrate import simpson, trapezoid


BOHR_TO_CM: float = 5.29177210903e-9
A0_SQ_TO_CM2: float = BOHR_TO_CM**2
A0_SQ_TO_MB: float = A0_SQ_TO_CM2 / 1.0e-18


def _coerce_1d_float_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _integrate_weighted_probability(
    b_grid: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate P(b) * b over the impact-parameter grid."""

    if probabilities.ndim == 1:
        weighted = probabilities * b_grid
        sigma_simpson = 2.0 * np.pi * simpson(weighted, x=b_grid)
        sigma_trapezoid = 2.0 * np.pi * trapezoid(weighted, x=b_grid)
    elif probabilities.ndim == 2:
        if probabilities.shape[0] == b_grid.size:
            weighted = probabilities * b_grid[:, None]
            axis = 0
        elif probabilities.shape[1] == b_grid.size:
            weighted = probabilities * b_grid[None, :]
            axis = 1
        else:
            raise ValueError(
                "2D probabilities must have one dimension equal to len(b_grid)"
            )

        sigma_simpson = 2.0 * np.pi * simpson(weighted, x=b_grid, axis=axis)
        sigma_trapezoid = 2.0 * np.pi * trapezoid(weighted, x=b_grid, axis=axis)
    else:
        raise ValueError("probabilities must be one-dimensional or two-dimensional")

    denom = np.maximum(np.abs(sigma_simpson), 1.0e-30)
    rel_error = np.abs(sigma_simpson - sigma_trapezoid) / denom
    return sigma_simpson, sigma_trapezoid, rel_error


def _load_processed_circuit_data(
    processed_data: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(processed_data, (str, Path)):
        import json

        with Path(processed_data).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return dict(processed_data)


def calculate_processed_circuit_cross_sections(
    processed_data: Mapping[str, Any] | str | Path,
    strict: bool = False,
    integration_tolerance: float = 5.0e-2,
) -> pd.DataFrame:
    """Integrate excitation probabilities from a processed-circuit JSON file.

    Parameters
    ----------
    processed_data
        Path to a file in ``data/processed_circuits/`` or an in-memory mapping
        with ``impact_parameters_bohr`` and ``excitation_probabilities_vs_b``.
    strict
        If ``True``, raise on unconverged Simpson/trapezoid disagreement. For
        the coarse five-point sweeps in ``data/processed_circuits/``, the
        default is ``False`` so a usable estimate is still returned.
    integration_tolerance
        Relative disagreement threshold used to flag convergence.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by excited-state number with cross sections in
        ``a_0^2``, cm^2, and Mb plus convergence diagnostics.
    """

    sweep = _load_processed_circuit_data(processed_data)

    b_grid = _coerce_1d_float_array(sweep["impact_parameters_bohr"], "impact_parameters_bohr")
    probability_map = sweep["excitation_probabilities_vs_b"]
    if not isinstance(probability_map, Mapping):
        raise ValueError(
            "excitation_probabilities_vs_b must be a mapping of state labels to arrays"
        )

    excited_state_keys = sorted(
        (
            key
            for key in probability_map.keys()
            if key.startswith("state_") and key != "state_0"
        ),
        key=lambda item: int(item.split("_", 1)[1]),
    )
    if not excited_state_keys:
        raise ValueError("No state_* excitation probabilities were found")

    probability_matrix = np.column_stack(
        [np.asarray(probability_map[key], dtype=float) for key in excited_state_keys]
    )

    sigma_au, sigma_trapezoid, rel_error = _integrate_weighted_probability(
        b_grid=b_grid,
        probabilities=probability_matrix,
    )

    converged = rel_error <= integration_tolerance
    if strict and not bool(np.all(converged)):
        raise RuntimeError(
            "Impact-parameter integration is not converged; refine the sweep or "
            "increase the sampled impact-parameter range"
        )

    frame = pd.DataFrame(
        {
            "state_label": excited_state_keys,
            "sigma_au": sigma_au,
            "sigma_cm2": sigma_au * A0_SQ_TO_CM2,
            "sigma_mb": sigma_au * A0_SQ_TO_MB,
            "sigma_trapezoid_au": sigma_trapezoid,
            "integration_relative_error": rel_error,
            "integration_converged": converged,
            "impact_parameter_min_bohr": float(np.min(b_grid)),
            "impact_parameter_max_bohr": float(np.max(b_grid)),
            "incident_energy_ev": float(sweep["incident_energy_ev"]),
        }
    )
    frame.insert(0, "state_index", np.arange(1, len(excited_state_keys) + 1, dtype=int))
    return frame.set_index("state_index")


def save_processed_circuit_cross_sections(
    processed_data: Mapping[str, Any] | str | Path,
    output_path: Path | str,
    strict: bool = False,
    integration_tolerance: float = 5.0e-2,
) -> Path:
    """Calculate and save processed-circuit cross sections to JSON."""

    frame = calculate_processed_circuit_cross_sections(
        processed_data=processed_data,
        strict=strict,
        integration_tolerance=integration_tolerance,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_json(output_file, orient="records", indent=2)
    return output_file


if __name__ == "__main__":
    default_path = Path("data/processed_circuits/populations_Be_E40eV.json")
    output_path = Path("data/cross_sections/cross_sections_Be_E40eV.json")
    result = calculate_processed_circuit_cross_sections(default_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.reset_index().to_json(output_path, orient="records", indent=2)
    print(f"[+] Saved cross sections to: {output_path}")
    print(result[["state_label", "sigma_au", "sigma_mb", "integration_relative_error", "integration_converged"]])