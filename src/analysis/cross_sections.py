r"""
Cross-section and Branching Ratio calculations for processed impact-parameter sweeps.

This module reads the probability sweeps stored under `data/processed_circuits/`
and integrates them over impact parameters:
    \sigma_{0 \to j}(E_inc) = 2\pi \int_{b_min}^{b_max} P_{0 \to j}(b, E_inc) \, b \, db

And computes the energy-dependent Branching Ratios:
    \mathcal{B}_{0 \to j}(E_inc) = \frac{\sigma_{0 \to j}(E_inc)}{\sum_{k \neq 0} \sigma_{0 \to k}(E_inc)}

Cross-sections are calculated in atomic units (a_0^2), cm^2, and Megabarns (Mb, 1 Mb = 10^-18 cm^2).
"""

from __future__ import annotations

import json
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
=======
from typing import Any, Mapping
import json

>>>>>>> main
import numpy as np

# Physical conversion constants
BOHR_TO_CM: float = 5.29177210903e-9
A0_SQ_TO_CM2: float = BOHR_TO_CM ** 2
A0_SQ_TO_MB: float = A0_SQ_TO_CM2 / 1.0e-18


# ============================================================================
# Section 1: Robust Numerical Quadrature (Simpson & Trapezoid)
# ============================================================================

def _trapezoid_quadrature(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Composite trapezoidal numerical integration compatible with NumPy 1.x/2.x and SciPy."""
    try:
        from scipy.integrate import trapezoid
        return trapezoid(y, x=x, axis=axis)
    except ImportError:
        pass

    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x=x, axis=axis)

    # Pure NumPy manual fallback
    if y.ndim == 1:
        return 0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x))
    dx = np.diff(x)
    if axis == 0:
        return 0.5 * np.sum((y[1:, :] + y[:-1, :]) * dx[:, None], axis=0)
    else:
        return 0.5 * np.sum((y[:, 1:] + y[:, :-1]) * dx[None, :], axis=1)


def _simpson_quadrature(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Composite Simpson's rule numerical integration with linear fallback."""
    try:
        from scipy.integrate import simpson
        return simpson(y, x=x, axis=axis)
    except ImportError:
        pass

    # Pure NumPy manual Simpson's rule implementation
    n = y.shape[axis] if y.ndim > 1 else len(y)
    if n < 3:
        return _trapezoid_quadrature(y, x=x, axis=axis)

    dx = np.diff(x)
    if np.allclose(dx, dx[0]):
        h = float(dx[0])
        weights = np.ones(n)
        weights[1:-1:2] = 4.0
        weights[2:-2:2] = 2.0
        if y.ndim == 1:
            return (h / 3.0) * np.sum(weights * y)
        if axis == 0:
            return (h / 3.0) * np.sum(weights[:, None] * y, axis=0)
        return (h / 3.0) * np.sum(weights[None, :] * y, axis=1)

    return _trapezoid_quadrature(y, x=x, axis=axis)


def _integrate_weighted_probability(
    b_grid: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Integrates 2 * \pi * P(b) * b over the impact-parameter grid.
    """
    if probabilities.ndim == 1:
        weighted = probabilities * b_grid
        sigma_simpson = 2.0 * np.pi * _simpson_quadrature(weighted, x=b_grid)
        sigma_trapezoid = 2.0 * np.pi * _trapezoid_quadrature(weighted, x=b_grid)
    elif probabilities.ndim == 2:
        if probabilities.shape[0] == b_grid.size:
            weighted = probabilities * b_grid[:, None]
            axis = 0
        elif probabilities.shape[1] == b_grid.size:
            weighted = probabilities * b_grid[None, :]
            axis = 1
        else:
            raise ValueError("2D probabilities must have one dimension equal to len(b_grid)")

        sigma_simpson = 2.0 * np.pi * _simpson_quadrature(weighted, x=b_grid, axis=axis)
        sigma_trapezoid = 2.0 * np.pi * _trapezoid_quadrature(weighted, x=b_grid, axis=axis)
    else:
        raise ValueError("probabilities must be one-dimensional or two-dimensional")

    denom = np.maximum(np.abs(sigma_simpson), 1.0e-30)
    rel_error = np.abs(sigma_simpson - sigma_trapezoid) / denom
    return sigma_simpson, sigma_trapezoid, rel_error


<<<<<<< HEAD
# ============================================================================
# Section 2: Core Cross-Section and Branching Ratio Calculator
# ============================================================================

def _load_processed_circuit_data(processed_data: Union[Mapping[str, Any], str, Path]) -> Dict[str, Any]:
    """Loads a processed circuit JSON file or dictionary."""
=======
def _load_processed_circuit_data(
    processed_data: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
>>>>>>> main
    if isinstance(processed_data, (str, Path)):
        with Path(processed_data).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return dict(processed_data)


<<<<<<< HEAD
def calculate_cross_sections_and_branching_ratios(
    processed_data: Union[Mapping[str, Any], str, Path],
    strict: bool = False,
    integration_tolerance: float = 0.05,
) -> List[Dict[str, Any]]:
=======
def _write_json_records(frame: pd.DataFrame, output_file: Path) -> None:
    """Write a DataFrame as a JSON array of records with native Python scalars."""

    records: list[dict[str, Any]] = []
    for record in frame.reset_index().to_dict(orient="records"):
        clean_record: dict[str, Any] = {}
        for key, value in record.items():
            if value is None:
                clean_record[key] = None
            elif isinstance(value, np.generic):
                native_value = value.item()
                clean_record[key] = None if pd.isna(native_value) else native_value
            elif pd.isna(value):
                clean_record[key] = None
            else:
                clean_record[key] = value
        records.append(clean_record)

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


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
        DataFrame indexed by state number with cross sections in ``a_0^2``,
        cm^2, and Mb plus convergence diagnostics.
>>>>>>> main
    """
    Integrates excitation probabilities and calculates Cross Sections & Branching Ratios.

    Returns:
    --------
    records : List[Dict[str, Any]]
        List of dictionaries with cross sections, branching ratios, and convergence diagnostics.
    """
    sweep = _load_processed_circuit_data(processed_data)

    species = sweep.get("species", "Unknown")
    incident_energy_ev = float(sweep.get("incident_energy_ev", 0.0))
    b_grid = np.asarray(sweep["impact_parameters_bohr"], dtype=float)
    probability_map = sweep["excitation_probabilities_vs_b"]

<<<<<<< HEAD
    excited_state_keys = sorted(
        (key for key in probability_map.keys() if key.startswith("state_") and key != "state_0"),
        key=lambda item: int(item.split("_", 1)[1]),
    )
    if not excited_state_keys:
        raise ValueError("No excited state_* probabilities found in processed data.")
=======
    state_keys = sorted(
        (key for key in probability_map.keys() if key.startswith("state_")),
        key=lambda item: int(item.split("_", 1)[1]),
    )
    if not state_keys:
        raise ValueError("No state_* excitation probabilities were found")
>>>>>>> main

    probability_matrix = np.column_stack(
        [np.asarray(probability_map[key], dtype=float) for key in state_keys]
    )

    probability_matrix = np.clip(probability_matrix, 0.0, 1.0)

    sigma_au, sigma_trapezoid, rel_error = _integrate_weighted_probability(
        b_grid=b_grid,
        probabilities=probability_matrix,
    )

    # Calculate Total Inelastic Excitation Cross Section: \sigma_tot = \sum_{j \neq 0} \sigma_j
    total_excitation_sigma_au = float(np.sum(sigma_au))
    total_excitation_sigma_mb = total_excitation_sigma_au * A0_SQ_TO_MB

    # Compute Branching Ratios: B_{0 -> j} = \sigma_j / \sigma_tot
    if total_excitation_sigma_au > 1e-30:
        branching_ratios = sigma_au / total_excitation_sigma_au
    else:
        branching_ratios = np.zeros_like(sigma_au)

    converged = rel_error <= integration_tolerance
    if strict and not bool(np.all(converged)):
        raise RuntimeError("Impact-parameter integration did not converge within tolerance.")

<<<<<<< HEAD
    records = []
    for idx, key in enumerate(excited_state_keys):
        state_num = int(key.split("_")[1])
        rec = {
            "species": species,
            "incident_energy_ev": incident_energy_ev,
            "state_index": state_num,
            "state_label": key,
            "sigma_au": float(sigma_au[idx]),
            "sigma_cm2": float(sigma_au[idx] * A0_SQ_TO_CM2),
            "sigma_mb": float(sigma_au[idx] * A0_SQ_TO_MB),
            "branching_ratio": float(branching_ratios[idx]),
            "sigma_trapezoid_au": float(sigma_trapezoid[idx]),
            "integration_relative_error": float(rel_error[idx]),
            "integration_converged": bool(converged[idx]),
=======
    frame = pd.DataFrame(
        {
            "state_label": state_keys,
            "sigma_au": sigma_au,
            "sigma_cm2": sigma_au * A0_SQ_TO_CM2,
            "sigma_mb": sigma_au * A0_SQ_TO_MB,
            "sigma_trapezoid_au": sigma_trapezoid,
            "integration_relative_error": rel_error,
            "integration_converged": converged,
>>>>>>> main
            "impact_parameter_min_bohr": float(np.min(b_grid)),
            "impact_parameter_max_bohr": float(np.max(b_grid)),
            "total_excitation_sigma_mb": total_excitation_sigma_mb,
        }
        records.append(rec)

    return records


def calculate_processed_circuit_cross_sections(
    processed_data: Union[Mapping[str, Any], str, Path],
    strict: bool = False,
    integration_tolerance: float = 0.05,
) -> Any:
    """
    Wrapper returning pandas DataFrame when pandas is installed, or dict list.
    """
    records = calculate_cross_sections_and_branching_ratios(
        processed_data=processed_data,
        strict=strict,
        integration_tolerance=integration_tolerance,
    )
<<<<<<< HEAD
    try:
        import pandas as pd
        frame = pd.DataFrame(records)
        return frame.set_index("state_index")
    except ImportError:
        return records
=======
    frame.insert(0, "state_index", np.arange(0, len(state_keys), dtype=int))

    return frame.set_index("state_index")
>>>>>>> main


def save_processed_circuit_cross_sections(
    processed_data: Union[Mapping[str, Any], str, Path],
    output_path: Union[Path, str],
    strict: bool = False,
    integration_tolerance: float = 0.05,
) -> Path:
    """Calculate and save cross sections and branching ratios to JSON."""
    records = calculate_cross_sections_and_branching_ratios(
        processed_data=processed_data,
        strict=strict,
        integration_tolerance=integration_tolerance,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
=======
    _write_json_records(frame, output_file)
    return output_file


def calculate_branching_ratios(
    cross_section_data: pd.DataFrame | Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    r"""Calculate pairwise branching ratios from cross sections.

    The pairwise branching ratio between two states ``i`` and ``j`` is defined
    here as

    .. math::

       \mathcal{B}_{i\to j}(E_{\mathrm{inc}}) = \frac{\sigma_i(E_{\mathrm{inc}})}{\sigma_j(E_{\mathrm{inc}})}.

    For the Be-like output files this yields columns such as
    ``branching_ratio_3_2``, ``branching_ratio_3_1``, ``branching_ratio_3_0``,
    ``branching_ratio_2_1``, ``branching_ratio_2_0``, and
    ``branching_ratio_1_0``.
    """

    if isinstance(cross_section_data, pd.DataFrame):
        frame = cross_section_data.copy()
    else:
        if isinstance(cross_section_data, (str, Path)):
            import json

            with Path(cross_section_data).open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        else:
            records = cross_section_data

        frame = pd.DataFrame(records)

    if "state_index" in frame.columns:
        frame = frame.set_index("state_index")

    if "sigma_au" not in frame.columns:
        raise ValueError("cross_section_data must contain a sigma_au column")

    ordered_frame = frame.copy()
    ordered_frame = ordered_frame.sort_index()

    state_indices = [int(value) for value in ordered_frame.index.to_list()]
    sigma_values = ordered_frame["sigma_au"].to_numpy(dtype=float)

    result = ordered_frame.copy()
    result["incident_energy_ev"] = result.get("incident_energy_ev", np.nan)

    for higher_pos, higher_index in enumerate(state_indices):
        higher_sigma = float(sigma_values[higher_pos])
        for lower_pos in range(higher_pos):
            lower_index = state_indices[lower_pos]
            lower_sigma = float(sigma_values[lower_pos])
            ratio_column = f"branching_ratio_{higher_index}_{lower_index}"
            result[ratio_column] = np.nan
            if lower_sigma > 0.0:
                result.loc[higher_index, ratio_column] = higher_sigma / lower_sigma

    return result


def build_cross_section_records(
    cross_section_data: pd.DataFrame | Mapping[str, Any] | str | Path,
) -> pd.DataFrame:
    """Return cross-section records with branching-ratio columns merged in."""

    return calculate_branching_ratios(cross_section_data)


def save_processed_circuit_cross_sections(
    processed_data: Mapping[str, Any] | str | Path,
    output_path: Path | str,
    strict: bool = False,
    integration_tolerance: float = 5.0e-2,
) -> Path:
    """Calculate and save processed-circuit cross sections with branching ratios to JSON."""

    frame = calculate_processed_circuit_cross_sections(
        processed_data=processed_data,
        strict=strict,
        integration_tolerance=integration_tolerance,
    )
    frame = build_cross_section_records(frame)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json_records(frame, output_file)
>>>>>>> main
    return output_file


# ============================================================================
# Section 3: CLI Entrypoint
# ============================================================================

if __name__ == "__main__":
<<<<<<< HEAD
    import argparse

    parser = argparse.ArgumentParser(description="Cross Section & Branching Ratio Calculator")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed_circuits/populations_Be_E50eV.json",
        help="Input processed circuit JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cross_sections/cross_sections_Be_E50eV.json",
        help="Output cross-section JSON file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        saved_file = save_processed_circuit_cross_sections(input_path, args.output)
        print(f"[+] Saved cross sections & branching ratios to: {saved_file}")
        recs = calculate_cross_sections_and_branching_ratios(input_path)
        print("\nResults:")
        for r in recs:
            print(f"  [{r['state_label']}] Cross-Section: {r['sigma_mb']:.6e} Mb | Branching Ratio: {r['branching_ratio']*100:.2f}% | Converged: {r['integration_converged']}")
    else:
        print(f"[!] Input file not found: {input_path}")
=======
    default_path = Path("data/processed_circuits/populations_Be_E40eV.json")
    output_path = Path("data/cross_sections/cross_sections_Be_E40eV.json")
    result = build_cross_section_records(calculate_processed_circuit_cross_sections(default_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_records(result, output_path)
    print(f"[+] Saved cross sections to: {output_path}")
    print(result[["state_label", "sigma_au", "sigma_mb", "integration_relative_error", "integration_converged"]])
>>>>>>> main
