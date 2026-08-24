r"""Time-dependent dipole interaction and manifold Hamiltonian."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.collision.trajectories import TrajectoryResult, generate_trajectory


def electric_field(position: ArrayLike, *,
                   softening_bohr: float = 0.0) -> NDArray[np.float64]:
    r"""Field of an incident electron: ``E=-R/(R^2+s^2)^(3/2)``."""
    points = np.asarray(position, dtype=float)
    if points.shape == (3,):
        scalar, points = True, points[None, :]
    elif points.ndim >= 2 and points.shape[-1] == 3:
        scalar = False
    else:
        raise ValueError("position must have shape (3,) or (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("position must contain only finite values")
    if not np.isfinite(softening_bohr) or softening_bohr < 0:
        raise ValueError("softening_bohr must be finite and non-negative")
    radius_squared = np.sum(points * points, axis=-1) + softening_bohr**2
    if np.any(radius_squared == 0):
        raise ValueError("electric field is singular at R=0; use softening_bohr")
    field = points / radius_squared[..., None] ** 1.5
    return field[0] if scalar else field


def _dipole_tensor(dipoles: ArrayLike | Mapping[str, Any],
                   n_states: int) -> NDArray[np.complex128]:
    if isinstance(dipoles, Mapping):
        try:
            tensor = np.stack([dipoles[f"dipole_matrix_{axis}"]
                               for axis in ("x", "y", "z")])
        except KeyError as error:
            raise ValueError(f"missing dipole component {error.args[0]!r}") from error
    else:
        tensor = np.asarray(dipoles)
        # Component-first is the public contract.  Prefer it in the ambiguous
        # three-state case where both layouts have shape (3, 3, 3).
        if (tensor.shape != (3, n_states, n_states) and
                tensor.shape == (n_states, n_states, 3)):
            tensor = np.moveaxis(tensor, -1, 0)
    tensor = np.asarray(tensor, dtype=complex)
    if tensor.shape != (3, n_states, n_states):
        raise ValueError(f"dipoles must have shape (3, {n_states}, {n_states})")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("dipoles must contain only finite values")
    if not np.allclose(tensor, tensor.conj().transpose(0, 2, 1), atol=1e-10):
        raise ValueError("each dipole component must be Hermitian")
    return tensor


def interaction_matrix(dipoles: ArrayLike | Mapping[str, Any],
                       field: ArrayLike) -> NDArray[np.complex128]:
    r"""Return ``V_ij=-sum_alpha mu[alpha,i,j] E_alpha``."""
    values = np.asarray(field, dtype=float)
    if values.shape == (3,):
        scalar, values = True, values[None, :]
    elif values.ndim >= 2 and values.shape[-1] == 3:
        scalar = False
    else:
        raise ValueError("field must have shape (3,) or (..., 3)")
    if isinstance(dipoles, Mapping):
        n_states = len(dipoles.get("energies_au", dipoles["dipole_matrix_x"]))
    else:
        shape = np.shape(dipoles)
        n_states = shape[1] if len(shape) == 3 and shape[0] == 3 else shape[0]
    tensor = _dipole_tensor(dipoles, n_states)
    result = -np.einsum("...a,aij->...ij", values, tensor, optimize=True)
    return result[0] if scalar else result


def hamiltonian(energies_au: Sequence[float],
                dipoles: ArrayLike | Mapping[str, Any], position: ArrayLike,
                *, softening_bohr: float = 0.0) -> NDArray[np.complex128]:
    """Build ``H(t)=diag(E)+V(t)`` at one or many positions."""
    energies = np.asarray(energies_au, dtype=float)
    if energies.ndim != 1 or energies.size == 0 or not np.all(np.isfinite(energies)):
        raise ValueError("energies_au must be a non-empty finite 1-D sequence")
    tensor = _dipole_tensor(dipoles, energies.size)
    coupling = interaction_matrix(
        tensor, electric_field(position, softening_bohr=softening_bohr))
    return coupling + np.diag(energies)


def collision_hamiltonian(time: ArrayLike, energies_au: Sequence[float],
                          dipoles: ArrayLike | Mapping[str, Any],
                          impact_parameter_bohr: float,
                          incident_energy_ev: float, *,
                          trajectory_type: str = "straight_line",
                          ion_charge: float = 0.0,
                          softening_bohr: float = 0.0,
                          return_trajectory: bool = False):
    """Generate a trajectory and its complete time-dependent Hamiltonian."""
    path = generate_trajectory(
        time, impact_parameter_bohr, incident_energy_ev,
        trajectory_type=trajectory_type, ion_charge=ion_charge)
    matrices = hamiltonian(energies_au, dipoles, path.position,
                           softening_bohr=softening_bohr)
    return (matrices, path) if return_trajectory else matrices


def load_manifold(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an electronic manifold JSON file."""
    with Path(path).open(encoding="utf-8") as stream:
        manifold = json.load(stream)
    required = {"energies_au", "dipole_matrix_x", "dipole_matrix_y",
                "dipole_matrix_z"}
    missing = required.difference(manifold)
    if missing:
        raise ValueError(f"manifold is missing required keys: {sorted(missing)}")
    _dipole_tensor(manifold, len(manifold["energies_au"]))
    return manifold


build_hamiltonian = collision_hamiltonian
build_time_dependent_hamiltonian = collision_hamiltonian
