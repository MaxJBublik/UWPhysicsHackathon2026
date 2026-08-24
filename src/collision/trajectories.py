r"""Classical electron trajectories in atomic units."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

HARTREE_TO_EV = 27.211386245988


def incident_velocity(incident_energy_ev: float) -> float:
    """Return the asymptotic electron speed in atomic units."""
    if not np.isfinite(incident_energy_ev) or incident_energy_ev <= 0:
        raise ValueError("incident_energy_ev must be finite and positive")
    return float(np.sqrt(2.0 * incident_energy_ev / HARTREE_TO_EV))


@dataclass(frozen=True)
class TrajectoryResult:
    """Sampled time, position, and velocity (arrays end in Cartesian axes)."""
    time: NDArray[np.float64]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]

    @property
    def radius(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.position, axis=-1)


def _times(time: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    values = np.asarray(time, dtype=float)
    if values.ndim > 1 or not np.all(np.isfinite(values)):
        raise ValueError("time must be a finite scalar or one-dimensional array")
    return np.atleast_1d(values), values.ndim == 0


def _shape(result: TrajectoryResult, scalar: bool) -> TrajectoryResult:
    if not scalar:
        return result
    return TrajectoryResult(result.time[0], result.position[0], result.velocity[0])


def straight_line_trajectory(time: ArrayLike, impact_parameter_bohr: float,
                             incident_energy_ev: float) -> TrajectoryResult:
    """Return ``R(t)=(b,0,v*t)`` with closest approach at ``t=0``."""
    if not np.isfinite(impact_parameter_bohr) or impact_parameter_bohr < 0:
        raise ValueError("impact_parameter_bohr must be finite and non-negative")
    t, scalar = _times(time)
    speed = incident_velocity(incident_energy_ev)
    position = np.column_stack((np.full(t.size, impact_parameter_bohr),
                                np.zeros(t.size), speed * t))
    velocity = np.column_stack((np.zeros(t.size), np.zeros(t.size),
                                np.full(t.size, speed)))
    return _shape(TrajectoryResult(t, position, velocity), scalar)


def _solve_hyperbolic_kepler(mean_anomaly: NDArray[np.float64],
                             eccentricity: float) -> NDArray[np.float64]:
    u = np.arcsinh(mean_anomaly / eccentricity)
    for _ in range(80):
        residual = eccentricity * np.sinh(u) - u - mean_anomaly
        step = residual / (eccentricity * np.cosh(u) - 1.0)
        step = np.clip(step, -1.0, 1.0)
        u_next = u - step
        if np.max(np.abs(step)) < 2e-13:
            return u_next
        u = u_next
    raise RuntimeError("hyperbolic Kepler equation failed to converge")


def coulomb_hyperbolic_trajectory(time: ArrayLike,
                                  impact_parameter_bohr: float,
                                  incident_energy_ev: float,
                                  ion_charge: float) -> TrajectoryResult:
    r"""Exact attractive Rutherford orbit for ``V(r)=-ion_charge/r``.

    The orbit lies in the x-z plane and reaches periapsis at ``t=0``.
    ``impact_parameter_bohr`` is the asymptotic impact parameter.
    """
    if not np.isfinite(impact_parameter_bohr) or impact_parameter_bohr <= 0:
        raise ValueError(
            "impact_parameter_bohr must be finite and positive for a hyperbolic orbit"
        )
    if not np.isfinite(ion_charge) or ion_charge <= 0:
        raise ValueError("ion_charge must be finite and positive")
    t, scalar = _times(time)
    speed = incident_velocity(incident_energy_ev)
    a = ion_charge / speed**2
    eccentricity = np.sqrt(1.0 + (impact_parameter_bohr / a) ** 2)
    time_scale = a / speed
    anomaly = _solve_hyperbolic_kepler(t / time_scale, eccentricity)
    root = np.sqrt(eccentricity**2 - 1.0)
    position = np.column_stack((a * (eccentricity - np.cosh(anomaly)),
                                np.zeros(t.size),
                                a * root * np.sinh(anomaly)))
    du_dt = 1.0 / (time_scale * (eccentricity * np.cosh(anomaly) - 1.0))
    velocity = np.column_stack((-a * np.sinh(anomaly) * du_dt,
                                np.zeros(t.size),
                                a * root * np.cosh(anomaly) * du_dt))
    return _shape(TrajectoryResult(t, position, velocity), scalar)


def generate_trajectory(time: ArrayLike, impact_parameter_bohr: float,
                        incident_energy_ev: float, *,
                        trajectory_type: str = "straight_line",
                        ion_charge: float = 0.0) -> TrajectoryResult:
    """Dispatch to a straight-line or Coulomb-hyperbolic path."""
    kind = trajectory_type.lower().replace("-", "_")
    if kind in {"straight", "straight_line"}:
        return straight_line_trajectory(time, impact_parameter_bohr,
                                        incident_energy_ev)
    if kind in {"coulomb", "hyperbolic", "coulomb_hyperbolic"}:
        return coulomb_hyperbolic_trajectory(
            time, impact_parameter_bohr, incident_energy_ev, ion_charge)
    raise ValueError(f"unknown trajectory_type: {trajectory_type!r}")


straight_line = straight_line_trajectory
coulomb_hyperbolic = coulomb_hyperbolic_trajectory
coulomb_trajectory = coulomb_hyperbolic_trajectory
