"""
Species configurations for the Beryllium isoelectronic sequence (4-electron systems).
"""

from typing import Dict, Any

SPECIES_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Be": {
        "symbol": "Be",
        "name": "Neutral Beryllium",
        "atomic_number": 4,
        "charge": 0,
        "spin": 0,  # 2S = 0 (singlet ground state)
        "num_electrons": 4,
        "recommended_basis": "cc-pvdz",
        "active_orbitals": 2,  # e.g., 2s, 2px, 2py, 2pz
        "plasma_regime": "Low-temperature / Edge Plasma",
        "trajectory_type": "straight_line",
    },
    "C2+": {
        "symbol": "C",
        "name": "Be-like Carbon Ion (C 2+)",
        "atomic_number": 6,
        "charge": 2,
        "spin": 0,
        "num_electrons": 4,
        "recommended_basis": "cc-pvdz",
        "active_orbitals": 4,
        "active_electrons": 2,
        "plasma_regime": "Warm / Divertor / Astrophysical Plasma",
        "trajectory_type": "coulomb_hyperbolic",
    },
    "Fe22+": {
        "symbol": "Fe",
        "name": "Be-like Iron Ion (Fe 22+)",
        "atomic_number": 26,
        "charge": 22,
        "spin": 0,
        "num_electrons": 4,
        "recommended_basis": "def2-tzvp",
        "active_orbitals": 4,
        "active_electrons": 2,
        "plasma_regime": "Tokamak Core / Solar Flare / X-Ray Plasma",
        "trajectory_type": "coulomb_hyperbolic",
    },
}

def get_species_config(species_key: str) -> Dict[str, Any]:
    """Retrieve species configuration dictionary."""
    if species_key not in SPECIES_CONFIGS:
        raise ValueError(
            f"Unknown species: {species_key}. Supported species are: {list(SPECIES_CONFIGS.keys())}"
        )
    return SPECIES_CONFIGS[species_key]
