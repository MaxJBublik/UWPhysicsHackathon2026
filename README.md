# Quantum Mechanical Collisional Excitation of Plasma Neutrals and Ions

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/ab--initio-PySCF-orange.svg)](https://pyscf.org/)
[![PennyLane](https://img.shields.io/badge/quantum-PennyLane-9cf.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid quantum-classical modeling framework coupling **multi-root electronic structure calculations in PySCF** with **multi-level quantum time-evolution circuits in PennyLane** to simulate electron-impact collisional excitation dynamics across plasma neutrals and ions.

---

## 🌟 Scientific Overview

In fusion and astrophysical plasmas, electron collisions drive non-adiabatic transitions, cascade excitation, and ionization. This project investigates the **Beryllium isoelectronic sequence (4 electrons)** across three distinct plasma regimes:

1. **Neutral Beryllium ($\text{Be}$, $Z=4, q=0$)**: Low-temperature edge plasma / laboratory plasma.
2. **Weakly Charged Carbon ($\text{C}^{2+}$, $Z=6, q=2$)**: Divertor / warm astrophysical plasma.
3. **Highly Charged Iron ($\text{Fe}^{22+}$, $Z=26, q=22$)**: Tokamak core / solar flares / high-energy X-ray plasma.

```
  ┌─────────────────────────────┐        ┌─────────────────────────────┐
  │     PySCF Electronic        │        │     Collision Physics       │
  │     Structure Engine        │        │   Trajectories & Pulses     │
  │  E_0, E_1, ..., E_N & μ_ij  │        │   b, E_inc, Rutherford      │
  └──────────────┬──────────────┘        └──────────────┬──────────────┘
                 │                                      │
                 └──────────────────┬───────────────────┘
                                    │
                                    v
                     ┌──────────────────────────────┐
                     │   Hamiltonian Pauli Mapper   │
                     │    H(t) = H_0 + V_coll(t)    │
                     │    Decomposed into Paulis    │
                     └──────────────┬───────────────┘
                                    │
                                    v
                     ┌──────────────────────────────┐
                     │  PennyLane Quantum Circuit   │
                     │  Trotterized Time Evolution  │
                     │  Measures State Populations  │
                     └──────────────┬───────────────┘
                                    │
                                    v
                     ┌──────────────────────────────┐
                     │     Physics Deliverables     │
                     │   1. Multi-State P_i(t)      │
                     │   2. Branching Ratios        │
                     │   3. Isoelectronic Z-Scaling │
                     └──────────────────────────────┘
```

---

## 📁 Repository Structure

```
UWPhysics2026/
├── README.md                    # Project documentation and guide
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment definition
├── .gitignore                   # Git ignore configuration
├── data/
│   ├── raw_pyscf/               # Electronic structure JSONs (energies, dipoles)
│   └── processed_circuits/      # Simulation traces and benchmarks
├── src/
│   ├── __init__.py
│   ├── electronic_structure/    # Track 1: PySCF multi-root calculations
│   │   ├── __init__.py
│   │   ├── species_configs.py   # Isoelectronic configs (Be, C2+, Fe22+)
│   │   └── pyscf_runner.py      # Multi-root CI & dipole moment solver
│   ├── collision/               # Track 2: Collision trajectories & V(t)
│   │   ├── __init__.py
│   │   ├── trajectories.py      # Straight line & Coulomb hyperbolic paths
│   │   └── interaction.py       # Time-dependent dipole interaction pulse
│   ├── quantum/                 # Track 3: PennyLane Hamiltonian & Circuit
│   │   ├── __init__.py
│   │   ├── mapper.py            # Manifold to Pauli string decomposition
│   │   └── time_evolution.py    # Trotterized quantum time evolution
│   └── analysis/                # Track 4: Observable analysis & scaling
│       ├── __init__.py
│       ├── cross_sections.py    # Impact parameter integration & branching ratios
│       └── scaling_laws.py      # Z-scaling trends across species
├── notebooks/                   # Interactive Jupyter walkthroughs
│   ├── 01_pyscf_manifold_demo.ipynb
│   ├── 02_pennylane_collision_demo.ipynb
│   └── 03_isoelectronic_scaling_study.ipynb
├── tests/                       # Unit and integration tests
│   └── __init__.py
└── app.py                       # Interactive Streamlit dashboard
```

---

## ⚙️ Environment Setup

You can set up a custom Python environment using **Conda** (recommended for scientific dependencies) or standard **venv**.

### Option A: Using Conda (Recommended)

```bash
# 1. Create the environment from environment.yml
conda env create -f environment.yml

# 2. Activate the environment
conda activate quantum_plasma

# 3. Verify installation
python -c "import pyscf, pennylane; print('PySCF:', pyscf.__version__, '| PennyLane:', pennylane.__version__)"
```

### Option B: Using Python `venv`

```bash
# 1. Create a virtual environment with Python 3.10 or 3.11
python3 -m venv .venv

# 2. Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate on Windows

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quickstart & Usage

### 1. Run Multi-Root Electronic Structure (PySCF)

Generate the energy manifold and transition dipole matrices for all target species:

```bash
# Run for all species (Be, C 2+, Fe 22+)
python -m src.electronic_structure.pyscf_runner --species all --n-states 4

# Or run for a specific species
python -m src.electronic_structure.pyscf_runner --species Be --n-states 6 --basis cc-pvdz
```

Results are saved as JSON files in `data/raw_pyscf/manifold_{species}.json`.

---

## 👥 Team & Subagent Division of Labor

| Track / Role | Focus Area | Primary Files |
| :--- | :--- | :--- |
| **Track 1: Electronic Structure** | PySCF multi-root state-averaged CI, dipole matrices | `src/electronic_structure/` |
| **Track 2: Collision Physics** | Straight & Coulomb hyperbolic trajectories, $V(t)$ | `src/collision/` |
| **Track 3: Quantum Circuits** | Hamiltonian Pauli decomposition, PennyLane Trotter evolution | `src/quantum/` |
| **Track 4: Analysis & Visuals** | Branching ratios vs $E_{\text{inc}}$, $Z$-scaling plots, Streamlit app | `src/analysis/`, `app.py` |

---

## 🌿 Git Collaboration Workflow

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd UWPhysics2026
   ```
2. Create a feature branch:
   ```bash
   git checkout -b feature/<feature-name>
   ```
3. Commit and push your changes:
   ```bash
   git add .
   git commit -m "feat: description of work"
   git push origin feature/<feature-name>
   ```
4. Open a Pull Request into `main`.
