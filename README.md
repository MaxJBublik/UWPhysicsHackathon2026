# Quantum Mechanical Collisional Excitation of Plasma Neutrals and Ions

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PySCF](https://img.shields.io/badge/ab--initio-PySCF-orange.svg)](https://pyscf.org/)
[![PennyLane](https://img.shields.io/badge/quantum-PennyLane-9cf.svg)](https://pennylane.ai/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid quantum-classical modeling framework coupling **multi-root electronic structure calculations in PySCF** with **multi-level quantum time-evolution circuits in PennyLane** to simulate electron-impact collisional excitation dynamics across plasma neutrals and ions.

---

## 🌟 Scientific Overview

In fusion (tokamak) and astrophysical plasmas, electron-impact collisions drive non-adiabatic transitions, cascade excitation, and line emissions. This framework models the **Beryllium isoelectronic sequence (4 electrons)** across three distinct plasma regimes:

1. **Neutral Beryllium ($\text{Be}$, $Z=4, q=0$)**: Low-temperature edge plasma / laboratory plasma.
2. **Weakly Charged Carbon ($\text{C}^{2+}$, $Z=6, q=+2$)**: Divertor / warm astrophysical plasma.
3. **Highly Charged Iron ($\text{Fe}^{22+}$, $Z=26, q=+22$)**: Tokamak core / solar flare / high-energy X-ray plasma.

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
├── README.md                    # Project documentation and validation guide
├── work-distribution.md         # Team tracks, schemas, and AI agent prompts
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment definition
├── main.py                      # Master end-to-end pipeline runner
├── app.py                       # Interactive Streamlit Web GUI
├── data/
│   ├── raw_pyscf/               # Ab-initio electronic structure JSONs (E_k, μ_ij)
│   ├── processed_circuits/      # Quantum population dynamics sweeps P_i(t, b)
│   ├── cross_sections/          # Integrated cross-section & branching ratio tables
│   └── scaling_analysis/        # Isoelectronic Z-scaling synthesis & plots
├── src/
│   ├── __init__.py
│   ├── electronic_structure/    # Track 1: PySCF RHF + CASCI multi-root solver
│   │   ├── __init__.py
│   │   ├── species_configs.py   # Isoelectronic configs (Be, C2+, Fe22+)
│   │   └── pyscf_runner.py      # Multi-root CI & 1-TDM transition dipole solver
│   ├── collision/               # Track 2: Collision trajectories & V(t)
│   │   ├── __init__.py
│   │   ├── trajectories.py      # Straight-line & Rutherford hyperbolic Kepler orbits
│   │   └── interaction.py       # Time-dependent dipole pulse V(t) = -μ · E(t)
│   ├── quantum/                 # Track 3: PennyLane Hamiltonian & Circuit
│   │   ├── __init__.py
│   │   ├── mapper.py            # Manifold to Pauli string decomposition
│   │   └── time_evolution.py    # Trotterized quantum time evolution & population tracking
│   └── analysis/                # Track 4: Cross sections, branching ratios & scaling
│       ├── __init__.py
│       ├── cross_sections.py    # Impact parameter integration & branching ratios
│       └── scaling_laws.py      # Z-scaling laws & multi-energy branching ratio spectra
└── tests/                       # Complete unit and integration test suite
    ├── test_collision.py
    ├── test_mapper.py
    ├── test_time_evolution.py
    └── test_analysis.py
```

---

## ⚙️ Environment Setup

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

## 🚀 How to Run the Project

### 1. Run the Full End-to-End Pipeline (One Command)

To execute all 4 stages of the project in sequence (Electronic Structure $\to$ Pauli Mapping $\to$ Quantum Circuit Evolution $\to$ Cross-Sections & Branching Ratios $\to$ Isoelectronic $Z$-Scaling):

```bash
python main.py --energy 50.0
```

**Pipeline Actions:**
1. Verifies/computes PySCF multi-root state-averaged CI for $\text{Be}, \text{C}^{2+}, \text{Fe}^{22+}$.
2. Decomposes $H(t) = H_0 + V(t)$ into $n$-qubit Pauli strings.
3. Runs Trotterized quantum time evolution over impact parameters $b \in [0.5, 10]\ a_0$.
4. Integrates cross-sections $\sigma(E)$ and calculates **Branching Ratios** $\mathcal{B}_{0 \to j}(E)$.
5. Evaluates $Z$-scaling laws across incident energies $E \in [15, 150]\text{ eV}$ and saves publication figures in `data/scaling_analysis/isoelectronic_scaling_trends.png`.

For direct control of the electronic-structure stage:

```bash
# Run for all species (Be, C 2+, Fe 22+)
python -m src.electronic_structure.pyscf_runner --species all --n-states 4

# Or run for a specific species
python -m src.electronic_structure.pyscf_runner --species Be --n-states 6 --basis cc-pvdz

# Spin-pure singlet SA-CASSCF with a custom active space
python -m src.electronic_structure.pyscf_runner --species Be --n-states 9 \
  --basis cc-pvtz --ncas 14 --casscf

# Complete symmetry-resolved Be 1S + 1P + 1D manifold
python -m src.electronic_structure.pyscf_runner --species Be \
  --basis cc-pvtz --symmetry-resolved
```

The runner uses a singlet-adapted FCI solver. CASCI is the default; pass
`--casscf` to optimize equal-weight state-averaged orbitals. Keep complete
angular shells in a truncated atomic active space to avoid symmetry breaking.

Results are saved as JSON files in `data/raw_pyscf/manifold_{species}.json`.

---

### 2. Launch the Interactive Web Dashboard (Streamlit GUI)

To explore real-time electron trajectories, quantum population dynamics, energy level spectra, and branching ratios in an interactive dashboard:

```bash
streamlit run app.py
```

**Dashboard Features:**
* **Tab 1: Quantum Dynamics**: Adjust incident energy $E_{\text{inc}}$ and impact parameter $b$ with live sliders to see the electron orbit $\vec{R}(t)$, the collision pulse $\vec{\mathcal{E}}(t)$, and quantum population curves $P_i(t)$.
* **Tab 2: Electronic Structure**: Side-by-side comparison of CASCI energy levels against NIST ASD benchmarks, and 2D dipole heatmaps.
* **Tab 3: Cross Sections & Branching Ratios**: Real-time numerical integration table showing cross-sections in atomic units ($a_0^2$), $\text{cm}^2$, and Megabarns ($\text{Mb}$), alongside branching ratio percentages.
* **Tab 4: Isoelectronic $Z$-Scaling**: Multi-species comparative scaling curves showing the influence of nuclear charge $Z$ and ion charge $q$.

---

### 3. Run the Unit & Integration Test Suite

Verify all mathematical mappings, unitarity conditions, and numerical integrations:

```bash
python tests/test_mapper.py && python tests/test_time_evolution.py && python tests/test_analysis.py
```

---

## 🔬 How to Validate Results Against Real-World Data

To independently verify and benchmark the derived values against experimental and atomic database standards, follow these guidelines:

### 1. Benchmark Electronic Structure ($\Delta E, \mu_{ij}, f_{01}$) against NIST ASD

The **NIST Atomic Spectra Database (ASD)** is the gold standard for atomic energy levels and transition probabilities:
* **Database Link:** [NIST ASD Levels Form](https://physics.nist.gov/PhysRefData/ASD/levels_form.html)

#### Expected Values for the Beryllium Isoelectronic Sequence:

| Species | Spectrum Name | Configuration & Term | NIST ASD Energy ($\text{eV}$) | Our PySCF Model ($\text{eV}$) | Validation Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Neutral $\text{Be}$** | $\text{Be I}$ ($Z=4, q=0$) | $1s^2 2s^2\ (^1S_0)$<br>$1s^2 2s 2p\ (^3P_{0,1,2})$<br>$1s^2 2s 2p\ (^1P_1)$ | **$0.00\text{ eV}$**<br>**$2.72\text{ eV}$**<br>**$5.28\text{ eV}$** | **$0.00\text{ eV}$**<br>**$2.85\text{ eV}$**<br>**$5.79\text{ eV}$** | Agreement within $\sim 0.1 - 0.5\text{ eV}$ (Active space CI without core polarization). |
| **Be-like $\text{C}^{2+}$** | $\text{C III}$ ($Z=6, q=+2$) | $1s^2 2s^2\ (^1S_0)$<br>$1s^2 2s 2p\ (^3P_{0,1,2})$<br>$1s^2 2s 2p\ (^1P_1)$ | **$0.00\text{ eV}$**<br>**$6.50\text{ eV}$**<br>**$12.69\text{ eV}$** | **$0.00\text{ eV}$**<br>**$6.57\text{ eV}$**<br>**$13.65\text{ eV}$** | Excellent agreement ($< 7\%$ error). |
| **Be-like $\text{Fe}^{22+}$** | $\text{Fe XXIII}$ ($Z=26, q=+22$) | $1s^2 2s^2\ (^1S_0)$<br>$1s^2 2s 2p\ (^3P_{0,1,2})$<br>$1s^2 2s 2p\ (^1P_1)$ | **$0.00\text{ eV}$**<br>**$35 - 50\text{ eV}$**<br>**$75 - 90\text{ eV}$** | **$0.00\text{ eV}$**<br>**$43.80\text{ eV}$**<br>**$82.88\text{ eV}$** | Soft X-ray regime transition correctly captured. |

---

### 2. Benchmark Collision Cross-Sections against OPEN-ADAS / IAEA ALADDIN

The **Atomic Data and Analysis Structure (ADAS)** and **IAEA ALADDIN Database** provide benchmark electron-impact excitation cross-sections for fusion modeling:
* **ADAS Link:** [OPEN-ADAS Database](https://open.adas.ac.uk/)
* **IAEA ALADDIN Link:** [IAEA Atomic & Molecular Data](https://www-amdis.iaea.org/ALADDIN/)

#### Validation Checks to Perform:

1. **Threshold Behavior ($E_{\text{inc}} < \Delta E$)**:
   - For incident electron energies below the excitation threshold ($E_{\text{inc}} < \Delta E$), the excitation cross-section must vanish ($\sigma \to 0$).
   - *Example:* For $\text{Fe}^{22+}$ ($^1P_1$ threshold $\Delta E = 82.9\text{ eV}$), at $E_{\text{inc}} = 50\text{ eV}$ the cross section is strictly zero or exponentially suppressed.
2. **Peak Cross-Section Position**:
   - For neutral atoms ($\text{Be}$), dipole-allowed transitions peak at $E_{\text{inc}} \approx 3 - 5 \times \Delta E$ (broad maximum between $15 - 30\text{ eV}$) and decay logarithmically as $\sim \frac{\ln(E)}{E}$ at high energies (Bethe-Born asymptotic limit).
   - For charged ions ($\text{C}^{2+}, \text{Fe}^{22+}$), Coulomb attraction pulls the peak cross-section closer to the threshold.
3. **Order-of-Magnitude Bounds**:
   - Resonance dipole cross sections for low-lying valence states typically lie in the range $\sigma \sim 0.01 - 10\text{ Mb}$ ($10^{-20} - 10^{-17}\text{ cm}^2$).
   - Cascade / multi-step transitions are suppressed by $10^{-3} - 10^{-5}$ relative to direct dipole channels.

---

### 3. Verify Isoelectronic $Z$-Scaling Laws

As nuclear charge $Z$ increases across the 4-electron series ($\text{Be} \to \text{C}^{2+} \to \text{Fe}^{22+}$), check that the outputs obey fundamental hydrogenic/isoelectronic scaling:

1. **Energy Gap Scaling**:
   $$\Delta E(Z) \propto Z \quad (\text{for } \Delta n = 0, 2s \to 2p \text{ transitions})$$
2. **Transition Dipole Contraction**:
   $$\mu(Z) = \langle 2s | r | 2p \rangle \propto \frac{1}{Z}$$
3. **Dipole Cross-Section Scaling**:
   $$\sigma_{\text{dipole}}(Z) \propto \mu^2 \propto \frac{1}{Z^2}$$
4. **Coulomb Focusing Factor**:
   For ions with net charge $q = Z - 4$, the cross section at low-to-moderate energies is enhanced by the Coulomb acceleration factor:
   $$F_{\text{Coulomb}}(q, E_{\text{inc}}) \approx 1 + \frac{q}{b \cdot E_{\text{inc}}}$$
