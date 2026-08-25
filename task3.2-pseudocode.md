# Task 3.2 — `src/analysis/scaling_laws.py` Pseudocode & Code Flow

## Module-level layout

```
src/analysis/scaling_laws.py
│
├── CONSTANTS
├── Section 1: Data loading        (load_manifold, load_cross_sections, load_species_summary, build_scaling_table)
├── Section 2: Power-law fitting   (fit_power_law)
├── Section 3: Scaling analyses    (analyze_energy_scaling, analyze_dipole_scaling, analyze_coulomb_focusing)
├── Section 4: Plotting            (plot_energy_scaling, plot_dipole_scaling, plot_sigma_ratios)
├── Section 5: Export & main       (export_summary, run_full_analysis, main)
```

## Overall code flow

```
main()
  └── run_full_analysis(data_dir, output_dir)
        ├── for species in [Be, C2+, Fe22+]:
        │     └── load_species_summary(species)          # Step 1
        │           ├── load_manifold(species)           # energies, dipoles, Z, q
        │           └── load_cross_sections(species)     # sigma per state, per energy
        ├── table = build_scaling_table(summaries)       # one tidy DataFrame
        ├── e_fit  = analyze_energy_scaling(table)       # Step 2: ΔE ∝ Z^α
        ├── mu_fit = analyze_dipole_scaling(table)       # Step 3: μ ∝ Z^β
        ├── ratios = analyze_coulomb_focusing(table)     # Step 4: σ_ion/σ_neutral
        ├── plot_energy_scaling / plot_dipole_scaling / plot_sigma_ratios   # Step 5
        └── export_summary(table, e_fit, mu_fit, ratios) # data/scaling/scaling_summary.json
```

---

## Section 0 — Constants

```pseudocode
CONSTANT SPECIES_LIST = ["Be", "C2+", "Fe22+"]
CONSTANT DIPOLE_NOISE_FLOOR = 1e-8         # treat |μ| below this as zero
CONSTANT MANIFOLD_DIR = "data/raw_pyscf"
CONSTANT SIGMA_DIR    = "data/cross_sections"
CONSTANT OUTPUT_DIR   = "data/scaling"
```

---

## Section 1 — Data loading

```pseudocode
FUNCTION load_manifold(species) -> dict
    """Read data/raw_pyscf/manifold_{species}.json."""
    path = MANIFOLD_DIR / f"manifold_{species}.json"
    IF path does not exist: RAISE FileNotFoundError with helpful message
    raw = json.load(path)
    RETURN {
        species:  raw["species"],
        Z:        raw["atomic_number"],
        charge:   raw["charge"],
        n_states: raw["n_states"],
        energies_ev: array(raw["excitation_energies_ev"]),
        dipole_xyz:  stack(raw["dipole_matrix_x"], _y, _z)   # shape (3, N, N)
    }


FUNCTION dominant_dipole_magnitudes(dipole_xyz, n_states) -> array[N]
    """|μ_0j| = sqrt(μx² + μy² + μz²) from ground state to each state j."""
    FOR j in 0..N-1:
        mu[j] = sqrt( sum over axis a of dipole_xyz[a, 0, j]**2 )
        IF mu[j] < DIPOLE_NOISE_FLOOR: mu[j] = 0.0     # kill 1e-15 noise
    RETURN mu


FUNCTION load_cross_sections(species) -> DataFrame
    """Glob data/cross_sections/cross_sections_{species}_E*eV.json.

    Globbing (not a fixed 50 eV path) means new energies from Track B
    are picked up automatically."""
    frames = []
    FOR file in glob(SIGMA_DIR / f"cross_sections_{species}_E*eV.json"):
        records = json.load(file)          # list of per-state dicts
        frames.append(DataFrame(records))
    IF frames empty: WARN and RETURN empty DataFrame
    RETURN concat(frames)                  # columns include state_index,
                                           # sigma_au, incident_energy_ev,
                                           # integration_converged


FUNCTION load_species_summary(species) -> dict
    manifold = load_manifold(species)
    mu       = dominant_dipole_magnitudes(manifold.dipole_xyz, manifold.n_states)
    sigma_df = load_cross_sections(species)

    # pick the "dominant channel": excited state j with the largest |μ_0j|
    j_dom = argmax(mu[1:]) + 1

    RETURN {
        **manifold,
        mu_per_state: mu,
        dominant_state: j_dom,
        delta_E_dominant_ev: manifold.energies_ev[j_dom],
        mu_dominant: mu[j_dom],
        sigma_table: sigma_df,
    }


FUNCTION build_scaling_table(summaries: list[dict]) -> DataFrame
    """One row per (species, excited state, incident energy)."""
    rows = []
    FOR s in summaries:
        FOR each row r in s.sigma_table:
            rows.append({
                species, Z, charge,
                state_index: r.state_index,
                delta_E_ev:  s.energies_ev[r.state_index],
                mu_au:       s.mu_per_state[r.state_index],
                sigma_au:    r.sigma_au,
                E_inc_ev:    r.incident_energy_ev,
                converged:   r.integration_converged,
                is_dominant: (r.state_index == s.dominant_state),
            })
    RETURN DataFrame(rows)
```

---

## Section 2 — Power-law fitting

```pseudocode
FUNCTION fit_power_law(x, y) -> dict
    """Fit y = A * x^alpha via linear regression in log-log space."""
    ASSERT all(x > 0) and all(y > 0)       # drop zero/negative before calling
    slope, intercept = polyfit(log(x), log(y), degree=1)
    y_pred = exp(intercept) * x**slope
    r_squared = 1 - SS_res / SS_tot        # goodness of fit
    RETURN {alpha: slope, prefactor: exp(intercept), r_squared}
```

---

## Section 3 — Scaling analyses

```pseudocode
FUNCTION analyze_energy_scaling(table) -> dict          # Step 2
    sub = table WHERE is_dominant, one row per species (dedupe on species)
    fit = fit_power_law(x=sub.Z, y=sub.delta_E_ev)
    # sanity check: expect alpha > 0 (gap grows with Z)
    IF fit.alpha <= 0: WARN "unexpected energy scaling — check state ordering"
    RETURN {points: (Z, delta_E) per species, fit}


FUNCTION analyze_dipole_scaling(table) -> dict          # Step 3
    sub = table WHERE is_dominant, dedupe on species
    fit = fit_power_law(x=sub.Z, y=sub.mu_au)
    # sanity check: expect alpha < 0 (dipole shrinks ~1/Z)
    IF fit.alpha >= 0: WARN "unexpected dipole scaling"
    RETURN {points, fit}


FUNCTION analyze_coulomb_focusing(table, reference="Be") -> dict   # Step 4
    # only compare at energies ALL species share (currently 50 eV)
    common_E = intersection of E_inc_ev across species
    results = []
    FOR E in common_E:
        sigma_ref = sigma of reference species, dominant channel, at E
        FOR species in table (excluding reference):
            ratio = sigma_species_dominant / sigma_ref
            flag  = NOT converged for either -> mark unreliable
            results.append({species, E, ratio, reliable: not flag})
    RETURN results
```

---

## Section 4 — Plotting  (each: figure -> save PNG -> return path)

```pseudocode
FUNCTION plot_energy_scaling(analysis, output_dir)
    log-log scatter of (Z, delta_E) with species labels
    overlay fitted line, annotate "ΔE ∝ Z^{alpha:.2f} (R²=...)"
    save output_dir / "deltaE_vs_Z.png"

FUNCTION plot_dipole_scaling(analysis, output_dir)
    same shape, annotate "μ ∝ Z^{alpha:.2f}"
    save output_dir / "dipole_vs_Z.png"

FUNCTION plot_sigma_ratios(focusing, output_dir)
    bar chart: x = species (C2+, Fe22+), y = sigma/sigma_Be at common E
    hatch or grey-out bars flagged unreliable (non-converged integration)
    horizontal line at ratio = 1 (no enhancement)
    save output_dir / "sigma_ratio.png"
```

---

## Section 5 — Export & entry point

```pseudocode
FUNCTION export_summary(table, e_fit, mu_fit, focusing, output_dir)
    payload = {
        generated_from: list of input files,
        per_species_table: table.to_dict(),
        energy_scaling:  e_fit,
        dipole_scaling:  mu_fit,
        coulomb_focusing: focusing,
        caveats: ["single common energy (50 eV)",
                  "Fe22+ state 3 integration not converged", ...]
    }
    write output_dir / "scaling_summary.json"


FUNCTION run_full_analysis(data_dir=".", output_dir=OUTPUT_DIR) -> dict
    mkdir(output_dir)
    summaries = [load_species_summary(s) FOR s in SPECIES_LIST]   # Step 1
    table  = build_scaling_table(summaries)
    e_fit  = analyze_energy_scaling(table)                        # Step 2
    mu_fit = analyze_dipole_scaling(table)                        # Step 3
    focus  = analyze_coulomb_focusing(table)                      # Step 4
    plot_energy_scaling(e_fit, output_dir)                        # Step 5
    plot_dipole_scaling(mu_fit, output_dir)
    plot_sigma_ratios(focus, output_dir)
    export_summary(table, e_fit, mu_fit, focus, output_dir)
    print human-readable summary (exponents, ratios)
    RETURN {table, e_fit, mu_fit, focus}


FUNCTION main()
    argparse: --data-dir, --output-dir, --reference-species (default Be)
    run_full_analysis(parsed args)

IF __name__ == "__main__": main()
```

---

## Tests — `tests/test_scaling_laws.py`  (Step 6)

```pseudocode
TEST test_loader_returns_all_species:
    summaries = [load_species_summary(s) for s in SPECIES_LIST]
    ASSERT Z values == [4, 6, 26]
    ASSERT each has n_states >= 2 and positive delta_E_dominant_ev

TEST test_fit_power_law_recovers_known_exponent:
    x = [4, 6, 26]; y = 2.0 * x**1.5        # synthetic, no data dependence
    fit = fit_power_law(x, y)
    ASSERT fit.alpha ≈ 1.5 AND fit.prefactor ≈ 2.0

TEST test_scaling_directions:
    table = build_scaling_table(real data)
    ASSERT analyze_energy_scaling(table).fit.alpha > 0    # gap grows
    ASSERT analyze_dipole_scaling(table).fit.alpha < 0    # dipole shrinks

TEST test_end_to_end(tmp_path):
    run_full_analysis(output_dir=tmp_path)
    ASSERT tmp_path/"scaling_summary.json" exists
    ASSERT all three PNGs exist
```

## Design decisions baked in

1. **Glob, don't hardcode energies** — new `cross_sections_*_E*eV.json` files from Track B flow in automatically.
2. **Dominant channel by dipole strength, not state index** — degenerate states can reorder between species; the physics lives in the dipole-allowed transition.
3. **Noise floor on dipoles** — manifold JSONs carry ~1e-15 numerical junk; zero it before argmax.
4. **Convergence flags propagate** — Fe22+ state 3 has `integration_converged: false`; ratios built on it are marked unreliable, never silently used.
5. **JSON summary as the contract** — the Streamlit app (task 3.3) reads `scaling_summary.json` instead of recomputing.
