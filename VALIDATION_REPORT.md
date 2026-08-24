# Experimental Validation Report — Task 3.2

**Scope:** verify the computed electronic manifolds and the isoelectronic
Z-scaling laws against measured atomic data.
**Reference:** NIST Atomic Spectra Database (Be I, C III, Fe XXIII), retrieved
2026-08-24, plus a published C III oscillator strength.
**Reproduce with:** `python -m src.analysis.nist_benchmark`

---

## 1. Excitation energies vs NIST

The CASCI calculation is non-relativistic, so the computed `3P` level is
compared against the statistically weighted J-average
`E(3P) = [E(J=0) + 3E(J=1) + 5E(J=2)] / 9`, which is the quantity a
spin-orbit-free method actually approximates.

| Species | Spectrum | Term | Computed (eV) | NIST (eV) | Error |
|---|---|---|---:|---:|---:|
| Be | Be I | 2s2p ³P | 2.8505 | 2.7252 | **+4.6 %** |
| Be | Be I | 2s2p ¹P | 5.7889 | 5.2774 | **+9.7 %** |
| C²⁺ | C III | 2s2p ³P | 6.5713 | 6.4992 | **+1.1 %** |
| C²⁺ | C III | 2s2p ¹P | 13.6520 | 12.6900 | **+7.6 %** |
| Fe²²⁺ | Fe XXIII | 2s2p ³P | 43.8044 | 52.9614 | **−17.3 %** |
| Fe²²⁺ | Fe XXIII | 2s2p ¹P | 82.8806 | 93.2870 | **−11.2 %** |

**Verdict: the physics is right, the accuracy is Z-dependent.** Every state is
correctly identified — the dipole-forbidden triplet sits below the
dipole-allowed singlet in all three species, exactly as observed. Agreement is
good at low Z (1–10 %) and degrades systematically as Z grows, flipping sign
between C²⁺ and Fe²²⁺. This is the expected signature of a non-relativistic
method: at Z = 26 the inner electrons are relativistic and the calculation
under-binds the excited configuration.

The manifold energies are therefore fit for the Be and C²⁺ results, and
Fe²²⁺ numbers should be quoted with a ~15 % caveat.

## 2. Transition dipoles vs reference oscillator strengths

| Species | f computed | f reference | \|μ\| computed (a.u.) | \|μ\| reference (a.u.) | ratio |
|---|---:|---:|---:|---:|---:|
| Be | 0.5413 | 1.34 | 1.9536 | 3.2193 | **0.607** |
| C²⁺ | 0.2911 | 0.767 | 0.9330 | 1.5707 | **0.594** |
| Fe²²⁺ | 0.0521 | *not available* | 0.1602 | — | — |

**Verdict: a systematic ~40 % underestimate, not a random error.** The
computed dipole is 0.60 × the reference for *both* species that could be
checked. A near-constant multiplicative offset has two consequences:

- **Absolute cross sections are too small by roughly 0.6² ≈ 0.36**, since
  σ scales with |μ|². Any σ quoted from this pipeline is a lower bound.
- **The Z-scaling exponent survives**, because a constant factor cancels in
  a power-law fit. The shape of the scaling result is trustworthy even though
  the magnitude is not.

The cause is the small active space: a (4e, 4o) CASCI in a cc-pVDZ basis gives
the 2p orbital too little radial extent. Enlarging the active space or the
basis would recover most of the missing dipole strength.

## 3. Z-scaling exponent — the headline Task 3.2 result

| Quantity | Computed | NIST reference | Difference |
|---|---:|---:|---:|
| ΔE ∝ Z^α (¹P resonance) | **+1.37** | **+1.49** | −0.12 |

**Verdict: validated.** The computed energy-gap scaling exponent reproduces the
exponent extracted from measured NIST energies to within 0.12 — about 8 %.
The Task 3.2 conclusion that excitation energies grow slightly faster than
linearly in Z is confirmed by experiment.

The dipole exponent could not be validated over the full range because no
open-source reference f-value for Fe XXIII was obtainable. Over the two
species that do have reference values (Be → C²⁺), the computed and reference
dipole exponents are −1.82 and −1.77 — agreement to 3 %, again supporting the
μ ∝ 1/Z picture.

---

## 4. Two defects found by the comparison

### 4.1 Degenerate multiplets are truncated — cross sections undercounted

Both the ³P and ¹P terms are **threefold spatially degenerate**, but the
4-state manifolds keep only 2 of 3 triplet components and **1 of 3 singlet
components** for every species.

This is not cosmetic. The excitation cross section into the ¹P term is the sum
over its three components, so the pipeline is currently reporting roughly
**one third of the true resonance-channel cross section**, on top of the 0.36
factor from the dipole error.

**Fix:** rerun the PySCF manifolds with `n_states = 7` or more (1 ground + 3
triplet + 3 singlet components) — Track A owns this.

### 4.2 Missing relativistic fine structure at high Z

NIST shows the Fe XXIII ³P term split across **43.17 → 58.49 eV, a 15.3 eV
spread**, comparable to the excitation energy itself. The non-relativistic
CASCI returns a single degenerate level and cannot see this at all.

For a solar-flare / tokamak-core plasma these components are separately
observable lines, so any Fe²²⁺ branching-ratio claim from this pipeline should
be labelled as a non-relativistic approximation.

---

## 5. What to say in the presentation

Defensible as-is:

- The **Z-scaling exponents** — validated against NIST to within 8 %.
- **Relative** comparisons between species (branching ratios, focusing ratios),
  where the systematic dipole error largely cancels.
- Low-Z (Be, C²⁺) excitation energies, good to ~10 %.

Needs a caveat:

- **Absolute cross sections** — low by ~3× from multiplet truncation and a
  further ~2.8× from the dipole underestimate.
- **Fe²²⁺ specifics** — energies off by 11–17 %, fine structure absent.

---

## Sources

- NIST Atomic Spectra Database, Levels Form — Be I, C III, Fe XXIII levels:
  <https://physics.nist.gov/asd>
- S. N. Nahar, *Fine structure radiative transitions in C II and C III*
  (C III 2s² ¹S₀ – 2s2p ¹P₁ oscillator strength):
  <https://www.astronomy.ohio-state.edu/nahar.1/papers/adndt-fjj-c2-c3.pdf>
- Be I resonance-line f-value: standard NIST ASD tabulated value (f = 1.34);
  quoted from the literature, not re-fetched in this session.
