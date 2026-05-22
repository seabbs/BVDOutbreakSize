# Continuous-time explicit-convolution v2 architecture

Prototype proposal implementing [issue #5](https://github.com/epiforecasts/BVDOutbreakSize/issues/5).
This is the lowest-risk redesign option: it keeps the deterministic
continuous-time growth and the numerical-integral observation models of
the current code, and changes only what the latent state *means* and how
each stream maps onto it.

## Motivation: resolve the onsets/infections inconsistency

The current model treats the latent trajectory `C(s) = exp(r·s)`
inconsistently across streams (issue #5).
The deaths convolution treats `C` as cumulative *onsets*: it convolves
`C` with an onset-to-death delay, which only makes sense if `C` counts
symptom onsets.
The exports window treats `C` as cumulative *infections*: the detection
window `w` is defined by McCabe et al. as "incubation + onset-to-detection",
which only makes sense if the clock starts at infection.
The same `C(s)` cannot be both onsets and infections at once, so the
current model lives in a pragmatic middle ground and the CFR is
ambiguously "fraction of infections-or-cases that die".

v2 fixes this by making the latent state unambiguously cumulative
*infections* and giving every observation stream its own explicit delay
convolution from infection.
A new incubation (infection-to-onset) delay sits between infections and
every downstream onset-clocked quantity, so onsets, deaths, reports and
exports are all derived consistently from one infection process.
CFR becomes unambiguously the fraction of *onsets* that die.

## Mathematics

Latent state, single zoonotic seed at `s = 0`, exponential growth at
rate `r = log(2)/τ`:

```math
I(t)        = e^{r t}                          \quad\text{cumulative infections}
i_{inf}(t)  = r\,e^{r t}                        \quad\text{infection incidence}
```

Onset incidence is infection incidence convolved with the incubation
density `f_inc`:

```math
i_{onset}(t) = \int_0^t r\,e^{r s}\, f_{inc}(t - s)\, ds .
```

The three onset-clocked observation expectations then each convolve
`i_onset` against a delay:

```math
\mathbb{E}[\text{exports}] = p_U\, q \int_{T-w}^{T} i_{onset}(s)\, ds
```
```math
\mathbb{E}[\text{deaths}]  = \mathrm{CFR} \int_0^T i_{onset}(s)\, F_{otd}(T-s)\, ds
```
```math
\mathbb{E}[\text{reports}] = p_{DRC} \int_0^T i_{onset}(s)\, F_{otr}(T-s)\, ds
```

with `q` the per-capita travel rate, `w` the onset-to-detection window,
`F_otd` the onset-to-death CDF, `F_otr` the onset-to-report CDF, and
`p_U`, `p_DRC` the pooled ascertainment fractions.

Two structural improvements over the current model fall out of this:

- The exports window `w` now applies to *onset* incidence, so it is
  unambiguously onset-to-detection, not the incubation-plus-detection
  mixture the current detection-window prior tries to span.
- Reporting is now *delayed*: the current model reports an instantaneous
  fraction `p_DRC · C(T)`, but v2 convolves with an onset-to-report
  delay, so very recent infections are not yet fully ascertained. This
  is more realistic and couples the reported-case stream to the same
  delay machinery as deaths.

The deaths and reports expectations are **nested** convolutions: the
inner `i_onset(s)` is itself a convolution, integrated again against an
outer delay CDF. This is the central cost of v2 (see Runtime).

## Mapping the four data streams plus TMRCA

| Stream | v2 expectation | Likelihood |
|---|---|---|
| Exported cases (Uganda) | `expected_exports_v2` over the onset-to-detection window | Poisson |
| Suspected deaths (DRC) | `expected_deaths_v2`, onset-to-death convolution | NegBinomial (shared `k`) |
| Reported cases (DRC) | `expected_reports_v2`, onset-to-report convolution | NegBinomial (shared `k`) |
| Deaths among exports | reuse the current `expected_exports_deaths` machinery on the onset window (see Migration) | binned Poisson |
| Genetic TMRCA | soft lower bound on `T`, `censored(Normal(T, σ); upper = g)` | unchanged |

The growth, CFR, traveller-volume, dispersion and pooled-ascertainment
priors are unchanged from the current model and are injected into the
v2 composer rather than re-defined.

## New incubation prior

The Imperial report cites a 6-11 day incubation range across three
PubMed studies but no Bayesian posterior exists anywhere, and the
vendored bdbv-linelist analysis does not fit incubation (its line list
has onset dates only).
The prior is constructed from the narrative range as a gamma delay with
weakly-informative truncated-Normal priors on its parameters:

```math
\alpha_{inc} \sim \mathrm{Normal}^+(11, 3), \qquad
\theta_{inc} \sim \mathrm{Normal}^+(0.74, 0.25),
```

giving `Gamma(α, θ)` a mean near 8 days, covering the cited 6-11 day
span.
Both parameters are expected to be prior-dominated (see Identifiability).

The onset-to-report delay carries the same caveat as in issue #4: the
bdbv-linelist Isiro 2012 onset-to-notification estimate is 19.7 d
(13.7-30.1), but Charniga 2024 flags a 30-day-cap truncation bias, so
the prior is centred near 18 days and used with a strong caveat.

## Numerical implications and runtime

The smoke prototype (`scripts/smoke_v2.jl`) measured the nested-integral
cost directly against the current single-convolution helper, at
`r = log(2)/14`, `T = 90`, on this machine:

| Quantity | Cost |
|---|---|
| `OnsetIncidence` build (65-point grid) | ~460 µs/draw |
| v2 deaths integral (reusing the tabulated onset curve) | ~4 µs/call |
| current single deaths convolution | ~6 µs/call |

The onset-incidence tabulation dominates: it pays the inner incubation
convolution once per grid node (65 clustered 64-node convolutions).
Reusing the tabulated curve across all three observation integrals keeps
the per-draw observation cost near the current model; without the
precompute, the inner convolution would be paid at every outer
quadrature node of every stream (~`64 × 3` times per draw instead of
65).

Net per-draw forward cost is therefore roughly **one extra inner
convolution layer**, i.e. order ~0.5 ms/draw on top of the current
model's tens of µs. Against the current ~6 min full fit this points to a
**rough doubling of wall-clock**, dominated by the onset tabulation
rather than the outer integrals. The grid size (`ONSET_GRID_POINTS = 65`)
trades build cost against tail accuracy: cumulative onsets are accurate
to <0.1% at 65 points versus a 1025-point reference, with the largest
pointwise error confined to the exponential tail near `T` that the
deaths and reports convolutions weight.

The clustered onset convolution is exact against an adaptive QuadGK
reference (relative error ~1e-15 in the smoke check), so the only v2
approximation beyond the current model is the linear interpolation of
the tabulated onset curve.

## Mooncake-AD findings

The full conditioned joint model (16 sampled parameters: τ, m, the two
incubation params, the two onset-to-death params, the two onset-to-report
params, CFR, dispersion, the four ascertainment params, the window)
builds a Mooncake `LogDensityFunction`, returns a finite log density,
and a **finite Mooncake gradient** (‖grad‖ ≈ 24 at a prior draw).
A short NUTS smoke fit (50 warmup + 50 samples, single chain) completes
without error, including compilation, in under 30 s.

The AD-safety pattern from the current code carries over unchanged: the
deaths and reports convolutions differentiate through the *density*
alone, never the gamma-CDF shape-parameter derivative (which Mooncake
does not support). v2 reuses the existing `ExportDeathDelay` precomputed
CDF holder for both the onset-to-death and onset-to-report CDFs,
extending its grid to cover `[0, T]` for the cumulative streams.
The incubation convolution differentiates through the incubation density
only.

## Identifiability of the weakly-supported incubation params

The two incubation parameters `(α_inc, θ_inc)` are informed only
indirectly: no stream observes onsets directly, so the data constrain
the incubation lag mainly through the lag between infections and the
delayed reported-case and death signals.
With a handful of aggregate counts this is weak, so both are expected to
be prior-dominated, tracking the constructed 6-11 day prior closely.
This is an honest limitation of v2: it adds two parameters the data
barely speak to, in exchange for a coherent latent state.
The onset-to-report parameters are similarly weak and inherit the
30-day-cap caveat.

## Drift from the McCabe et al. replication

v2 stays the closest of the redesign options to McCabe et al.: it keeps
deterministic exponential growth, the same numerical-integral
observation structure, and the same priors for everything except the new
incubation and onset-to-report delays.
The drift is the inserted incubation layer (which McCabe et al. fold
into the window `w`) and the now-delayed reporting.
For the exports stream, separating incubation out of `w` shifts the
window's interpretation but not the published sensitivity range; the
deaths stream is essentially unchanged because onset-to-death already
clocked from onset.
The expected drift in the posterior `C(T)` (now reported as cumulative
infections `I_T`, with cumulative onsets `onsets_T` slightly lower) is
small relative to the renewal or compartmental alternatives.

## Migration effort from current code

Low. The v2 layer reuses the existing Gauss-Legendre `integrate`
helpers, the `ExportDeathDelay` precomputed-CDF holder, the
`_delay_scale` clustering, and all the unchanged building-block
submodels.
The new code is two files added alongside the current model without
touching it:

- `src/convolution_v2.jl` — the explicit-convolution forward layer
  (`infection_incidence`, `onset_incidence`, `OnsetIncidence`,
  `expected_onsets_v2`, `expected_exports_v2`, `expected_deaths_v2`,
  `expected_reports_v2`).
- `src/models_v2.jl` — the Turing building-block and observation
  submodels and the `bvd_joint_v2` composer.

The deaths-among-exports and export-detection-timing terms are not yet
ported in the prototype; they reuse the current
`expected_exports_deaths` / `expected_exports` machinery and would be
rewired to take the tabulated onset curve, a small follow-up.

## What v2 does and does not buy

v2 **does**:

- resolve the onsets/infections inconsistency with a coherent latent
  state;
- make CFR unambiguously fraction-of-onsets;
- give the exports window a clean onset-to-detection meaning;
- add realistic reporting delay;
- stay close to the McCabe et al. replication and reuse almost all
  existing code and integrators.

v2 **does not**:

- replace deterministic growth with a stochastic incidence process (the
  trajectory is still `exp(r·t)`; only observation counts carry noise);
- add a mechanistic transmission model, a renewal equation, or
  compartments;
- add spatial structure;
- relax the single-zoonotic-seed assumption.

It also costs roughly a doubling of runtime and adds two weakly-identified
delay parameters.
If the goal is a genuinely different generative process (stochastic
transmission, `R_t`, saturation), v2 is not it; the renewal or
compartmental options are.
v2 is the right incremental step when the priority is internal
consistency and defensible delay semantics without abandoning the
McCabe et al.-anchored structure.

## Optional time-varying growth extension

The design is not locked to constant `r`.
Because the whole forward layer is driven by the infection incidence
`i_inf(s)`, a time-varying growth is a drop-in: replace the scalar `r`
with a piecewise-constant or random-walk log-growth `r(s)` and integrate
`i_inf(s) = I'(s)` numerically, where `I(s) = exp(∫_0^s r(u) du)`.
The onset convolution, the `OnsetIncidence` tabulation, and all three
observation integrals are unchanged because they only ever call
`i_onset(t)`.
With four aggregate counts a full random walk is not identifiable, but a
single breakpoint (two-segment piecewise `r`) is cheap to add and would
let the design express a slowdown since seeding.
Not implemented in the prototype to keep the smoke test focused on the
core convolution change; flagged here so the architecture is not
mistaken for being constant-growth-only.
