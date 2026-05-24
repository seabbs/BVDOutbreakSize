# Explicit-convolution architecture

Design proposal for the model redesign tracked in
[issue #5](https://github.com/epiforecasts/BVDOutbreakSize/issues/5).
This is the lowest-risk redesign option: it keeps the deterministic
continuous-time growth and the numerical-integral observation models of
the current code, and changes only what the latent state *means* and
how each stream maps onto it.

## Motivation: resolve the onsets/infections inconsistency

The current model treats the latent trajectory `C(s) = exp(r·s)`
inconsistently across streams (issue #5).
The deaths convolution treats `C` as cumulative *onsets*: it convolves
`C` with an onset-to-death delay, which only makes sense if `C` counts
symptom onsets.
The exports window treats `C` as cumulative *infections*: the detection
window `w` is defined by McCabe et al. as
"incubation + onset-to-detection", which only makes sense if the clock
starts at infection.
The same `C(s)` cannot be both onsets and infections at once, so the
current model lives in a pragmatic middle ground and the CFR is
ambiguously "fraction of infections-or-cases that die".

The explicit-convolution architecture fixes this by making the latent
state unambiguously cumulative *infections* and giving every
observation stream its own explicit delay convolution from infection.
A new incubation (infection-to-onset) delay sits between infections
and every downstream onset-clocked quantity, so onsets, deaths,
reports and exports are all derived consistently from one infection
process.
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

with `q` the per-capita travel rate, `w` the onset-to-detection
window, `F_otd` the onset-to-death CDF, `F_otr` the onset-to-report
CDF, and `p_U`, `p_DRC` the pooled ascertainment fractions.

Two structural improvements over the current model fall out of this:

- The exports window `w` now applies to *onset* incidence, so it is
  unambiguously onset-to-detection, not the incubation-plus-detection
  mixture the current detection-window prior tries to span.
- Reporting is now *delayed*: the current model reports an
  instantaneous fraction `p_DRC · C(T)`, but the explicit-convolution
  model convolves with an onset-to-report delay, so very recent
  infections are not yet fully ascertained. This is more realistic
  and couples the reported-case stream to the same delay machinery as
  deaths.

The deaths and reports expectations are **nested** convolutions: the
inner `i_onset(s)` is itself a convolution, integrated again against
an outer delay CDF. This is the central cost of the architecture (see
Runtime).

## Mapping the four data streams plus TMRCA

| Stream | Expectation | Likelihood |
|---|---|---|
| Exported cases (Uganda) | `expected_exports_onset_staged` over the onset-to-detection window | Poisson |
| Suspected deaths (DRC) | `expected_deaths_onset_staged`, onset-to-death convolution | NegBinomial (shared `k`) |
| Reported cases (DRC) | `expected_reports_onset_staged`, onset-to-report convolution | NegBinomial (shared `k`) |
| Deaths among exports | reuse the current `expected_exports_deaths` machinery on the onset window (see Migration) | binned Poisson |
| Genetic TMRCA | soft lower bound on `T`, `censored(Normal(T, σ); upper = g)` (now in `genetic_seeding_bound_model`) | unchanged |

The growth, CFR, traveller-volume, dispersion and pooled-ascertainment
priors are unchanged from the current model and are injected into the
composer rather than re-defined.

## Modularity: every prior is its own submodel, every obs is injected

Design invariants (project-owner directives), all satisfied in this
implementation and enforced by tests in
`test_explicit_convolution_models.jl`:

1. **Every prior is a submodel.** No literal `Normal`/`Gamma`/etc.
   constants are buried in a model body. Each prior lives in its own
   `@model` submodel, the composer accepts it as an injectable
   keyword, and the only literal distributions are the submodel
   defaults (kwargs) — the same pattern as the current model's
   `delay_model`, `cfr_model`, `surveillance_dispersion_model`,
   `traveller_volume_model`, `detection_window_model`,
   `pooled_ascertainment_model`. The TMRCA genetic-seeding term is
   lifted into its own `genetic_seeding_bound_model` submodel; pass
   `genetic = nothing` to drop it.
2. **Every delay is specified via a prior**, including the
   onset-to-detection window (now in
   `onset_to_detection_window_model`). No delay distribution and no
   generation time is a fixed constant. The growth rate `r` is itself
   sampled via `τ` and `m`, so the timescale of spread is not a fixed
   input either.
3. **Every observation distribution is injected** (with a sensible
   default). `exports_onset_staged_obs`, `deaths_onset_staged_obs`
   and `cases_onset_staged_obs` each take an `obs` constructor
   keyword: a callable that returns the observation Distribution from
   the expected count (and shared `k` for NegBinomial). Defaults are
   Poisson for exports and the shared safe-NegBinomial for deaths and
   reports. Tests swap in Poisson everywhere to guard against a
   hardcoded distribution sneaking back.
4. **The onset-incidence curve is built once per draw**
   (`OnsetIncidence`) and threaded into the three observation
   submodels. A source-level test asserts the constructor appears
   exactly one place in `explicit_convolution_models.jl`.

The four time scales the observation expectations consume:

| Delay | Submodel | Prior | Status |
|---|---|---|---|
| Infection → onset (incubation) | `infection_to_onset_delay_model` | `α_inc ~ Normal⁺(11, 3)`, `θ_inc ~ Normal⁺(0.74, 0.25)` | new, prior from narrative range |
| Onset → death | `onset_to_death_delay_model` | `α ~ Normal⁺(4.3, 1.22)`, `θ ~ Normal⁺(2.6, 0.82)` | prior-based (bdbv-linelist reanalysis) |
| Onset → report | `onset_to_report_delay_model` | `α_otr ~ Normal⁺(4, 1.5)`, `θ_otr ~ Normal⁺(4.5, 1.5)` | new, prior with 30-day-cap caveat |
| Onset → detection (window) | `onset_to_detection_window_model` | `w ~ Normal⁺(7, 3)` | new submodel, was inline previously |

The growth timescale enters through the sampled `τ` and `m`, so the
rate of spread is not a fixed input either.
The regression test (`test_explicit_convolution_models.jl`, "every
delay parameter is sampled") asserts `α_inc, θ_inc, α, θ, α_otr,
θ_otr, w, τ, m` all appear as sampled variables, so a future fixed
delay cannot slip in unnoticed.

The forward-layer helpers (`expected_deaths_onset_staged`,
`expected_reports_onset_staged`, `expected_exports_onset_staged`,
`onset_incidence`) take the delay *distribution* as an argument and
hold no constants; the fixed `Gamma(...)` values in the unit tests
exist only to pin the deterministic integrals and never enter the
fitted model.

The incubation prior is constructed from the Imperial report's
narrative 6-11 day range across three PubMed studies (no Bayesian
posterior exists anywhere, and the vendored bdbv-linelist analysis
does not fit incubation since its line list has onset dates only).
`Gamma(α_inc, θ_inc)` then has a mean near 8 days, covering that
span; both parameters are expected to be prior-dominated (see
Identifiability).
The onset-to-report delay carries the same caveat as in issue #4: the
bdbv-linelist Isiro 2012 onset-to-notification estimate is 19.7 d
(13.7-30.1), but Charniga 2024 flags a 30-day-cap truncation bias, so
the prior is centred near 18 days and used with a strong caveat.

## Non-centred parameterisations

The composer reuses the existing `pooled_ascertainment_model`, which
already samples the ascertainment fractions in non-centred form
(`z_drc, z_uganda ~ Normal(0, 1)`, `logit p = μ + τ·z`); this avoids
the funnel geometry of the centred form that gave hundreds of
divergent transitions in the current code.
The implementation does not introduce any other hierarchical or
random-walk structure (deterministic exponential growth, no
time-varying `r`), so no further non-centred reparameterisation is
needed at present.
If the optional time-varying-`r` extension (sketched below) is ever
implemented, its log-growth random walk would follow the same
non-centred pattern: sample `z_t ~ Normal(0, 1)` and form
`log r_t = log r_0 + σ_w · cumsum(z_t)`.

## Daily-bin observation discretisation

The three observation submodels in this implementation
(`exports_onset_staged_obs`, `deaths_onset_staged_obs`,
`cases_onset_staged_obs`) all observe a **cumulative aggregate count**
at the cut-off, with no daily binning, so no
continuous-delay-to-daily-bin discretisation is required.
The current model's `exports_deaths_model` does observe a per-day
series and is the only place a double-censored daily-bin
discretisation would arise; that submodel has not yet been ported to
the explicit-convolution composer (it stays usable through the
current composer).
The cross-track verdict that `CensoredDistributions.jl` does not
differentiate under Mooncake therefore does not bite this design:
there is no discretisation step on the AD path.
When an explicit-convolution analogue of `exports_deaths_model` is
added, the same daily-bin construction the current model uses (an
inhomogeneous Poisson with bin means `Λ(t_d) − Λ(t_{d-1})` from the
cumulative intensity, see `exports_deaths_model` in `analysis.jl`)
carries over unchanged: it discretises the *cumulative intensity* by
finite differences, not the delay distribution itself, so it does not
need a double-censored delay primitive and inherits the same AD
safety as the rest of the layer.
If a future iteration ever needs an explicit double-censored daily
delay primitive and lacks AD support, the bias of any single-side
stopgap (rounding down, say) must be characterised honestly before
inclusion.

## Numerical implications and runtime

The smoke script (`scripts/smoke_explicit_convolution.jl`) measured
the nested-integral cost directly against the current
single-convolution helper, at `r = log(2)/14`, `T = 90`, on this
machine:

| Quantity | Cost |
|---|---|
| `OnsetIncidence` build (65-point grid) | ~460 µs/draw |
| Onset-staged deaths integral (reusing the tabulated onset curve) | ~4 µs/call |
| current single deaths convolution | ~6 µs/call |

The onset-incidence tabulation dominates: it pays the inner
incubation convolution once per grid node (65 clustered 64-node
convolutions).
Reusing the tabulated curve across all three observation integrals
keeps the per-draw observation cost near the current model; without
the precompute, the inner convolution would be paid at every outer
quadrature node of every stream (~`64 × 3` times per draw instead of
65).

Net per-draw forward cost is therefore roughly **one extra inner
convolution layer**, i.e. order ~0.5 ms/draw on top of the current
model's tens of µs. Against the current ~6 min full fit this points
to a **rough doubling of wall-clock**, dominated by the onset
tabulation rather than the outer integrals. The grid size
(`ONSET_GRID_POINTS = 65`) trades build cost against tail accuracy:
cumulative onsets are accurate to <0.1% at 65 points versus a
1025-point reference, with the largest pointwise error confined to
the exponential tail near `T` that the deaths and reports
convolutions weight.

The clustered onset convolution is exact against an adaptive QuadGK
reference (relative error ~1e-15 in the smoke check), so the only
approximation beyond the current model is the linear interpolation of
the tabulated onset curve.

## Mooncake-AD findings

The full conditioned joint model (16 sampled parameters: τ, m, the
two incubation params, the two onset-to-death params, the two
onset-to-report params, CFR, dispersion, the four ascertainment
params, the window) builds a Mooncake `LogDensityFunction`, returns
a finite log density, and a **finite Mooncake gradient** (‖grad‖ ≈ 24
at a prior draw).
A short NUTS smoke fit (50 warmup + 50 samples, single chain)
completes without error, including compilation, in under 30 s.

The AD-safety pattern from the current code carries over unchanged:
the deaths and reports convolutions differentiate through the
*density* alone, never the gamma-CDF shape-parameter derivative
(which Mooncake does not support). The implementation reuses the
existing `ExportDeathDelay` precomputed CDF holder for both the
onset-to-death and onset-to-report CDFs, extending its grid to cover
`[0, T]` for the cumulative streams.
The incubation convolution differentiates through the incubation
density only.

## Identifiability of the weakly-supported incubation params

The two incubation parameters `(α_inc, θ_inc)` are informed only
indirectly: no stream observes onsets directly, so the data constrain
the incubation lag mainly through the lag between infections and the
delayed reported-case and death signals.
With a handful of aggregate counts this is weak, so both are expected
to be prior-dominated, tracking the constructed 6-11 day prior
closely.
This is an honest limitation of the design: it adds two parameters
the data barely speak to, in exchange for a coherent latent state.
The onset-to-report parameters are similarly weak and inherit the
30-day-cap caveat.

## Drift from the McCabe et al. replication

The explicit-convolution model stays the closest of the redesign
options to McCabe et al.: it keeps deterministic exponential growth,
the same numerical-integral observation structure, and the same
priors for everything except the new incubation and onset-to-report
delays.
The drift is the inserted incubation layer (which McCabe et al. fold
into the window `w`) and the now-delayed reporting.
For the exports stream, separating incubation out of `w` shifts the
window's interpretation but not the published sensitivity range; the
deaths stream is essentially unchanged because onset-to-death already
clocked from onset.
The expected drift in the posterior `C(T)` (now reported as
cumulative infections `I_T`, with cumulative onsets `onsets_T`
slightly lower) is small relative to the renewal or compartmental
alternatives.

## Migration effort from current code

Low. The explicit-convolution layer reuses the existing
Gauss-Legendre `integrate` helpers, the `ExportDeathDelay`
precomputed-CDF holder, the `_delay_scale` clustering, and all the
unchanged building-block submodels.
The new code is two files added alongside the current model without
touching it:

- `src/explicit_convolution.jl` — the forward layer
  (`infection_incidence`, `onset_incidence`, `OnsetIncidence`,
  `expected_onsets_staged`, `expected_exports_onset_staged`,
  `expected_deaths_onset_staged`, `expected_reports_onset_staged`).
- `src/explicit_convolution_models.jl` — the Turing building-block and
  observation submodels and the `bvd_joint_explicit_convolution`
  composer.

The deaths-among-exports and export-detection-timing terms are not
yet ported in this iteration; they reuse the current
`expected_exports_deaths` / `expected_exports` machinery and would be
rewired to take the tabulated onset curve, a small follow-up.

## What this architecture does and does not buy

It **does**:

- resolve the onsets/infections inconsistency with a coherent latent
  state;
- make CFR unambiguously fraction-of-onsets;
- give the exports window a clean onset-to-detection meaning;
- add realistic reporting delay;
- stay close to the McCabe et al. replication and reuse almost all
  existing code and integrators.

It **does not**:

- replace deterministic growth with a stochastic incidence process
  (the trajectory is still `exp(r·t)`; only observation counts carry
  noise);
- add a mechanistic transmission model, a renewal equation, or
  compartments;
- add spatial structure;
- relax the single-zoonotic-seed assumption.

It also costs roughly a doubling of runtime and adds two
weakly-identified delay parameters.
If the goal is a genuinely different generative process (stochastic
transmission, `R_t`, saturation), this design is not it; the renewal
or compartmental options are.
The explicit-convolution architecture is the right incremental step
when the priority is internal consistency and defensible delay
semantics without abandoning the McCabe et al.-anchored structure.

## Optional time-varying growth extension

The design is not locked to constant `r`.
Because the whole forward layer is driven by the infection incidence
`i_inf(s)`, a time-varying growth is a drop-in: replace the scalar
`r` with a piecewise-constant or random-walk log-growth `r(s)` and
integrate `i_inf(s) = I'(s)` numerically, where
`I(s) = exp(∫_0^s r(u) du)`.
The onset convolution, the `OnsetIncidence` tabulation, and all three
observation integrals are unchanged because they only ever call
`i_onset(t)`.
With four aggregate counts a full random walk is not identifiable,
but a single breakpoint (two-segment piecewise `r`) is cheap to add
and would let the design express a slowdown since seeding.
Not implemented here to keep the focus on the core convolution
change; flagged so the architecture is not mistaken for being
constant-growth-only.
