# Stochastic latent infection process (issue #48)

This is one candidate architecture for the issue #48 redesign, developed on the `arch-stochastic` branch.
It is additive: nothing in the production model changes.

## Motivation

The baseline model represents early growth as a deterministic cumulative-incidence curve `C(s) = exp(r s)`, seeded by a single zoonotic case at `s = 0`.
Every observation likelihood builds its expected counts from that one smooth curve.
When cumulative cases are O(1–10) the real trajectory is dominated by demographic stochasticity: who infects whom, and exactly when the first case and first death occur.
A deterministic `exp(r s)` has zero variance there, so the model is over-confident about the early trajectory and about everything that leans on it: the elapsed time `T`, the timing of the first case and first death, and the small exported-case counts.

The concrete cost is that the earliest known onset (24 April 2026) is unusable under the baseline.
Under the deterministic single-seed curve the implied first case is detection-limited, and a single smooth curve cannot simultaneously place an onset that early and produce 131 deaths roughly 24 days later without an implausible growth rate.
So the date is discarded.
A stochastic latent process lets an early sporadic case sit above the smooth mean, so the onset can be old without forcing the entire curve to be old, and the date enters as an informative "at-or-before" bound on `T`.

## Architecture: log-Gaussian / linear-noise relaxation on log-incidence

I model the latent log-*incidence rate* as a continuous Gaussian random walk with exponential-growth drift, evaluated on a fixed grid of `n` knots over `[0, T]` with step `Δ = T / (n - 1)`:

```
log i(s_1)   = log r                              (seed: i(0) = r so the
                                                   σ=0 limit recovers
                                                   dC/ds = r·exp(r·s))
log i(s_j+1) = log i(s_j) + r·Δ + σ·√Δ·z_j,   z_j ~ Normal(0, 1),
```

and obtain cumulative infections by trapezoid integration on the knot grid, `C(s) = ∫_0^s i(u) du`.
Working on log-incidence (not log-cumulative) keeps `i(s) > 0` everywhere, so `C(s)` is monotone by construction — the property we want of a cumulative-infection process.
The drift `r·Δ` reproduces deterministic exponential growth in expectation; the diffusion term `σ·√Δ·z_j` injects the early-phase variance the deterministic curve lacks.

Why this process and not a branching process or exact negative-binomial renewal:

- A branching process or stochastic renewal with overdispersed offspring has *discrete* latent counts.
NUTS cannot differentiate through discrete latent state, so those formulations force either marginalisation (tractable only in special cases) or a particle/SMC sampler.
- The log-Gaussian relaxation is the linear-noise approximation (LNA) of those processes in continuous form: it keeps the early-phase variance but replaces the discrete counts with a continuous Gaussian, so the whole model stays differentiable and NUTS + Mooncake samples it directly.
- It nests the baseline exactly: `σ = 0` recovers `i(s) = r · exp(r s)` and hence `C(s) = exp(r s) - 1`, so the architecture is a strict generalisation and the existing results are the `σ → 0` limit.

The default uses a constant per-step log-variance `σ²Δ`.
A fuller LNA would scale the per-step variance by `1/C(s)` so demographic noise is loudest at O(1–10) counts and tightens as the outbreak grows.
That is the more faithful approximation and the natural next step; it complicates the gradient and is not needed to demonstrate the inference route, so it is left as aspiration.

The non-centred parameterisation (sample standard-Normal `z`, scale by `σ` inside the model) avoids the funnel between `σ` and the increments, the same trick the baseline already uses for the pooled ascertainment offsets.

Prior on the noise scale: `σ ~ Normal⁺(0, 0.3)`, weakly informative and centred at the deterministic limit, so the data must argue for early-phase variance.

## Modularity

Every prior — for `τ`, `m`, `σ`, the latent increments, the incubation delay, the onset-timing SD, the genetic TMRCA bound, the onset-to-death delay, CFR, the detection window, traveller volume, surveillance dispersion, ascertainment — lives in its own `@model` submodel.
The joint composer takes them as keyword arguments with sensible defaults, so any one can be swapped without editing model bodies.
No `Normal(...) / Gamma(...) / Beta(...)` literals sit inline in the composer or in `stochastic_growth_model`.

Observation distributions are likewise injected.
The composer takes `exports_dist`, `deaths_dist`, `cases_dist` callables and uses them via `obs ~ exports_dist(μ, k)` etc., with the package's `Poisson` / safe-`NegativeBinomial` choices as defaults.

## Onset staging

Observations stage as **infections → onset → death / report / detection**.
Infections are never convolved directly with the downstream delays:

1. The latent LNA process produces infection incidence `i(s)` (positive by construction).
2. The infection cumulative `C(s) = ∫_0^s i(u) du` is built once by trapezoid on the knot grid.
3. The **onset cumulative** is the single intermediate quantity, computed once via `onset_cumulative` and reused by every downstream stream:
```
C_o(t) = ∫_0^t i(s) · F_inc(t - s) ds
```
where `F_inc` is the incubation CDF (sampled).
4. Downstream streams condition on the onset stage:
   - DRC deaths: `μ_deaths(T) = CFR · ∫_0^T i_o(u) · f_d(T - u) du`, convolving onset incidence `i_o(t)` against the onset-to-death density.
   - DRC reported cases: `p_DRC · C_o(T)`.
   - Uganda exports: `p_Uganda · q · ∫_{T-w}^{T} C_o(s) ds`.
   - Earliest-onset bound: `T ≥ onset_delta` via a censored Normal on `T`.

## Every delay is prior-based

Per project rule, no observation submodel uses a fixed delay distribution or a fixed generation time.
Every delay samples its parameters from a prior and carries that uncertainty into the fit:

- Incubation (infection-to-onset): `α_inc ~ Normal⁺(3, 1)`, `θ_inc ~ Normal⁺(3, 1.5)` (Gamma).
- Onset-to-death: `α ~ Normal⁺(4.3, 1.22)`, `θ ~ Normal⁺(2.6, 0.82)`.
- Detection window `w` (onset-to-detection at the border): `w ~ Normal⁺(15, 5)` days.
- Earliest-onset timing SD: `σ_o ~ Normal⁺(14, 5)` days.
- No generation interval / renewal kernel is hardcoded: growth is driven by the sampled rate `r` (from the doubling-time prior) plus the latent stochastic increments.
- Onset-to-report: not explicitly modelled here; DRC reported cases are a thinning `p_DRC · C_o(T)` (ascertainment, not a delay).
If a reporting delay is added later it must likewise be sampled.

## Censoring

Only single-side `censored(Normal(T, σ); upper = g)` is used (earliest-onset bound, genetic TMRCA bound).
The two-sided `CensoredDistributions` form does not differentiate under Mooncake; the one-sided form reduces to a `logcdf` call on the underlying `Normal`, which Mooncake handles.
The honest cost is that the bound only uses the upper-tail probability `P(T ≥ g)`.
An interval bound (e.g. censoring also from below if a known absence-of-cases date were available) would need a manually-coded logcdf difference rather than `censored`.

## How it unlocks the earliest onset date

The earliest onset is the first confirmed symptom-onset date.
It is an "at-or-before" statement: the outbreak must be at least `onset_delta` days old (`onset_delta` = days from that onset to the cut-off), because a case had already shown symptoms by then.
I model it as a soft one-sided bound on `T`, reusing the package's existing genetic-TMRCA construction: observe `onset_delta` at the upper censoring point of `Normal(T, σ_o)`, contributing

```
p_onset(T) = P[Normal(T, σ_o) ≥ onset_delta] = Φ((T − onset_delta) / σ_o).
```

`σ_o` is the SD on the location of the bound.
It absorbs the infection-to-onset (incubation) delay and onset-date recording uncertainty, so it is itself a delay quantity: sampled, not fixed.
Passing `onset_delta = missing` makes the term a no-op.

The reason this date is usable under the stochastic process but not the deterministic one is the early-phase variance: the bound only requires that some case had onset by `onset_delta`, which the stochastic trajectory can satisfy via an early fluctuation above the mean, whereas the deterministic curve would have to shift its whole mean trajectory (and hence `r` or `T`) to honour the same date, conflicting with the death count.

The date itself belongs in `data/observations.toml` as a new `first_onset_date` block, loaded into an `onset_delta` offset exactly like `first_export_detection_date` already is.
The current demonstration script uses a placeholder offset pending that data entry.

## Inference strategy and the make-or-break verdict

The hard question for issue #48 is whether latent stochastic trajectories are tractable in Turing with Mooncake/NUTS.
The options:

1. Continuous (log-Gaussian/LNA) relaxation, sampled with NUTS + Mooncake.
The latent state is `n − 1` continuous standard-Normal increments, so the model is fully differentiable.
This is the route here.
2. Exact discrete process (branching / NB renewal) with marginalisation.
Tractable only for special structures; in general the marginal likelihood of aggregate counts under a stochastic process has no closed form here.
3. Exact discrete process with particle Gibbs / SMC (Turing supports `PG`, `SMC`).
This handles discrete latent counts but is far slower, mixes poorly for the continuous nuisance parameters (`τ`, `m`, `CFR`, delays, dispersion, ascertainment), and would need careful tuning of particle counts.

Verdict: the continuous relaxation (option 1) is tractable.
The demonstration script shows it end-to-end:

- The full joint model compiles and prior-samples to finite, positive `C_T`.
- Mooncake returns a finite gradient of the full log-density through the latent trajectory, the trapezoid cumulative, the staged onset convolutions, and every downstream integral.
- A short NUTS run samples without error and gives finite posterior `C_T` and `T`.

A note on Mooncake support: `Distributions.cdf(::Gamma, x)` differentiated in the Gamma shape does not work under Mooncake, so every CDF used inside the staged pipeline is computed as the inner integral of the density (the same workaround the package's deaths-among-exports likelihood uses).

The exact discrete process via SMC (option 3) is a longer-term bet, not tractable in the time available and likely much slower; it is not recommended as the next step.
The continuous relaxation captures the early-phase variance issue #48 actually cares about while staying inside the existing NUTS + Mooncake machinery.

## Expected runtime versus the current ~6 min

The latent dimension grows from ~10 to ~10 + (n − 1) for the trajectory plus the new submodel parameters (incubation `α_inc`, `θ_inc`; onset-timing `σ_o`; process noise `σ`).
At 24 knots and the staged pipeline the joint model sits around 39 dimensions.
Each leapfrog step also evaluates the cumulative integral, the staged onset convolutions and the export and death integrals over a piecewise trajectory, a modest constant-factor cost.
Realistic expectation: roughly 3–5× the current ~6 min full fit (so ~20–30 min), dominated by the larger gradient, the longer NUTS trajectories the higher dimension needs, and the nested onset-to-death integrals.
Knot count trades resolution against cost and can be tuned down if 24 is more than the early phase needs.

## Identifiability risks

- `σ` versus the count noise.
With only aggregate counts, the process noise `σ` and the surveillance dispersion `k` both inflate the spread of the observed totals and are only weakly separated.
The prior on `σ` carries the inference; the streams cannot pin it precisely.
This is acceptable for the stated goal (propagating early-phase uncertainty) but `σ` should be read as prior-driven.
- Heavy-tailed `C_T`.
The compounding log-Gaussian variance gives `C_T` a heavy right tail under the prior.
The likelihoods pull this back, but the prior on `σ` and the upper bound on `m` matter more than in the deterministic model and need a prior-predictive check.
- The mean-incidence approximation.
The deaths convolution now uses the staged *onset* incidence rather than the exponential mean, so the early-phase variance is propagated into the death stream.
The constant log-variance LNA is still an approximation to a fuller `1/C`-scaled LNA; that refinement is the natural next step.
- Knot count and grid.
Too few knots under-resolve the early phase; too many add cost and near-redundant dimensions.
24 is a starting guess, not tuned.

## Migration effort

Low to moderate, and incremental:

- Done: `stochastic_growth_model`, `lna_trajectory`, `lna_incidence`, `lna_logi`, `onset_cumulative`, `onset_incidence`, `incubation_model`, `onset_timing_model`, `tmrca_timing_model`, plus the prior sub-submodels (`doubling_time_model`, `multiplier_model`, `process_noise_model`, `latent_increments_model`, `onset_sd_model`) in `src/stochastic_growth.jl`, exported and tested; a joint-composer demonstration in `scripts/prototype_stochastic.jl`.
- Next: add `first_onset_date` to `data/observations.toml` and an `onset_delta` field to `load_observations`, mirroring `first_export_detection_date`.
- Next: once the observation `@model` blocks move into the package (issue #81), the demonstration composer collapses to a thin variant of `bvd_joint` that swaps the growth submodel and adds the onset-staging pipeline.
- Aspiration: the `1/C` LNA variance scaling and an interval-censored earliest-onset bound (a manually coded logcdf difference, since Mooncake does not differentiate the two-sided `censored` form).
- Throughout: prior-predictive checks on `σ` and `C_T`, and a runtime/identifiability comparison against the deterministic baseline before this replaces it.

The deterministic model stays the default until those checks pass; this architecture is a strict generalisation, so the comparison is the `σ → 0` limit.

## Files

- `src/stochastic_growth.jl` — building-block submodels and trajectory helpers (package code, importable, tested).
- `test/test_stochastic_growth.jl` — unit tests for the trajectory builder, the growth-submodel prior draws, the onset-staging helpers, and the onset bound.
- `scripts/prototype_stochastic.jl` — joint-composer demonstration with prior-predictive, gradient and NUTS smoke tests.
- `docs/src/proposals/stochastic-latent.md` — this proposal.
