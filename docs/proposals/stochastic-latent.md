# Proposal: stochastic latent infection process (issue #48)

Status: prototype.
This is one candidate architecture for the issue #48 redesign, prototyped on the `arch-stochastic` branch.
It is additive and changes no existing behaviour.

## Motivation

The baseline model represents early growth as a deterministic cumulative-incidence curve `C(s) = exp(r s)`, seeded by a single zoonotic case at `s = 0`.
Every observation likelihood builds its expected counts from that one smooth curve.
When cumulative cases are O(1–10) the real trajectory is dominated by demographic stochasticity: who infects whom, and exactly when the first case and first death occur.
A deterministic `exp(r s)` has zero variance there, so the model is over-confident about the early trajectory and about everything that leans on it: the elapsed time `T`, the timing of the first case and first death, and the small exported-case counts.

The concrete cost is that the earliest known onset (24 April 2026) is currently unusable.
Under the deterministic single-seed curve the implied first case is detection-limited, and a single smooth curve cannot simultaneously place an onset that early and produce 131 deaths roughly 24 days later without an implausible growth rate.
So the date is discarded.
A stochastic latent process lets an early sporadic case sit above the smooth mean, so the onset can be old without forcing the entire curve to be old, and the date enters as an informative "at-or-before" bound on `T`.

## Chosen process: log-Gaussian / linear-noise relaxation

I model the latent log-cumulative incidence as a continuous Gaussian random walk with exponential-growth drift, evaluated on a fixed grid of `n` knots over `[0, T]` with step `Δ = T / (n - 1)`:

```
log C(s_1)   = 0                                  (single seed, C(0) = 1)
log C(s_j+1) = log C(s_j) + r·Δ + σ·√Δ·z_j,   z_j ~ Normal(0, 1).
```

The drift `r·Δ` reproduces deterministic exponential growth in expectation.
The diffusion term `σ·√Δ·z_j` injects the early-phase variance the deterministic curve lacks.
The continuous-time trajectory `C(s)` is the exponential of the piecewise-linear interpolant of these log-levels, so it is positive everywhere and differentiable except at the knots.

Why this process and not a branching process or an exact negative-binomial renewal:

- A branching process or stochastic renewal with overdispersed offspring has *discrete* latent counts.
NUTS cannot differentiate through discrete latent state, so those formulations force either marginalisation (tractable only in special cases) or a particle/SMC sampler (see the inference verdict below).
- The log-Gaussian relaxation is the linear-noise approximation (LNA) of those processes in continuous form: it keeps the early-phase variance but replaces the discrete counts with a continuous Gaussian, so the whole model stays differentiable and NUTS + Mooncake samples it directly.
- It nests the baseline exactly: `σ = 0` recovers `C(s) = exp(r s)`, so the prototype is a strict generalisation and the existing results are the `σ → 0` limit.

The prototype uses a constant per-step log-variance `σ²Δ`.
A fuller LNA would scale the per-step variance by `1/C(s)`, so demographic noise is loudest at O(1–10) counts and tightens as the outbreak grows.
That is the more faithful approximation and is the natural next step; it is left as aspiration here because it complicates the gradient and is not needed to demonstrate the inference route.

The non-centred parameterisation (sample standard-Normal `z`, scale by `σ` inside the model) avoids the funnel between `σ` and the increments, the same trick the baseline already uses for the pooled ascertainment offsets.

Prior on the noise scale: `σ ~ Normal⁺(0, 0.3)`, weakly informative and centred at the deterministic limit, so the data must argue for early-phase variance.

## How it unlocks the earliest onset date

The earliest onset is the first confirmed symptom-onset date.
It is an "at-or-before" statement: the outbreak must be at least `onset_delta` days old (`onset_delta` = days from that onset to the cut-off), because a case had already shown symptoms by then.
I model it as a soft one-sided bound on `T`, reusing exactly the construction the package already uses for the genetic TMRCA (`genetic_seeding_model`): observe `onset_delta` at the upper censoring point of `Normal(T, σ_o)`, contributing

```
p_onset(T) = P[Normal(T, σ_o) ≥ onset_delta] = Φ((T − onset_delta) / σ_o).
```

`σ_o` is the SD on the location of the bound.
It absorbs the infection-to-onset (incubation) delay and onset-date recording uncertainty, so it is itself a delay quantity.
Per the project rule that every delay carries prior uncertainty into the fit, `σ_o` is *sampled*, not fixed: `σ_o ~ Normal⁺(14, 5)` days, a weakly-informative prior spanning plausible BVD incubation-plus-recording spreads where direct data is thin.
Passing `onset_delta = missing` makes the term a no-op (and skips sampling `σ_o`), so it is safe to add unconditionally.

This is implemented as `onset_timing_model` in `src/stochastic_growth.jl` and exercised in the prototype joint composer.
The reason it is *usable* under the stochastic process but not the deterministic one is the early-phase variance: the bound only requires that *some* case had onset by `onset_delta`, which the stochastic trajectory can satisfy via an early fluctuation above the mean, whereas the deterministic curve would have to shift its whole mean trajectory (and hence `r` or `T`) to honour the same date, conflicting with the death count.

The date itself belongs in `data/observations.toml` as a new `first_onset_date` block, loaded into an `onset_delta` offset exactly like `first_export_detection_date` already is.
The prototype script uses a placeholder offset pending that data entry.

## How each data stream maps in

The stochastic growth submodel exposes the same `growth_state` interface as the deterministic one: the NamedTuple `(; τ, r, m, T, C_T, cumulative)`, where `cumulative` is the random trajectory `C(s)` and `C_T = C(T)` is read off its endpoint.
Because every observation submodel only reads `cumulative`, `C_T`, `r` and `T`, all four streams map in unchanged:

- Exports (Method 1): `expected_exports` integrates the at-risk person-time `∫_{T−w}^{T} C(s) ds` over the stochastic trajectory directly. The detection window `w` (incubation + onset-to-detection) is a delay, so it is sampled from a prior (`w ~ Normal⁺(15, 5)` days), not fixed.
- DRC reported cases: ascertained endpoint `p_DRC · C(T)`.
- DRC deaths (Method 2): the onset-to-death convolution `CFR · ∫_0^T C(s) f(T−s) ds`.
The package `expected_deaths` currently assumes `exp(r s)` internally; the prototype passes the matching `r` so the mean drift agrees.
A faithful version would convolve against the stochastic incidence `i(s) = C'(s)` rather than the exponential mean (aspiration; see risks).
- Deaths among exports: the binned-Poisson timing likelihood weights the same person-time by the onset-to-death CDF, again on `cumulative`.
- Genetic TMRCA: the censored lower bound on `T` is unchanged.
- Earliest onset: the new `onset_timing_model` bound, above.

So the stochastic process drops in at exactly one seam and the rest of the model is reused, not rewritten.

### Every delay is prior-based

Per the project rule, no observation submodel uses a fixed delay distribution or a fixed generation time; every delay samples its parameters from a prior and carries that uncertainty into the fit.

- Onset-to-death (deaths, deaths-among-exports): the Gamma shape and scale are sampled (`α ~ Normal⁺(4.3, 1.22)`, `θ ~ Normal⁺(2.6, 0.82)`), as in the baseline `delay_model`.
- Detection window `w` (incubation + onset-to-detection, exports and deaths-among-exports): sampled, `w ~ Normal⁺(15, 5)` days, as in the baseline `detection_window_model`.
- Infection-to-onset (incubation) timing on the earliest-onset bound: enters through `σ_o`, which is sampled from `Normal⁺(14, 5)` days rather than fixed.
- No generation-interval / renewal kernel is hardcoded: growth is driven by the sampled rate `r` (from the doubling-time prior) plus the latent stochastic increments, so there is no fixed generation time to specify.
- Onset-to-report: the prototype does not model an explicit reporting delay (DRC reported cases are an ascertained fraction `p_DRC · C(T)`, a thinning not a delay), so there is no fixed reporting-delay constant. If a reporting delay is added later it must likewise be sampled.

## Inference strategy and the make-or-break verdict

The hard question for issue #48 is whether latent stochastic trajectories are tractable in Turing with Mooncake/NUTS.
The honest options:

1. Continuous (log-Gaussian/LNA) relaxation, sampled with NUTS + Mooncake.
The latent state is `n − 1` continuous standard-Normal increments, so the model is fully differentiable.
This is the route prototyped here.
2. Exact discrete process (branching / NB renewal) with marginalisation.
Tractable only for special structures; in general the marginal likelihood of aggregate counts under a stochastic process has no closed form here.
3. Exact discrete process with particle Gibbs / SMC (Turing supports `PG`, `SMC`).
This handles discrete latent counts but is far slower, mixes poorly for the continuous nuisance parameters (`τ`, `m`, `CFR`, delay, dispersion, ascertainment), and would need careful tuning of particle counts.

Verdict: the continuous relaxation (option 1) is tractable now.
The prototype demonstrates it end-to-end:

- The full joint model (37 latent dimensions, including 23 trajectory increments and the sampled detection window, traveller volume and onset-timing SD) compiles.
- A prior-predictive draw gives finite, positive `C_T` with a sensible median (~100).
- Mooncake returns a finite value and a finite gradient of the full 37-dimensional log-density, so NUTS has a usable gradient through the latent trajectory, the interpolation, and every downstream integral.
- A short NUTS run samples without error and gives finite posterior `C_T` and a posterior `T` median around 130 days.

The exact discrete process via SMC (option 3) is a longer-term bet, not tractable in the time available and likely much slower; it is not recommended as the next step.
The continuous relaxation captures the early-phase variance that issue #48 actually cares about while staying inside the existing NUTS + Mooncake machinery.

## Expected runtime versus the current ~6 min

The latent dimension grows from ~10 to ~10 + (n − 1).
At the prototype's 24 knots that is +23 dimensions.
Each leapfrog step also evaluates the export and death integrals over a piecewise-linear trajectory (a closure lookup per quadrature node) rather than a single `exp`, a modest constant-factor cost.
Realistic expectation: roughly 2–4× the current ~6 min full fit (so ~15–25 min), dominated by the larger gradient and the longer NUTS trajectories the higher dimension needs.
Knot count trades resolution against cost and can be tuned down if 24 is more than the early phase needs.

## Identifiability risks

- `σ` versus the count noise.
With only aggregate counts, the process noise `σ` and the surveillance dispersion `k` both inflate the spread of the observed totals and are only weakly separated.
The prior on `σ` carries the inference; the streams cannot pin it precisely.
This is acceptable for the stated goal (propagating early-phase uncertainty) but `σ` should be read as prior-driven.
- Heavy-tailed `C_T`.
The compounding log-Gaussian variance gives `C_T` a heavy right tail under the prior (the prototype prior reached `C_T` ~ 10⁸ in the extreme draws before any data).
The likelihoods pull this back, but the prior on `σ` and the upper bound on `m` matter more than in the deterministic model and need a prior-predictive check.
- The mean-incidence approximation in the deaths convolution.
Until the death and export integrals convolve against the *realised* stochastic incidence rather than the exponential mean, the early-phase variance is only partly propagated into the death stream.
This is the main fidelity gap between the prototype and the aspiration.
- Knot count and grid.
Too few knots under-resolve the early phase; too many add cost and near-redundant dimensions.
24 is a starting guess, not tuned.

## Migration effort

Low to moderate, and incremental:

- Done (prototype): `stochastic_growth_model`, `lna_trajectory`, `lna_logC`, `onset_timing_model` in `src/stochastic_growth.jl`, exported and tested; a joint-composer demonstration in `scripts/prototype_stochastic.jl`.
- Next: add `first_onset_date` to `data/observations.toml` and an `onset_delta` field to `load_observations`, mirroring `first_export_detection_date`.
- Next: once the observation `@model` blocks move into the package (issue #81), the prototype composer collapses to a thin variant of `bvd_joint` that swaps the growth submodel and adds the onset term, with no stand-in likelihoods.
- Aspiration: the `1/C` LNA variance scaling and the stochastic-incidence death/export convolutions.
- Throughout: prior-predictive checks on `σ` and `C_T`, and a runtime/identifiability comparison against the deterministic baseline before this replaces it.

The deterministic model stays the default until those checks pass; the stochastic process is a strict generalisation, so the comparison is the `σ → 0` limit.

## Files

- `src/stochastic_growth.jl` — the new building-block submodels and trajectory helpers (package code, importable, tested).
- `test/test_stochastic_growth.jl` — unit tests for the trajectory builder, the growth-submodel prior draws, and the onset bound.
- `scripts/prototype_stochastic.jl` — the joint-composer demonstration with prior-predictive, gradient and NUTS smoke tests.
- `docs/proposals/stochastic-latent.md` — this proposal.
