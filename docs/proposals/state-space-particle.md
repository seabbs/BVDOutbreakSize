# Proposal: state-space stochastic infection process with particle-filter inference

Status: prototype on the `arch-state-space` branch.
Additive to the existing model; the deterministic exponential-growth baseline is unaffected.

## Motivation and contrast with the sibling prototypes

The baseline model represents the latent infection process as a deterministic continuous-time curve `C(s) = exp(r·s)`.
Every observation likelihood is built from that one smooth trajectory.
At the start of the outbreak the population of infected people is O(1-10) and the real trajectory is dominated by demographic stochasticity: who infects whom, and exactly when each early case occurs.
A deterministic curve has zero variance there.

Two sibling prototypes already address this gap.

- The renewal prototype (`arch-renewal`, PR #103) replaces `exp(r·s)` with a deterministic discrete-time renewal `I_t = R_t · sum(I_{t-s} · g_s)` driven by a weekly random-walk `R_t`.
The latent infection process is still deterministic given `R_t`; only `R_t` carries the noise.
- The stochastic-latent prototype (`arch-stochastic`, PR #101) adds Gaussian process noise to `log C(s)`, i.e. a log-Gaussian / linear-noise approximation (LNA) of the underlying stochastic infection process.
This keeps the early-phase variance but represents it as a continuous Gaussian relaxation, so the whole model stays differentiable and NUTS + Mooncake samples it directly.

This proposal takes the next step honestly.
The exact discrete-time stochastic infection process is a branching process with overdispersed offspring: each infected case generates a random integer number of secondaries drawn from a negative binomial, and the secondaries' infection times are drawn from the generation-interval PMF.
That process has *discrete* latent counts, so NUTS cannot differentiate through it.
The Gaussian relaxation in `arch-stochastic` is one way out; the other is to write the model as an explicit state-space model and infer the latent trajectory by particle methods (particle filtering / particle Gibbs).
This prototype is that route.

Comparison axis the owner asked for: when does the exact discrete latent pay off in inference about the early phase, versus the Gaussian relaxation?
The answer is honest and largely negative for this outbreak; see the verdict.

## Generative model

### Latent state

A daily grid `t = 1, …, n`, where `n` is the data cut-off (so day `n` is the as-of date).
Day 1 represents the seeding event, with a small seed `I_1`.
Conditional on the per-day reproduction number `R_t` (continuous, sampled by a weekly random walk) and a fixed offspring overdispersion `φ`, the *force of infection* on day `t` is

```
λ_t = R_t · sum_{s ≥ 1} I_{t-s} · g_s
```

with `g` the generation-interval PMF.
The latent infection count on day `t` is

```
I_t | I_{<t}, R_t  ~  NegativeBinomial(mean = λ_t, dispersion = φ),
```

with `Var(I_t) = λ_t + λ_t² / φ`.
This is the exact discrete branching-process step at the population level: the negative binomial is the marginal for sum of NB-offspring across infectors when each infector's offspring distribution is NB.
It collapses to a Poisson branching process as `φ → ∞`.
A Poisson alternative (`φ = ∞`) is left selectable by prior on `φ` (a wide half-Normal with substantial mass at large values), so the Poisson nesting is recovered.

`I_1` is the *seed size* with a small-integer prior, allowing for a small primary cluster rather than a hard single zoonotic case.
The default prior is `I_1 ~ 1 + Poisson(0.5)` (median 1, occasional 2-3), so the single-seed baseline is the prior mode.

### Observation staging — same as the conv-v2 / renewal route

The observation submodels never touch the daily infections directly.
Infections are first convolved with the incubation PMF to give the onset incidence `O_t`, and onsets are convolved with the relevant onset-to-X delay PMF to give the daily count of the observed event timed at observation:

```
O_t                 = sum_{d ≥ 0} I_{t-d}    · h_incub[d]
deaths_t            = CFR · sum_{d ≥ 0} O_{t-d} · h_death[d]
reports_t           = p_DRC · sum_{d ≥ 0} O_{t-d} · h_report[d]
exports_t           = p_Uganda · q · sum_{d ≥ 0} O_{t-d} · h_detect[d]
export_deaths_t     = CFR · p_Uganda · q · sum_{d ≥ 0} O_{t-d} · h_death[d]
```

This is the same onset-staging the discrete-renewal prototype uses; it makes detection and death consistently timed from onset.

### Observation likelihoods

For the prototype, observations condition on the available *cumulative* totals through to the data cut-off, exactly as the baseline model does:

```
exported_cases  ~ Poisson(sum(exports_t))
total_deaths    ~ NegBinomial(sum(deaths_t), k)
reported_cases  ~ NegBinomial(sum(reports_t), k)
exports_deaths  ~ Poisson(sum(export_deaths_t))
```

Where time-resolved data are available — for example the dated export deaths the package already loads, or the sitrep vintage trajectory of PR #107 — the same `deaths_t` / `export_deaths_t` series plugs in as the intensity of a binned-Poisson likelihood (matching the existing `exports_deaths_model`).
The prototype demonstrates the cumulative version end-to-end; the daily version is a drop-in replacement on the same vectors.

### Genetic TMRCA

The soft lower bound on the outbreak age `T = n` is added as a censored upper-tail term, identical to the baseline `genetic_seeding_model`:

```
log p_gen(T) = log Φ((n - tmrca_days) / tmrca_days_sd).
```

Because `n` is fixed in this prototype (set by the daily-grid length), the bound is effectively a one-sided likelihood on `n`.
A version that samples `n` (the outbreak age) is possible but materially complicates the particle filter (the latent grid length itself becomes a random dimension).

## Design rules — applied across the prototype

Mirroring the main model and the renewal prototype:

- Every prior is a submodel, passed in.
The composer accepts `gi`, `incubation`, `onset_to_death`, `onset_to_report`, `onset_to_detection`, `cfr`, `dispersion`, `ascertainment`, `traveller`, `window`, `genetic`, `seed`, `rt`, and `offspring` building-block submodels as keyword arguments with sensible defaults.
- All delays carry uncertainty.
Generation interval, incubation, onset-to-death, onset-to-report and onset-to-detection are each sampled with priors on their distribution parameters, so no fixed delay constants are buried in the model body.
The generation interval defaults to a Gamma with `α, θ` priors mirroring the onset-to-death delay shape.
- Observation distributions are injected.
The Poisson and NegBinomial likelihoods come from a small per-stream observation-error submodel where the dispersion / link is parameterised; this keeps the dispersion `k` swappable.
- No inline `using` / `import` in `src/state_space.jl`.
All imports stay on the module page; the file relies on names already imported by `BVDOutbreakSize.jl`.
- Onset staging once, reused: `O_t` is computed once and convolved separately for each downstream stream.
- Non-centred parameterisation for the weekly random-walk knots and the pooled ascertainment offsets, matching the baseline pattern.
- Delay discretisation by hand-rolled trapezoidal integration over daily bins with the density at zero forced to zero, the same AD-safe trick the renewal prototype uses (CensoredDistributions' analytical primary-censored CDF is not Mooncake-differentiable; see the renewal track's verdict).
This is a single-interval bin rather than a full double-interval-censored PMF; the bias from the missing second censoring is small (a few percent on the central mass) and is documented honestly here rather than chased into AD-unfriendly machinery.

## Inference plan

The latent state is the daily infection trajectory `I_{1:n}`, a vector of non-negative integers.
NUTS cannot sample it.
The continuous nuisance parameters (the weekly log-`R_t` knots, the random-walk SD, `CFR`, the delay shape/scale parameters, the ascertainment hyperparameters and offsets, the dispersion, the traveller volume, the detection window, the seed size's continuous embedding when used) are differentiable.

The intended sampler is therefore a Turing `Gibbs(:I => PG(N), :nuisances => NUTS(adtype))` block:

- `PG(N)` (particle Gibbs) updates the latent integer trajectory by running a conditional SMC sweep with `N` particles, conditional on the current nuisance values.
- `NUTS` updates the continuous nuisances by Hamiltonian Monte Carlo with Mooncake gradients, conditional on the latent trajectory.

In Turing 0.45 the cleanest way to expose the latent counts to PG is to declare them inside the `@model` as

```julia
for t in 2:n
    I[t] ~ NegativeBinomial(mean = λ_t, dispersion = φ)
end
```

i.e. via the `~` distribution, so PG can step them.
PG then resamples ancestor particles at each `~` step.

### Realistic feasibility (the verdict)

The prototype implements the full generative model and demonstrates:

- It compiles end-to-end as a Turing `@model`.
- A prior-predictive draw gives a finite, integer-valued `I_{1:n}` trajectory and finite expected counts for every stream.
- A short particle-Gibbs smoke test runs to completion (a handful of Gibbs sweeps with a small particle count) and records finite log-likelihoods, demonstrating the inference route is mechanically operational.
- A full posterior fit is not run here; the per-iteration cost dominates.

The cost calculation is honest.
PG's per-Gibbs-sweep cost scales as `O(N · n)` for a `n`-day grid with `N` particles, and each "particle step" is a likelihood evaluation across all streams' daily contributions.
For `n ≈ 130 days` and `N = 200` particles (a defensible minimum for a moderately overdispersed daily process before the particle population collapses), one Gibbs sweep is roughly `n · N ≈ 26000` forward-model evaluations of the daily kernel, against a single forward-model evaluation per NUTS leapfrog in the LNA prototype.
A NUTS step typically does 10-200 leapfrog evaluations, so one PG-NUTS sweep costs roughly `10-100×` a single NUTS step in the LNA prototype.
Translated to wall clock, expect:

- LNA prototype full fit: ~15-25 min (per its own measurements).
- PG + NUTS full fit on the same problem: ~5-25 h for 1000 post-warmup samples per chain, depending on particle count and acceptance rates.
This is a back-of-envelope figure, not a measurement, and is honest about being a rough order of magnitude.

Particle Gibbs is also known to mix poorly when the latent path is long and the observation noise is small relative to the process noise — exactly the regime here where four aggregate counts pin a 130-day daily trajectory with high latent overdispersion.
Practically this means the path-mixing rate of PG (the fraction of latent days that change between consecutive sweeps) tends to be low at the beginning of the path, where there is the least observational anchor.
That is precisely the early phase the architecture is meant to learn about, so the asymptotic-mixing argument cuts the wrong way.

### Why not pure SMC / particle marginal Metropolis-Hastings?

A particle marginal MH would marginalise the latent path and step the nuisances on the SMC marginal likelihood.
This is a viable alternative but trades latent-path mixing for nuisance-step acceptance, and the same per-sweep cost applies.
We do not implement it here because the diagnostic and tuning machinery is less mature in Turing.

## Comparison to the LNA prototype (`arch-stochastic`, #101)

What the exact discrete process buys, in principle:

- Early-phase non-Gaussianity.
Branching processes have heavy tails: in any one realisation a small early cluster of three to five secondaries can occur and persist as a fluctuation above the mean.
The Gaussian LNA squashes that into a log-normal tail, which slightly under-represents the chance of an early cluster.
For inference about the *minimum* outbreak age that is consistent with the data, the exact process gives marginally more posterior mass to "old outbreak with a slow early phase that happened to fluctuate up early".
- Integer realism at small counts.
The LNA's continuous trajectory passes through fractional infection counts (e.g. `I_3 = 1.7`) that have no integer interpretation, while the state-space model produces integer trajectories the whole way.
For observation likelihoods that condition on aggregate counts this is irrelevant; for inferences that read off the exact day-of-first-case it is not.

What it does not buy, for this outbreak:

- Cumulative-count likelihoods.
All four primary streams are aggregate cumulatives.
Sums smooth over the discrete jumps of the latent path almost completely.
By the central limit theorem the cumulative is approximately Gaussian regardless of whether the daily steps are exact NB or LNA, so the Gaussian relaxation gives essentially the same posterior on `C(T)`.
- The earliest-onset bound.
The LNA prototype shows that the censored upper-tail construction is what makes the earliest onset usable.
That bound only depends on `T`, not on the latent path's discreteness, so the exact process gives no additional handle on it.
- The genetic TMRCA bound.
Same as the earliest-onset bound: it constrains `T`, not the path's discreteness.

The honest read-out: the LNA prototype gets most of what the exact discrete process gives, at much lower cost, because all the available observations are aggregates.
A time-resolved daily stream that resolves the early phase (e.g. dated infection-onset records day-by-day from the first onset) would tilt the comparison toward the exact process, because then the daily-count tail-behaviour starts to matter.
The current data does not have such a stream.

## Identifiability risks

- Offspring overdispersion `φ` versus process noise versus surveillance dispersion `k`.
With aggregate counts only, `φ`, `k`, and any random-walk noise on `R_t` all add to the spread of the observed totals.
The prior on `φ` will carry the inference; the streams cannot separate it from `k`.
- Seed size `I_1`.
The seed is poorly identified.
A larger seed `I_1` and a smaller growth rate `R_1` are nearly degenerate in their effect on `C(T)`, just as `T` and `r` are in the baseline.
- Particle population collapse.
For long latent trajectories with mild observation noise, the ancestry collapses and effective sample size drops at the start of the path.
This is a well-known pathology; remedies (stratified resampling, ancestor sampling) are partial.
- Drift from the deterministic baseline.
At `R_t = exp(r)` constant and `φ = ∞` and `I_1 = 1`, the discrete branching process has expectation `E[I_t] = R^{t-1}` and `E[C_t] = (R^t - 1)/(R - 1)`, which differs from the continuous `exp(r·s)` by a small constant factor near the seed.
This is intentional and is a feature of moving to discrete time, not a bug.

## Files

- `src/state_space.jl` — package code: building-block submodels, the discrete NB-branching renewal step, the onset-staging convolutions, the `state_space_joint` composer.
- `test/test_state_space.jl` — unit tests for the renewal step, the delay discretisation, the convolution helper, the rt walk, the composer's prior-predictive, and an end-to-end short PG smoke test.
- `scripts/prototype_state_space.jl` — driver: prior-predictive sample, gradient check on the continuous block, short particle-Gibbs run, console summary.
- `docs/proposals/state-space-particle.md` — this proposal.

## Verdict and recommendation

The architecture is mechanically operational: it compiles, draws from the prior, and a short particle-Gibbs run completes.
The cost is roughly an order of magnitude or more above the LNA prototype, and the inferential pay-off on the available data is small because all the observations are aggregates that the LNA already captures faithfully.

Recommendation: do not pursue this as the default architecture for the current data.
The LNA prototype (`arch-stochastic`, #101) is the right route for production while the streams remain aggregate.
The state-space prototype is worth keeping as a reference point: if a future data drop adds genuinely time-resolved early-phase records (a day-by-day first-onset trail, or sequenced cases with sampling dates that anchor the early branching), the discreteness of the latent path starts to matter, and the prototype here is the seed of the right route at that point.
