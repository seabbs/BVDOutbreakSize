# # Joint estimation of BVD outbreak size
#
# This walkthrough fits a single joint Bayesian model to the data
# used by McCabe et al. ([Imperial College London, 18 May 2026](https://doi.org/10.25560/130007))
# to estimate the size of the 2026 Bundibugyo virus disease (BVD)
# outbreak in the Democratic Republic of the Congo. The Imperial
# report runs two independent analyses — geographic spread from
# cases detected in Uganda, and back-calculation from suspected
# deaths in DRC — and reports a sensitivity sweep over the nuisance
# parameters. Here those streams are combined in one model, the
# nuisance parameters are given priors, and the output is a single
# posterior over the latent cumulative case count `C_T`.
#
# An ascertainment extension takes the reported suspected-case
# count as a third data stream, sharing the surveillance NegBinomial
# dispersion with the deaths likelihood.
#
# **→ Jump to the [joint posterior results](#Joint-model-and-results).**
#
# ## What we do differently from McCabe et al.
#
# - *Joint posterior, not 15 scenario estimates.* The doubling time
#   `τ`, CFR, onset-to-death shape and scale, detection window `w`,
#   daily traveller volume and surveillance dispersion all have
#   priors and are sampled jointly. McCabe et al. fix each and
#   sweep.
# - *Exact cumulative integral for exports* —
#   `(daily_travellers / source_pop) · ∫_{T−w}^{T} C(s) ds` — rather
#   than the small-`rw` simplification `q · w · C(T)` used by
#   McCabe et al. (and by Imai et al. 2020 before them). The two
#   forms agree as `r → 0`.
# - *Numerical (not closed-form) deaths convolution.* For a gamma
#   delay the convolution integral has an exact closed form; McCabe
#   et al. derive it and use the large-`T` simplification
#   `D_T ≈ CFR · C_T · (1 + r/β)^{−α}`. We evaluate the integral
#   numerically instead — not for accuracy (the gamma case is
#   exact either way) but so the onset-to-death distribution can be
#   swapped for any other family without re-deriving the integral.
# - *Onset-to-death prior anchored on the Bayesian reanalysis* of
#   the same Isiro 2012 line list McCabe et al. cite for their
#   point estimates ([sbfnk/bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis)),
#   so the priors carry the published 95% credible intervals on
#   `α` and `θ` rather than collapsing onto Rosello's point
#   estimate.
# - *NegBinomial likelihood on deaths and reported cases* with a
#   single shared surveillance dispersion `k`. McCabe et al. use
#   Poisson for deaths and do not have a cases-ascertainment
#   model at all. Exports stay Poisson because two observations
#   would not identify a separate dispersion. The McCabe et al.
#   "exact NegBinomial CIs" on Method 1 are the conventional
#   binomial-inversion procedure, not an estimated dispersion.
# - *Ascertainment extension* (not in McCabe et al.). A `Beta(2, 6)`
#   prior on the reporting fraction, applied to the latent `C_T`,
#   gives a joint posterior over the reported suspected-case
#   count alongside deaths and exports.
# - *No-onward-transmission counterfactual* (not in McCabe et al.).
#   Projects the committed deaths from cases already infected
#   by `T`, integrating `i(s) · (1 − F(T − s))` per draw — a
#   lower bound on the eventual death toll if every onward
#   transmission stopped today.
#
# ## Limitations
#
# - *Fitted only to aggregate reported counts.* The data are a
#   handful of summary figures — total suspected cases, total
#   suspected deaths, and cases (and one death) detected in Uganda —
#   from press and situation reports. There is no line list and no
#   temporal information: no onset dates, no epidemic curve, no
#   per-case data. The model also has no knowledge of the situation
#   on the ground (case definitions, testing capacity, affected
#   areas, interventions, reporting completeness). Every estimate is
#   a model-based extrapolation from sparse summary statistics under
#   strong assumptions rather than a measurement.
# - *LLM-driven reimplementation.* The model code, priors,
#   convolution implementation and walkthrough were drafted by a
#   language model from the published Imperial report and the
#   companion delay reanalysis, then reviewed and revised. Not
#   independently replicated against the authors' code.
# - *Prior-driven inference where data is scarce.* Two exports,
#   ~10² deaths, and a single reported-case total give little
#   information about `τ`, `m`, the surveillance dispersion, or
#   the reporting fraction individually. Posteriors track their
#   priors closely.
# - *Inherits Imperial's epidemiological assumptions and core
#   model structures.* Exponential growth from a single zoonotic
#   seed; the underlying case trajectory is treated as a
#   deterministic function of the latent state (only the
#   observation counts carry sampling noise via Poisson / NegBinomial)
#   rather than a stochastic incidence process; the cumulative-case
#   / deaths convolution structure for Method 2; the geographic-
#   spread / detection-window structure for Method 1; no spatial
#   structure beyond the Ituri / Nord Kivu split; no time series
#   of cases or deaths.
# - *Onset-to-death delay anchored on Isiro 2012.* A single-
#   outbreak fit; the delay distribution reporting here follows
#   [charniga2024](@cite) but cross-outbreak heterogeneity is
#   unmodelled.
# - *Detection-window definition is loose.* `w` lumps incubation
#   and onset-to-detection together — both poorly characterised
#   for BVD.
# - *Selection bias in deaths-among-exports.* The deaths-among-
#   exports likelihood assumes Uganda's surveillance retains detected
#   exports through to any subsequent death. If the system loses
#   cases that progress to death, the observed count is biased
#   downward and the constraint it places on `T` is overstated.
# - *Ascertainment partially pooled, not separately identified.*
#   Uganda's exported-case ascertainment `p_uganda` and DRC's
#   reported-case ascertainment `p_drc` share a logit-scale
#   hyperprior. With only two exports and one export death the Uganda
#   fraction is weakly identified and leans on the pooled mean and
#   the DRC side.
#
# ## How the model is built up
#
# The model is assembled from small reusable Turing submodels rather
# than written as one monolithic block. Reading top to bottom:
#
# 1. **Building-block submodels** — one per epi parameter family
#    (growth, onset-to-death delay, CFR, detection window,
#    surveillance dispersion, ascertainment). Each samples its own
#    priors and returns a small NamedTuple of values.
# 2. **Forward integrals** — the gamma onset-to-death convolution
#    used by the deaths likelihood and the at-risk person-time
#    integral used by the exports likelihood.
# 3. **Observation submodels** — `exports_model`, `deaths_model`,
#    `cases_model`, `exports_deaths_model`. Each takes the growth
#    state as input, plugs into one of the integrals, and ties one
#    data stream to the latent `C_T`.
# 4. **Top-level composers** — `exports_only_model`,
#    `deaths_only_model`, `cases_only_model`,
#    `exports_deaths_only_model`, `bvd_joint`. Each is a thin wrapper
#    that samples the building blocks and the relevant observation
#    submodels.
# 5. **Inference** — prior predictive, the four NUTS fits, posterior
#    summaries, posterior-predictive plots.
# 6. **Counterfactual and sense check** — a no-onward-transmission
#    lower bound on cumulative deaths, and a `Turing.fix`-pinned
#    reproduction of Imperial's Method 2 main scenario.
#
# Composition is via `~ to_submodel(...)`. Replacing one submodel
# (e.g. swapping the Gamma onset-to-death for a LogNormal, or
# swapping exponential growth for a logistic curve) requires editing
# only that submodel — the integrals and composers do not change.

using Turing
using Turing: to_submodel
using Distributions
using StatsFuns: logit, logistic
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
using DataFrames: DataFrame
import CSV
using Random
using BVDOutbreakSize

Random.seed!(20260518)

# ## Data
#
# Observations are loaded from `data/observations.toml`. The source
# population is treated as fixed (census data); the daily outbound
# traveller volume is given a normal prior centred at the Imperial
# figure with an SD covering point-of-entry variation.

#md # ```@raw html
#md # <details><summary>Loading observations and building the data table</summary>
#md # ```

obs = load_observations()
observations_table = DataFrame(
    field = [
        "exported_cases",
        "exports_deaths",
        "total_deaths",
        "reported_cases",
        "daily_outbound_travellers",
        "daily_outbound_travellers_sd",
        "source_population",
    ],
    value = [
        obs.exported_cases,
        obs.exports_deaths,
        obs.total_deaths,
        obs.reported_cases,
        obs.daily_outbound_travellers,
        obs.daily_outbound_travellers_sd,
        obs.source_population,
    ],
    source = [
        obs.sources.exported_cases,
        obs.sources.exports_deaths,
        obs.sources.total_deaths,
        obs.sources.reported_cases,
        obs.sources.daily_outbound_travellers,
        obs.sources.daily_outbound_travellers_sd,
        obs.sources.source_population,
    ],
);

#md # ```@raw html
#md # </details>
#md # ```

observations_table

#md # ```@raw html
#md # <details><summary>Bind observation values to module-level constants</summary>
#md # ```

const ITURI_POPULATION    = obs.source_population
const ITURI_DAILY_TRAVEL  = obs.daily_outbound_travellers
const EXPORTED_CASES      = obs.exported_cases
const EXPORTS_DEATHS      = obs.exports_deaths
const TOTAL_DEATHS        = obs.total_deaths
const REPORTED_CASES      = obs.reported_cases

#md # ```@raw html
#md # </details>
#md # ```

# ## Building-block submodels
#
# The joint model is composed of swappable Turing submodels. Each
# submodel owns its priors; replacing one (a different delay study,
# a different growth assumption) requires no edits to the joint
# structure. The code conventions used here (Mooncake AD,
# `Integrals.jl` integrand pattern, NB safe-clamp, FlexiChains,
# PairPlots `plot_pair`, AlgebraOfGraphics density layouts) follow
# the hantavirus modelling project [hantavirus_2026](@cite).

# ### Growth — exponential
#
# The outbreak is seeded `T` days ago by a single zoonotic case and
# grows exponentially with doubling time `τ`. Imperial varies the
# doubling time over a sensitivity sweep of 7 / 14 / 21 days; here
# we put a LogNormal prior on `τ` centred at the main scenario
# (14 d) with log-SD 0.4, giving a 95% prior interval of roughly
# `(6, 31)` d that encompasses the full sweep.
#
# Rather than sampling `τ` and `T` directly (which are ridge-
# correlated through `C_T = exp(r T)`), the model samples `log τ`
# and the *doubling-time multiplier* `m = T / τ`. Then `C_T = 2^m`
# is near-orthogonal to `τ`. The growth rate `r = log(2) / τ` and
# `T = m · τ` are exposed as `:=` deterministics so they still
# appear in posterior tables and pair plots.
#
# `m` is centred at 7 (`C_T = 2^7 = 128`) with SD 2.5, giving 95%
# prior support of roughly `m ∈ (2, 12)` → `C_T ∈ (4, 4000)`. This
# brackets Imperial's headline scenario range on the log scale —
# their lowest reported `C_T = 235` corresponds to `m ≈ 7.9`, their
# highest `1008` to `m ≈ 10`. The hard upper bound at 13 caps
# `C_T` at ~8000, well above Imperial's tail.

#md # ```@raw html
#md # <details><summary>Submodel: exponential_growth_model</summary>
#md # ```

@model function exponential_growth_model(;
        tau_prior = LogNormal(log(14), 0.4),
        m_prior   = truncated(Normal(7.0, 2.5);
                              lower = 0, upper = 13.0))
    τ ~ tau_prior
    m ~ m_prior
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    cumulative = s -> exp(r * s)
    return (; τ, r, m, T, C_T, cumulative)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Onset-to-death delay
#
# Symptom-onset to death is gamma distributed with shape `α` and
# scale `θ`, giving mean `α·θ` and SD `√α · θ`.
#
# The Imperial report uses the Rosello et al. 2015 point estimate
# (α = 4.42, β = 0.388/day, θ ≈ 2.58 day). The companion Bayesian
# reanalysis of the same Isiro line list
# ([sbfnk/bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis))
# gives 95% credible intervals of roughly `(2.4, 7.2)` for α and
# `(1.6, 4.8)` for θ. The priors here are Normals centred on the
# bdbv-linelist-analysis posterior mean with SD matching the
# half-width of the published 95% CrIs
# (`1.22 = (7.2 − 2.4) / 3.92`, `0.82 = (4.8 − 1.6) / 3.92`),
# truncated at zero to keep `Gamma(α, θ)` defined. Reporting and
# inference of epidemiological delays follow the recommendations in
# [charniga2024](@cite).

#md # ```@raw html
#md # <details><summary>Submodel: delay_model</summary>
#md # ```

@model function delay_model(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Case-fatality ratio
#
# The CDC summary for the two previous BVD outbreaks is 55 deaths /
# 169 cases ≈ 33% with confidence bands spanning roughly 24-40%. The
# companion BDBV reanalysis reports a baseline of 0.47
# (95% CrI 0.31-0.65) for non-HCW confirmed cases. The prior is
# `Beta(6, 14)` (mean 0.30, 95% interval roughly 0.13-0.51).

#md # ```@raw html
#md # <details><summary>Submodel: cfr_model</summary>
#md # ```

@model function cfr_model(; cfr_prior = Beta(6.0, 14.0))
    CFR ~ cfr_prior
    return (; CFR)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Detection window
#
# The probability that an infectious case is detected after travelling
# to Uganda is
#
# ```math
# p_{\text{detect}} = w \cdot
#     \frac{\text{daily travellers}}{\text{source population}}
# ```
#
# where `w` is the mean time during which a case is still infectious
# and detectable abroad (incubation + onset-to-detection).

#md # ```@raw html
#md # <details><summary>Submodel: detection_window_model</summary>
#md # ```

@model function detection_window_model(;
        window_prior = truncated(Normal(15.0, 5.0); lower = 0))
    w ~ window_prior
    return (; w)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Surveillance dispersion
#
# Both NegBinomial likelihoods (deaths and reported cases) share a
# single dispersion parameter `k`, since both arise from the same
# passive-surveillance system.
#
# ```math
# Y \sim \mathrm{NegBinomial}(\mu,\ k),
# ```
#
# with variance `μ + μ²/k`. The dispersion captures passive-
# surveillance noise (under-reporting that varies by district,
# weekend reporting effects, batched updates), not transmission
# heterogeneity. The default prior `1/√k ~ Exponential(2)` is weak,
# giving `k` a prior median near 4 (mild overdispersion).
# Imperial uses Poisson on deaths (Method 2; their reported CIs
# are Poisson CIs) and does not model reported suspected cases at
# all; the switch to NegBinomial here is an intentional deviation
# from their Method 2 choice.

#md # ```@raw html
#md # <details><summary>Submodel: surveillance_dispersion_model</summary>
#md # ```

@model function surveillance_dispersion_model(;
        inv_sqrt_k_prior = Exponential(2.0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Ascertainment — partial pooling between DRC and Uganda
#
# Two surveillance systems detect cases: DRC passive surveillance
# (which produces the reported suspected-case count) and Uganda's
# point-of-entry / hospital surveillance (which produces the
# exported-case count). Each captures only a fraction of the true
# cases passing through it. Treating the two fractions as identical
# conflates two different systems; treating them as independent
# wastes the structural similarity and leaves the Uganda fraction
# almost wholly prior-driven given only two exports.
#
# Partial pooling sits between the two. Both ascertainment fractions
# share a logit-scale hyperprior with mean `μ_logit` and SD
# `τ_logit`:
#
# ```math
# \mu_{\text{logit}} \sim \mathrm{Normal}(\mathrm{logit}(0.25),\ 1),
# \qquad
# \tau_{\text{logit}} \sim \mathrm{Normal}^{+}(0,\ 0.5),
# ```
#
# ```math
# \mathrm{logit}(p_{\text{DRC}}) \sim
#     \mathrm{Normal}(\mu_{\text{logit}},\ \tau_{\text{logit}}),
# \qquad
# \mathrm{logit}(p_{\text{Uganda}}) \sim
#     \mathrm{Normal}(\mu_{\text{logit}},\ \tau_{\text{logit}}),
# ```
#
# with `p = logistic(logit p)`. The hyperprior mean is centred on
# `logit(0.25)`, matching the previous `Beta(2, 6)` mean of 0.25.
# `τ_logit` is the pooling strength: small `τ_logit` pulls the two
# fractions together (the shared-`p_report` limit), large `τ_logit`
# lets them move independently (the separate-`p` limit). The data
# decide where on that spectrum the fit lands. `cases_model` uses
# `p_drc`; `exports_model` uses `p_uganda`.

#md # ```@raw html
#md # <details><summary>Submodel: pooled_ascertainment_model</summary>
#md # ```

@model function pooled_ascertainment_model(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0, 0.5); lower = 1e-4))
    μ_logit  ~ mu_prior
    τ_logit  ~ tau_prior
    logit_p_drc    ~ Normal(μ_logit, τ_logit)
    logit_p_uganda ~ Normal(μ_logit, τ_logit)
    p_drc    := logistic(logit_p_drc)
    p_uganda := logistic(logit_p_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

#md # ```@raw html
#md # </details>
#md # ```

# ## Forward convolution for deaths
#
# Expected cumulative deaths by time `T` from a single seeding case:
#
# ```math
# \mathbb{E}[D_T] = \mathrm{CFR} \cdot
#     \int_0^T e^{r s}\, f(T - s;\, \alpha, \theta)\, ds
# ```
#
# where `f(·; α, θ)` is the gamma onset-to-death density. The
# integral is evaluated by Gauss-Legendre quadrature with `n = 64`.
# For a gamma delay this integral has an exact closed form;
# McCabe et al. derive it and use the large-`T` simplification
# `D_T ≈ CFR · C_T · (1 + r/β)^{−α}` (valid for `T ⪞ 12/(β+r)`).
# We integrate numerically instead — not because the gamma case
# needs it, but so swapping `Gamma(α, θ)` in `delay_model` for any
# other continuous distribution requires no changes to the
# quadrature.

const DEATH_INTEGRAL_ALG = GaussLegendre(; n = 64)

#md # ```@raw html
#md # <details><summary>Integral helpers — deaths convolution</summary>
#md # ```

function _death_integrand(u, p)
    s = p.halfwidth * (u + 1)
    τ = p.T - s
    τ <= 0 && return zero(p.r)
    return exp(p.r * s) * pdf(p.delay_dist, τ)
end

function expected_deaths(CFR, r, T, delay_dist;
        alg = DEATH_INTEGRAL_ALG)
    halfwidth = T / 2
    params = (; r, T, halfwidth, delay_dist)
    prob = IntegralProblem(_death_integrand, (-1.0, 1.0), params)
    return CFR * halfwidth * solve(prob, alg).u
end

#md # ```@raw html
#md # </details>
#md # ```

# ## At-risk person-time for export detection
#
# Each case in the source population travels to Uganda on any given
# day with probability `q = daily_travellers / source_pop`, treating
# cases as exchangeable with the general population. A case is
# *detection-eligible* for `w` days from infection (incubation +
# onset-to-detection). For a case infected at time `s ≤ T`, the
# accumulated probability of being detected in Uganda by `T` is
#
# ```math
# P(\text{detected by } T \mid \text{infected at } s) = q \cdot \min(T - s, w).
# ```
#
# Splitting at `s = T - w`: ``\min(T-s, w) = w`` for cases infected
# before `T-w` (full window elapsed), and ``\min(T-s, w) = T - s``
# for cases still inside their window. Summing across all incidence
# gives the **full export integral**:
#
# ```math
# \mathbb{E}[\text{exports by }T]
#     = q \cdot \left[ w \cdot C(T-w)
#                      + \int_{T-w}^{T} i(s) \, (T - s) \, ds \right].
# ```
#
# Integration by parts collapses this to the cleaner form
#
# ```math
# \boxed{\,\mathbb{E}[\text{exports by }T] = q \cdot \int_{T-w}^{T} C(s)\, ds\,}
# ```
#
# — the daily travel rate times the cumulative-cases trajectory
# integrated over the detection window. For exponential growth this
# evaluates in closed form to `q · (C(T) - C(T-w)) / r`; as `r → 0`
# it recovers Imperial's small-`rw` simplification `q · w · C(T)`.
# For any other `growth_state.cumulative` callable (logistic,
# two-phase, empirical) the integral has no closed form, so we
# evaluate it numerically by Gauss-Legendre quadrature with `n = 32`,
# mapping `[T-w, T]` to the reference domain `[-1, 1]`.

const _CUM_INTEGRAL_ALG = GaussLegendre(; n = 32)

#md # ```@raw html
#md # <details><summary>Integral helpers — at-risk person-time</summary>
#md # ```

function _cumulative_integrand(u, p)
    s = p.halfwidth * (u + 1) + p.lower
    return p.cumulative(s)
end

function integrate_cumulative(cumulative, lower, upper;
        alg = _CUM_INTEGRAL_ALG)
    upper <= lower && return zero(upper - lower)
    halfwidth = (upper - lower) / 2
    params = (; cumulative, halfwidth, lower)
    prob = IntegralProblem(_cumulative_integrand, (-1.0, 1.0), params)
    return halfwidth * solve(prob, alg).u
end

#md # ```@raw html
#md # </details>
#md # ```

# ## Deaths-among-exports convolution
#
# The deaths-among-exports likelihood needs the expected number of
# exported cases that have *died* by `T`, rather than merely been
# detected. An export detected after being infected at outbreak age
# `s` has died by `T` with probability `F_d(T - s)`, the gamma
# onset-to-death CDF. Weighting the at-risk cumulative-cases
# integrand by that survival-to-death probability gives
#
# ```math
# \int_{T-w}^{T} C(s)\, F_d(T - s)\, ds,
# ```
#
# the same detection window `[T-w, T]` as the exports likelihood but
# with each case down-weighted by how likely it is to have died yet.
# Rather than calling the gamma CDF directly — whose derivative with
# respect to the shape parameter is not supported by the reverse-mode
# AD backend — we write `F_d(T - s)` as the inner integral of the
# gamma density `\int_0^{T-s} f_d(u)\,du`, so the whole expression
# differentiates through the density `f_d` alone, exactly as the
# deaths convolution does. The result is a nested Gauss-Legendre
# quadrature: the outer integral runs over `[T-w, T]` and, for each
# outer node `s`, an inner integral evaluates `F_d(T - s)` over
# `[0, T - s]`. Both map to the reference domain `[-1, 1]`.

#md # ```@raw html
#md # <details><summary>Integral helpers — deaths-among-exports</summary>
#md # ```

function _exports_deaths_cdf_integrand(u, p)
    v = p.inner_halfwidth * (u + 1)        # u ∈ [-1, 1] → v ∈ [0, T-s]
    return pdf(p.delay_dist, v)
end

function _exports_deaths_cdf(delay_dist, upper; alg = _CUM_INTEGRAL_ALG)
    upper <= zero(upper) && return zero(upper)
    inner_halfwidth = upper / 2
    params = (; delay_dist, inner_halfwidth)
    prob = IntegralProblem(
        _exports_deaths_cdf_integrand, (-1.0, 1.0), params)
    return inner_halfwidth * solve(prob, alg).u
end

function _exports_deaths_integrand(u, p)
    s = p.halfwidth * (u + 1) + p.lower
    return p.cumulative(s) * _exports_deaths_cdf(p.delay_dist, p.T - s)
end

function integrate_exports_deaths(cumulative, delay_dist, lower, upper, T;
        alg = _CUM_INTEGRAL_ALG)
    upper <= lower && return zero(upper - lower)
    halfwidth = (upper - lower) / 2
    params = (; cumulative, delay_dist, halfwidth, lower, T)
    prob = IntegralProblem(_exports_deaths_integrand, (-1.0, 1.0), params)
    return halfwidth * solve(prob, alg).u
end

#md # ```@raw html
#md # </details>
#md # ```

# Helper used by the deaths and cases observation submodels:
# `NegativeBinomial` parameterised by mean `μ` and dispersion `k`
# (so `variance = μ + μ²/k`), with NaN / Inf-safe clamping on the
# success probability so extreme NUTS proposals during warmup don't
# trip the distribution domain check.

#md # ```@raw html
#md # <details><summary>Function: safe_nbinomial</summary>
#md # ```

function safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

#md # ```@raw html
#md # </details>
#md # ```

# ## Observation submodels
#
# With the latent state (growth) and the integral machinery in
# place, we now build the three observation models that tie data
# to the latent state. Each takes the growth state as input and
# returns its expected count and likelihood: `exports_model`
# implements Imperial's Method 1, `deaths_model` implements
# Method 2, and `cases_model` is the ascertainment extension.
# These are the units the top-level composers will glue together
# into the per-stream fits and the joint fit.

# ### Exports — Method 1 (Imperial Method 1, geographic spread)
#
# Each case in the source population travels to Uganda on any given
# day with probability `q = daily_travellers / source_pop`, treating
# cases as exchangeable with the general population. A case is
# *detection-eligible* for `w` days from infection (incubation +
# onset-to-detection). For a case infected at time `s ≤ T`, the
# accumulated probability of being detected in Uganda by `T` is
#
# ```math
# P(\text{detected by } T \mid \text{infected at } s) = q \cdot \min(T - s, w).
# ```
#
# Splitting at `s = T - w`:
# ``\min(T - s, w) = w`` for cases infected before `T-w` (full window
# elapsed), and ``\min(T - s, w) = T - s`` for cases still inside their
# window. Summing across all incidence gives the **full export
# integral**:
#
# ```math
# \mathbb{E}[\text{exports by }T]
#     = q \cdot \left[ w \cdot C(T-w)
#                      + \int_{T-w}^{T} i(s) \, (T - s) \, ds \right].
# ```
#
# Integration by parts collapses this to the cleaner form
#
# ```math
# \boxed{\,\mathbb{E}[\text{exports by }T] = q \cdot \int_{T-w}^{T} C(s)\, ds\,}
# ```
#
# — the daily travel rate times the cumulative-cases trajectory
# integrated over the detection window. For exponential growth
# this evaluates to `q · (C(T) - C(T-w)) / r`; as `r → 0` it
# recovers Imperial's small-`rw` simplification `q · w · C(T)`.
#
# We evaluate the boxed integral numerically over
# `growth_state.cumulative`, so the same form works for any growth
# parameterisation (logistic, two-phase, empirical), and apply a
# Poisson likelihood `Y_{\text{exports}} \sim \mathrm{Poisson}(\mu_e)`.
#
# !!! note "Comparison with Imperial / Imai 2020"
#     Imperial use the small-`rw` simplification
#     `μ_e ≈ q · w · C(T)`, the limit of the integral as
#     `r → 0`. For BVD's prior range `rw ∈ 0.33 − 2.0` the
#     simplification under-estimates `C_T` by roughly 15-57%.
#     Both Imperial and we use `Binomial(N, p)`-style observation
#     models; with `p ≈ q·w ≈ 6·10⁻³` our Poisson is
#     indistinguishable from Imperial's Binomial in the small-`p`
#     regime.
#
# **Daily traveller prior.** Imperial Table 3 records mean weekly
# passenger counts across seven PoEs from one to four weekly sitreps
# per PoE. The Ituri-side daily total of 1871 is a sample mean
# across roughly 15-21 PoE-weeks. The prior here is a Normal centred
# on 1871 with SD 200 (≈ 10% CV), truncated at zero. The SD covers
# point-of-entry-to-point-of-entry variation and the sampling
# uncertainty implicit in the sitrep schedule; source population is
# kept fixed (census).

#md # ```@raw html
#md # <details><summary>Submodel: exports_model</summary>
#md # ```

@model function exports_model(
        exported_cases::Union{Missing, Integer},
        growth_state, p_uganda::Real;
        travellers_mean::Real   = ITURI_DAILY_TRAVEL,
        travellers_sd::Real     = ITURI_DAILY_TRAVEL_SD,
        source_population::Real = ITURI_POPULATION,
        window                  = detection_window_model())

    cumulative = growth_state.cumulative
    T          = growth_state.T

    window_state ~ to_submodel(window, false)
    w = window_state.w

    daily_travellers ~ truncated(
        Normal(travellers_mean, travellers_sd);
        lower = 0)

    window_start = max(T - w, zero(T))
    cumulative_window_integral := integrate_cumulative(
        cumulative, window_start, T)
    expected_exports := max(
        p_uganda * (daily_travellers / source_population) *
            cumulative_window_integral,
        eps(typeof(daily_travellers * one(T) * p_uganda)),
    )

    exported_cases ~ Poisson(expected_exports)

    return (; w, daily_travellers, p_uganda,
              cumulative_window_integral, expected_exports)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Deaths — Method 2 (Imperial Method 2, backcalculation from deaths)
#
# The death convolution is integrated against the latent growth
# trajectory and weighted by the case-fatality ratio. The dispersion
# `k` is supplied by the composer rather than sampled inside, so it
# can be shared with the cases likelihood.

#md # ```@raw html
#md # <details><summary>Submodel: deaths_model</summary>
#md # ```

@model function deaths_model(
        total_deaths::Union{Missing, Integer},
        growth_state, k::Real;
        delay = delay_model(),
        cfr   = cfr_model())

    C_T = growth_state.C_T
    r   = growth_state.r
    T   = growth_state.T

    delay_state ~ to_submodel(delay, false)
    cfr_state   ~ to_submodel(cfr, false)

    CFR = cfr_state.CFR

    ## NaN-safe guards. Extreme NUTS proposals during warmup can
    ## push `expected_deaths_T` or `p_nb_d` to NaN / Inf; clamping
    ## NaN-safe clamp: extreme NUTS proposals during warmup can
    ## push `expected_deaths_T` to NaN / Inf.
    raw_deaths         = expected_deaths(CFR, r, T, delay_state.dist)
    expected_deaths_T := isfinite(raw_deaths) ?
        max(raw_deaths, eps(typeof(raw_deaths))) :
        eps(typeof(raw_deaths))

    total_deaths ~ safe_nbinomial(k, expected_deaths_T)

    return (; CFR, delay_dist = delay_state.dist, expected_deaths_T)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Cases — Extension beyond Imperial
#
# Imperial Methods 1 and 2 use exports and deaths only. Reported
# suspected cases from the same passive-surveillance system carry
# information about `C_T` once an ascertainment fraction is
# introduced:
#
# ```math
# \mu_c = p_{\text{DRC}} \cdot C_T,
# \qquad
# Y_{\text{cases}} \sim \mathrm{NegBinomial}(\mu_c,\ k).
# ```
#
# The DRC ascertainment fraction `p_drc` comes from the pooled
# ascertainment submodel, sampled once by the composer and shared
# with the Uganda-side exports likelihood through `p_uganda`. The
# dispersion `k` is shared with the deaths likelihood the same way.

#md # ```@raw html
#md # <details><summary>Submodel: cases_model</summary>
#md # ```

@model function cases_model(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real)

    C_T = growth_state.C_T

    raw_reports        = p_drc * C_T
    expected_reports := isfinite(raw_reports) ?
        max(raw_reports, eps(typeof(raw_reports))) :
        eps(typeof(raw_reports))

    reported_cases ~ safe_nbinomial(k, expected_reports)

    return (; p_drc, expected_reports)
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Deaths-among-exports — fourth observation likelihood
#
# Uganda reports a single death among its detected exports. The zero
# (when there were none) and the small count here are informative:
# if the exports happened long ago, more of them would have died by
# now under the onset-to-death gamma, so the observed deaths-among-
# exports bound how recently the exports occurred and so help
# constrain `T` (and `C_T`).
#
# The expected count reuses the at-risk export integral but weights
# each case by its probability of having died by `T`, then scales by
# the CFR, the daily travel rate `q = daily_travellers / source_pop`
# and the Uganda ascertainment fraction `p_uganda`:
#
# ```math
# \mathbb{E}[D_{\text{Uganda}}]
#     = \mathrm{CFR} \cdot p_{\text{Uganda}} \cdot q
#       \cdot \int_{T-w}^{T} C(s)\, F_d(T - s)\, ds.
# ```
#
# The detection window `w` and daily traveller volume are sampled by
# the exports likelihood and threaded in here so the two Uganda-side
# observations share the same person-time. A Poisson likelihood ties
# the observed count to the expected.
#
# !!! note "Selection-bias caveat"
#     This assumes Uganda's surveillance retains detected exports
#     through to any subsequent death. If the system instead loses
#     cases that progress to death, the observed deaths-among-exports
#     count is selection-biased downward and the constraint it places
#     on `T` is overstated.

#md # ```@raw html
#md # <details><summary>Submodel: exports_deaths_model</summary>
#md # ```

@model function exports_deaths_model(
        exports_deaths::Union{Missing, Integer},
        growth_state, CFR::Real, delay_dist, p_uganda::Real;
        window::Real,
        daily_travellers::Real,
        source_population::Real = ITURI_POPULATION)

    cumulative = growth_state.cumulative
    T          = growth_state.T

    window_start = max(T - window, zero(T))
    exports_deaths_integral := integrate_exports_deaths(
        cumulative, delay_dist, window_start, T, T)
    q = daily_travellers / source_population
    raw = CFR * p_uganda * q * exports_deaths_integral
    expected_exports_deaths := isfinite(raw) ?
        max(raw, eps(typeof(raw))) : eps(typeof(raw))

    exports_deaths ~ Poisson(expected_exports_deaths)

    return (; expected_exports_deaths)
end

#md # ```@raw html
#md # </details>
#md # ```

# ## Top-level composers
#
# These composers stitch the building blocks into the **full
# generative models** for each Imperial analysis. Imperial
# inverts a deterministic summary formula at fixed nuisance
# parameters; here we sample the entire generative process — growth,
# delay, CFR, detection window, dispersion — and condition on the
# observed counts. The per-stream composers (`exports_only_model`,
# `deaths_only_model`) reproduce Imperial Methods 1 and 2 as
# stand-alone Bayesian fits; `bvd_joint` is the same generative
# process conditioned on all four observed streams simultaneously.
#
# The joint composer samples a single `surveillance_dispersion_model`
# and passes that same `k` to both deaths and cases likelihoods, so
# they share one passive-surveillance noise scale. It also samples a
# single `pooled_ascertainment_model`, threading `p_drc` to the cases
# likelihood and `p_uganda` to the two Uganda-side likelihoods
# (exports and deaths-among-exports). The window `w` and daily
# traveller volume sampled by the exports likelihood are reused by
# the deaths-among-exports likelihood so the two share person-time.
# Each per-stream composer instantiates its own pooled submodel.

# ### Exports-only fit — Imperial Method 1 analogue

#md # ```@raw html
#md # <details><summary>Composer: exports_only_model</summary>
#md # ```

@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        growth        = exponential_growth_model(),
        exports       = exports_model,
        ascertainment = pooled_ascertainment_model())

    growth_state ~ to_submodel(growth, false)
    asc_state    ~ to_submodel(ascertainment, false)

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state, asc_state.p_uganda), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Deaths-only fit — Imperial Method 2 analogue

#md # ```@raw html
#md # <details><summary>Composer: deaths_only_model</summary>
#md # ```

@model function deaths_only_model(
        total_deaths::Union{Missing, Integer};
        growth     = exponential_growth_model(),
        deaths     = deaths_model,
        dispersion = surveillance_dispersion_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    k = dispersion_state.k

    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Cases-only fit — ascertainment extension (no Imperial counterpart)

#md # ```@raw html
#md # <details><summary>Composer: cases_only_model</summary>
#md # ```

@model function cases_only_model(
        reported_cases::Union{Missing, Integer};
        growth        = exponential_growth_model(),
        cases         = cases_model,
        dispersion    = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k = dispersion_state.k

    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k, asc_state.p_drc), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Deaths-among-exports-only fit (no Imperial counterpart)

#md # ```@raw html
#md # <details><summary>Composer: exports_deaths_only_model</summary>
#md # ```

@model function exports_deaths_only_model(
        exports_deaths::Union{Missing, Integer};
        growth        = exponential_growth_model(),
        delay         = delay_model(),
        cfr           = cfr_model(),
        window        = detection_window_model(),
        exports_deaths_lik = exports_deaths_model,
        ascertainment = pooled_ascertainment_model(),
        travellers_mean::Real   = ITURI_DAILY_TRAVEL,
        travellers_sd::Real     = ITURI_DAILY_TRAVEL_SD,
        source_population::Real = ITURI_POPULATION)

    growth_state ~ to_submodel(growth, false)
    delay_state  ~ to_submodel(delay, false)
    cfr_state    ~ to_submodel(cfr, false)
    window_state ~ to_submodel(window, false)
    asc_state    ~ to_submodel(ascertainment, false)

    daily_travellers ~ truncated(
        Normal(travellers_mean, travellers_sd); lower = 0)

    exports_deaths_state ~ to_submodel(
        exports_deaths_lik(exports_deaths, growth_state,
            cfr_state.CFR, delay_state.dist, asc_state.p_uganda;
            window           = window_state.w,
            daily_travellers = daily_travellers,
            source_population = source_population),
        false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Joint fit — full posterior over `C_T` from all data streams

#md # ```@raw html
#md # <details><summary>Composer: bvd_joint</summary>
#md # ```

@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer} = missing,
        exports_deaths::Union{Missing, Integer} = missing;
        growth        = exponential_growth_model(),
        exports       = exports_model,
        deaths        = deaths_model,
        cases         = cases_model,
        exports_deaths_lik = exports_deaths_model,
        dispersion    = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model(),
        source_population::Real = ITURI_POPULATION)

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k        = dispersion_state.k
    p_drc    = asc_state.p_drc
    p_uganda = asc_state.p_uganda

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state, p_uganda), false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)
    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k, p_drc), false)
    exports_deaths_state ~ to_submodel(
        exports_deaths_lik(exports_deaths, growth_state,
            deaths_state.CFR, deaths_state.delay_dist, p_uganda;
            window           = exports_state.w,
            daily_travellers = exports_state.daily_travellers,
            source_population = source_population),
        false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ## Prior predictive check
#
# Before any observation is taken into account, what does the joint
# prior imply about replicated exports, deaths, reported cases and
# deaths among exports? Draws from the prior over the unobserved data
# should bracket the observed `Y = 2` exports, `D = 136` deaths,
# `C = 514` reported suspected cases and one death among exports.

prior_chn = sample(bvd_joint(missing, missing, missing, missing),
                   Prior(), 2_000; progress = false);

#md # ```@raw html
#md # <details><summary>Prior summary table</summary>
#md # ```

prior_C_table = summary_table(prior_chn, [:cumulative_cases]; digits = 0);

#md # ```@raw html
#md # </details>
#md # ```

prior_C_table

# Pair plot of the prior over the latent quantities — useful for
# spotting prior correlations before any data has been seen.

#md # ```@raw html
#md # <details><summary>Prior pair plot</summary>
#md # ```

prior_pair_fig = plot_pair(prior_chn,
    [:τ, :m, :cumulative_cases, :CFR, :w, :k,
     :p_drc, :p_uganda, :τ_logit]);

#md # ```@raw html
#md # </details>
#md # ```

prior_pair_fig

# ## Fitting the joint model
#
# NUTS with Mooncake reverse-mode AD, four chains, 1000 post-warmup
# draws each, `target_accept = 0.9`. Chains initialise from the prior
# to keep the sampler away from the boundary of `r` and `m`.

#md # ```@raw html
#md # <details><summary>Run the joint NUTS fit</summary>
#md # ```

chn_joint = nuts_sample(
    bvd_joint(obs.exported_cases, obs.total_deaths,
              obs.reported_cases, obs.exports_deaths));

#md # ```@raw html
#md # </details>
#md # ```

# Also fit the three single-stream models so the four posteriors can
# be compared side by side.

#md # ```@raw html
#md # <details><summary>Run the per-stream NUTS fits</summary>
#md # ```

chn_exports = nuts_sample(exports_only_model(obs.exported_cases));
chn_deaths  = nuts_sample(deaths_only_model(obs.total_deaths));
chn_cases   = nuts_sample(cases_only_model(obs.reported_cases));

#md # ```@raw html
#md # </details>
#md # ```

# ## Posterior summary
#
# The first table reports equal-tailed 30%, 60% and 90% credible
# intervals on the key joint-fit parameters: growth rate `r`,
# doubling-time multiplier `m`, days since seeding `T`, CFR, the DRC
# and Uganda ascertainment fractions `p_drc` and `p_uganda`, the
# pooling SD `τ_logit`, and cumulative cases `C_T`. The second table
# puts the four posteriors over `C_T` side-by-side — the three
# single-stream fits and the joint — to show how each data stream
# constrains the latent outbreak size on its own and what the joint
# combination buys.

posterior_C_joint   = vec(Array(chn_joint[:cumulative_cases]));
posterior_C_exports = vec(Array(chn_exports[:cumulative_cases]));
posterior_C_deaths  = vec(Array(chn_deaths[:cumulative_cases]));
posterior_C_cases   = vec(Array(chn_cases[:cumulative_cases]));

#md # ```@raw html
#md # <details><summary>Joint posterior summary table</summary>
#md # ```

joint_summary = summary_table(chn_joint,
    [:r, :m, :T, :CFR, :p_drc, :p_uganda, :τ_logit,
     :cumulative_cases]; digits = 2);

#md # ```@raw html
#md # </details>
#md # ```

joint_summary

# Comparing `C_T` across the four fits:

#md # ```@raw html
#md # <details><summary>Per-stream C_T table</summary>
#md # ```

streams_C_table = streams_table(
    "exports-only" => posterior_C_exports,
    "deaths-only"  => posterior_C_deaths,
    "cases-only"   => posterior_C_cases,
    "joint"        => posterior_C_joint);

#md # ```@raw html
#md # </details>
#md # ```

streams_C_table

# ## Posterior predictive
#
# A posterior predictive check draws replicated observations from
# the fitted model and compares them to the observed counts. If a
# fit is reasonable the observed value (red line) should sit inside
# the bulk of its replicate distribution. The 2×3 grid below has
# replicates from the per-stream fits on the top row and the joint
# fit on the bottom row — comparable column-wise so it's easy to
# see what each per-stream fit constrains and how the joint
# combination shifts the predictives.

pp_exports_only = vec(Array(predict(
    exports_only_model(missing), chn_exports)[:exported_cases]))
pp_deaths_only  = vec(Array(predict(
    deaths_only_model(missing),  chn_deaths )[:total_deaths]))
pp_cases_only   = vec(Array(predict(
    cases_only_model(missing),   chn_cases  )[:reported_cases]))

pp_joint   = predict(
    bvd_joint(missing, missing, missing, missing), chn_joint)
pp_exports = vec(Array(pp_joint[:exported_cases]))
pp_deaths  = vec(Array(pp_joint[:total_deaths]))
pp_cases   = vec(Array(pp_joint[:reported_cases]))

#md # ```@raw html
#md # <details><summary>Posterior predictive grid plot</summary>
#md # ```

ppc_grid_fig = plot_posterior_predictive_grid(;
    individual = (; exports = pp_exports_only,
                    deaths  = pp_deaths_only,
                    cases   = pp_cases_only),
    joint      = (; exports = pp_exports,
                    deaths  = pp_deaths,
                    cases   = pp_cases),
    observed   = (; exports = obs.exported_cases,
                    deaths  = obs.total_deaths,
                    cases   = obs.reported_cases),
);

#md # ```@raw html
#md # </details>
#md # ```

ppc_grid_fig

# ## Overlaid posterior densities for `C_T`

#md # ```@raw html
#md # <details><summary>Overlaid C_T density plot</summary>
#md # ```

cumulative_density_fig = plot_cumulative_cases(
    "exports-only" => posterior_C_exports,
    "deaths-only"  => posterior_C_deaths,
    "cases-only"   => posterior_C_cases,
    "joint"        => posterior_C_joint);

#md # ```@raw html
#md # </details>
#md # ```

cumulative_density_fig

# ## Counterfactual: lower bound under no further transmission
#
# Suppose every onward transmission stopped today. The cohort
# already infected by time `T` still carries committed deaths in
# the onset-to-death tail: a case infected at outbreak age `s` has
# died by `T` with probability `F_d(T - s)`, so a fraction
# `1 - F_d(T - s)` of its CFR-weighted contribution has not yet
# been observed in the reported death count. Integrating against
# the incidence `i(s) = r·exp(r·s)` under exponential growth gives
# the additional committed deaths
#
# ```math
# \Delta D = \mathrm{CFR} \cdot \int_0^T r\,\exp(r\,s)
#            \,\bigl(1 - F_d(T - s)\bigr)\,ds,
# ```
#
# and a lower bound on the cumulative-death endpoint of
# `D_T + \Delta D`, evaluated per posterior draw.

no_onward = predict_no_onward_deaths(chn_joint; obs_deaths = TOTAL_DEATHS);

#md # ```@raw html
#md # <details><summary>No-onward projected-deaths summary table</summary>
#md # ```

no_onward_table = streams_table(
    "no-onward total" => no_onward.total_projected;
    digits = 0);

#md # ```@raw html
#md # </details>
#md # ```

no_onward_table

# Density of the projected total, with the observed death count
# marked as a dashed black rule.

#md # ```@raw html
#md # <details><summary>No-onward projected-deaths plot</summary>
#md # ```

no_onward_fig = plot_no_onward_deaths(no_onward; obs_deaths = TOTAL_DEATHS);

#md # ```@raw html
#md # </details>
#md # ```

no_onward_fig

# ## One-week-ahead forecast
#
# If the fitted model is taken at face value, what should the next
# week's situation reports show? We continue the fitted exponential
# growth `horizon = 7` days past the cut-off `T` and apply the same
# observation models to forecast the cumulative reported cases (DRC),
# deaths (DRC) and exports (Uganda) by `T + 7`, and the *new* counts
# expected over the coming week (cumulative at `T + 7` minus the
# count already observed). These are posterior-predictive: each draw
# yields a replicated integer count, so the intervals carry both
# parameter and observation uncertainty.
#
# This assumes growth continues unchanged for the week — no
# interventions and no saturation — so it is a "no-change"
# projection, not a considered forecast.

#md # ```@raw html
#md # <details><summary>Generate the one-week-ahead forecast</summary>
#md # ```

forecast = forecast_reported(chn_joint;
    horizon           = 7,
    daily_travellers  = ITURI_DAILY_TRAVEL,
    source_population = ITURI_POPULATION,
    obs_cases         = REPORTED_CASES,
    obs_deaths        = TOTAL_DEATHS,
    obs_exports       = EXPORTED_CASES);
forecast_summary = forecast_table(forecast);

#md # ```@raw html
#md # </details>
#md # ```

forecast_summary

# New counts expected over the coming week, by stream:

#md # ```@raw html
#md # <details><summary>One-week-ahead forecast plot</summary>
#md # ```

forecast_fig = plot_forecast(forecast);

#md # ```@raw html
#md # </details>
#md # ```

forecast_fig

# ## Imperial report sense check
#
# A quick sanity check that the model can recover Imperial's Method 2
# headline when given Imperial's inputs. We use `deaths_only_model`,
# conditioned on Imperial's 16 May 2026 deaths snapshot (88), since
# Method 2 backcalculates outbreak size from deaths alone and does
# not touch the exports, reported-cases or deaths-among-exports
# streams (so the pooled ascertainment and the deaths-among-exports
# likelihood play no part here). We `Turing.fix` the doubling time,
# CFR, gamma shape and scale to the Method 2 main-scenario values
# (`τ = 14 d, CFR = 30%, α = 4.42, β = 0.388/d`) and pin the
# NegBinomial dispersion at `inv_sqrt_k = 0` so the deaths likelihood
# collapses to Poisson — Imperial's actual choice (Table 2 reports
# Poisson CIs). The only sampled latent is `m`, the number of
# doublings since seeding; `C_T = 2^m`. The same `m_prior` used in
# the joint fit applies here without modification.

imperial_fixed = Turing.fix(
    deaths_only_model(88),                              # Imperial 16 May 2026 snapshot
    (τ = 14.0, CFR = 0.30, α = 4.42, θ = 1/0.388,
     inv_sqrt_k = 0.0),
)

#md # ```@raw html
#md # <details><summary>Run the Imperial sense-check fit</summary>
#md # ```

chn_imperial = nuts_sample(imperial_fixed; samples = 500, chains = 2);
posterior_C_imperial = vec(Array(chn_imperial[:cumulative_cases]));

#md # ```@raw html
#md # </details>
#md # ```

#md # ```@raw html
#md # <details><summary>Imperial sense-check summary table</summary>
#md # ```

imperial_summary = summary_table(chn_imperial,
    [:m, :T, :cumulative_cases]; digits = 1);

#md # ```@raw html
#md # </details>
#md # ```

imperial_summary

# ### Side-by-side: Imperial reported vs our two analogues
#
# Compare what Imperial *actually reported* for their two main
# scenarios (Method 1 Ituri w=15 d, and Method 2 τ=14 d / CFR 30%)
# against the two `C_T` estimates we derive: the model conditioned
# on Imperial's central assumptions, and the full joint posterior.
# Imperial's 95% intervals are the exact NegBinomial CIs (Method 1)
# and Poisson CIs (Method 2) reported in Tables 1 and 2 of the
# report.

#md # ```@raw html
#md # <details><summary>Building the main comparison table</summary>
#md # ```

joint_C_credibles    = posterior_summary(posterior_C_joint)
imperial_C_credibles = posterior_summary(posterior_C_imperial)

main_comparison = DataFrame(
    source = [
        "Imperial Method 1 (Ituri, w=15 d)",
        "Imperial Method 2 (τ=14 d, CFR=30%)",
        "Our model | Imperial main assumptions",
        "Our joint posterior",
    ],
    C_T_central = [313.0, 501.0,
                   round(quantile(posterior_C_imperial, 0.5); digits = 0),
                   round(quantile(posterior_C_joint,    0.5); digits = 0)],
    CrI_lower = [39.0, 402.0,
                 round(imperial_C_credibles.lo90; digits = 0),
                 round(joint_C_credibles.lo90;    digits = 0)],
    CrI_upper = [870.0, 612.0,
                 round(imperial_C_credibles.hi90; digits = 0),
                 round(joint_C_credibles.hi90;    digits = 0)],
);

#md # ```@raw html
#md # </details>
#md # ```

main_comparison

# Joint posterior coverage of all 15 published Imperial scenarios:

#md # ```@raw html
#md # <details><summary>Joint coverage table</summary>
#md # ```

coverage_table = comparison_table(posterior_C_joint);

#md # ```@raw html
#md # </details>
#md # ```

coverage_table

# ## Saving results
#
# The tables above are written to an `output/` directory at the repo
# root so they can be archived and shared. On every push to `main` a
# GitHub Actions workflow regenerates these files and publishes them
# as a GitHub Release, downloadable from the repository's releases
# page (<https://github.com/epiforecasts/BVDOutbreakSize/releases>).
# The release bundles the four summary tables, a thinned set of
# posterior draws, and a copy of the input `observations.toml` so the
# exact data that produced each result is recorded alongside it.

#md # ```@raw html
#md # <details><summary>Write outputs to output/</summary>
#md # ```

output_dir = joinpath(pkgdir(BVDOutbreakSize), "output")
mkpath(output_dir)

CSV.write(joinpath(output_dir, "posterior_summary.csv"), joint_summary)
CSV.write(joinpath(output_dir, "cumulative_cases_by_stream.csv"),
          streams_C_table)
CSV.write(joinpath(output_dir, "imperial_comparison.csv"), main_comparison)
CSV.write(joinpath(output_dir, "scenario_coverage.csv"), coverage_table)

## Copy the input data so the release records what produced these
## results.
cp(joinpath(pkgdir(BVDOutbreakSize), "data", "observations.toml"),
   joinpath(output_dir, "observations.toml"); force = true)

## Thinned posterior draws of the key joint parameters (every 10th
## draw) so downstream users can recompute their own summaries.
posterior_draws = DataFrame(
    τ                = vec(Array(chn_joint[:τ])),
    r                = vec(Array(chn_joint[:r])),
    m                = vec(Array(chn_joint[:m])),
    T                = vec(Array(chn_joint[:T])),
    CFR              = vec(Array(chn_joint[:CFR])),
    p_drc            = vec(Array(chn_joint[:p_drc])),
    p_uganda         = vec(Array(chn_joint[:p_uganda])),
    cumulative_cases = vec(Array(chn_joint[:cumulative_cases])),
)[1:10:end, :]
CSV.write(joinpath(output_dir, "posterior_draws.csv"), posterior_draws)

#md # ```@raw html
#md # </details>
#md # ```

