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
# ## What we do differently from Imperial
#
# - *Joint posterior, not 15 scenario estimates.* The doubling time
#   `τ`, CFR, onset-to-death shape and scale, detection window `w`,
#   daily traveller volume and surveillance dispersion all have
#   priors and are sampled jointly. Imperial fixes each and sweeps.
# - *Exact cumulative integral for exports* —
#   `(daily_travellers / source_pop) · ∫_{T−w}^{T} C(s) ds` — rather
#   than Imperial's small-`rw` simplification `q · w · C(T)`. The
#   two forms agree as `r → 0`; over the BVD prior range
#   `rw ∈ 0.33-2.0` Imperial's approximation under-estimates `C_T`
#   by 15-57%.
# - *Exact gamma convolution for deaths.* Imperial uses the
#   closed-form approximation `D_T ≈ CFR · C_T · (1 + r/β)^{−α}`;
#   we evaluate the integral numerically so the delay family
#   stays a runtime parameter.
# - *Reparameterise growth.* Sampling `log τ` (doubling time) and
#   `m = T / τ` (number of doublings since seeding), with
#   `C_T := 2^m` and `r := log(2) / τ` as deterministics,
#   decorrelates `r` from `T` (the natural ridge in `C_T = e^{rT}`).
# - *Onset-to-death prior anchored on the Bayesian reanalysis* of
#   the same Isiro 2012 line list Imperial cites for its point
#   estimates ([sbfnk/bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis)),
#   so the priors carry the published 95% credible intervals on
#   `α` and `θ` rather than collapsing onto Rosello's point
#   estimate.
# - *NegBinomial likelihood on deaths and reported cases* with a
#   single shared surveillance dispersion `k`, vs Imperial's
#   Poisson on both. Exports stay Poisson because two observations
#   would not identify a separate dispersion. Imperial's "exact
#   NegBinomial CIs" on Method 1 are the conventional binomial-
#   inversion procedure, not an estimated dispersion.
# - *Ascertainment extension* (not in Imperial). A `Beta(2, 6)`
#   prior on the reporting fraction, applied to the latent `C_T`,
#   gives a joint posterior over the reported suspected-case
#   count alongside deaths and exports.
# - *No-onward-transmission counterfactual* (not in Imperial).
#   Projects the committed deaths from cases already infected
#   by `T`, integrating `i(s) · (1 − F(T − s))` per draw — a
#   lower bound on the eventual death toll if every onward
#   transmission stopped today.
#
# ## Limitations
#
# - *LLM-driven reimplementation.* The model code, priors,
#   convolution implementation and walkthrough were drafted by a
#   language model from the published Imperial report and the
#   companion delay reanalysis. Not independently replicated
#   against the authors' code; not for public-health decisions.
# - *Prior-driven inference where data is scarce.* Two exports and
#   ~10² deaths give little information about `τ`, `m`, the
#   dispersion, or the reporting fraction individually.
#   Posteriors track their priors closely.
# - *Inherits Imperial's epidemiological assumptions.* Exponential
#   growth from a single zoonotic seed, no spatial structure
#   beyond the Ituri / Nord Kivu split, no time series.
# - *Onset-to-death delay anchored on Isiro 2012.* A single-
#   outbreak fit; the delay distribution reporting here follows
#   [charniga2024](@cite) but cross-outbreak heterogeneity is
#   unmodelled.
# - *Detection-window definition is loose.* `w` lumps incubation
#   and onset-to-detection together — both poorly characterised
#   for BVD.
# - *Convolution numerics.* `GaussLegendre(n = 64)` is accurate
#   for `T ≲ 200 d`.

using Turing
using Turing: to_submodel
using Distributions
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
import SpecialFunctions
using DataFrames: DataFrame
using Random
using BVDOutbreakSize

Random.seed!(20260518)

# ## Data
#
# Observations are loaded from `data/observations.toml`. The source
# population is treated as fixed (census data); the daily outbound
# traveller volume is given a normal prior centred at the Imperial
# figure with an SD covering point-of-entry variation.

obs = load_observations()

println("Loaded observations")
println("-------------------")
println("  exported_cases               = ", obs.exported_cases)
println("  total_deaths                 = ", obs.total_deaths)
println("  daily_outbound_travellers    = ",
        obs.daily_outbound_travellers,
        " (sd = ", obs.daily_outbound_travellers_sd, ")")
println("  source_population            = ", obs.source_population)

const ITURI_POPULATION    = obs.source_population
const ITURI_DAILY_TRAVEL  = obs.daily_outbound_travellers
const EXPORTED_CASES      = obs.exported_cases
const TOTAL_DEATHS        = obs.total_deaths
const REPORTED_CASES      = obs.reported_cases

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

@model function exponential_growth_model(;
        log_τ_prior = Normal(log(14), 0.4),
        m_prior     = truncated(Normal(7.0, 2.5);
                                lower = 0, upper = 13.0))
    log_τ ~ log_τ_prior
    m     ~ m_prior
    τ   := exp(log_τ)
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    cumulative = s -> exp(r * s)
    return (; log_τ, τ, r, m, T, C_T, cumulative)
end

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

@model function delay_model(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

# ### Case-fatality ratio
#
# The CDC summary for the two previous BVD outbreaks is 55 deaths /
# 169 cases ≈ 33% with confidence bands spanning roughly 24-40%. The
# companion BDBV reanalysis reports a baseline of 0.47
# (95% CrI 0.31-0.65) for non-HCW confirmed cases. The prior is
# `Beta(6, 14)` (mean 0.30, 95% interval roughly 0.13-0.51).

@model function cfr_model(; cfr_prior = Beta(6.0, 14.0))
    CFR ~ cfr_prior
    return (; CFR)
end

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

@model function detection_window_model(;
        window_prior = truncated(Normal(15.0, 5.0); lower = 0))
    w ~ window_prior
    return (; w)
end

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
# giving `k` a prior median near 4 (mild overdispersion). Imperial
# use Poisson here; the switch to NegBinomial is an intentional
# deviation.

@model function surveillance_dispersion_model(;
        inv_sqrt_k_prior = Exponential(2.0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

# ### Ascertainment
#
# Of every true case `C_T`, only a fraction `p_report` is captured
# by passive surveillance and appears as a reported suspected case.
# The prior `p_report ~ Beta(2, 6)` has mean 0.25 and a 95% interval
# of roughly `(0.03, 0.65)`, leaving most of the unit interval open
# while gently downweighting near-perfect ascertainment.

@model function ascertainment_model(; p_prior = Beta(2.0, 6.0))
    p_report ~ p_prior
    return (; p_report)
end

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
# Imperial use the closed-form approximation
# `D_T ≈ CFR · C_T · (1 + r/β)^{−α}` (valid for
# `T ⪞ 12 / (β + r)`); we evaluate the exact integral so the delay
# family stays a runtime parameter and the approximation error is
# avoided.

const DEATH_INTEGRAL_ALG = GaussLegendre(; n = 64)

function _death_integrand(u, p)
    s = p.halfwidth * (u + 1)
    τ = p.T - s
    τ <= 0 && return zero(p.r)
    return exp(p.r * s) * τ^(p.α - 1) * exp(-τ / p.θ)
end

function expected_deaths(CFR, r, T, α, θ;
        alg = DEATH_INTEGRAL_ALG)
    halfwidth = T / 2
    params = (; r, T, α, θ, halfwidth)
    prob = IntegralProblem(_death_integrand, (-1.0, 1.0), params)
    integral = halfwidth * solve(prob, alg).u
    log_norm = -α * log(θ) - SpecialFunctions.loggamma(α)
    return CFR * exp(log_norm) * integral
end

# Numerical integral of the cumulative-incidence trajectory `C(s)`
# over `[lower, upper]`, used by the exports likelihood to express
# `∫_{T-w}^{T} C(s) ds` for arbitrary growth submodels.

const _CUM_INTEGRAL_ALG = GaussLegendre(; n = 32)

function _cumulative_integrand(u, p)
    s = p.halfwidth * (u + 1) + p.lower
    return p.cumulative(s)
end

function _integrate_cumulative(cumulative, lower, upper;
        alg = _CUM_INTEGRAL_ALG)
    upper <= lower && return zero(upper - lower)
    halfwidth = (upper - lower) / 2
    params = (; cumulative, halfwidth, lower)
    prob = IntegralProblem(_cumulative_integrand, (-1.0, 1.0), params)
    return halfwidth * solve(prob, alg).u
end

# ## Observation submodels
#
# Three observation submodels take the latent growth state as their
# first non-data argument. They own their own nested submodels and
# the likelihood. The NegBinomial dispersion `k` is shared between
# the deaths and cases likelihoods and is supplied by the composer,
# so the joint fit sees a single surveillance noise scale.

# ### Exports likelihood — Method 1 (Imperial Method 1, geographic spread)
#
# Each case has a daily probability `q = daily_travellers / source_pop`
# of travelling to Uganda, accumulated across the `w` days they are
# in their detection-eligible window. Per case the chance of being
# detected in Uganda by time `T` is `q · min(T − s, w)` (with `s` the
# case's infection time), so the cumulative expected exports are
#
# ```math
# \mu_e = q \cdot \int_{T - w}^{T} C(s)\, ds,
# ```
#
# the daily travel rate times the cumulative cases integrated across
# the detection window. We evaluate the integral numerically over
# `growth_state.cumulative` and apply a Poisson likelihood
# `Y_{\text{exports}} \sim \mathrm{Poisson}(\mu_e)`.
#
# Imperial / Imai 2020 use the small-`rw` simplification
# `μ_e ≈ q · w · C(T)`; this is the limit of the integral as
# `r → 0`. For BVD's prior range `rw ∈ 0.33 − 2.0` the simplification
# under-estimates `C_T` by roughly 15-57%. Both Imperial and we use
# `Binomial(N, p)`-style observation models; with `p ≈ q·w ≈ 6·10⁻³`
# the Poisson approximation we use is indistinguishable from
# Imperial's Binomial in the small-`p` regime.
#
# **Daily traveller prior.** Imperial Table 3 records mean weekly
# passenger counts across seven PoEs from one to four weekly sitreps
# per PoE. The Ituri-side daily total of 1871 is a sample mean
# across roughly 15-21 PoE-weeks. The prior here is a Normal centred
# on 1871 with SD 200 (≈ 10% CV), truncated at zero. The SD covers
# point-of-entry-to-point-of-entry variation and the sampling
# uncertainty implicit in the sitrep schedule; source population is
# kept fixed (census).

@model function exports_model(
        exported_cases::Union{Missing, Integer},
        growth_state;
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
    cumulative_window_integral := _integrate_cumulative(
        cumulative, window_start, T)
    expected_exports := max(
        (daily_travellers / source_population) *
            cumulative_window_integral,
        eps(typeof(daily_travellers * one(T))),
    )

    exported_cases ~ Poisson(expected_exports)

    return (; w, daily_travellers,
              cumulative_window_integral, expected_exports)
end

# ### Deaths likelihood — Method 2 (Imperial Method 2, backcalculation from deaths)
#
# The death convolution is integrated against the latent growth
# trajectory and weighted by the case-fatality ratio. The dispersion
# `k` is supplied by the composer rather than sampled inside, so it
# can be shared with the cases likelihood.

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

    α   = delay_state.α
    θ   = delay_state.θ
    CFR = cfr_state.CFR

    raw_deaths         = expected_deaths(CFR, r, T, α, θ)
    expected_deaths_T := isfinite(raw_deaths) ?
        max(raw_deaths, eps(typeof(raw_deaths))) :
        eps(typeof(raw_deaths))

    p_nb_d_raw = k / (k + expected_deaths_T)
    p_nb_d = isfinite(p_nb_d_raw) ?
        clamp(p_nb_d_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    total_deaths ~ NegativeBinomial(k, p_nb_d)

    return (; α, θ, CFR, expected_deaths_T)
end

# ### Cases likelihood — Extension beyond Imperial (ascertainment)
#
# Imperial Methods 1 and 2 use exports and deaths only. Reported
# suspected cases from the same passive-surveillance system carry
# information about `C_T` once an ascertainment fraction is
# introduced:
#
# ```math
# \mu_c = p_{\text{report}} \cdot C_T,
# \qquad
# Y_{\text{cases}} \sim \mathrm{NegBinomial}(\mu_c,\ k).
# ```
#
# The reporting-delay convolution is deferred to v2 (issue #5). The
# dispersion `k` is shared with the deaths likelihood through the
# composer.

@model function cases_model(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real;
        ascertainment = ascertainment_model())

    C_T = growth_state.C_T

    asc_state ~ to_submodel(ascertainment, false)
    p_report  = asc_state.p_report

    raw_reports        = p_report * C_T
    expected_reports := isfinite(raw_reports) ?
        max(raw_reports, eps(typeof(raw_reports))) :
        eps(typeof(raw_reports))

    p_nb_c_raw = k / (k + expected_reports)
    p_nb_c = isfinite(p_nb_c_raw) ?
        clamp(p_nb_c_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    reported_cases ~ NegativeBinomial(k, p_nb_c)

    return (; p_report, expected_reports)
end

# ## Top-level composers
#
# Thin composer models stitch the submodels together: per-stream
# fits for exports, deaths and reported cases, plus the joint fit.
# The joint composer samples a single `surveillance_dispersion_model`
# and passes that same `k` to both deaths and cases likelihoods, so
# they share one passive-surveillance noise scale.

# ### Exports-only fit — Imperial Method 1 analogue

@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        growth  = exponential_growth_model(),
        exports = exports_model)

    growth_state ~ to_submodel(growth, false)
    exports_state ~ to_submodel(
        exports(exported_cases, growth_state), false)

    cumulative_cases := growth_state.C_T
end

# ### Deaths-only fit — Imperial Method 2 analogue

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

# ### Cases-only fit — ascertainment extension (no Imperial counterpart)

@model function cases_only_model(
        reported_cases::Union{Missing, Integer};
        growth     = exponential_growth_model(),
        cases      = cases_model,
        dispersion = surveillance_dispersion_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    k = dispersion_state.k

    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end

# ### Joint fit — full posterior over `C_T` from all data streams

@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer} = missing;
        growth     = exponential_growth_model(),
        exports    = exports_model,
        deaths     = deaths_model,
        cases      = cases_model,
        dispersion = surveillance_dispersion_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    k = dispersion_state.k

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state), false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)
    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end

# ## Prior predictive check
#
# Before either observation is taken into account, what does the
# joint prior imply about replicated exports, deaths and reported
# cases? Draws from the prior over the unobserved data should bracket
# the observed `Y = 2` exports, `D = 130` deaths and `C = 500`
# reported suspected cases.

prior_chn = sample(bvd_joint(missing, missing, missing), Prior(), 2_000;
                   progress = false)
println("Prior C_T summary:")
println(summary_table(prior_chn, [:cumulative_cases]; digits = 0))

# ## Fitting the joint model
#
# NUTS with Mooncake reverse-mode AD, four chains, 1000 post-warmup
# draws each, `target_accept = 0.9`. Chains initialise from the prior
# to keep the sampler away from the boundary of `r` and `m`.

chn_joint = nuts_sample(
    bvd_joint(obs.exported_cases, obs.total_deaths, obs.reported_cases))

# Also fit the three single-stream models so the four posteriors can
# be compared side by side.

chn_exports = nuts_sample(exports_only_model(obs.exported_cases))
chn_deaths  = nuts_sample(deaths_only_model(obs.total_deaths))
chn_cases   = nuts_sample(cases_only_model(obs.reported_cases))

# ## Posterior summary

posterior_C_joint   = vec(Array(chn_joint[:cumulative_cases]))
posterior_C_exports = vec(Array(chn_exports[:cumulative_cases]))
posterior_C_deaths  = vec(Array(chn_deaths[:cumulative_cases]))
posterior_C_cases   = vec(Array(chn_cases[:cumulative_cases]))

println("Joint posterior summary:")
println(summary_table(chn_joint,
        [:r, :m, :T, :CFR, :p_report, :cumulative_cases]; digits = 2))

println()
println("C_T by data stream:")
println(streams_table(
    "exports-only" => posterior_C_exports,
    "deaths-only"  => posterior_C_deaths,
    "cases-only"   => posterior_C_cases,
    "joint"        => posterior_C_joint))

# ## Posterior predictive
#
# Replicated observations from each fit, checked against the
# corresponding observed counts.

# Exports-only fit: one panel for the exported case stream.

pp_exports_chn  = predict(exports_only_model(missing), chn_exports)
pp_exports_only = vec(Array(pp_exports_chn[:exported_cases]))
plot_posterior_predictive(pp_exports_only, nothing,
                          obs.exported_cases, nothing)

# Deaths-only fit: one panel for cumulative deaths.

pp_deaths_chn  = predict(deaths_only_model(missing), chn_deaths)
pp_deaths_only = vec(Array(pp_deaths_chn[:total_deaths]))
plot_posterior_predictive(nothing, pp_deaths_only,
                          nothing, obs.total_deaths)

# Joint fit: three panels covering exports, deaths and reported cases.

pp_chn     = predict(bvd_joint(missing, missing, missing), chn_joint)
pp_exports = vec(Array(pp_chn[:exported_cases]))
pp_deaths  = vec(Array(pp_chn[:total_deaths]))
pp_cases   = vec(Array(pp_chn[:reported_cases]))

plot_posterior_predictive(pp_exports, pp_deaths,
                          obs.exported_cases, obs.total_deaths;
                          pp_cases = pp_cases,
                          obs_cases = obs.reported_cases)

# ## Overlaid posterior densities for `C_T`

plot_cumulative_cases(
    "exports-only" => posterior_C_exports,
    "deaths-only"  => posterior_C_deaths,
    "cases-only"   => posterior_C_cases,
    "joint"        => posterior_C_joint)

# ## Counterfactual: lower bound under no further transmission
#
# Suppose every onward transmission stopped today. The cohort
# already infected by time `T` still carries committed deaths in
# the onset-to-death tail: a case infected at outbreak age `s` has
# only died by `T` with probability `F_d(T - s)`, so a fraction
# `1 - F_d(T - s)` of its CFR-weighted contribution is still in
# flight. Integrating against the incidence `i(s) = r·exp(r·s)`
# under exponential growth gives the additional committed deaths
#
# ```math
# \Delta D = \mathrm{CFR} \cdot \int_0^T r\,\exp(r\,s)
#            \,\bigl(1 - F_d(T - s)\bigr)\,ds,
# ```
#
# and a lower bound on the cumulative-death endpoint of
# `D_T + \Delta D`, evaluated per posterior draw.

no_onward = predict_no_onward_deaths(chn_joint; obs_deaths = TOTAL_DEATHS)

println("Projected total deaths under no onward transmission:")
println(streams_table(
    "no-onward total" => no_onward.total_projected;
    digits = 0))

# Density of the projected total, with the observed death count
# marked as a dashed black rule.

plot_no_onward_deaths(no_onward; obs_deaths = TOTAL_DEATHS)

# ## Imperial report sense check
#
# A quick sanity check: fix the growth, CFR, delay and travel
# parameters at Imperial's main-scenario point estimates and sample
# only the doubling-time multiplier `m`. Pinning `log τ = log(14)`
# (the Imperial main scenario, doubling time 14 days) and centring
# `m` on 8 (so `T ≈ 112 d`) brings the model into the range needed
# to reproduce Imperial's Method 2 headline figure of `C_T = 501`
# (which requires `T ≈ 126 d`). The NegBinomial dispersion is pinned
# at `inv_sqrt_k = 0` so the deaths likelihood collapses to
# Poisson — Imperial's actual choice (Table 2 reports Poisson CIs).

imperial_growth = exponential_growth_model(
    m_prior = truncated(Normal(8.0, 2.0);
                        lower = 0, upper = 13.0),
)

imperial_fixed = Turing.fix(
    deaths_only_model(88; growth = imperial_growth),  # Imperial 16 May 2026 snapshot
    (log_τ = log(14), CFR = 0.30, α = 4.42, θ = 1/0.388,
     inv_sqrt_k = 0.0),
)

chn_imperial = nuts_sample(imperial_fixed; samples = 500, chains = 2)
posterior_C_imperial = vec(Array(chn_imperial[:cumulative_cases]))

println("Imperial sense check (r and delays fixed):")
println(summary_table(chn_imperial,
        [:m, :T, :cumulative_cases]; digits = 1))

# ### Side-by-side: Imperial reported vs our two analogues
#
# Compare what Imperial *actually reported* for their two main
# scenarios (Method 1 Ituri w=15 d, and Method 2 τ=14 d / CFR 30%)
# against the two `C_T` estimates we derive: the model conditioned
# on Imperial's central assumptions, and the full joint posterior.
# Imperial's 95% intervals are the exact NegBinomial CIs (Method 1)
# and Poisson CIs (Method 2) reported in Tables 1 and 2 of the
# report.

posterior_C_joint    = vec(Array(chn_joint[:cumulative_cases]))
joint_summary        = posterior_summary(posterior_C_joint)
imperial_fit_summary = posterior_summary(posterior_C_imperial)

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
                 round(imperial_fit_summary.lo90; digits = 0),
                 round(joint_summary.lo90;        digits = 0)],
    CrI_upper = [870.0, 612.0,
                 round(imperial_fit_summary.hi90; digits = 0),
                 round(joint_summary.hi90;        digits = 0)],
)
println()
println("Main comparison vs Imperial:")
println(main_comparison)

println()
println("Joint posterior coverage of all 15 published scenarios:")
println(comparison_table(posterior_C_joint))

# ## Notes
#
# - `C_T` here is cumulative incidence since the seeding zoonotic
#   case; it is comparable to the "Number of cases" column in
#   Table 1 / Table 2 of the report, not to the *reported* suspected
#   cases.
# - The convolution is integrated on `[0, T]` with `T` itself a
#   derived quantity (`T = m · log(2) / r`). `T` is identified
#   jointly by the death and export likelihoods through `C_T = 2^m`
#   and the integral.
