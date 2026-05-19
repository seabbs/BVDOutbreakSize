# # Joint estimation of BVD outbreak size
#
# This walkthrough fits a single joint Bayesian model to the two data
# streams used by McCabe et al. ([Imperial College London, 18 May 2026](https://doi.org/10.25560/130007))
# to estimate the size of the 2026 Bundibugyo virus disease (BVD)
# outbreak in the Democratic Republic of the Congo:
#
# 1. Two BVD cases detected in Uganda, with population movement data
#    from Ituri Province across seven points of entry.
# 2. Suspected BVD deaths reported in DRC.
#
# The report runs each stream through an independent estimator
# and reports a sensitivity sweep over detection windows, doubling
# times and case fatality ratios. Here those nuisance parameters are
# given priors and fitted jointly, returning a single posterior over
# the latent cumulative case count `C_T`. The closed-form deaths
# approximation `D_T = CFR · C_T · (1 + r/β)^(-α)` is replaced by
# the full convolution integral evaluated numerically, so the delay
# family is a runtime parameter rather than a baked-in analytic form.
#
# **→ Jump to the [joint posterior results](#Joint-model-and-results).**
#
# !!! warning
#     This is an LLM-driven reimplementation as a methodological
#     experiment. It is not intended to inform public health decisions.
#
# ## Summary and limitations
#
# **What this does.** A single Turing model fits the two data streams
# in the report — exported cases in Uganda and suspected BVD deaths in
# DRC — to a shared latent total case count `C_T`. The death stream
# uses the full convolution of exponential growth with the
# onset-to-death gamma, not the report's closed-form approximation.
# Uncertainty in the doubling time, CFR, onset-to-death delay,
# detection window and daily traveller volume is expressed through
# priors rather than scenario sweeps, so there is a single posterior
# over `C_T` rather than 15 scenario-conditional point estimates.
#
# **Limitations.**
#
# - *LLM-driven reimplementation.* The model code, priors, convolution
#   implementation and walkthrough were drafted by a language model
#   from the published Imperial report and the companion delay
#   reanalysis. It has not been independently replicated against the
#   authors' code. It should not inform public-health decisions.
# - *Prior-driven inference where data is scarce.* Two exports and
#   ~10² deaths give almost no information about the dispersion or
#   about `r` and the doubling-time multiplier `m` separately.
#   Posteriors on these parameters will track their priors closely.
# - *Inherits the report's epidemiological assumptions.* Exponential
#   growth from a single seeding case, no spatial structure beyond
#   the Ituri / Nord Kivu split, no time series of cases or deaths.
# - *Onset-to-death delay prior anchored on Isiro 2012.* Centred on
#   the [bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis)
#   posterior, which is a single-outbreak fit. The delay distribution
#   reporting here follows the recommendations in [charniga2024](@cite),
#   but cross-outbreak heterogeneity remains unmodelled.
# - *Deaths likelihood is NegBinomial; exports likelihood is Poisson.*
#   Imperial uses Poisson for both. Here cumulative deaths are given
#   a NegBinomial likelihood to absorb passive-surveillance
#   overdispersion; exports stay Poisson because two observations do
#   not identify an extra dispersion parameter.
# - *Exact convolution for exports.* Imperial / Imai 2020 use the
#   small-`rw` approximation
#   `expected_exports ≈ C_T · w · daily_travellers / source_pop`.
#   We use the exact form
#   `(C(T) − C(T − w)) · daily_travellers / source_pop`, which
#   differs by a factor `(1 − e^{−rw}) / (rw)`. Over the BVD prior
#   range `rw ∈ 0.33-2.0`, the approximation under-estimates `C_T`
#   by 15-57%.
# - *Detection-window definition is loose.* `w` lumps incubation and
#   onset-to-detection into one delay; both are poorly characterised
#   for Bundibugyo virus.
# - *Convolution numerics.* `GaussLegendre(n = 64)` on `[0, T]` is
#   accurate for `T ≲ 200 d`.

using Turing
using Turing: to_submodel
using Distributions
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
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

const ITURI_POPULATION    = obs.source_population_size
const ITURI_DAILY_TRAVEL  = obs.daily_outbound_travellers
const EXPORTED_CASES      = obs.exported_cases
const TOTAL_DEATHS        = obs.total_deaths

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
                                lower = 0.5, upper = 13.0))
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
        window_prior = truncated(Normal(15.0, 5.0); lower = 2.0))
    w ~ window_prior
    return (; w)
end

# ### NegBinomial dispersion for deaths
#
# Cumulative deaths follow
#
# Exported case count uses
# ```math
# Y_{\text{deaths}} \sim
#     \mathrm{NegBinomial}(\mu = \mathbb{E}[D_T],\ k_d),
# ```
#
# with variance `μ + μ²/k_d`. The dispersion captures
# passive-surveillance noise (under-reporting that varies by
# district, weekend reporting effects, batched updates), not
# transmission heterogeneity, so transmission-`k` literature is not
# used. The default prior `1/√k_d ~ Exponential(2)` is weak, giving
# `k_d` prior median near 4 (mild overdispersion).
# Imperial use Poisson here; the switch to NegBinomial is an
# intentional deviation.

@model function deaths_dispersion_model(;
        inv_sqrt_k_prior = Exponential(2.0))
    inv_sqrt_k_d ~ inv_sqrt_k_prior
    k_d := 1.0 / (inv_sqrt_k_d^2 + eps(typeof(inv_sqrt_k_d)))
    return (; k_d, inv_sqrt_k_d)
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
# The normalisation `1 / (θ^α · Γ(α))` is factored out of the
# integrand to keep AD off `loggamma` inside the quadrature loop.

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
    params = (; T, halfwidth,
              cumulative = growth_state.cumulative,
              delay_dist)
    prob = IntegralProblem(_death_integrand, (-1.0, 1.0), params)
    return CFR * halfwidth * solve(prob, alg).u
end

# ## Method 1 — exports-only model
#
# Method 1 of the report (geographic spread) as a stand-alone Turing
# model, mirroring the approach of Imai et al. [imai2020](@cite). Only
# growth, detection-window and NB dispersion submodels are invoked
# because the exports likelihood does not depend on the delay or
# CFR. `C_T` is identified through the product `r · T`, so the
# marginals on `r` and `T` separately stay close to their priors.

@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        daily_travellers::Real  = ITURI_DAILY_TRAVEL,
        source_population::Real = ITURI_POPULATION,
        growth     = exponential_growth_model(),
        window     = detection_window_model(),
        dispersion = nb_dispersion_model())

    growth_state     ~ to_submodel(growth, false)
    window_state     ~ to_submodel(window, false)
    dispersion_state ~ to_submodel(dispersion, false)

    C_T = growth_state.C_T
    w   = window_state.w
    k   = dispersion_state.k

    p_detect         := (daily_travellers / source_population) * w
    expected_exports := C_T * p_detect
    cumulative_cases := C_T

    p_nb = max(k / (k + expected_exports), eps(typeof(k)))
    exported_cases ~ NegativeBinomial(k, p_nb)
end

# ## Method 2 — deaths-only model
#
# Method 2 of the report (backcalculation from deaths) as a
# stand-alone Turing model. Growth, delay and CFR submodels are
# invoked; the detection window and dispersion submodels are not.

@model function deaths_only_model(
        total_deaths::Union{Missing, Integer};
        growth = exponential_growth_model(),
        delay  = delay_model(),
        cfr    = cfr_model())

    growth_state ~ to_submodel(growth, false)
    delay_state  ~ to_submodel(delay,  false)
    cfr_state    ~ to_submodel(cfr,    false)

    raw_deaths        = expected_deaths(cfr_state.CFR, growth_state,
                                        delay_state.dist)
    expected_deaths_T := max(raw_deaths, eps(typeof(raw_deaths)))
    cumulative_cases  := growth_state.C_T

    total_deaths ~ Poisson(expected_deaths_T)
end

# ## Observation submodels
#
# Two observation submodels take the latent `C_T` (and `r`, `T` for
# the deaths convolution) as inputs from the growth submodel. They
# own their own nested submodels (delay, CFR, detection window,
# dispersion) and the likelihood.

# ### Exports
#
# The expected number of exported cases is the incidence produced
# over the last `w` days, scaled by the per-case travel-to-Uganda
# probability:
#
# ```math
# \mu_e = \frac{\text{daily travellers}}{\text{source pop}}
#         \cdot (C(T) - C(T - w)),
# ```
#
# with `Y_exports ~ Poisson(μ_e)`. The Imperial / Imai 2020
# Method 1 simplification
# `μ_e = C_T · w · daily_travellers / source_pop` differs by a
# factor `(1 - e^{-rw}) / (rw)`, which for the BVD prior range
# `rw ∈ 0.33-2.0` sits between 0.43 and 0.85, so the simplification
# under-estimates `C_T` by roughly 15-57%. We use the convolution
# form throughout. Adding NegBinomial overdispersion is not
# identified by two exports.

@model function exports_model(
        exported_cases::Union{Missing, Integer},
        cumulative, T::Real;
        travellers_mean::Real   = ITURI_DAILY_TRAVEL,
        travellers_sd::Real     = ITURI_DAILY_TRAVEL_SD,
        source_population::Real = ITURI_POPULATION,
        window                  = detection_window_model())

    window_state ~ to_submodel(window, false)
    w = window_state.w

    daily_travellers ~ truncated(
        Normal(travellers_mean, travellers_sd);
        lower = travellers_mean * 0.3)

    window_start      = max(T - w, zero(T))
    cumulative_window := cumulative(T) - cumulative(window_start)
    expected_exports  := max(
        (daily_travellers / source_population) * cumulative_window,
        eps(typeof(daily_travellers * one(T))),
    )

    exported_cases ~ Poisson(expected_exports)

    return (; w, daily_travellers, cumulative_window,
              expected_exports)
end

# ### Deaths

@model function deaths_model(
        total_deaths::Union{Missing, Integer},
        C_T::Real, r::Real, T::Real;
        delay      = delay_model(),
        cfr        = cfr_model(),
        dispersion = deaths_dispersion_model())

    delay_state      ~ to_submodel(delay, false)
    cfr_state        ~ to_submodel(cfr, false)
    dispersion_state ~ to_submodel(dispersion, false)

    α   = delay_state.α
    θ   = delay_state.θ
    CFR = cfr_state.CFR
    k_d = dispersion_state.k_d

    raw_deaths         = expected_deaths(CFR, r, T, α, θ)
    expected_deaths_T := max(raw_deaths, eps(typeof(raw_deaths)))

    p_nb_d = max(k_d / (k_d + expected_deaths_T),
                 eps(typeof(k_d)))
    total_deaths ~ NegativeBinomial(k_d, p_nb_d)

    return (; α, θ, CFR, k_d, expected_deaths_T)
end

# ## Top-level composers
#
# Three thin composer models stitch the submodels together: an
# exports-only fit, a deaths-only fit, and the joint fit applying
# both observation submodels.

@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        growth  = exponential_growth_model(),
        exports = exports_model)

    growth_state ~ to_submodel(growth, false)
    exports_state ~ to_submodel(
        exports(exported_cases,
                growth_state.cumulative, growth_state.T), false)

    cumulative_cases := growth_state.C_T
    return (; r = growth_state.r, m = growth_state.m,
              T = growth_state.T, C_T = growth_state.C_T,
              w = exports_state.w,
              daily_travellers = exports_state.daily_travellers,
              cumulative_cases,
              expected_exports = exports_state.expected_exports)
end

@model function deaths_only_model(
        total_deaths::Union{Missing, Integer};
        growth = exponential_growth_model(),
        deaths = deaths_model)

    growth_state ~ to_submodel(growth, false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state.C_T,
               growth_state.r,  growth_state.T), false)

    cumulative_cases := growth_state.C_T
    return (; r = growth_state.r, m = growth_state.m,
              T = growth_state.T, C_T = growth_state.C_T,
              α = deaths_state.α, θ = deaths_state.θ,
              CFR = deaths_state.CFR, k_d = deaths_state.k_d,
              cumulative_cases,
              expected_deaths_T = deaths_state.expected_deaths_T)
end

@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer};
        growth  = exponential_growth_model(),
        exports = exports_model,
        deaths  = deaths_model)

    growth_state ~ to_submodel(growth, false)
    exports_state ~ to_submodel(
        exports(exported_cases,
                growth_state.cumulative, growth_state.T), false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state.C_T,
               growth_state.r,  growth_state.T), false)

    cumulative_cases := growth_state.C_T
    return (; r = growth_state.r, m = growth_state.m,
              T = growth_state.T, C_T = growth_state.C_T,
              w = exports_state.w,
              daily_travellers = exports_state.daily_travellers,
              α = deaths_state.α, θ = deaths_state.θ,
              CFR = deaths_state.CFR, k_d = deaths_state.k_d,
              cumulative_cases,
              expected_exports = exports_state.expected_exports,
              expected_deaths_T = deaths_state.expected_deaths_T)
end

# ## Prior predictive check
#
# Before either observation is taken into account, what does the
# joint prior imply about replicated exports and deaths? Draws from
# the prior over the unobserved data; should bracket the observed
# `Y = 2` exports and `D = 88` deaths.

prior_chn = sample(bvd_joint(missing, missing), Prior(), 2_000;
                   progress = false)
println("Prior C_T summary:")
println(summary_table(prior_chn, [:cumulative_cases]; digits = 0))

# ## Fitting the joint model
#
# NUTS with Mooncake reverse-mode AD, four chains, 1000 post-warmup
# draws each, `target_accept = 0.9`. Chains initialise from the prior
# to keep the sampler away from the boundary of `r` and `m`.

chn_joint = nuts_sample(
    bvd_joint(obs.exported_cases, obs.total_deaths))

# Also fit the two single-stream models so the three posteriors can
# be compared side by side.

chn_exports = nuts_sample(exports_only_model(obs.exported_cases))
chn_deaths  = nuts_sample(deaths_only_model(obs.total_deaths))

# ## Posterior summary

posterior_C_joint   = vec(Array(chn_joint[:cumulative_cases]))
posterior_C_exports = vec(Array(chn_exports[:cumulative_cases]))
posterior_C_deaths  = vec(Array(chn_deaths[:cumulative_cases]))

println("Joint posterior summary:")
println(summary_table(chn_joint,
        [:r, :m, :T, :CFR, :cumulative_cases]; digits = 2))

println()
println("C_T by data stream:")
println(streams_table(
    "exports-only" => posterior_C_exports,
    "deaths-only"  => posterior_C_deaths,
    "joint"        => posterior_C_joint))

# ## Posterior predictive
#
# Draw replicated `(exports, deaths)` from the fitted posterior so
# the two observation streams can be checked against the data they
# were fitted to.

pp_chn     = predict(bvd_joint(missing, missing), chn_joint)
pp_exports = vec(Array(pp_chn[:exported_cases]))
pp_deaths  = vec(Array(pp_chn[:total_deaths]))

plot_posterior_predictive(pp_exports, pp_deaths,
                          obs.exported_cases, obs.total_deaths)

# ## Overlaid posterior densities for `C_T`

plot_cumulative_cases(
    "exports-only" => posterior_C_exports,
    "deaths-only"  => posterior_C_deaths,
    "joint"        => posterior_C_joint)

# ## Imperial report sense check
#
# A quick sanity check: fix the growth, CFR, delay and travel
# parameters at Imperial's main-scenario point estimates and sample
# only the doubling-time multiplier `m` and the dispersion. Pinning
# `log τ = log(14)` (the Imperial main scenario, doubling time
# 14 days) and centring `m` on 8 (so `T ≈ 112 d`) brings the model
# into the range needed to reproduce Imperial's Method 2 headline
# figure of `C_T = 501` (which requires `T ≈ 126 d`).

imperial_growth = exponential_growth_model(
    m_prior = truncated(Normal(8.0, 2.0);
                        lower = 0.5, upper = 13.0),
)

imperial_fixed = Turing.fix(
    deaths_only_model(88; growth = imperial_growth),  # Imperial 16 May 2026 snapshot
    (log_τ = log(14), CFR = 0.30, α = 4.42, θ = 1/0.388),
)

chn_imperial = nuts_sample(imperial_fixed; samples = 500, chains = 2)
posterior_C_imperial = vec(Array(chn_imperial[:cumulative_cases]))

println("Imperial sense check (r and delays fixed):")
println(summary_table(chn_imperial,
        [:m, :T, :cumulative_cases]; digits = 1))

println()
println("Comparison to published scenarios (joint model):")
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
