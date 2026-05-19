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

# ## Observation submodels
#
# Two observation submodels take the latent growth state as input
# and apply the likelihood for one data stream each. Their priors
# live in nested submodels (delay, CFR, detection window,
# dispersion). Each maps directly onto one of the two analyses in
# the Imperial report.

# ### Exports likelihood — Imperial Method 1 (geographic spread)
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
# `μ_e = C_T · w · daily_travellers / source_pop` differs from this
# by a factor `(1 − e^{−rw}) / (rw)`, which for the BVD prior range
# `rw ∈ 0.33 − 2.0` sits between 0.43 and 0.85, so the simplification
# under-estimates `C_T` by 15-57%. We use the convolution form.
#
# Imperial's explicit likelihood is `Binomial(N, p)`; their reported
# "exact NegBinomial CIs" are the standard confidence-interval
# procedure for inferring `N` from a binomial sample with known `p`.
# In the small-`p` regime here (`p ≈ 6 × 10⁻³`) Poisson and Binomial
# coincide to within numerical noise, so we use Poisson and avoid
# the extra `N`-as-integer machinery.

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
        lower = 0)

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

# ### Deaths likelihood — Imperial Method 2 (backcalculation from deaths)

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
    expected_deaths_T := isfinite(raw_deaths) ?
        max(raw_deaths, eps(typeof(raw_deaths))) :
        eps(typeof(raw_deaths))

    p_nb_d_raw = k_d / (k_d + expected_deaths_T)
    p_nb_d = isfinite(p_nb_d_raw) ?
        clamp(p_nb_d_raw, eps(typeof(k_d)), one(k_d) - eps(typeof(k_d))) :
        eps(typeof(k_d))
    total_deaths ~ NegativeBinomial(k_d, p_nb_d)

    return (; α, θ, CFR, k_d, expected_deaths_T)
end

# ## Top-level composers
#
# Three thin composer models stitch the building blocks together,
# each mapping onto one analysis from the Imperial report.

# ### Exports-only fit — Imperial Method 1 analogue

@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        growth  = exponential_growth_model(),
        exports = exports_model)

    growth_state ~ to_submodel(growth, false)
    exports_state ~ to_submodel(
        exports(exported_cases,
                growth_state.cumulative, growth_state.T), false)

    cumulative_cases := growth_state.C_T
end

# ### Deaths-only fit — Imperial Method 2 analogue

@model function deaths_only_model(
        total_deaths::Union{Missing, Integer};
        growth = exponential_growth_model(),
        deaths = deaths_model)

    growth_state ~ to_submodel(growth, false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state.C_T,
               growth_state.r,  growth_state.T), false)

    cumulative_cases := growth_state.C_T
end

# ### Joint fit — both data streams, single posterior over `C_T`

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
# at `inv_sqrt_k_d = 0` so the deaths likelihood collapses to
# Poisson — Imperial's actual choice (Table 2 reports Poisson CIs).

imperial_growth = exponential_growth_model(
    m_prior = truncated(Normal(8.0, 2.0);
                        lower = 0, upper = 13.0),
)

imperial_fixed = Turing.fix(
    deaths_only_model(88; growth = imperial_growth),  # Imperial 16 May 2026 snapshot
    (log_τ = log(14), CFR = 0.30, α = 4.42, θ = 1/0.388,
     inv_sqrt_k_d = 0.0),
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
