# # Joint estimation of BVD outbreak size
#
# A single joint Bayesian model fits the two data streams used by
# McCabe et al. ([Imperial College London, 18 May 2026](https://doi.org/10.25560/130007))
# to estimate the size of the 2026 Bundibugyo virus disease (BVD)
# outbreak in the Democratic Republic of the Congo:
#
# 1. Two BVD cases detected in Uganda, with population movement data
#    from Ituri Province across seven points of entry.
# 2. 88 suspected BVD deaths reported in DRC as of 16 May 2026.
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

using Turing
using Turing: to_submodel
using Distributions
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
using CairoMakie
using Random
using BVDOutbreakSize

Random.seed!(20260518)

# ## Data
#
# Source population, daily travel volume, and observed counts come
# directly from Tables 1 and 3 of the report and are loaded from
# `data/observations.toml`. As the situation evolves and more cases
# and deaths are reported, edit that file (not the model code) and
# rebuild — the literate walkthrough picks up the new numbers
# automatically.

obs = load_observations()

const ITURI_POPULATION    = obs.source_population_size
const ITURI_DAILY_TRAVEL  = obs.daily_outbound_travellers
const EXPORTED_CASES      = obs.exported_cases
const TOTAL_DEATHS        = obs.total_deaths

# ## Building-block submodels
#
# The joint model is composed of swappable Turing submodels. Each
# submodel owns its priors; replacing one (a different delay study,
# a different growth assumption) requires no edits to the joint
# structure.

# ### Growth — exponential
#
# A single zoonotic case seeds the outbreak `T` days ago and the
# cumulative incidence grows exponentially:
#
# ```math
# C(s) = \exp(r \cdot s), \qquad C_T = C(T).
# ```
#
# The growth submodel returns the latent parameters and a closure
# `cumulative(s)` so the deaths convolution downstream is agnostic
# to the chosen parametric form. Swapping in logistic, two-phase or
# empirical growth only requires writing a new submodel returning
# its own `cumulative` callable.

@model function exponential_growth_model(;
        r_prior = truncated(Normal(0.05, 0.04); lower = 1e-3),
        T_prior = truncated(Normal(60.0, 20.0); lower = 14.0, upper = 180.0))
    r ~ r_prior
    T ~ T_prior
    cumulative = s -> exp(r * s)
    C_T := cumulative(T)
    return (; r, T, C_T, cumulative)
end

# ### Onset-to-death delay
#
# Returns any continuous distribution; the numerical convolution
# downstream doesn't care which family it is. The Imperial report
# uses Rosello et al. 2015's point estimate (α = 4.42, θ ≈ 2.58 d).
# The Bayesian reanalysis of the same line list
# ([sbfnk/bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis),
# vendored as a submodule) gives an `onset → death` marginal of
# mean 11.7 d (95% CrI 9.4-14.6), P95 24.0 d, implying α ≈ 2.4,
# θ ≈ 4.8 — the priors here.

@model function delay_model(;
        alpha_prior = truncated(Normal(2.4, 0.7); lower = 0.5),
        theta_prior = truncated(Normal(4.8, 1.2); lower = 0.2))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

# ### Case-fatality ratio
#
# `Beta(6, 14)` has mean 0.30 and 95% interval (0.13, 0.51), covering
# both the cross-outbreak CDC range (24-40%) and the Isiro baseline
# of 0.47 (95% CrI 0.31-0.65) from the BDBV reanalysis.

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
# p_{\text{detect}} = w \cdot \frac{\text{daily travellers}}{\text{source population}}.
# ```
#
# `w` lumps incubation and onset-to-detection. The report explores
# `w ∈ {10, 15, 20}` days.

@model function detection_window_model(;
        window_prior = truncated(Normal(15.0, 5.0); lower = 2.0))
    w ~ window_prior
    return (; w)
end

# ### NegBinomial dispersion
#
# Exported case count uses
# ```math
# Y_{\text{exports}} \sim \mathrm{NegBinomial}(\mu = C_T \cdot p_{\text{detect}},\ k),
# ```
# with variance `μ + μ²/k`. `1/√k ~ Exponential(1)` follows Stan's
# prior-choice guidance and places ~68% mass on `k > 0.18`,
# consistent with Lloyd-Smith 2005 SARS (0.16) and Althaus 2015 West
# Africa Ebola (0.18). With only two observed exports the prior
# dominates.

@model function nb_dispersion_model(; inv_sqrt_k_prior = Exponential(1.0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

# ## Forward convolution for deaths
#
# Expected cumulative deaths by time `T` from a single seeding case:
#
# ```math
# \mathbb{E}[D_T] = \mathrm{CFR} \cdot \int_0^T C(s)\, f(T - s)\, ds,
# ```
#
# where `C(s)` is supplied by the growth submodel and `f` is the
# onset-to-death density. The integral is evaluated by Gauss-Legendre
# quadrature on `[0, T]` re-mapped to `[-1, 1]`.

const _DEATH_INTEGRAL_ALG = GaussLegendre(; n = 64)

function _death_integrand(u, p)
    s = p.halfwidth * (u + 1)   # u ∈ [-1, 1] → s ∈ [0, T]
    return p.cumulative(s) * pdf(p.delay_dist, p.T - s)
end

function expected_deaths(CFR, growth_state, delay_dist;
        alg = _DEATH_INTEGRAL_ALG)
    T = growth_state.T
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
# model. Only growth, detection-window and NB dispersion submodels
# are invoked because the exports likelihood does not depend on the
# delay or CFR. `C_T` is identified through the product `r · T`, so
# the marginals on `r` and `T` separately stay close to their priors.

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

# ## Joint model
#
# Both observation likelihoods share the same `C_T = exp(r · T)`. All
# five submodels are sampled. The death stream constrains `T`, `r`,
# and the delay distribution; the export stream constrains the
# detection window and dispersion; both share `C_T`.

@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer};
        daily_travellers::Real  = ITURI_DAILY_TRAVEL,
        source_population::Real = ITURI_POPULATION,
        growth     = exponential_growth_model(),
        delay      = delay_model(),
        cfr        = cfr_model(),
        window     = detection_window_model(),
        dispersion = nb_dispersion_model())

    growth_state     ~ to_submodel(growth,     false)
    delay_state      ~ to_submodel(delay,      false)
    cfr_state        ~ to_submodel(cfr,        false)
    window_state     ~ to_submodel(window,     false)
    dispersion_state ~ to_submodel(dispersion, false)

    C_T = growth_state.C_T
    w   = window_state.w
    k   = dispersion_state.k

    p_detect         := (daily_travellers / source_population) * w
    expected_exports := C_T * p_detect
    raw_deaths        = expected_deaths(cfr_state.CFR, growth_state,
                                        delay_state.dist)
    expected_deaths_T := max(raw_deaths, eps(typeof(raw_deaths)))
    cumulative_cases := C_T

    p_nb = max(k / (k + expected_exports), eps(typeof(k)))
    exported_cases ~ NegativeBinomial(k, p_nb)
    total_deaths   ~ Poisson(expected_deaths_T)
end

# ## Prior predictive check
#
# Before either observation is taken into account, what does the
# joint prior imply about replicated exports and deaths? Draws from
# the prior over the unobserved data; should bracket the observed
# `Y = 2` exports and `D = 88` deaths.

prior_chn = sample(bvd_joint(missing, missing), Prior(), 2_000;
                   progress = false)
plot_prior_predictive(
    vec(Array(prior_chn[:exported_cases])),
    vec(Array(prior_chn[:total_deaths])),
    EXPORTED_CASES,
    TOTAL_DEATHS,
)

# ## Fits
#
# Four parallel chains of 1000 post-warmup NUTS draws each, Mooncake
# reverse-mode AD, chains initialised from the prior. `nuts_sample`
# is defined in `BVDOutbreakSize`.

chn_exports = nuts_sample(exports_only_model(EXPORTED_CASES))
chn_deaths  = nuts_sample(deaths_only_model(TOTAL_DEATHS))
chn         = nuts_sample(bvd_joint(EXPORTED_CASES, TOTAL_DEATHS))
nothing

# ## Joint model and results
#
# Posterior summaries for the joint fit. All intervals are
# equal-tailed at three widths (30%, 60%, 90%); no point estimate is
# reported because the posterior is strongly prior-driven on
# `r`, `T` and the dispersion.

summary_table(chn, [:cumulative_cases, :r, :T, :CFR, :α, :θ, :w, :k];
              digits = 3)

# ### Side-by-side `C_T` from each fit
#
# How tightly each data stream alone constrains `C_T`, compared to
# the joint fit:

streams_table(
    "Method 1 (exports only)" => vec(Array(chn_exports[:cumulative_cases])),
    "Method 2 (deaths only)"  => vec(Array(chn_deaths[:cumulative_cases])),
    "Joint"                   => vec(Array(chn[:cumulative_cases])),
)

# ### Posterior `C_T` plot
#
# Densities from the three fits overlaid, with the 15 published
# Imperial scenario point estimates shown as faint dashed lines.

plot_cumulative_cases(
    "Method 1 (exports only)" => vec(Array(chn_exports[:cumulative_cases])),
    "Method 2 (deaths only)"  => vec(Array(chn_deaths[:cumulative_cases])),
    "Joint"                   => vec(Array(chn[:cumulative_cases]));
    xmax = 2_500,
)

# ### Posterior predictive
#
# Replicated `(exports, deaths)` from the joint posterior, with
# observed values (`Y = 2`, `D = 88`) shown in red.

pp_chn = predict(bvd_joint(missing, missing), chn)

plot_posterior_predictive(
    vec(Array(pp_chn[:exported_cases])),
    vec(Array(pp_chn[:total_deaths])),
    EXPORTED_CASES,
    TOTAL_DEATHS,
)

# ### Joint posterior over key parameters
#
# The joint distribution over `C_T`, the growth and delay parameters,
# the CFR, the detection window and the importation dispersion.
# Diagonal panels show marginals; off-diagonal panels show pairwise
# joint contours.

plot_pair(chn, [:cumulative_cases, :r, :T, :CFR, :α, :θ, :w, :k])

# ### Comparison to the published report
#
# McCabe et al. report 15 point estimates of `C_T` across the two
# methods and their sensitivity sweeps (Tables 1 and 2). The joint
# posterior collapses those 15 numbers to a single distribution. The
# table below shows the narrowest joint credible interval (30, 60 or
# 90%) that contains each published scenario, or "outside 90%" if
# the joint fit places the scenario in the tail.

comparison_table(vec(Array(chn[:cumulative_cases])))

# ## Summary and limitations
#
# **What this does.** A single Turing model fits the two data streams
# in the report — 2 exported cases in Uganda, 88 suspected BVD
# deaths in DRC — to a shared latent cumulative case count `C_T`.
# The death stream uses the full convolution of cumulative incidence
# with the onset-to-death delay, integrated numerically so the
# growth and delay families can be swapped without editing the
# quadrature. Uncertainty in the doubling time, CFR, onset-to-death
# delay, detection window and importation overdispersion is
# expressed through priors rather than scenario sweeps, so there is
# a single posterior over `C_T` rather than 15 scenario-conditional
# point estimates.
#
# **Limitations.**
#
# - *LLM-driven reimplementation.* The model code, priors and
#   walkthrough were drafted by a language model from the published
#   Imperial report and the companion delay reanalysis. It has not
#   been independently replicated against the authors' code. It
#   should not inform public-health decisions.
# - *Prior-driven inference where data is scarce.* Two exported
#   cases and 88 deaths give little information about the
#   NegBinomial dispersion `k` or about `r` and `T` separately.
#   Posteriors on these parameters will track their priors closely.
# - *Inherits the report's epidemiological assumptions.* Exponential
#   growth from a single seeding case, no spatial structure, no
#   time series of cases or deaths, no spillover beyond the index
#   zoonotic case, no Uganda-side importation dynamics beyond a
#   fixed daily travel rate.
# - *Onset-to-death delay anchored on Isiro 2012.* A single-outbreak
#   fit (n=22 onset → death observations). The prior widens this
#   but cannot reflect cross-outbreak variation.
# - *Deaths likelihood is Poisson.* The Funk / Camacho / Kucharski
#   line of work uses NegBinomial for both reported cases and
#   deaths. Switching to NegBinomial for deaths is a reasonable
#   robustness check.
# - *Detection-window definition is loose.* `w` lumps incubation and
#   onset-to-detection; both are poorly characterised for
#   Bundibugyo virus and the Imperial report flags this explicitly.
# - *Convolution numerics.* `GaussLegendre(n = 64)` is accurate for
#   `T ≲ 200 d`.

# ## Sense check against Imperial Method 2, main scenario
#
# To check the model recovers a published Method 2 estimate, we
# condition the deaths-only model on the same fixed inputs as the
# report's main scenario:
#
# - Doubling time `τ = 14 d`, so `r = ln(2) / 14 ≈ 0.0495 / day`.
# - CFR = 30% (the middle of the report's CFR sweep).
# - Onset-to-death gamma = Rosello et al. 2015 point estimate
#   (shape `α = 4.42`, rate `β = 0.388 / day`, hence scale
#   `θ ≈ 2.578 day`), the exact distribution the report uses.
#
# All four parameters are pinned via `Turing.fix`. The only latent
# left to sample is `T`, the time since outbreak start; `C_T` is a
# deterministic function of `T` and the fixed `r`. The Imperial
# report's main-scenario estimate is 501 (95% CI 402-612).

imperial_fixed = Turing.fix(
    deaths_only_model(TOTAL_DEATHS),
    (
        r   = log(2) / 14,
        CFR = 0.30,
        α   = 4.42,
        θ   = 1 / 0.388,
    ),
)
chn_imperial = nuts_sample(imperial_fixed)

streams_table(
    "Imperial main (τ=14, CFR=30%, Rosello delay)" =>
        vec(Array(chn_imperial[:cumulative_cases])),
    "Joint (priors on everything)" =>
        vec(Array(chn[:cumulative_cases])),
)
