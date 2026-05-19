# # Joint estimation of BVD outbreak size
#
# This walkthrough fits a single joint Bayesian model to the two data
# streams used by McCabe et al. ([Imperial College London, 18 May 2026](https://doi.org/10.25560/130007))
# to estimate the size of the 2026 Bundibugyo virus disease (BVD) outbreak
# in the Democratic Republic of the Congo:
#
# 1. Two BVD cases detected in Uganda, with population movement data
#    from Ituri Province across seven points of entry.
# 2. 88 suspected BVD deaths reported in DRC as of 16 May 2026.
#
# The original report runs each data stream through an independent
# estimator and reports a sensitivity sweep over detection windows,
# doubling times and case fatality ratios.
# Here those nuisance parameters are given priors and fitted jointly,
# returning a single posterior over the latent total case count `C_T`
# that propagates uncertainty from delays, growth rate and CFR.
# The closed-form approximation
# `D_T = CFR · C_T · (1 + r/β)^(-α)`
# is replaced by the full convolution integral, evaluated by
# Gauss-Legendre quadrature.
#
# !!! warning
#     This is an LLM-driven reimplementation as a methodological
#     experiment. It is not intended to inform public health decisions.

using Turing
using Turing: to_submodel
using Distributions
using ADTypes: AutoMooncake
using Mooncake: Mooncake
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
using SpecialFunctions: loggamma
using Random
using BVDOutbreakSize

Random.seed!(20260518)

# ## Data
#
# Source population, daily travel volume, and observed counts taken
# directly from Tables 1 and 3 of the report. The two exported cases
# travelled from Ituri Province specifically to seek care in Kampala.

const ITURI_POPULATION    = 4_392_200
const ITURI_DAILY_TRAVEL  = 1_871
const EXPORTED_CASES      = 2
const TOTAL_DEATHS        = 88

# ## Submodels and the maths behind them
#
# The joint model is composed of swappable Turing submodels. Each
# submodel owns its priors so they can be replaced (e.g. when wiring
# in a different delay study) without editing the joint structure.

# ### Exponential growth
#
# The outbreak is seeded `T` days ago by a single zoonotic case and
# grows exponentially at rate `r`, so cumulative incidence to date is
#
# ```math
# C_T = \exp(r \cdot T).
# ```
#
# `r` is centred on the report's main scenario (doubling time 14 days,
# `r = ln(2)/14 ≈ 0.05/day`) with a half-normal SD that spans the
# 7-day and 21-day sensitivity scenarios. `T` is constrained to be
# at least two weeks old (consistent with the timing of the first
# laboratory-confirmed Ituri samples) and is given a relatively
# loose prior on the upper end so the death stream can pull `T`
# toward the value implied by the observed cumulative deaths.

@model function growth_model(;
        r_prior = truncated(Normal(0.05, 0.04); lower = 1e-3),
        T_prior = truncated(Normal(60.0, 20.0); lower = 14.0, upper = 180.0))
    r ~ r_prior
    T ~ T_prior
    C_T := exp(r * T)
    return (; r, T, C_T)
end

# ### Onset-to-death delay
#
# Symptom-onset to death is gamma distributed with shape `α` and
# scale `θ`, giving mean `α·θ` and SD `√α · θ`.
#
# The Imperial report's gamma is the point-estimate of Rosello et al.
# 2015 (mean 11.37, SD 5.41; α = 4.42, β = 0.388/day, θ ≈ 2.58 day).
# The companion Bayesian reanalysis of the same Isiro line list
# ([sbfnk/bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis))
# gives an `onset → death` marginal of mean 11.7 d (95% CrI 9.4-14.6)
# with P95 24.0 d (95% CrI 19.0-32.4). Reading off the implied second
# moment gives `α ≈ 2.4`, `θ ≈ 4.8`. Both are used to centre the
# priors here.

@model function delay_model(;
        alpha_prior = truncated(Normal(2.4, 0.7); lower = 0.5),
        theta_prior = truncated(Normal(4.8, 1.2); lower = 0.2))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

# ### Case-fatality ratio
#
# The CDC summary for the two previous BVD outbreaks is 55 deaths /
# 169 cases ≈ 33% (the report's 30% main scenario), with confidence
# bands spanning roughly 24-40%. The companion BDBV reanalysis
# stratifies CFR by HCW status and case definition and reports a
# baseline of 0.47 (95% CrI 0.31-0.65) for non-HCW confirmed cases.
# The prior here is a `Beta(6, 14)` (mean 0.30, 95% interval roughly
# 0.13-0.51) which covers both the cross-outbreak CDC range and the
# Isiro baseline.

@model function cfr_model(; cfr_prior = Beta(6.0, 14.0))
    CFR ~ cfr_prior
    return (; CFR)
end

# ### Detection window
#
# The probability that a single infectious case is detected after
# travelling to Uganda is
#
# ```math
# p_{\text{detect}} = w \cdot \frac{\text{daily travellers}}{\text{source population}}
# ```
#
# where `w` is the mean time during which a case is still infectious
# and detectable abroad (incubation + onset-to-detection). The report
# explores `w ∈ {10, 15, 20}` days; this prior is centred at the
# midpoint with an SD that covers the full sweep within one
# standard deviation.

@model function detection_window_model(;
        window_prior = truncated(Normal(15.0, 5.0); lower = 2.0))
    w ~ window_prior
    return (; w)
end

# ### NegBinomial dispersion for exports
#
# The exported case count is modelled as
#
# ```math
# Y_{\text{exports}} \sim \mathrm{NegBinomial}(\mu = C_T \cdot p_{\text{detect}},\ k),
# ```
#
# with variance `μ + μ²/k`. Following the Stan prior-choice
# guidance, the dispersion enters through `1/√k ~ Exponential(1)`.
# That places ~68% prior mass on `k > 0.18`, consistent with the
# Lloyd-Smith 2005 SARS estimate (k ≈ 0.16) and the Althaus 2015
# West-Africa Ebola estimate (k ≈ 0.18), while admitting the Poisson
# limit at `1/√k → 0`. With only two observed exports the prior
# dominates, so the choice matters.

@model function nb_dispersion_model(; inv_sqrt_k_prior = Exponential(1.0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

# ## Forward convolution for deaths
#
# Under exponential growth from a single seeding case the expected
# cumulative deaths by time `T` is
#
# ```math
# \mathbb{E}[D_T] = \mathrm{CFR} \cdot \int_0^T e^{r s}\, f(T - s;\, \alpha, \theta)\, ds
# ```
#
# where `f(\cdot; α, θ)` is the gamma onset-to-death density. The
# report evaluates this with the closed-form approximation
# `D_T ≈ CFR · C_T · (1 + r/β)^{-α}`; here the integral is computed
# directly by Gauss-Legendre quadrature.
#
# The normalisation `1 / (θ^α · Γ(α))` is factored out of the
# integrand and applied as `exp(-α log θ - loggamma(α))` after the
# quadrature. `loggamma` has a well-supported Mooncake rrule
# through the SpecialFunctions extension, so AD passes through the
# special function exactly once per draw rather than at every
# quadrature node.

const _DEATH_INTEGRAL_ALG = GaussLegendre(; n = 64)

function _death_integrand(u, p)
    ## Map u ∈ [-1, 1] to s ∈ [0, T] with Jacobian halfwidth = T/2.
    s = p.halfwidth * (u + 1)
    τ = p.T - s
    τ <= 0 && return zero(p.r)
    ## Unnormalised gamma pdf in τ; the constant 1/(θ^α · Γ(α)) is
    ## applied outside the integrand to keep AD off the gamma
    ## function inside a quadrature loop.
    return exp(p.r * s) * τ^(p.α - 1) * exp(-τ / p.θ)
end

function expected_deaths(CFR, r, T, α, θ; alg = _DEATH_INTEGRAL_ALG)
    halfwidth = T / 2
    params = (; r, T, halfwidth, α, θ)
    prob = IntegralProblem(_death_integrand, (-1.0, 1.0), params)
    raw = halfwidth * solve(prob, alg).u
    log_norm = -α * log(θ) - loggamma(α)
    return CFR * exp(log_norm) * raw
end

# ## Joint model
#
# All five submodels are composed under one `bvd_joint` model. The
# observation likelihoods (`NegBinomial` for exports, `Poisson` for
# deaths) close over the latent state returned by the submodels.

@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer};
        daily_travellers::Real = ITURI_DAILY_TRAVEL,
        source_population::Real = ITURI_POPULATION,
        growth      = growth_model(),
        delay       = delay_model(),
        cfr         = cfr_model(),
        window      = detection_window_model(),
        dispersion  = nb_dispersion_model())

    growth_state     ~ to_submodel(growth, false)
    delay_state      ~ to_submodel(delay, false)
    cfr_state        ~ to_submodel(cfr, false)
    window_state     ~ to_submodel(window, false)
    dispersion_state ~ to_submodel(dispersion, false)

    C_T   = growth_state.C_T
    r     = growth_state.r
    T     = growth_state.T
    α     = delay_state.α
    θ     = delay_state.θ
    CFR   = cfr_state.CFR
    w     = window_state.w
    k     = dispersion_state.k

    p_detect          := (daily_travellers / source_population) * w
    expected_exports  := C_T * p_detect
    raw_deaths         = expected_deaths(CFR, r, T, α, θ)
    expected_deaths_T := max(raw_deaths, eps(typeof(raw_deaths)))
    cumulative_cases  := C_T

    ## NegBinomial parameterised by mean μ and dispersion k:
    ## variance = μ + μ²/k. The clamp on `p` mirrors the `safe_nb`
    ## guard in the hantavirus realtime work; without it an extreme
    ## proposal can collapse the success probability to zero and
    ## break AD.
    p_nb = max(k / (k + expected_exports), eps(typeof(k)))
    exported_cases ~ NegativeBinomial(k, p_nb)

    total_deaths ~ Poisson(expected_deaths_T)

    return (; r, T, α, θ, CFR, w, k,
            cumulative_cases, expected_exports, expected_deaths_T)
end

# ## Prior predictive check
#
# Drawing from the prior alone (with the observation likelihoods
# turned off via `missing` data) gives a sense of what the priors
# imply about `C_T` before either observation has been taken into
# account.

prior_model = bvd_joint(missing, missing)
prior_chn   = sample(prior_model, Prior(), 2_000; progress = false)
prior_C     = vec(Array(prior_chn[:cumulative_cases]))
println(format_summary("Prior C_T", summarise(prior_C); digits = 0))

# ## Fitting
#
# NUTS with Mooncake reverse-mode AD, four chains, 1000 post-warmup
# draws each, `target_accept = 0.9`. Chains initialise from the prior
# to keep the sampler away from the boundary of `r` and `T`.

const ADTYPE = AutoMooncake(; config = Mooncake.Config())

chn = sample(
    bvd_joint(EXPORTED_CASES, TOTAL_DEATHS),
    NUTS(0.9; adtype = ADTYPE),
    MCMCThreads(),
    1_000, 4;
    initial_params = fill(Turing.DynamicPPL.InitFromPrior(), 4),
    progress = false,
)

# ## Posterior summary
#
# Headline quantity: posterior over total cumulative cases `C_T` to
# date in DRC.

posterior_C   = vec(Array(chn[:cumulative_cases]))
posterior_r   = vec(Array(chn[:r]))
posterior_CFR = vec(Array(chn[:CFR]))
posterior_T   = vec(Array(chn[:T]))

c_summary   = summarise(posterior_C)
r_summary   = summarise(posterior_r)
cfr_summary = summarise(posterior_CFR)
T_summary   = summarise(posterior_T)

println(format_summary("Posterior C_T", c_summary;   digits = 0))
println(format_summary("Posterior r",   r_summary;   digits = 3))
println(format_summary("Posterior CFR", cfr_summary; digits = 3))
println(format_summary("Posterior T",   T_summary;   digits = 1))

# ## Posterior predictive
#
# Draw replicated `(exports, deaths)` from the fitted posterior so the
# two observation streams can be checked against the data they were
# fitted to.

pp_chn     = predict(bvd_joint(missing, missing), chn)
pp_exports = vec(Array(pp_chn[:exported_cases]))
pp_deaths  = vec(Array(pp_chn[:total_deaths]))

println("Replicated exports (obs=", EXPORTED_CASES, "): ",
        format_summary("", summarise(pp_exports); digits = 1))
println("Replicated deaths  (obs=", TOTAL_DEATHS,  "): ",
        format_summary("", summarise(pp_deaths);  digits = 1))

# ## Comparison to the published report
#
# McCabe et al. report 15 point estimates of cumulative cases `C_T`
# across the two methods and their sensitivity sweeps (Tables 1 and 2).
# The joint posterior collapses those 15 numbers to a single
# `C_T` posterior. The comparison below shows which of the published
# point estimates fall inside the joint 95% credible interval.

print_comparison(c_summary)

# ## Summary and limitations
#
# **What this does.** A single Turing model fits the two data streams
# in the report — 2 exported cases in Uganda, 88 suspected BVD deaths
# in DRC — to a shared latent total case count `C_T`. The death
# stream uses the full convolution of exponential growth with the
# onset-to-death gamma, not the report's closed-form approximation.
# Uncertainty in the doubling time, CFR, onset-to-death delay,
# detection window and importation overdispersion is expressed
# through priors rather than scenario sweeps, so there is a single
# posterior over `C_T` rather than 15 scenario-conditional point
# estimates.
#
# **Limitations.**
#
# - *LLM-driven reimplementation.* The model code, priors,
#   convolution implementation and walkthrough were drafted by a
#   language model from the published Imperial report and the
#   companion delay reanalysis. It has not been independently
#   replicated against the authors' code. It should not inform
#   public-health decisions.
# - *Prior-driven inference where data is scarce.* `Y_exports = 2`
#   and `D = 88` give almost no information about the NegBinomial
#   dispersion `k` or about `r` and `T` separately. Posteriors on
#   these parameters will track their priors closely; the priors
#   cover the report's sensitivity ranges but are otherwise a
#   judgement call.
# - *Inherits the report's epidemiological assumptions.* Exponential
#   growth from a single seeding case, no spatial structure beyond
#   the Ituri / Nord Kivu split, no time series of cases or deaths,
#   no spillover beyond the index zoonotic case, no Uganda-side
#   importation dynamics beyond a fixed daily travel rate.
# - *Onset-to-death delay prior anchored on Isiro 2012.* Centred
#   on the [bdbv-linelist-analysis](https://github.com/sbfnk/bdbv-linelist-analysis)
#   posterior, which itself is a single-outbreak fit (n=22 onset →
#   death observations). The prior widens this but cannot reflect
#   cross-outbreak variation.
# - *Deaths likelihood is Poisson.* The Funk / Camacho / Kucharski
#   line of work uses NegBinomial for both reported cases and
#   reported deaths, on the grounds that passive surveillance is
#   overdispersed. Switching to NegBinomial for deaths is a
#   reasonable robustness check; the current model uses Poisson by
#   design choice.
# - *Detection-window definition is loose.* `w` lumps incubation and
#   onset-to-detection into one delay; both are poorly characterised
#   for Bundibugyo virus and the Imperial report flags this
#   explicitly.
# - *Convolution numerics.* `GaussLegendre(n = 64)` on `[0, T]` is
#   accurate for `T ≲ 200 d`; pushing `T` orders of magnitude further
#   would warrant a domain split.
#
# ## Notes
#
# - `C_T` here is cumulative incidence since the seeding zoonotic
#   case; it is comparable to the "Number of cases" column in
#   Table 1 / Table 2 of the report, not to the 336 *reported*
#   suspected cases.
# - The convolution is integrated on `[0, T]` with `T` itself a
#   random variable. `T` is identified jointly by the death and
#   export likelihoods through `C_T = exp(r·T)` and the integral.
