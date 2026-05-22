# Stochastic latent growth prototype (issue #48) — demonstration script.
#
# Wires the new `stochastic_growth_model` and `onset_timing_model`
# (in `src/stochastic_growth.jl`) into a joint composer over all four
# data streams plus the genetic TMRCA bound and the newly-usable
# earliest-onset date, then runs:
#
#   1. a prior-predictive draw,
#   2. a gradient smoke test under NUTS + Mooncake, and
#   3. a short NUTS sampling smoke test.
#
# The observation `@model` blocks live in the literate, not the package
# (issue #81), so — exactly as the test suite does — this script defines
# minimal stand-ins for the four likelihoods. The point of the prototype
# is the GROWTH process and its INFERENCE route, not re-deriving the
# (already tested) likelihoods. The stand-ins consume the same
# `growth_state` interface the real submodels do.
#
# Run from the repo root:
#   julia --project=. scripts/prototype_stochastic.jl

using BVDOutbreakSize
using BVDOutbreakSize: stochastic_growth_model, onset_timing_model,
                       expected_exports, expected_deaths,
                       default_adtype, load_observations,
                       ITURI_POPULATION
using Turing: Turing, @model, to_submodel, sample, Prior, NUTS,
              filldist
using Turing: DynamicPPL
import Turing.DynamicPPL.LogDensityProblems as LDP
using Distributions: Normal, LogNormal, truncated, Beta, Poisson,
                     NegativeBinomial, Gamma
using StatsFuns: logit, logistic
using Random: MersenneTwister
import FlexiChains

# --- Minimal observation stand-ins (mirror the test-suite pattern) ------

@model function _delay()
    α ~ truncated(Normal(4.3, 1.22); lower = 0)
    θ ~ truncated(Normal(2.6, 0.82); lower = 0)
    return (; α, θ, dist = Gamma(α, θ))
end

@model function _cfr()
    CFR ~ Beta(6.6, 13.4)
    return (; CFR)
end

## Detection window (incubation + onset-to-detection). A delay, so it is
## sampled from a weakly-informative prior, not fixed — mirrors the
## baseline `detection_window_model`.
@model function _window()
    w ~ truncated(Normal(15.0, 5.0); lower = 0)
    return (; w)
end

## Daily traveller volume, sampled (the per-capita travel rate `q` is not
## a delay, but the volume is an estimated input with its own prior).
@model function _travel(; mean = 1871.0, sd = 200.0)
    daily_travellers ~ truncated(Normal(mean, sd); lower = 0)
    return (; daily_travellers)
end

@model function _dispersion()
    inv_sqrt_k ~ truncated(Normal(0.6, 0.2); lower = 1e-4)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k)
end

@model function _ascertainment()
    μ_logit  ~ Normal(logit(0.25), 1.0)
    τ_logit  ~ truncated(Normal(0, 0.5); lower = 1e-4)
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)
    return (; p_drc, p_uganda)
end

_safe_nb(k, μ) = begin
    μs = max(μ, eps(typeof(μ)))
    p  = clamp(k / (k + μs), eps(typeof(k)), one(k) - eps(typeof(k)))
    return NegativeBinomial(k, p)
end

# --- Joint composer over the stochastic growth process ------------------

@model function _bvd_joint_stochastic(
        exported_cases, total_deaths, reported_cases;
        onset_delta, tmrca_days, tmrca_sd,
        source_population = ITURI_POPULATION)

    growth_state ~ to_submodel(stochastic_growth_model(), false)
    r = growth_state.r; T = growth_state.T
    C_T = growth_state.C_T; cumulative = growth_state.cumulative

    ## Genetic TMRCA lower bound on T (censored, as in the package).
    tmrca_days ~ Turing.censored(Normal(T, tmrca_sd); upper = tmrca_days)

    ## Newly-usable earliest-onset "at-or-before" bound on T.
    onset_state ~ to_submodel(
        onset_timing_model(T; onset_delta = onset_delta), false)

    disp_state ~ to_submodel(_dispersion(), false); k = disp_state.k
    asc_state  ~ to_submodel(_ascertainment(), false)
    delay_state ~ to_submodel(_delay(), false)
    cfr_state   ~ to_submodel(_cfr(), false)
    window_state ~ to_submodel(_window(), false)
    travel_state ~ to_submodel(_travel(), false)

    w = window_state.w
    q = travel_state.daily_travellers / source_population

    ## Exports: at-risk person-time on the stochastic trajectory.
    μ_e = expected_exports(cumulative, asc_state.p_uganda, q, T, w)
    exported_cases ~ Poisson(μ_e)

    ## DRC deaths: CFR-weighted onset-to-death convolution. The package
    ## `expected_deaths` assumes exp(r·s); here we pass the same r so the
    ## mean drift matches, demonstrating the seam. A full version would
    ## convolve against the stochastic incidence directly (aspiration).
    μ_d = expected_deaths(cfr_state.CFR, r, T, delay_state.dist)
    total_deaths ~ _safe_nb(k, μ_d)

    ## DRC reported cases: ascertained C(T) off the stochastic endpoint.
    reported_cases ~ _safe_nb(k, asc_state.p_drc * C_T)

    cumulative_cases := C_T
end

# --- Build the model on the bundled observations ------------------------

obs = load_observations()
onset_delta = 24  # placeholder: earliest onset 24 Apr 2026 vs 18 May
                  # cut-off ≈ 24 days; real value belongs in the TOML.

model = _bvd_joint_stochastic(
    obs.exported_cases, obs.total_deaths, obs.reported_cases;
    onset_delta = onset_delta,
    tmrca_days  = obs.genetic_tmrca_days,
    tmrca_sd    = obs.genetic_tmrca_days_sd)

# --- 1. Prior-predictive draw -------------------------------------------

println("== Prior-predictive draw ==")
prior_model = _bvd_joint_stochastic(
    missing, missing, missing;
    onset_delta = missing,
    tmrca_days  = obs.genetic_tmrca_days,
    tmrca_sd    = obs.genetic_tmrca_days_sd)
pp = sample(prior_model, Prior(), 200;
            chain_type = FlexiChains.VNChain, progress = false)
CT = vec(Array(pp[:cumulative_cases]))
println("  C_T prior: n=", length(CT), " finite=", all(isfinite, CT),
        " >0=", all(CT .> 0))
println("  C_T median ≈ ", round(sort(CT)[length(CT) ÷ 2]))

# --- 2. Gradient smoke test (NUTS + Mooncake) ---------------------------

println("== Gradient smoke test (Mooncake) ==")
## Build the AD-wrapped log-density and evaluate value + gradient at an
## in-support point drawn from the prior. A finite gradient of the full
## dimension confirms Mooncake differentiates through the latent
## log-trajectory, the interpolation, and every downstream integral.
vi = DynamicPPL.VarInfo(model)
vi = DynamicPPL.link(vi, model)
ldf = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, vi;
                                    adtype = default_adtype())
x0 = vi[:]
lp, grad = LDP.logdensity_and_gradient(ldf, x0)
println("  logdensity finite=", isfinite(lp),
        " grad finite=", all(isfinite, grad),
        " dim=", length(grad), " (expected dim=", LDP.dimension(ldf),
        ")")

# --- 3. Short NUTS sampling smoke test ----------------------------------

println("== NUTS sampling smoke test ==")
chn = sample(MersenneTwister(20260518), model,
             NUTS(0.8; adtype = default_adtype()),
             50; chain_type = FlexiChains.VNChain, progress = false)
CTp = vec(Array(chn[:cumulative_cases]))
Tp  = vec(Array(chn[:T]))
println("  posterior C_T: n=", length(CTp), " finite=",
        all(isfinite, CTp))
println("  posterior T median ≈ ", round(sort(Tp)[length(Tp) ÷ 2]),
        " days")
println("DONE")
