# Stochastic latent infection process (issue #48) — demonstration script.
#
# Wires the new building-block submodels in `src/stochastic_growth.jl`
# into a joint composer over all four data streams plus the genetic
# TMRCA bound and the newly-usable earliest-onset date, then runs:
#
#   1. a prior-predictive draw,
#   2. a gradient smoke test under NUTS + Mooncake,
#   3. a short NUTS sampling smoke test.
#
# Cross-project directives followed:
#   * Every prior is a submodel passed in to the composer; no inline
#     Normal/Gamma constants live in `@model` bodies.
#   * Observation distributions are injected via keyword arguments
#     (`exports_dist`, `deaths_dist`, `cases_dist`) with sensible
#     defaults.
#   * All delays — incubation, onset-to-death, detection window — are
#     prior-based. No fixed generation interval is used (growth is the
#     sampled rate `r` plus latent increments).
#   * Observations stage as INFECTIONS → ONSET → death/report/detection:
#     the latent LNA process gives infection incidence; the onset
#     cumulative `C_o(t)` is convolved off it once via
#     `onset_cumulative`, and every downstream stream conditions on the
#     same `C_o`. Infections are never convolved directly to deaths /
#     reports / detection.
#   * Non-centred latent increments (z ~ Normal(0,1), scaled by σ).
#   * Only single-side `censored` is used (Mooncake-differentiable);
#     two-sided / `CensoredDistributions` is avoided.
#
# The observation `@model` blocks live in the literate (issue #81), so
# this script defines minimal stand-ins for them, mirroring the test
# suite's convention.
#
# Run:  julia --project=. scripts/prototype_stochastic.jl

using BVDOutbreakSize
using BVDOutbreakSize: stochastic_growth_model, onset_timing_model,
                       tmrca_timing_model, incubation_model,
                       onset_cumulative,
                       default_adtype, load_observations,
                       ITURI_POPULATION, integrate
using Turing: Turing, @model, to_submodel, sample, Prior, NUTS,
              filldist
using Turing: DynamicPPL
import Turing.DynamicPPL.LogDensityProblems as LDP
using Distributions: Normal, LogNormal, truncated, Beta, Gamma,
                     Poisson, NegativeBinomial, pdf
using StatsFuns: logit, logistic
using Random: MersenneTwister
import FlexiChains

# ----- Prior submodels for the observation block ----------------------
#
# Every prior is a named submodel. No `Normal/Gamma/Beta` constants
# sit inline in the composer body.

@model function _onset_to_death(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    dist = Gamma(α, θ)
    f = let d = dist; x -> pdf(d, max(x, zero(x))); end
    return (; α, θ, dist, f)
end

@model function _cfr(; cfr_prior = Beta(6.6, 13.4))
    CFR ~ cfr_prior
    return (; CFR)
end

@model function _window(;
        window_prior = truncated(Normal(15.0, 5.0); lower = 0))
    w ~ window_prior
    return (; w)
end

@model function _travel(;
        prior = truncated(Normal(1871.0, 200.0); lower = 0))
    daily_travellers ~ prior
    return (; daily_travellers)
end

@model function _dispersion(;
        inv_sqrt_k_prior = truncated(Normal(0.6, 0.2); lower = 1e-4))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k)
end

@model function _ascertainment(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0.0, 0.5); lower = 1e-4),
        z_prior   = Normal(0, 1))
    μ_logit  ~ mu_prior
    τ_logit  ~ tau_prior
    z_drc    ~ z_prior
    z_uganda ~ z_prior
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)
    return (; p_drc, p_uganda)
end

# ----- Injectable observation distributions ---------------------------
#
# Distributions are passed in via keyword arguments. The defaults below
# match the package's own choices (Poisson exports, safe NegBinomial
# deaths/cases). The composer never names a specific distribution.

function safe_nbinomial(k, μ)
    μs = max(μ, eps(typeof(μ)))
    p  = clamp(k / (k + μs), eps(typeof(k)), one(k) - eps(typeof(k)))
    return NegativeBinomial(k, p)
end

_default_exports_dist(μ, _k) = Poisson(max(μ, eps(typeof(μ))))
_default_deaths_dist(μ, k)   = safe_nbinomial(k, μ)
_default_cases_dist(μ, k)    = safe_nbinomial(k, μ)

# ----- Joint composer --------------------------------------------------
#
# Stages observations as infections → onset → downstream streams. The
# stochastic growth submodel returns an `incidence(s)` closure for
# infections; the incubation submodel produces an onset CDF/density;
# `onset_cumulative` convolves them once into `C_o(t)`, and every
# downstream stream uses `C_o`, not raw infections.

@model function _bvd_joint_stochastic(
        exported_cases, total_deaths, reported_cases;
        obs,
        growth        = stochastic_growth_model(),
        incubation    = incubation_model(),
        onset_to_death = _onset_to_death(),
        cfr           = _cfr(),
        window        = _window(),
        travel        = _travel(),
        dispersion    = _dispersion(),
        ascertainment = _ascertainment(),
        tmrca_timing  = tmrca_timing_model,
        onset_timing  = onset_timing_model,
        exports_dist  = _default_exports_dist,
        deaths_dist   = _default_deaths_dist,
        cases_dist    = _default_cases_dist,
        source_population::Real = ITURI_POPULATION)

    growth_state ~ to_submodel(growth, false)
    r = growth_state.r; T = growth_state.T
    incidence = growth_state.incidence

    ## Timing terms (single-side censored — Mooncake-safe).
    tmrca_state ~ to_submodel(
        tmrca_timing(T;
            tmrca_days    = obs.genetic_tmrca_days,
            tmrca_days_sd = obs.genetic_tmrca_days_sd), false)
    onset_state ~ to_submodel(
        onset_timing(T; onset_delta = obs.onset_delta), false)

    ## Building blocks (each a submodel; no inline priors here).
    incub_state ~ to_submodel(incubation, false)
    od_state    ~ to_submodel(onset_to_death, false)
    cfr_state   ~ to_submodel(cfr, false)
    window_state ~ to_submodel(window, false)
    travel_state ~ to_submodel(travel, false)
    disp_state  ~ to_submodel(dispersion, false)
    asc_state   ~ to_submodel(ascertainment, false)

    CFR = cfr_state.CFR
    k   = disp_state.k
    w   = window_state.w
    q   = travel_state.daily_travellers / source_population

    ## Incubation density / CDF closures from the incubation submodel.
    ## `F_inc` is the package's CDF-as-density-integral workaround
    ## (Mooncake cannot differentiate `cdf(::Gamma, x)` in α).
    f_inc = incub_state.f
    F_inc = incub_state.F
    f_d   = od_state.f

    ## --- Stage 1: infections → onset ----------------------------------
    ## Onset cumulative C_o(t) = ∫_0^t i(s) F_inc(t - s) ds. Computed
    ## once from the latent infection incidence and reused below.
    C_o_T = onset_cumulative(incidence, F_inc, T)

    ## Onset cumulative as a callable for the exports person-time
    ## integral (re-uses the same incubation CDF).
    C_o = let incidence = incidence, F_inc = F_inc
        s -> onset_cumulative(incidence, F_inc, s)
    end

    ## Onset incidence i_o(t) = ∫_0^t i(s) f_inc(t - s) ds, as a closure
    ## reused by the deaths convolution.
    i_o = let incidence = incidence, f_inc = f_inc
        t -> begin
            t <= zero(t) && return zero(t)
            g = let inc = incidence, fi = f_inc, t = t
                s -> inc(s) * fi(t - s)
            end
            return integrate(g, zero(t), t)
        end
    end

    ## --- Stage 2a: onset → deaths ------------------------------------
    ## μ_deaths(T) = CFR · ∫_0^T i_o(u) f_d(T - u) du. Convolves the
    ## ONSET incidence with the onset-to-death density, NOT infections.
    g_d = let i_o = i_o, f_d = f_d, T = T
        u -> i_o(u) * f_d(T - u)
    end
    μ_d = CFR * integrate(g_d, zero(T), T)
    total_deaths ~ deaths_dist(μ_d, k)

    ## --- Stage 2b: onset → reports -----------------------------------
    ## μ_cases(T) = p_DRC · C_o(T). Conditions on the onset cumulative,
    ## not on raw infection cumulative.
    reported_cases ~ cases_dist(asc_state.p_drc * C_o_T, k)

    ## --- Stage 2c: onset → detection (exports) -----------------------
    ## μ_exports = p_Uganda · q · ∫_{T - w}^{T} C_o(s) ds. The at-risk
    ## person-time uses the ONSET cumulative (a case must have had
    ## onset to be detection-eligible at the border).
    lo = max(T - w, zero(T))
    μ_e = asc_state.p_uganda * q * integrate(C_o, lo, T)
    exported_cases ~ exports_dist(μ_e, k)

    cumulative_cases := C_o_T
end

# ----- Build the model on the bundled observations --------------------

raw = load_observations()
# Add a placeholder onset offset until first_onset_date is added to the
# TOML. 24 Apr 2026 → 18 May cut-off ≈ 24 days.
obs = (; raw..., onset_delta = 24)

model = _bvd_joint_stochastic(
    obs.exported_cases, obs.total_deaths, obs.reported_cases; obs = obs)

# ----- 1. Prior-predictive draw ---------------------------------------

println("== Prior-predictive draw ==")
prior_obs = (; obs..., onset_delta = missing)
prior_model = _bvd_joint_stochastic(
    missing, missing, missing; obs = prior_obs)
pp = sample(prior_model, Prior(), 200;
            chain_type = FlexiChains.VNChain, progress = false)
CT = vec(Array(pp[:cumulative_cases]))
println("  C_T(onset) prior: n=", length(CT),
        " finite=", all(isfinite, CT),
        " >0=", all(CT .>= 0))
println("  C_T(onset) median ≈ ", round(sort(CT)[length(CT) ÷ 2]))

# ----- 2. Gradient smoke test (Mooncake) ------------------------------

println("== Gradient smoke test (Mooncake) ==")
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

# ----- 3. Short NUTS smoke test ---------------------------------------

println("== NUTS sampling smoke test ==")
chn = sample(MersenneTwister(20260518), model,
             NUTS(0.8; adtype = default_adtype()),
             50; chain_type = FlexiChains.VNChain, progress = false)
CTp = vec(Array(chn[:cumulative_cases]))
Tp  = vec(Array(chn[:T]))
println("  posterior C_T(onset): n=", length(CTp),
        " finite=", all(isfinite, CTp))
println("  posterior T median ≈ ", round(sort(Tp)[length(Tp) ÷ 2]),
        " days")
println("DONE")
