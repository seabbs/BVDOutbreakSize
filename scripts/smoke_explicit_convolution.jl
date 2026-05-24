## Smoke script for the explicit-convolution architecture (issue #5).
## Exercises: onset-incidence convolution accuracy, the joint model
## compiling, a prior-predictive draw, a Mooncake gradient, runtime of
## the nested integral vs the current single convolution, and a short
## NUTS smoke fit. Run with:
## julia --project=. scripts/smoke_explicit_convolution.jl

using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint_explicit_convolution, OnsetIncidence,
                       onset_incidence,
                       expected_deaths_onset_staged, expected_deaths,
                       expected_onsets_staged, ExportDeathDelay,
                       nuts_sample, default_adtype
using Turing
using Turing: to_submodel
using Turing.DynamicPPL: LogDensityFunction, VarInfo, link!!,
                         getlogjoint_internal
import Turing.DynamicPPL.LogDensityProblems as LDP
using Distributions
using StatsFuns: logit, logistic
using Integrals: IntegralProblem, QuadGKJL, solve
using Random
using Printf: @printf

Random.seed!(20260518)

## Reuse the current model's unchanged building blocks via local
## copies (the package model takes them as injected submodels).
@model function growth_bb()
    τ ~ LogNormal(log(14), 0.4)
    m ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
    r   := log(2) / τ
    T   := m * τ
    I_T := 2.0 ^ m
    return (; τ, r, m, T, I_T)
end
@model function cfr_bb()
    CFR ~ Beta(6.6, 13.4)
    return (; CFR)
end
@model function disp_bb()
    inv_sqrt_k ~ truncated(Normal(0.6, 0.2); lower = 0)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end
@model function asc_bb()
    μ_logit  ~ Normal(logit(0.25), 1.0)
    τ_logit  ~ truncated(Normal(0, 0.5); lower = 1e-4)
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)
    return (; p_drc, p_uganda)
end
@model function travel_bb()
    daily_travellers ~ truncated(Normal(1871.0, 200.0); lower = 0)
    return (; daily_travellers)
end

build(obs_e, obs_d, obs_c; tmrca = missing) =
    bvd_joint_explicit_convolution(obs_e, obs_d, obs_c,
        growth_bb(), cfr_bb(), disp_bb(), asc_bb(), travel_bb();
        tmrca_days = tmrca)

println("== 1. Onset-incidence convolution accuracy ==")
let r = log(2) / 14, incub = Gamma(11.0, 0.74), t = 60.0
    ref(s, p) = (t - s) > 0 ? r * exp(r * s) * pdf(incub, t - s) : 0.0
    prob = IntegralProblem(ref, (0.0, t), nothing)
    exact = solve(prob, QuadGKJL(); reltol = 1e-10).u
    got = onset_incidence(r, incub, t)
    relerr = abs(got - exact) / exact
    @printf "  i_onset(60) approx=%.6g exact=%.6g rel.err=%.2e\n" (
        got) exact relerr
end

println("== 2. Tabulated OnsetIncidence vs direct ==")
let r = log(2) / 14, incub = Gamma(11.0, 0.74), T = 90.0
    oi = OnsetIncidence(r, incub, T)
    maxerr = 0.0
    for t in 5.0:5.0:85.0
        direct = onset_incidence(r, incub, t)
        maxerr = max(maxerr, abs(oi(t) - direct) / max(direct, 1e-12))
    end
    @printf "  max interpolation rel.err over grid = %.2e\n" maxerr
    @printf "  cumulative onsets C_onset(T) = %.4g  (I_T = %.4g)\n" (
        expected_onsets_staged(oi)) exp(r * T)
end

println("== 3. Joint model compiles + prior-predictive draw ==")
gen = build(missing, missing, missing)
draw = gen()
@printf "  prior-predictive returns: %s\n" string(keys(draw))
@printf("  I_T=%.1f onsets_T=%.1f E[exp]=%.3g E[deaths]=%.3g " *
        "E[rep]=%.3g\n",
        draw.I_T, draw.onsets_T, draw.expected_exports,
        draw.expected_deaths, draw.expected_reports)

println("== 4. Conditioned model + log density ==")
obs = load_observations()
model = build(obs.exported_cases, obs.total_deaths, obs.reported_cases;
              tmrca = obs.genetic_tmrca_days)
## A linked VarInfo gives a valid unconstrained parameter vector drawn
## from the prior; the log density is evaluated in that internal space
## (the same space NUTS samples in).
vi = link!!(VarInfo(model), model)
θ0 = vi[:]
ldf = LogDensityFunction(model, getlogjoint_internal, vi)
dim = LDP.dimension(ldf)
lp = LDP.logdensity(ldf, θ0)
@printf "  dimension=%d  logdensity(prior draw)=%.4g\n" dim lp

println("== 5. Mooncake gradient ==")
adtype = default_adtype()
ldf_ad = LogDensityFunction(model, getlogjoint_internal, vi;
                            adtype = adtype)
val, grad = LDP.logdensity_and_gradient(ldf_ad, θ0)
@printf "  logdensity=%.4g  grad finite=%s  ‖grad‖=%.4g\n" val (
    all(isfinite, grad)) sqrt(sum(abs2, grad))

println("== 6. Runtime: onset-staged deaths vs single-convolution ==")
let r = log(2) / 14, incub = Gamma(11.0, 0.74),
    death = Gamma(4.3, 2.6), T = 90.0, CFR = 0.33
    oi = OnsetIncidence(r, incub, T)
    dd = ExportDeathDelay(death, T)
    expected_deaths_onset_staged(oi, dd, CFR)        # warmup
    expected_deaths(CFR, r, T, death)
    build_oi() = OnsetIncidence(r, incub, T)
    n = 2000
    t_oi = @elapsed for _ in 1:n; build_oi(); end
    oi2 = build_oi()
    dd2 = ExportDeathDelay(death, T)
    t_new = @elapsed for _ in 1:n
        expected_deaths_onset_staged(oi2, dd2, CFR)
    end
    t_old = @elapsed for _ in 1:n
        expected_deaths(CFR, r, T, death)
    end
    @printf "  OnsetIncidence build : %7.2f µs/draw\n" 1e6 * t_oi / n
    @printf "  staged deaths int.   : %7.2f µs/call\n" 1e6 * t_new / n
    @printf "  current deaths conv  : %7.2f µs/call\n" 1e6 * t_old / n
    @printf "  per-draw staged cost (build once + 3 reuse) vs 3 convs:\n"
    @printf "    staged ≈ %7.2f µs   current ≈ %7.2f µs\n" (
        1e6 * (t_oi + 3 * t_new) / n) (1e6 * 3 * t_old / n)
end

println("== 7. Short NUTS smoke fit (50 warmup + 50 samples, 1 chain) ==")
t_fit = @elapsed chn = nuts_sample(model; samples = 50, chains = 1,
                                   target_accept = 0.8)
@printf "  fit ran in %.1f s; keys present: I_T=%s onsets_T=%s\n" (
    t_fit) (:I_T in keys(chn)) (:onsets_T in keys(chn))
println("  DONE")
