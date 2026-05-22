## Tests for the convolution-v2 Turing model (issue #5): the joint
## composer builds, draws from the prior, yields a finite Mooncake
## gradient, and runs a short NUTS smoke fit. Mirrors the wiring in
## scripts/smoke_v2.jl but kept small for CI.

using BVDOutbreakSize: bvd_joint_v2, growth_v2, incubation_v2,
                       onset_to_death_v2, onset_to_report_v2,
                       nuts_sample, default_adtype, load_observations
using Turing: @model
using Turing.DynamicPPL: LogDensityFunction, VarInfo, link!!,
                         getlogjoint_internal
import Turing.DynamicPPL.LogDensityProblems as LDP
using Distributions: Beta, Normal, truncated
using StatsFuns: logit, logistic

# Minimal unchanged building blocks injected into the v2 composer.
@model function _cfr_bb()
    CFR ~ Beta(6.6, 13.4)
    return (; CFR)
end
@model function _disp_bb()
    inv_sqrt_k ~ truncated(Normal(0.6, 0.2); lower = 0)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end
@model function _asc_bb()
    μ_logit  ~ Normal(logit(0.25), 1.0)
    τ_logit  ~ truncated(Normal(0, 0.5); lower = 1e-4)
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)
    return (; p_drc, p_uganda)
end
@model function _travel_bb()
    daily_travellers ~ truncated(Normal(1871.0, 200.0); lower = 0)
    return (; daily_travellers)
end

_v2_model(e, d, c; tmrca = missing) = bvd_joint_v2(
    e, d, c, growth_v2(), _cfr_bb(), _disp_bb(), _asc_bb(), _travel_bb();
    tmrca_days = tmrca)

@testset "every delay parameter is sampled (no fixed delays)" begin
    ## Project-owner invariant: all delays (incubation, onset-to-death,
    ## onset-to-report) and the onset-to-detection window must be drawn
    ## from priors, not fixed. Their parameters must therefore appear as
    ## sampled variables in the model's VarInfo.
    model = _v2_model(missing, missing, missing)
    vi = VarInfo(model)
    sampled = Set(Symbol.(string.(keys(vi))))
    ## Incubation gamma (infection→onset), onset→death gamma,
    ## onset→report gamma, and the onset→detection window.
    for v in (:α_inc, :θ_inc, :α, :θ, :α_otr, :θ_otr, :w)
        @test v in sampled
    end
    ## Growth timescale is sampled too (no fixed generation time).
    @test :τ in sampled
    @test :m in sampled
end

@testset "bvd_joint_v2 generates a prior-predictive draw" begin
    draw = _v2_model(missing, missing, missing)()
    @test haskey(draw, :I_T)
    @test haskey(draw, :onsets_T)
    @test draw.onsets_T < draw.I_T          # onsets lag infections
    @test draw.expected_deaths > 0
    @test draw.expected_reports > 0
end

@testset "bvd_joint_v2 yields a finite Mooncake gradient" begin
    obs = load_observations()
    model = _v2_model(obs.exported_cases, obs.total_deaths,
                      obs.reported_cases; tmrca = obs.genetic_tmrca_days)
    vi = link!!(VarInfo(model), model)
    θ0 = vi[:]
    ldf = LogDensityFunction(model, getlogjoint_internal, vi;
                             adtype = default_adtype())
    val, grad = LDP.logdensity_and_gradient(ldf, θ0)
    @test isfinite(val)
    @test all(isfinite, grad)
    @test length(grad) == LDP.dimension(ldf)
end

@testset "bvd_joint_v2 runs a short NUTS smoke fit" begin
    obs = load_observations()
    model = _v2_model(obs.exported_cases, obs.total_deaths,
                      obs.reported_cases; tmrca = obs.genetic_tmrca_days)
    chn = nuts_sample(model; samples = 30, chains = 1,
                      target_accept = 0.8)
    @test chn !== nothing
    Ts = vec(Array(chn[:T]))
    @test length(Ts) == 30
    @test all(isfinite, Ts)
    @test all(>(0), Ts)
end
