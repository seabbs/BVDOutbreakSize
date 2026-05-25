## Tests for the explicit-convolution Turing model (issue #5): the
## joint composer builds, draws from the prior, yields a finite
## Mooncake gradient, and runs a short NUTS smoke fit. Mirrors the
## wiring in scripts/smoke_explicit_convolution.jl but kept small for
## CI.

using BVDOutbreakSize:
    bvd_joint_explicit_convolution, exponential_growth_explicit,
    infection_to_onset_delay_model, onset_to_death_delay_model,
    onset_to_report_delay_model, onset_to_detection_window_model,
    genetic_seeding_bound_model, OnsetIncidence,
    nuts_sample, default_adtype, load_observations
using Turing: @model
using Turing.DynamicPPL: LogDensityFunction, VarInfo, link!!,
                         getlogjoint_internal
import Turing.DynamicPPL.LogDensityProblems as LDP
using Distributions: Beta, Gamma, Normal, Poisson, NegativeBinomial,
                     truncated
using StatsFuns: logit, logistic

# Minimal unchanged building blocks injected into the composer.
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

_explicit_model(e, d, c; tmrca = missing) =
    bvd_joint_explicit_convolution(e, d, c,
        exponential_growth_explicit(), _cfr_bb(), _disp_bb(),
        _asc_bb(), _travel_bb(); tmrca_days = tmrca)

@testset "every delay parameter is sampled (no fixed delays)" begin
    ## Project-owner invariant: all delays (incubation, onset-to-death,
    ## onset-to-report) and the onset-to-detection window must be drawn
    ## from priors, not fixed. Their parameters must therefore appear
    ## as sampled variables in the model's VarInfo.
    model = _explicit_model(missing, missing, missing)
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

@testset "delay and window submodels are independently usable" begin
    ## Each prior is its own submodel so a sensitivity analysis can
    ## instantiate it standalone and inject a different prior. This
    ## test exercises that interface without going through the joint.
    @test (infection_to_onset_delay_model())().dist isa Gamma
    @test (onset_to_death_delay_model())().dist isa Gamma
    @test (onset_to_report_delay_model())().dist isa Gamma
    @test (onset_to_detection_window_model())().w > 0
    ## genetic_seeding_bound_model conditions on `tmrca_days`; sample
    ## T=90 and check it returns the recorded fields without error.
    g = genetic_seeding_bound_model(90.0, 80; tmrca_days_sd = 20.0)()
    @test g.tmrca_days == 80
    @test g.tmrca_days_sd == 20.0
end

@testset "observation distributions are injectable" begin
    ## Swap each observation submodel's likelihood for an explicit
    ## Poisson and check the model still draws and produces
    ## non-negative counts. This guards against a hardcoded
    ## distribution sneaking back into the observation submodels.
    model = bvd_joint_explicit_convolution(missing, missing, missing,
        exponential_growth_explicit(), _cfr_bb(), _disp_bb(),
        _asc_bb(), _travel_bb();
        exports_obs = μ -> Poisson(max(μ, eps(typeof(μ)))),
        deaths_obs  = (k, μ) -> Poisson(max(μ, eps(typeof(μ)))),
        cases_obs   = (k, μ) -> Poisson(max(μ, eps(typeof(μ)))))
    draw = model()
    @test draw.expected_deaths > 0
    @test draw.expected_reports > 0
    @test draw.expected_exports > 0
end

@testset "OnsetIncidence is built once per draw (source check)" begin
    ## Item 5 of the project-owner directive: confirm the composer
    ## constructs the onset-incidence curve once and reuses it across
    ## the deaths/reports/exports submodels. Read the composer source
    ## and assert `OnsetIncidence(` appears exactly once.
    src = read(joinpath(pkgdir(BVDOutbreakSize), "src",
                        "explicit_convolution_models.jl"),
               String)
    n = length(collect(eachmatch(r"OnsetIncidence\(", src)))
    ## One call site (`oi = OnsetIncidence(r, ..., T)`); the type's
    ## constructor signature (`function OnsetIncidence(`) lives in
    ## explicit_convolution.jl, not here.
    @test n == 1
end

@testset "explicit-convolution composer generates a prior draw" begin
    draw = _explicit_model(missing, missing, missing)()
    @test haskey(draw, :I_T)
    @test haskey(draw, :onsets_T)
    @test draw.onsets_T < draw.I_T          # onsets lag infections
    @test draw.expected_deaths > 0
    @test draw.expected_reports > 0
end

@testset "explicit-convolution composer: finite Mooncake gradient" begin
    obs = load_observations()
    model = _explicit_model(obs.exported_cases, obs.total_deaths,
                            obs.reported_cases;
                            tmrca = obs.genetic_tmrca_days)
    vi = link!!(VarInfo(model), model)
    θ0 = vi[:]
    ldf = LogDensityFunction(model, getlogjoint_internal, vi;
                             adtype = default_adtype())
    val, grad = LDP.logdensity_and_gradient(ldf, θ0)
    @test isfinite(val)
    @test all(isfinite, grad)
    @test length(grad) == LDP.dimension(ldf)
end

@testset "explicit-convolution composer runs a short NUTS smoke fit" begin
    obs = load_observations()
    model = _explicit_model(obs.exported_cases, obs.total_deaths,
                            obs.reported_cases;
                            tmrca = obs.genetic_tmrca_days)
    chn = nuts_sample(model; samples = 30, chains = 1,
                      target_accept = 0.8)
    @test chn !== nothing
    Ts = vec(Array(chn[:T]))
    @test length(Ts) == 30
    @test all(isfinite, Ts)
    @test all(>(0), Ts)
end
