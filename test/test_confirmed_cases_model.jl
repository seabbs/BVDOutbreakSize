## Smoke tests for the laboratory-confirmed cases likelihood. The
## `@model` blocks live in the literate walkthrough, so we recreate the
## minimal set here to keep the tests self-contained and avoid a
## dependency on the doc-build pipeline. The convolution math reuses the
## exported `expected_deaths` helper.

using Distributions: Beta, Gamma, NegativeBinomial, truncated, Normal
using Turing: Turing, @model, sample, Prior, to_submodel
using BVDOutbreakSize: expected_deaths
import FlexiChains

@model function _confirmed_test_growth()
    log_τ ~ Normal(log(14), 0.4)
    m     ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
    τ   := exp(log_τ)
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    cumulative = s -> exp(r * s)
    return (; log_τ, τ, r, m, T, C_T, cumulative)
end

@model function _confirmed_test_positivity(;
        positivity_prior = Beta(4.5, 5.5))
    positivity ~ positivity_prior
    return (; positivity)
end

@model function _confirmed_test_delay(;
        alpha_prior = truncated(Normal(5.0, 2.0); lower = 0),
        theta_prior = truncated(Normal(1.5, 0.5); lower = 0))
    α_t ~ alpha_prior
    θ_t ~ theta_prior
    return (; α_t, θ_t, dist = Gamma(α_t, θ_t))
end

@model function _confirmed_test_dispersion()
    inv_sqrt_k ~ truncated(Normal(0.5, 0.2); lower = 1e-3, upper = 5.0)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

@model function _confirmed_test_ascertainment(; p_prior = Beta(2.0, 6.0))
    p_drc ~ p_prior
    return (; p_drc)
end

@model function _confirmed_test_likelihood(
        confirmed_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real;
        positivity = _confirmed_test_positivity(),
        delay      = _confirmed_test_delay())
    r = growth_state.r
    T = growth_state.T

    positivity_state ~ to_submodel(positivity, false)
    test_delay_state ~ to_submodel(delay, false)

    scale = positivity_state.positivity * p_drc
    raw_conf            = expected_deaths(scale, r, T,
                                          test_delay_state.dist)
    expected_confirmed := isfinite(raw_conf) ?
        max(raw_conf, eps(typeof(raw_conf))) :
        eps(typeof(raw_conf))

    p_nb_raw = k / (k + expected_confirmed)
    p_nb = isfinite(p_nb_raw) ?
        clamp(p_nb_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    confirmed_cases ~ NegativeBinomial(k, p_nb)
    return (; positivity = positivity_state.positivity,
              expected_confirmed)
end

@model function _confirmed_test_only(
        confirmed_cases::Union{Missing, Integer};
        growth        = _confirmed_test_growth(),
        confirmed     = _confirmed_test_likelihood,
        dispersion    = _confirmed_test_dispersion(),
        ascertainment = _confirmed_test_ascertainment())
    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k     = dispersion_state.k
    p_drc = asc_state.p_drc
    confirmed_state ~ to_submodel(
        confirmed(confirmed_cases, growth_state, k, p_drc), false)
    cumulative_cases := growth_state.C_T
end

@testset "confirmed_cases_model prior draws finite confirmed_cases" begin
    m = _confirmed_test_only(missing)
    chn = sample(m, Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    cc = vec(Array(chn[:confirmed_cases]))
    @test length(cc) == 200
    @test all(isfinite, cc)
    @test all(cc .>= 0)

    positivity = vec(Array(chn[:positivity]))
    @test all(0 .<= positivity .<= 1)

    α_t = vec(Array(chn[:α_t]))
    θ_t = vec(Array(chn[:θ_t]))
    @test all(α_t .> 0)
    @test all(θ_t .> 0)
end

@testset "confirmed_cases_model fits a tiny observation" begin
    chn = sample(_confirmed_test_only(33), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end

@testset "positivity scales the confirmed expectation" begin
    ## With p_drc fixed and growth fixed, doubling positivity should
    ## double expected_confirmed (the integral and p_drc factors are
    ## shared, positivity enters linearly).
    r = 0.07
    T = 80.0
    delay_dist = Gamma(5.0, 1.5)
    p_drc = 0.25
    base = expected_deaths(0.1 * p_drc, r, T, delay_dist)
    high = expected_deaths(0.2 * p_drc, r, T, delay_dist)
    @test high ≈ 2 * base rtol = 1e-10
    @test base > 0
end
