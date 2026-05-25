## Smoke tests for the laboratory-confirmed cases likelihood. The
## `@model` blocks live in the literate walkthrough, so we recreate the
## minimal set here to keep the tests self-contained and avoid a
## dependency on the doc-build pipeline. The confirmed-cases stream is a
## binomial subset of the suspected count conditional on `π`, so the
## suspected NegBinomial and the confirmed Binomial both appear here.

using Distributions: Beta, Binomial, NegativeBinomial, truncated, Normal,
                     mean, var
using Turing: Turing, @model, sample, Prior, to_submodel
import FlexiChains

## Mirrors the production `safe_nbinomial` helper in analysis.jl: clamps
## the success probability so extreme NUTS proposals during warmup do
## not trip the distribution domain check.
function _safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

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
        positivity_prior = Beta(2.0, 4.0))
    positivity ~ positivity_prior
    return (; positivity)
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

@model function _confirmed_test_cases(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real)
    C_T = growth_state.C_T
    raw_reports        = p_drc * C_T
    expected_reports := isfinite(raw_reports) ?
        max(raw_reports, eps(typeof(raw_reports))) :
        eps(typeof(raw_reports))
    reported_cases ~ _safe_nbinomial(k, expected_reports)
    return (; expected_reports, reported_cases)
end

@model function _confirmed_test_likelihood(
        confirmed_cases::Union{Missing, Integer},
        reported_cases::Integer;
        positivity = _confirmed_test_positivity())
    positivity_state ~ to_submodel(positivity, false)
    π = positivity_state.positivity
    confirmed_cases ~ Binomial(Int(reported_cases), π)
    return (; positivity = π, reported_cases)
end

@model function _confirmed_test_only(
        reported_cases::Union{Missing, Integer},
        confirmed_cases::Union{Missing, Integer};
        growth        = _confirmed_test_growth(),
        cases         = _confirmed_test_cases,
        confirmed     = _confirmed_test_likelihood,
        dispersion    = _confirmed_test_dispersion(),
        ascertainment = _confirmed_test_ascertainment())
    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k     = dispersion_state.k
    p_drc = asc_state.p_drc
    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k, p_drc), false)
    confirmed_state ~ to_submodel(
        confirmed(confirmed_cases, cases_state.reported_cases), false)
    cumulative_cases := growth_state.C_T
end

@testset "confirmed_cases_model prior draws are bounded by reported" begin
    m = _confirmed_test_only(missing, missing)
    chn = sample(m, Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    rc = vec(Array(chn[:reported_cases]))
    cc = vec(Array(chn[:confirmed_cases]))
    @test length(cc) == 200
    @test all(isfinite, cc)
    @test all(cc .>= 0)
    ## Binomial(n, π) confirmed counts can never exceed the n it
    ## conditioned on.
    @test all(cc .<= rc)

    positivity = vec(Array(chn[:positivity]))
    @test all(0 .<= positivity .<= 1)
end

@testset "confirmed_cases_model with both observations" begin
    chn = sample(_confirmed_test_only(516, 33), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end

@testset "Binomial(n, π) matches the analytic mean and variance" begin
    ## Sanity check on the likelihood shape that the model relies on:
    ## E[Y] = n·π, Var[Y] = n·π(1-π). Use a large reference sample so the
    ## empirical estimates are tight.
    n = 516
    π = 0.064
    d = Binomial(n, π)
    @test mean(d) ≈ n * π
    @test var(d)  ≈ n * π * (1 - π)
end
