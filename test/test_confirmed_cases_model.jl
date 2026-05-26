## Smoke tests for the truth-anchored cases and confirmed-cases
## likelihoods. C(T) is the latent eventually-confirmed pool; reported
## counts are inflated above it by 1/π and convolved through the
## infection-to-report delay; confirmed counts anchor it via the
## infection-to-confirmation delay f_conf = f_rep ∗ f_lab. The `@model`
## blocks live in the literate walkthrough, so we recreate the minimal
## set here.

using Distributions: Beta, Gamma, NegativeBinomial, truncated, Normal,
                     mean, var
using Turing: Turing, @model, sample, Prior, to_submodel
import FlexiChains
using BVDOutbreakSize: expected_deaths

function _safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

## Moment-matched Gamma mirroring `_convolved_gamma` in analysis.jl.
function _conv_gamma(d_rep::Gamma, d_lab::Gamma)
    μ  = mean(d_rep) + mean(d_lab)
    σ² = var(d_rep)  + var(d_lab)
    θ  = σ² / μ
    α  = μ  / θ
    return Gamma(α, θ)
end

@model function _confirmed_test_growth()
    log_τ ~ Normal(log(14), 0.4)
    m     ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
    τ   := exp(log_τ)
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    return (; log_τ, τ, r, m, T, C_T)
end

@model function _confirmed_test_background(;
        lambda_prior = truncated(Normal(0.0, 10.0); lower = 0))
    λ_bg ~ lambda_prior
    return (; λ_bg)
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

@model function _confirmed_test_report_delay()
    α_rep ~ truncated(Normal(4.0, 1.5); lower = 0)
    θ_rep ~ truncated(Normal(3.0, 1.0); lower = 0)
    return (; α = α_rep, θ = θ_rep, dist = Gamma(α_rep, θ_rep))
end

@model function _confirmed_test_lab_delay()
    α_lab ~ truncated(Normal(2.0, 1.0); lower = 0)
    θ_lab ~ truncated(Normal(1.5, 0.75); lower = 0)
    return (; α = α_lab, θ = θ_lab, dist = Gamma(α_lab, θ_lab))
end

@model function _confirmed_test_reported(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real,
        λ_bg::Real, f_rep::Gamma)
    r = growth_state.r
    T = growth_state.T
    conv_rep = expected_deaths(one(p_drc), r, T, f_rep)
    μ_BVD_raw = p_drc * conv_rep
    μ_bg_raw  = λ_bg * T
    μ_BVD := isfinite(μ_BVD_raw) ?
        max(μ_BVD_raw, eps(typeof(μ_BVD_raw))) :
        eps(typeof(μ_BVD_raw))
    μ_bg  := isfinite(μ_bg_raw) ?
        max(μ_bg_raw, eps(typeof(μ_bg_raw))) :
        eps(typeof(μ_bg_raw))
    expected_reports := μ_BVD + μ_bg
    positivity := μ_BVD / expected_reports
    reported_cases ~ _safe_nbinomial(k, expected_reports)
    return (; expected_reports, reported_cases, positivity)
end

@model function _confirmed_test_confirmed(
        confirmed_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real, f_rep::Gamma, f_lab::Gamma)
    f_conf = _conv_gamma(f_rep, f_lab)
    r = growth_state.r
    T = growth_state.T
    conv_conf = expected_deaths(one(p_drc), r, T, f_conf)
    raw_confirmed = p_drc * conv_conf
    expected_confirmed := isfinite(raw_confirmed) ?
        max(raw_confirmed, eps(typeof(raw_confirmed))) :
        eps(typeof(raw_confirmed))
    confirmed_cases ~ _safe_nbinomial(k, expected_confirmed)
    return (; expected_confirmed)
end

@model function _confirmed_test_only(
        reported_cases::Union{Missing, Integer},
        confirmed_cases::Union{Missing, Integer};
        growth        = _confirmed_test_growth(),
        background    = _confirmed_test_background(),
        report_delay  = _confirmed_test_report_delay(),
        lab_delay     = _confirmed_test_lab_delay(),
        dispersion    = _confirmed_test_dispersion(),
        ascertainment = _confirmed_test_ascertainment())
    growth_state     ~ to_submodel(growth, false)
    background_state ~ to_submodel(background, false)
    report_state     ~ to_submodel(report_delay, false)
    lab_state        ~ to_submodel(lab_delay, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k     = dispersion_state.k
    p_drc = asc_state.p_drc
    λ_bg  = background_state.λ_bg
    f_rep = report_state.dist
    f_lab = lab_state.dist
    reported_state ~ to_submodel(
        _confirmed_test_reported(
            reported_cases, growth_state, k, p_drc, λ_bg, f_rep), false)
    confirmed_state ~ to_submodel(
        _confirmed_test_confirmed(
            confirmed_cases, growth_state, k, p_drc, f_rep, f_lab), false)
    cumulative_cases := growth_state.C_T
end

@testset "truth-anchored streams: prior draws are finite and non-negative" begin
    m = _confirmed_test_only(missing, missing)
    chn = sample(m, Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    rc = vec(Array(chn[:reported_cases]))
    cc = vec(Array(chn[:confirmed_cases]))
    @test length(cc) == 200
    @test all(isfinite, rc)
    @test all(isfinite, cc)
    @test all(rc .>= 0)
    @test all(cc .>= 0)

    π = vec(Array(chn[:positivity]))
    @test all(0 .<= π .<= 1)
end

@testset "truth-anchored streams: fit with both observations" begin
    chn = sample(_confirmed_test_only(516, 33), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end

@testset "convolved Gamma matches the moments of the sum" begin
    ## E[X+Y] = E[X] + E[Y] and Var[X+Y] = Var[X] + Var[Y] by
    ## independence; `_conv_gamma` matches both by construction.
    d_rep = Gamma(4.0, 3.0)
    d_lab = Gamma(2.0, 1.5)
    d_conf = _conv_gamma(d_rep, d_lab)
    @test mean(d_conf) ≈ mean(d_rep) + mean(d_lab)
    @test var(d_conf)  ≈ var(d_rep)  + var(d_lab)
    ## Same-rate special case: the convolution is exactly Gamma.
    d_same_rep = Gamma(4.0, 2.5)
    d_same_lab = Gamma(2.0, 2.5)
    d_same_conf = _conv_gamma(d_same_rep, d_same_lab)
    @test d_same_conf.α ≈ 6.0
    @test d_same_conf.θ ≈ 2.5
end
