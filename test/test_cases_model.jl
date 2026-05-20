## Smoke tests for the cases ascertainment likelihood. The `@model`
## blocks live in the literate walkthrough, so we recreate the minimal
## set here to keep the tests self-contained and avoid a dependency on
## the doc-build pipeline.

@model function _cases_test_growth()
    log_τ ~ Normal(log(14), 0.4)
    m     ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
    τ   := exp(log_τ)
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    cumulative = s -> exp(r * s)
    return (; log_τ, τ, r, m, T, C_T, cumulative)
end

@model function _cases_test_ascertainment(; p_prior = Beta(2.0, 6.0))
    p_report ~ p_prior
    return (; p_report)
end

@model function _cases_test_dispersion()
    inv_sqrt_k ~ truncated(Normal(0.5, 0.2); lower = 1e-3, upper = 5.0)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

@model function _cases_test_likelihood(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real;
        ascertainment = _cases_test_ascertainment())
    C_T = growth_state.C_T
    asc_state ~ to_submodel(ascertainment, false)
    p_report  = asc_state.p_report
    raw_reports        = p_report * C_T
    expected_reports := isfinite(raw_reports) ?
        max(raw_reports, eps(typeof(raw_reports))) :
        eps(typeof(raw_reports))
    p_nb_raw = k / (k + expected_reports)
    p_nb = isfinite(p_nb_raw) ?
        clamp(p_nb_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    reported_cases ~ NegativeBinomial(k, p_nb)
    return (; p_report, expected_reports)
end

@model function _cases_test_only(
        reported_cases::Union{Missing, Integer};
        growth     = _cases_test_growth(),
        cases      = _cases_test_likelihood,
        dispersion = _cases_test_dispersion())
    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    k = dispersion_state.k
    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k), false)
    cumulative_cases := growth_state.C_T
end

@testset "cases_model prior draws finite reported_cases" begin
    m = _cases_test_only(missing)
    chn = sample(m, Prior(), 200;
                 chain_type = MCMCChains.Chains, progress = false)
    rc = vec(Array(chn[:reported_cases]))
    @test length(rc) == 200
    @test all(isfinite, rc)
    @test all(rc .>= 0)

    p_report = vec(Array(chn[:p_report]))
    @test all(0 .<= p_report .<= 1)
end

@testset "cases_only_model fits a tiny observation" begin
    chn = sample(_cases_test_only(50), Prior(), 200;
                 chain_type = MCMCChains.Chains, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end
