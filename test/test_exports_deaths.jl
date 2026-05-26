## Smoke tests for the deaths-among-exports likelihood.
## kept: the real `exports_deaths_model` is time-resolved (per-day
## Poisson over a daily vector) and exposes neither `:exports_deaths`
## nor `:expected_exports_deaths_T`; the helpers below condense the
## kernel to a single scalar-count likelihood with the same expected
## value at the cut-off, which is what these smoke tests assert against.

@testsnippet ExportsDeathsFixtures begin
    using Distributions: Gamma, Normal, Poisson, Beta, pdf, truncated
    using Integrals: IntegralProblem, GaussLegendre, solve
    using StatsFuns: logit, logistic
    using Turing: @model, to_submodel

    const _XD_ALG = GaussLegendre(; n = 32)

    function _xd_cdf_integrand(u, p)
        v = p.inner_halfwidth * (u + 1)
        return pdf(p.delay_dist, v)
    end

    function _xd_cdf(delay_dist, upper)
        upper <= zero(upper) && return zero(upper)
        inner_halfwidth = upper / 2
        prob = IntegralProblem(_xd_cdf_integrand, (-1.0, 1.0),
                               (; delay_dist, inner_halfwidth))
        return inner_halfwidth * solve(prob, _XD_ALG).u
    end

    function _xd_integrand(u, p)
        s = p.halfwidth * (u + 1) + p.lower
        return p.cumulative(s) * _xd_cdf(p.delay_dist, p.T - s)
    end

    function _xd_expected(growth_state, CFR, delay_dist, p_uganda,
            w, daily_travellers, source_population)
        cumulative = growth_state.cumulative
        T          = growth_state.T
        lower      = max(T - w, zero(T))
        upper      = T
        upper <= lower && return zero(T) + eps(typeof(T))
        halfwidth = (upper - lower) / 2
        params = (; cumulative, halfwidth, lower, T, delay_dist)
        prob = IntegralProblem(_xd_integrand, (-1.0, 1.0), params)
        integral_value = halfwidth * solve(prob, _XD_ALG).u
        q = daily_travellers / source_population
        return CFR * q * p_uganda * integral_value
    end

    @model function _xd_growth()
        log_τ ~ Normal(log(14), 0.4)
        m     ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
        τ   := exp(log_τ)
        r   := log(2) / τ
        T   := m * τ
        C_T := 2.0 ^ m
        cumulative = s -> exp(r * s)
        return (; log_τ, τ, r, m, T, C_T, cumulative)
    end

    @model function _xd_pooled(;
            mu_prior  = Normal(logit(0.25), 1.0),
            tau_prior = truncated(Normal(0.0, 0.5); lower = 1e-4))
        μ_logit  ~ mu_prior
        τ_logit  ~ tau_prior
        z_drc    ~ Normal(0, 1)
        z_uganda ~ Normal(0, 1)
        logit_p_drc    = μ_logit + τ_logit * z_drc
        logit_p_uganda = μ_logit + τ_logit * z_uganda
        p_drc    := logistic(logit_p_drc)
        p_uganda := logistic(logit_p_uganda)
        return (; μ_logit, τ_logit, p_drc, p_uganda)
    end

    @model function _xd_delay()
        α ~ truncated(Normal(4.3, 1.22); lower = 0)
        θ ~ truncated(Normal(2.6, 0.82); lower = 0)
        return (; α, θ, dist = Gamma(α, θ))
    end

    @model function _xd_cfr()
        CFR ~ Beta(6.0, 14.0)
        return (; CFR)
    end

    @model function _xd_window()
        w ~ truncated(Normal(15.0, 5.0); lower = 0)
        return (; w)
    end

    @model function _xd_likelihood(
            exports_deaths::Union{Missing, Integer},
            growth_state, CFR, delay_dist, p_uganda,
            w, daily_travellers; source_population = 4_392_200)
        raw = _xd_expected(growth_state, CFR, delay_dist, p_uganda,
                           w, daily_travellers, source_population)
        expected_exports_deaths_T := isfinite(raw) ?
            max(raw, eps(typeof(raw))) : eps(typeof(raw))
        exports_deaths ~ Poisson(expected_exports_deaths_T)
        return (; expected_exports_deaths_T)
    end

    @model function _xd_only(exports_deaths::Union{Missing, Integer})
        growth_state  ~ to_submodel(_xd_growth(), false)
        asc_state     ~ to_submodel(_xd_pooled(), false)
        delay_state   ~ to_submodel(_xd_delay(), false)
        cfr_state     ~ to_submodel(_xd_cfr(), false)
        window_state  ~ to_submodel(_xd_window(), false)
        daily_travellers ~ truncated(Normal(1871.0, 200.0); lower = 0)

        likelihood_state ~ to_submodel(
            _xd_likelihood(exports_deaths, growth_state,
                           cfr_state.CFR, delay_state.dist,
                           asc_state.p_uganda, window_state.w,
                           daily_travellers),
            false)

        cumulative_cases := growth_state.C_T
    end
end

@testitem "exports_deaths_model prior draws produce non-negative counts" tags=[:slow] setup=[ExportsDeathsFixtures] begin
    using Turing: sample, Prior
    import FlexiChains
    m = _xd_only(missing)
    chn = sample(m, Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    xd = vec(Array(chn[:exports_deaths]))
    @test length(xd) == 200
    @test all(isfinite, xd)
    @test all(xd .>= 0)

    expected = vec(Array(chn[:expected_exports_deaths_T]))
    @test all(expected .> 0)
    @test all(isfinite, expected)
end

@testitem "exports_deaths_only fits a zero observation" tags=[:slow] setup=[ExportsDeathsFixtures] begin
    using Turing: sample, Prior
    import FlexiChains
    chn = sample(_xd_only(0), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end
