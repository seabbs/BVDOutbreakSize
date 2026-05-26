## Accuracy of the onset-to-death convolution integrals when the delay
## density is narrow relative to the integration domain.
##
## The growth/delay model integrates a sampled `Gamma(α, θ)` delay over
## `[0, T]`, with `T = m·τ` reaching several hundred days. A uniform
## fixed-node Gauss-Legendre rule spread across the whole `[0, T]`
## under-resolves a narrow delay (small θ): the nodes near `T`, where the
## convolution mass sits, are too coarse to capture the delay peak, so
## the integral drifts off the true value (>20% error in the prior tail).
## The clustered `integrate(f, lo, hi, scale)` method packs the nodes
## towards `T` over a window set by the delay scale, restoring accuracy
## without dropping any of the domain. These tests pin the value against
## an adaptive QuadGK reference (over the full `[0, T]`) for both a
## typical and a narrow delay.

@testsnippet IntegrateRefs begin
    using Distributions: Gamma, ccdf, pdf
    using Integrals: IntegralProblem, QuadGKJL, solve

    function _expected_deaths_ref(CFR, r, T, delay_dist)
        f(s, p) = (T - s) > 0 ? exp(r * s) * pdf(delay_dist, T - s) : 0.0
        prob = IntegralProblem(f, (0.0, T), nothing)
        return CFR * solve(prob, QuadGKJL(); reltol = 1e-12, abstol = 1e-14).u
    end

    function _committed_deaths_ref(r, T, α, θ, CFR)
        d = Gamma(α, θ)
        f(s, p) = r * exp(r * s) * ccdf(d, T - s)
        prob = IntegralProblem(f, (0.0, T), nothing)
        return CFR * solve(prob, QuadGKJL(); reltol = 1e-12, abstol = 1e-14).u
    end
end

@testitem "clustered integrate: covers the full domain" begin
    using BVDOutbreakSize: integrate
    # A smooth integrand with no endpoint peak: clustering must not bias
    # the result, and a scale wider than the domain must reproduce the
    # uniform integrator exactly.
    f = s -> exp(0.01 * s)
    exact = (exp(0.01 * 50.0) - exp(0.01 * 10.0)) / 0.01
    @test isapprox(integrate(f, 10.0, 50.0, 5.0), exact; rtol = 1e-8)
    @test isapprox(integrate(f, 10.0, 50.0, 1.0e6),
                   integrate(f, 10.0, 50.0); rtol = 1e-10)
    @test integrate(f, 50.0, 50.0, 5.0) == 0.0
end

@testitem "expected_deaths: accurate for a typical delay" setup=[IntegrateRefs] begin
    using Distributions: Gamma
    using BVDOutbreakSize: expected_deaths
    CFR, r, T = 0.3, log(2) / 14, 100.0
    delay = Gamma(4.3, 2.6)
    val = expected_deaths(CFR, r, T, delay)
    ref = _expected_deaths_ref(CFR, r, T, delay)
    @test isapprox(val, ref; rtol = 1e-4)
end

@testitem "expected_deaths: accurate for a narrow delay over wide T" setup=[IntegrateRefs] begin
    using Distributions: Gamma
    using BVDOutbreakSize: expected_deaths
    # θ = 0.1 → delay std ≈ 0.21 d, peak ≈ 0.33 d before T; T = 360 d.
    # A uniform rule spreads 64 nodes over 360 d and resolves the delay
    # peak near T too coarsely (~20% error). The clustered rule must
    # still match the reference.
    CFR, r, T = 0.3, log(2) / 14, 360.0
    delay = Gamma(4.3, 0.1)
    val = expected_deaths(CFR, r, T, delay)
    ref = _expected_deaths_ref(CFR, r, T, delay)
    @test ref > 0
    @test isapprox(val, ref; rtol = 1e-3)
end

@testitem "committed-deaths integrand: accurate for a narrow delay" setup=[IntegrateRefs] begin
    using BVDOutbreakSize
    # Same `[0, T]` survival-weighted convolution behind
    # predict_no_onward_deaths; extreme narrow delay at wide T.
    r, T, CFR = log(2) / 14, 360.0, 0.3
    α, θ = 4.3, 0.05
    val = BVDOutbreakSize._committed_deaths_one(r, T, α, θ, CFR)
    ref = _committed_deaths_ref(r, T, α, θ, CFR)
    @test ref > 0
    @test isapprox(val, ref; rtol = 1e-3)
end
