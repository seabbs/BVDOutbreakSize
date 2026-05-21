## Accuracy of the onset-to-death convolution integrals when the delay
## density is narrow relative to the integration domain.
##
## The growth/delay model integrates a sampled `Gamma(α, θ)` delay over
## `[0, T]`, with `T = m·τ` reaching several hundred days. A fixed-node
## Gauss-Legendre rule spread across the whole `[0, T]` under-resolves a
## narrow delay (small θ): the nodes near `T`, where the convolution
## mass sits, are too coarse to capture the delay peak, so the integral
## drifts off the true value (>20% error in the prior tail). Bounding
## the domain to where the delay has mass (`T − (mean + K·std)`) clusters
## the nodes there and restores accuracy. These tests pin the value
## against an adaptive QuadGK reference for both a typical and a narrow
## delay.

import BVDOutbreakSize
using BVDOutbreakSize: expected_deaths
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

@testset "expected_deaths: accurate for a typical delay" begin
    CFR, r, T = 0.3, log(2) / 14, 100.0
    delay = Gamma(4.3, 2.6)
    val = expected_deaths(CFR, r, T, delay)
    ref = _expected_deaths_ref(CFR, r, T, delay)
    @test isapprox(val, ref; rtol = 1e-4)
end

@testset "expected_deaths: accurate for a narrow delay over wide T" begin
    # θ = 0.1 → delay std ≈ 0.21 d, peak ≈ 0.33 d before T; T = 360 d.
    # The non-adaptive rule spreads 64 nodes over 360 d and resolves the
    # delay peak near T too coarsely (~20% error). The adaptive rule
    # must still match the reference.
    CFR, r, T = 0.3, log(2) / 14, 360.0
    delay = Gamma(4.3, 0.1)
    val = expected_deaths(CFR, r, T, delay)
    ref = _expected_deaths_ref(CFR, r, T, delay)
    @test ref > 0
    @test isapprox(val, ref; rtol = 1e-3)
end

@testset "committed-deaths integrand: accurate for a narrow delay" begin
    # Same `[0, T]` survival-weighted convolution behind
    # predict_no_onward_deaths; extreme narrow delay at wide T.
    r, T, CFR = log(2) / 14, 360.0, 0.3
    α, θ = 4.3, 0.05
    val = BVDOutbreakSize._committed_deaths_one(r, T, α, θ, CFR)
    ref = _committed_deaths_ref(r, T, α, θ, CFR)
    @test ref > 0
    @test isapprox(val, ref; rtol = 1e-3)
end
