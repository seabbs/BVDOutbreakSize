## Tests for the export-death expectation helper and the
## first-export-death survival term. The helper backs both the count
## likelihood (evaluated at the cut-off `T`) and the timing survival
## term (evaluated at an earlier elapsed time).

using BVDOutbreakSize: expected_exports, expected_exports_deaths,
                       integrate_cumulative, integrate_exports_deaths,
                       CUMULATIVE_INTEGRAL_ALG
using Distributions: Gamma

@testset "expected_exports matches the at-risk person-time integral" begin
    r          = 0.05
    cumulative = s -> exp(r * s)
    p_uganda   = 0.25
    q          = 1871 / 4_392_200
    w          = 15.0
    T          = 90.0

    got  = expected_exports(cumulative, p_uganda, q, T, w)
    want = p_uganda * q * integrate_cumulative(cumulative, T - w, T)

    @test got ≈ want rtol = 1e-10
    @test got > 0
end

@testset "expected_exports grows with elapsed time" begin
    r          = 0.05
    cumulative = s -> exp(r * s)
    f(t) = expected_exports(cumulative, 0.25, 1871 / 4_392_200, t, 15.0)
    @test f(60.0) < f(90.0) < f(120.0)
    @test f(0.0) > 0
end

@testset "expected_exports_deaths matches the manual convolution" begin
    r          = 0.05
    cumulative = s -> exp(r * s)
    delay_dist = Gamma(4.3, 2.6)
    CFR        = 0.30
    p_uganda   = 0.25
    q          = 1871 / 4_392_200
    w          = 15.0
    T          = 90.0

    got = expected_exports_deaths(cumulative, delay_dist, CFR, p_uganda,
                                  q, T, w)

    ## Independent reconstruction from the documented integrand.
    integral = integrate_exports_deaths(cumulative, delay_dist,
                                        max(T - w, 0.0), T, T)
    want = CFR * p_uganda * q * integral

    @test got ≈ want rtol = 1e-10
    @test got > 0
end

@testset "expected_exports_deaths grows with elapsed time" begin
    r          = 0.05
    cumulative = s -> exp(r * s)
    delay_dist = Gamma(4.3, 2.6)
    f(t) = expected_exports_deaths(cumulative, delay_dist, 0.30, 0.25,
                                   1871 / 4_392_200, t, 15.0)
    ## More elapsed time => more at-risk export person-time => more
    ## expected export deaths. This monotonicity is what lets the
    ## survival term, exp(-E[D_u(t1)]), bound the elapsed time.
    @test f(60.0) < f(90.0) < f(120.0)
    ## Always strictly positive (clamped), even before the window opens.
    @test f(0.0) > 0
end
