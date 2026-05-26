## Tests for the export-death expectation helper and the
## first-export-death survival term. The helper backs both the count
## likelihood (evaluated at the cut-off `T`) and the timing survival
## term (evaluated at an earlier elapsed time).

@testitem "expected_exports matches the at-risk person-time integral" begin
    using BVDOutbreakSize: expected_exports, integrate_cumulative
    r = 0.05
    cumulative = s -> exp(r * s)
    p_uganda = 0.25
    q = 1871 / 4_392_200
    w = 15.0
    T = 90.0

    got = expected_exports(cumulative, p_uganda, q, T, w)
    want = p_uganda * q * integrate_cumulative(cumulative, T - w, T)

    @test got ≈ want rtol = 1e-10
    @test got > 0
end

@testitem "expected_exports grows with elapsed time" begin
    using BVDOutbreakSize: expected_exports
    r = 0.05
    cumulative = s -> exp(r * s)
    f(t) = expected_exports(cumulative, 0.25, 1871 / 4_392_200, t, 15.0)
    @test f(60.0) < f(90.0) < f(120.0)
    @test f(0.0) > 0
end

@testitem "expected_exports_deaths matches the manual convolution" begin
    using Distributions: Gamma
    using BVDOutbreakSize: expected_exports_deaths, integrate_exports_deaths
    r = 0.05
    cumulative = s -> exp(r * s)
    delay_dist = Gamma(4.3, 2.6)
    CFR = 0.30
    p_uganda = 0.25
    q = 1871 / 4_392_200
    w = 15.0
    T = 90.0

    got = expected_exports_deaths(cumulative, delay_dist, CFR, p_uganda,
        q, T, w)

    ## Independent reconstruction from the documented integrand.
    integral = integrate_exports_deaths(cumulative, delay_dist,
        max(T - w, 0.0), T, T)
    want = CFR * p_uganda * q * integral

    @test got ≈ want rtol = 1e-10
    @test got > 0
end

@testitem "expected_exports_deaths grows with elapsed time" begin
    using Distributions: Gamma
    using BVDOutbreakSize: expected_exports_deaths
    r = 0.05
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

## Gamma-specialised dispatch of `integrate_exports_deaths` evaluates the
## onset-to-death CDF in closed form via `_gamma_cdf` with rrule for AD
## instead of an inner quadrature. Pin value-equivalence to the generic dispatch and confirm
## the path is still reverse-mode differentiable via Mooncake.

@testitem "integrate_exports_deaths Gamma dispatch matches generic" begin
    using Distributions: Gamma
    using BVDOutbreakSize: integrate_exports_deaths

    _GAMMA_DISPATCH_GRID_SMOOTH = ((4.3, 2.6), (4.3, 1.0),
        (2.0, 3.0), (8.0, 1.5))

    cumulative = s -> exp(0.05 * s)
    w, T = 15.0, 90.0
    lo = max(T - w, 0.0)

    for (α, θ) in _GAMMA_DISPATCH_GRID_SMOOTH
        delay_dist = Gamma(α, θ)
        analytic = integrate_exports_deaths(cumulative, delay_dist, lo, T, T)
        numerical = invoke(integrate_exports_deaths,
            Tuple{Any, Any, Any, Any, Any},
            cumulative, delay_dist, lo, T, T)
        @test analytic ≈ numerical rtol = 1e-6
    end
end

## For α < 1 the generic dispatch's inner pdf-quadrature hits the
## t^(α-1) singularity at 0 and is no longer a trustworthy reference.
## So we test the AD path for the gamma dispatch at α < 1 against
## finite differences of the analytic form, against the full grid of
## (α, θ) values.

@testitem "integrate_exports_deaths Gamma dispatch has correct gradients" tags=[:ad] begin
    using Distributions: Gamma
    using FiniteDifferences: central_fdm, grad
    using Mooncake: Mooncake
    using BVDOutbreakSize: integrate_exports_deaths

    _GAMMA_DISPATCH_GRID = ((4.3, 2.6), (4.3, 1.0),
        (2.0, 3.0), (8.0, 1.5),
        (0.3, 2.6), (0.5, 1.0), (0.5, 3.0))

    cumulative = s -> exp(0.05 * s)
    w, T = 15.0, 90.0
    lo = max(T - w, 0.0)

    fast(x) = integrate_exports_deaths(
        cumulative, Gamma(x[1], x[2]), lo, T, T)

    cache = Mooncake.prepare_gradient_cache(fast, [1.0, 1.0])
    _mooncake_grad(x) = Mooncake.value_and_gradient!!(cache, fast, x)[2][2]
    _fd_grad(x) = grad(central_fdm(5, 1), fast, x)[1]

    for (α, θ) in _GAMMA_DISPATCH_GRID
        x = [α, θ]
        gf = _mooncake_grad(x)
        gd = _fd_grad(x)
        @test all(isfinite, gf)
        @test gf ≈ gd rtol = 1e-5
    end
end
