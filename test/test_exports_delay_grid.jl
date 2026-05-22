## Regression tests for the precomputed onset-to-death CDF
## (`ExportDeathDelay`): lock the gridded fast path to the distribution
## reference in value and gradient, and pin the gradient finite (the build
## must avoid the density at 0, whose Gamma shape derivative is NaN).

import BVDOutbreakSize
using BVDOutbreakSize: ExportDeathDelay, integrate_exports_deaths,
                       expected_exports_deaths
using Distributions: Gamma
using Mooncake: Mooncake

@testset "ExportDeathDelay matches the distribution integral" begin
    cumulative = s -> exp(0.05 * s)
    w, T       = 15.0, 90.0
    lo         = max(T - w, 0.0)
    for (α, θ) in ((4.3, 2.6), (4.3, 1.0), (2.0, 3.0), (8.0, 1.5))
        dist = Gamma(α, θ)
        ed   = ExportDeathDelay(dist, w)
        ref  = integrate_exports_deaths(cumulative, dist, lo, T, T)
        fast = integrate_exports_deaths(cumulative, ed,   lo, T, T)
        @test fast ≈ ref rtol = 1e-4
    end
end

@testset "ExportDeathDelay dispatches through expected_exports_deaths" begin
    cumulative = s -> exp(0.05 * s)
    dist       = Gamma(4.3, 2.6)
    CFR, p, q  = 0.30, 0.25, 1871 / 4_392_200
    w, T       = 15.0, 90.0
    ed   = ExportDeathDelay(dist, w)
    ref  = expected_exports_deaths(cumulative, dist, CFR, p, q, T, w)
    fast = expected_exports_deaths(cumulative, ed,   CFR, p, q, T, w)
    @test fast ≈ ref rtol = 1e-4
    @test fast > 0
end

@testset "ExportDeathDelay CDF is monotone and bounded" begin
    ed = ExportDeathDelay(Gamma(4.3, 2.6), 20.0)
    F  = [BVDOutbreakSize._cdf_to(ed, y) for y in 0.0:0.5:25.0]
    @test all(diff(F) .>= -1e-12)            # non-decreasing
    @test BVDOutbreakSize._cdf_to(ed, -1.0) == 0.0
    @test F[end] < 1.0                       # CDF below 1 within window
end

@testset "ExportDeathDelay gradient is finite and matches the reference" begin
    cumulative = s -> exp(0.05 * s)
    w, T = 15.0, 90.0
    lo   = max(T - w, 0.0)

    ref_fun(x)  = integrate_exports_deaths(
        cumulative, Gamma(x[1], x[2]), lo, T, T)
    fast_fun(x) = integrate_exports_deaths(
        cumulative, ExportDeathDelay(Gamma(x[1], x[2]), w), lo, T, T)

    grad(f, x) = begin
        cache = Mooncake.prepare_gradient_cache(f, x)
        _, gs = Mooncake.value_and_gradient!!(cache, f, x)
        gs[2]
    end

    x  = [4.3, 2.6]
    gr = grad(ref_fun, x)
    gf = grad(fast_fun, x)
    @test all(isfinite, gf)
    @test gf ≈ gr rtol = 1e-3

    ## Shape < 1 (singular density at 0) must still give a finite gradient.
    gf_lo = grad(fast_fun, [0.8, 3.0])
    @test all(isfinite, gf_lo)
end
