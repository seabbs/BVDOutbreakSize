## Tests for the analytic Gamma method of `delay_convolution`.


# Reference parameter values. α, θ match the Gamma onset-to-death
# prior means; CFR, r, T are mid-run state. x_cdf is the CDF argument
# that delay_convolution(::Gamma) actually feeds to `_gamma_cdf`,
# i.e. T*(1 + θ*r) — testing at this point keeps the AD checks
# aligned with the path the sampler will exercise.
#
# For α = 4.3, we do not have numerical stability issues in the α-derivative,
# see below for stronger tests on just the analytic form.
let α = 4.3, θ = 2.6, CFR = 0.3, r = 0.05, T = 30.0,
    x_cdf = T * (1 + θ * r)

    @testset "delay_convolution Gamma analytic matches integration" begin
        dist = Gamma(α, θ)
        analytic = delay_convolution(CFR, r, T, dist)
        numerical = invoke(delay_convolution,
                           Tuple{Any, Any, Any, Any},
                           CFR, r, T, dist) #avoid the analytic method dispatch
        @test analytic ≈ numerical rtol = 1e-6
    end

    # Found type-stability issue in `_gamma_cdf` and left here
    # to check against future regressions.
    test_opt(BVDOutbreakSize._gamma_cdf, (Float64, Float64, Float64);
             target_modules = (BVDOutbreakSize,))

    Mooncake.TestUtils.test_rule(
        MersenneTwister(20260520),
        BVDOutbreakSize._gamma_cdf, α, θ, x_cdf;
        is_primitive = true,
        perf_flag    = :none,
        mode         = Mooncake.ReverseMode,
    )
end

# Multi-α correctness test for the α-derivative from Stan. The
# derivative ∂_α P(α, z) is numerically delicate across the α range
# so we mirror Stan's grad_reg_inc_gamma_test.cpp grid — (α, z)
# points spanning z << α, z ≈ α, and z >> α — plus extra α < 1 cases that
# Stan skips but NUTS trajectories can reach during warmup. Tolerances follow Stan:
# atol = 1e-8 by default, loosened to 4e-7 at the z ≈ α crossover for
# α ≥ 5 (Stan's `(9, 10)` case). Each case is run through
# `Mooncake.TestUtils.test_rule`, which compares the lifted rrule
# against Richardson-extrapolated finite differences.
let θ = 1.0
    cases = [
        # (α,    z,      atol)
        (0.3,  13.04, 1e-8),  # small α, deep tail — NUTS warmup risk
        (0.5,  1.0,   1e-8),  # small α
        (0.5,  5.0,   1e-8),
        (1.1,  0.2,   1e-8),  # Stan grid
        (1.1,  2.0,   1e-8),
        (2.5,  1.3,   1e-8),
        (2.5,  30.0,  1e-8),  # Stan tail
        (9.0,  10.0,  4e-7),  # Stan crossover (loosened)
        (10.0, 9.0,   1e-8),
        (25.0, 13.04, 1e-8),
    ]
    for (α, z, atol) in cases
        x = z * θ
        Mooncake.TestUtils.test_rule(
            MersenneTwister(20260520),
            BVDOutbreakSize._gamma_cdf, α, θ, x;
            is_primitive = true,
            perf_flag    = :none,
            mode         = Mooncake.ReverseMode,
            atol         = atol,
            rtol         = atol,
        )
    end
end
