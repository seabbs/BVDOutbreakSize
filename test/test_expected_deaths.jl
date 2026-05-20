## Tests for the analytic Gamma method of `expected_deaths`.

using ChainRulesTestUtils: test_rrule
using JET: test_opt
import BVDOutbreakSize

# Reference parameter values. α, θ match the Gamma onset-to-death
# prior means; CFR, r, T are mid-run state. x_cdf is the CDF argument
# that expected_deaths(::Gamma) actually feeds to `_gamma_cdf`,
# i.e. T*(1 + θ*r) — testing at this point keeps the AD checks
# aligned with the path the sampler will exercise.
let α = 4.3, θ = 2.6, CFR = 0.3, r = 0.05, T = 30.0,
    x_cdf = T * (1 + θ * r)

    @testset "expected_deaths Gamma analytic matches integration" begin
        dist = Gamma(α, θ)
        analytic = expected_deaths(CFR, r, T, dist)
        numerical = invoke(expected_deaths,
                           Tuple{Any, Any, Any, Any},
                           CFR, r, T, dist) #avoid the analytic method dispatch
        @test analytic ≈ numerical rtol = 1e-6
    end

    # This is superceded by Mooncake.TestUtils.test_rule since
    # that tests the macro converted rrule!! used by Mooncake
    # But we keep it here for now
    test_rrule(BVDOutbreakSize._gamma_cdf, α, θ, x_cdf; rtol = 1e-7)

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
