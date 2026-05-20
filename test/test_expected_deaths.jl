## Tests for the analytic Gamma method of `expected_deaths`.

using Distributions: Gamma
using Mooncake: Mooncake
using BVDOutbreakSize: expected_deaths

@testset "expected_deaths Gamma analytic matches integration" begin
    CFR, r, T, α, θ = 0.3, 0.05, 30.0, 4.3, 2.6
    dist = Gamma(α, θ)
    analytic = expected_deaths(CFR, r, T, dist)
    numerical = invoke(expected_deaths,
                       Tuple{Any, Any, Any, Any},
                       CFR, r, T, dist)
    @test analytic ≈ numerical rtol = 1e-6
end

@testset "Mooncake AD on expected_deaths Gamma method matches FD" begin
    CFR, r, T = 0.3, 0.05, 30.0
    f = (α, θ) -> expected_deaths(CFR, r, T, Gamma(α, θ))
    α, θ = 4.3, 2.6
    rule = Mooncake.build_rrule(f, α, θ)
    val, grads = Mooncake.value_and_gradient!!(rule, f, α, θ)
    h = 1e-5
    fd_α = (f(α + h, θ) - f(α - h, θ)) / (2h)
    fd_θ = (f(α, θ + h) - f(α, θ - h)) / (2h)
    @test val ≈ f(α, θ)
    @test grads[2] ≈ fd_α rtol = 1e-6
    @test grads[3] ≈ fd_θ rtol = 1e-6
end
