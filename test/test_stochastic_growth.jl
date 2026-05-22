## Tests for the stochastic latent growth prototype (issue #48):
## the log-trajectory builder, the growth submodel prior draws, and the
## onset-timing bound. These exercise the new package-level pieces; the
## full joint composer is demonstrated in
## scripts/prototype_stochastic.jl (it depends on observation stand-ins,
## which by repo convention live outside the package; issue #81).

using BVDOutbreakSize: lna_logC, lna_trajectory,
                       stochastic_growth_model, onset_timing_model
using Distributions: Normal
using Turing: Turing, @model, sample, Prior, to_submodel
import FlexiChains

@testset "lna_logC recovers exponential growth when σ = 0" begin
    r = log(2) / 14
    T = 98.0
    z = zeros(23)
    logC = lna_logC(r, T, 0.0, z)
    @test length(logC) == 24
    @test logC[1] == 0.0
    ## With no noise the endpoint is exactly r·T (m = T/τ doublings).
    @test isapprox(logC[end], r * T; atol = 1e-10)
    @test exp(logC[end]) ≈ 2.0^(T / 14)
end

@testset "lna_logC injects variance when σ > 0" begin
    r = log(2) / 14
    T = 98.0
    z = [1.0; zeros(22)]
    logC = lna_logC(r, T, 0.5, z)
    ## A single positive shock lifts every level after the first knot.
    @test logC[3] > r * (2 * T / 23)
end

@testset "lna_trajectory is positive, continuous and flat outside" begin
    logC = lna_logC(log(2) / 14, 98.0, 0.0, zeros(23))
    C = lna_trajectory(logC, 98.0)
    @test C(0.0) == 1.0
    @test C(-5.0) == C(0.0)            # flat below 0
    @test C(200.0) == C(98.0)          # flat above T
    @test C(49.0) > C(0.0)             # increasing under growth
    @test all(C(s) > 0 for s in 0:10:98)
end

@testset "stochastic_growth_model prior draws are finite and > 0" begin
    chn = sample(stochastic_growth_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    CT = vec(Array(chn[:C_T]))
    @test length(CT) == 200
    @test all(isfinite, CT)
    @test all(CT .> 0)
    σ = vec(Array(chn[:σ]))
    @test all(σ .>= 0)
end

@testset "onset_timing_model is a no-op when delta is missing" begin
    @model function _onset_wrap(; onset_delta)
        T ~ Normal(100.0, 10.0)
        s ~ to_submodel(
            onset_timing_model(T; onset_delta = onset_delta), false)
        return T
    end
    ## With a missing delta the term adds no observation, so the prior on
    ## T is unchanged; with a delta it bounds T from below.
    free = sample(_onset_wrap(; onset_delta = missing), Prior(), 400;
                  chain_type = FlexiChains.VNChain, progress = false)
    @test all(isfinite, vec(Array(free[:T])))
end
