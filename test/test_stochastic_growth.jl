## Tests for the stochastic latent infection process (issue #48):
## the log-incidence trajectory builder, the growth submodel prior
## draws, the onset-staging helpers, and the onset-timing bound. These
## exercise the new package-level pieces; the full joint composer is
## demonstrated in scripts/prototype_stochastic.jl (it depends on
## observation stand-ins, which by repo convention live outside the
## package; issue #81).

using BVDOutbreakSize: lna_logi, lna_trajectory, lna_incidence,
                       onset_cumulative,
                       stochastic_growth_model, onset_timing_model,
                       incubation_model
using Distributions: Normal, Gamma, cdf, pdf
using Turing: Turing, @model, sample, Prior, to_submodel
import FlexiChains

@testset "lna_logi recovers exponential rate when σ = 0" begin
    r = log(2) / 14
    T = 98.0
    z = zeros(23)
    logi = lna_logi(r, T, 0.0, z)
    @test length(logi) == 24
    ## Seeded at log r so i(0) = r reproduces deterministic dC/ds at s=0.
    @test isapprox(logi[1], log(r); atol = 1e-12)
    ## With no noise the endpoint is log r + r·T.
    @test isapprox(logi[end], log(r) + r * T; atol = 1e-10)
end

@testset "lna_logi injects variance when σ > 0" begin
    r = log(2) / 14
    T = 98.0
    z = [1.0; zeros(22)]
    logi = lna_logi(r, T, 0.5, z)
    ## A single positive shock at j=1 lifts every level after the
    ## first knot above the deterministic mean.
    @test logi[3] > log(r) + r * (2 * T / 23)
end

@testset "lna_incidence is strictly positive on (0, T)" begin
    r = log(2) / 14
    T = 98.0
    logi = lna_logi(r, T, 0.5, randn(23))
    i = lna_incidence(logi, T)
    @test i(-1.0) == 0.0
    @test all(i(s) > 0 for s in 1.0:5.0:T)
end

@testset "lna_trajectory is monotone non-decreasing" begin
    r = log(2) / 14
    T = 98.0
    ## Even with negative draws, log-incidence stays finite and i(s)>0,
    ## so the cumulative trajectory is monotone by construction.
    logi = lna_logi(r, T, 0.5, randn(23))
    C = lna_trajectory(logi, T)
    samples = [C(s) for s in 0.0:5.0:T]
    @test all(diff(samples) .>= 0)
    @test C(0.0) == 0.0
    @test C(T + 5) == C(T)             # flat past T
end

@testset "lna_trajectory at σ=0 matches the exp(r·s) - 1 cumulative" begin
    r = log(2) / 14
    T = 98.0
    logi = lna_logi(r, T, 0.0, zeros(23))
    C = lna_trajectory(logi, T)
    ## Deterministic incidence i(s) = r·exp(r·s) integrates to
    ## C(s) = exp(r·s) - 1. The trapezoid on 24 knots over 98 days
    ## introduces a few-percent error, biggest at small s where the
    ## segment width is large relative to the local doubling time.
    for s in (10.0, 49.0, 90.0)
        @test isapprox(C(s), exp(r * s) - 1; rtol = 3e-2)
    end
end

@testset "stochastic_growth_model prior draws are finite and >= 0" begin
    chn = sample(stochastic_growth_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    CT = vec(Array(chn[:C_T]))
    @test length(CT) == 200
    @test all(isfinite, CT)
    @test all(CT .>= 0)
    σ = vec(Array(chn[:σ]))
    @test all(σ .>= 0)
end

@testset "onset_cumulative C_o(T) ≤ C(T) with sharp incubation" begin
    ## With a sharp incubation density, onsets track infections, so
    ## C_o(T) is bounded above by C(T) and close to it.
    r = log(2) / 14
    T = 98.0
    logi = lna_logi(r, T, 0.0, zeros(23))
    incidence = lna_incidence(logi, T)
    cumulative = lna_trajectory(logi, T)
    sharp = Gamma(2.0, 0.5)              # mean ≈ 1 day, sharp
    F_inc = x -> cdf(sharp, max(x, 0.0))
    C_o_T = onset_cumulative(incidence, F_inc, T)
    C_T = cumulative(T)
    @test 0 < C_o_T <= C_T * (1 + 1e-9)  # tiny quadrature slack
    @test C_o_T / C_T > 0.85
end

@testset "onset_cumulative is monotone in T" begin
    r = log(2) / 14
    T = 98.0
    logi = lna_logi(r, T, 0.0, zeros(23))
    incidence = lna_incidence(logi, T)
    F_inc = x -> cdf(Gamma(3.0, 2.0), max(x, 0.0))
    co50 = onset_cumulative(incidence, F_inc, 50.0)
    co70 = onset_cumulative(incidence, F_inc, 70.0)
    co90 = onset_cumulative(incidence, F_inc, 90.0)
    @test 0 < co50 < co70 < co90
end

@testset "incubation_model prior draws are positive and finite" begin
    chn = sample(incubation_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    α = vec(Array(chn[:α_inc]))
    θ = vec(Array(chn[:θ_inc]))
    @test all(α .> 0) && all(θ .> 0)
    @test all(isfinite, α) && all(isfinite, θ)
end

@testset "onset_timing_model: delta missing is no-op, else samples sd" begin
    @model function _onset_wrap(; onset_delta)
        T ~ Normal(100.0, 10.0)
        s ~ to_submodel(
            onset_timing_model(T; onset_delta = onset_delta), false)
        return T
    end
    ## With a missing delta the term adds no observation and no extra
    ## parameter, so the prior on T is unchanged.
    free = sample(_onset_wrap(; onset_delta = missing), Prior(), 400;
                  chain_type = FlexiChains.VNChain, progress = false)
    @test all(isfinite, vec(Array(free[:T])))

    ## With a delta the timing SD is sampled from its prior (a delay
    ## carried as prior uncertainty, not a fixed constant) and is > 0.
    bound = sample(_onset_wrap(; onset_delta = 120.0), Prior(), 400;
                   chain_type = FlexiChains.VNChain, progress = false)
    sd = vec(Array(bound[:onset_sd]))
    @test length(sd) == 400
    @test all(sd .> 0)
end
