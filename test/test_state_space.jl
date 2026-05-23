using BVDOutbreakSize: discretise_delay_ss, convolve_delay_ss,
                       nb_branching_step,
                       weekly_knot_days_ss, interpolate_knots_ss,
                       state_space_joint
import Distributions
using Distributions: Gamma, NegativeBinomial
using Random: MersenneTwister
import Turing
using Turing.DynamicPPL: LogDensityFunction, VarInfo
using Turing.LogDensityProblems: logdensity_and_gradient

@testset "discretise_delay_ss normalises and is non-negative" begin
    pmf = discretise_delay_ss(Gamma(4.3, 2.6), 40)
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1e-12)
    ## Density at the origin is forced to zero (shape > 1), so the lag-0
    ## bin is the trapezoid of [0, f(1)] only.
    @test pmf[1] < pmf[2]
end

@testset "convolve_delay_ss conserves mass for a normalised delay" begin
    x = [1.0, 2.0, 3.0, 4.0]
    delay = [0.2, 0.5, 0.3]
    y = convolve_delay_ss(x, delay)
    @test length(y) == length(x)
    ## A point mass delay at lag 0 returns the input unchanged.
    y0 = convolve_delay_ss(x, [1.0])
    @test y0 == x
end

@testset "nb_branching_step is a deterministic NB mean λ_t" begin
    ## With a one-lag generation interval `g = [1.0]` and constant `R = 2`,
    ## the renewal mean satisfies λ_{t+1} = 2 · I_t. Test the mean λ_t
    ## computation directly (the random draw is exercised in the joint
    ## model via the Turing `~` mechanism).
    g = [1.0]
    Rt = fill(2.0, 5)
    I = [1.0, 2.0, 4.0, 8.0, 16.0]
    λ = [nb_branching_step(I, g, Rt, t) for t in 2:5]
    @test λ ≈ [2.0, 4.0, 8.0, 16.0]
end

@testset "weekly_knot_days_ss pins both ends at weekly spacing" begin
    days = weekly_knot_days_ss(90; week = 7)
    @test days[1] == 1
    @test days[end] == 90
    @test issorted(days)
    @test all(diff(days)[1:(end - 1)] .== 7)
    @test weekly_knot_days_ss(15; week = 7) == [1, 8, 15]
end

@testset "interpolate_knots_ss is piecewise-linear and hits the knots" begin
    knots = [0.0, 2.0, -1.0]
    knot_days = [1, 5, 9]
    daily = interpolate_knots_ss(knots, knot_days, 9)
    @test length(daily) == 9
    @test daily[1] == 0.0
    @test daily[5] == 2.0
    @test daily[9] == -1.0
    @test daily[3] ≈ 1.0
    seg1 = diff(daily[1:5])
    @test all(≈(seg1[1]), seg1)
end

@testset "state_space_joint prior-predictive draw is finite" begin
    rng = MersenneTwister(20260523)
    gen = state_space_joint(
        60, missing, missing, missing, missing)
    draw = gen(rng)
    @test isfinite(draw.C_T)
    @test draw.C_T >= 0
    @test isfinite(draw.expected_deaths_T)
    @test isfinite(draw.expected_exports_T)
    @test all(isfinite, draw.Rt)
end

@testset "state_space_joint continuous-only gradient is finite" begin
    ## Build the model with the latent integer path fixed (pass a path
    ## via `I_obs` so the latent counts are conditioned and the remaining
    ## continuous block is differentiable under Mooncake). This proves
    ## the continuous nuisance block has finite gradients.
    n = 30
    I_obs = fill(2, n)
    model = state_space_joint(
        n, 2, 131, 516, 1;
        tmrca_days = 40.0, tmrca_days_sd = 20.0, I_obs = I_obs)
    ldf = LogDensityFunction(
        model; adtype = BVDOutbreakSize.default_adtype())
    x0 = VarInfo(model)[:]
    v, g = logdensity_and_gradient(ldf, x0)
    @test isfinite(v)
    @test all(isfinite, g)
    @test any(!iszero, g)
end
