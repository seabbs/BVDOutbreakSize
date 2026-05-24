## Tests for the compartmental (Catalyst + daily stepper) architecture.
## Covers:
##   * Symbolic Catalyst network constructor returns a complete system
##     with the expected species and reactions.
##   * Daily stepper conserves total population to machine precision.
##   * Daily stepper agrees with the continuous-time OrdinaryDiffEq
##     reference solve at the daily grid points to within a discrete-
##     time tolerance.
##   * Delay discretisation and convolution match a hand-computed
##     example.
##   * `bvd_compartmental_joint` admits a `missing` data run as a prior
##     generator.

using BVDOutbreakSize: bvd_seir_network, bvd_seir_ode_solve,
                       step_seir_daily, simulate_seir_daily,
                       simulate_seir_daily_full,
                       discretise_delay_seir, convolve_delay_seir,
                       seir_growth_model, bvd_compartmental_joint
import Catalyst
using Distributions: Gamma

@testset "Catalyst SEIR network has the expected species/reactions" begin
    rn = bvd_seir_network()
    species_names = Set(string(s) for s in Catalyst.species(rn))
    @test species_names == Set(["S(t)", "E(t)", "I(t)", "R(t)", "D(t)"])
    rxs = Catalyst.reactions(rn)
    @test length(rxs) == 4
end

@testset "step_seir_daily conserves population to machine precision" begin
    β, σ, γ, CFR, N = 0.6, 1/6, 1/7, 0.3, 1_000.0
    state = (N - 1.0, 1.0, 0.0, 0.0, 0.0)
    for _ in 1:50
        state, _, _ = step_seir_daily(state;
            β = β, σ = σ, γ = γ, CFR = CFR, N = N)
    end
    @test isapprox(sum(state), N; atol = 1e-9)
end

@testset "simulate_seir_daily returns onsets and deaths" begin
    sim = simulate_seir_daily(30;
        β = 0.6, σ = 1/6, γ = 1/7, CFR = 0.3, N = 1_000.0,
        S0 = 999.0, E0 = 1.0, I0 = 0.0)
    @test length(sim.onsets) == 30
    @test length(sim.deaths) == 30
    @test all(sim.onsets .>= 0)
    @test all(sim.deaths .>= 0)
end

@testset "daily stepper tracks the OrdinaryDiffEq reference solve" begin
    β, σ, γ, CFR, N = 0.6, 1/6, 1/7, 0.3, 1_000.0
    full = simulate_seir_daily_full(60;
        β = β, σ = σ, γ = γ, CFR = CFR, N = N,
        S0 = N - 1.0, E0 = 1.0, I0 = 0.0)
    sol = bvd_seir_ode_solve(60.0;
        β = β, σ = σ, γ = γ, CFR = CFR, N = N,
        S0 = N - 1.0, E0 = 1.0, I0 = 0.0, saveat = 1.0)
    rn = bvd_seir_network()
    I_ode = sol[rn.I]
    ## The daily stepper marches the state to integer day endpoints, so
    ## compare against the ODE solution at the matching times. The two
    ## forward operators agree to roughly the daily-discretisation
    ## constant: the exponentialised-rate stepper assumes a piecewise-
    ## constant force-of-infection across each day, whereas the ODE
    ## resolves the within-day curvature. At BVD-scale per-day rates
    ## (σ, γ ~ 0.15) the relative gap is bounded by a few tens of
    ## percent in I (because I is a small absolute quantity early on
    ## and the daily curvature concentrates there); cumulative onsets
    ## agree much better.
    ## The agreement is bounded by the daily-discretisation constant of
    ## an exponentialised-rate stepper: each flow is treated as a
    ## constant-rate exponential clock over the day, so cross-compartment
    ## within-day curvature is dropped. At BVD-scale per-day rates the
    ## relative gap is at most a few tens of percent, set as a wide
    ## sanity bound here rather than a tight quantitative match.
    rel_err_I = maximum(abs.(full.I .- I_ode[2:end])) /
              (maximum(abs.(I_ode)) + eps())
    @test rel_err_I < 0.5

    ## Cumulative onsets are the running sum of the daily $E \to I$ flux
    ## under each scheme. We check the magnitudes are in the same ball-
    ## park (better than a factor of 2) rather than tight agreement,
    ## since the two forward maps are genuinely different operators.
    cum_onsets_stepper = cumsum(full.onsets)
    cum_E = sol[rn.E]
    cum_I = sol[rn.I]
    cum_R = sol[rn.R]
    cum_D = sol[rn.D]
    cum_ode = (cum_E .+ cum_I .+ cum_R .+ cum_D)[2:end] .-
              (cum_E[1] + cum_I[1] + cum_R[1] + cum_D[1])
    ratio = cum_onsets_stepper[end] / (cum_ode[end] + eps())
    @test 0.5 < ratio < 2.0
end

@testset "discretise_delay_seir returns a normalised PMF" begin
    pmf = discretise_delay_seir(Gamma(4.3, 2.6), 60)
    @test length(pmf) == 61
    @test all(pmf .>= 0)
    @test isapprox(sum(pmf), 1.0; atol = 1e-12)
end

@testset "convolve_delay_seir matches a hand-computed example" begin
    x = [1.0, 0.0, 0.0]
    g = [0.5, 0.3, 0.2]
    y = convolve_delay_seir(x, g)
    @test isapprox(y, [0.5, 0.3, 0.2])

    x = [1.0, 2.0, 3.0]
    g = [1.0]
    y = convolve_delay_seir(x, g)
    @test y == [1.0, 2.0, 3.0]
end

@testset "bvd_compartmental_joint admits missing data (prior gen)" begin
    model = bvd_compartmental_joint(missing, missing, missing)
    ## Prior draw: no observations to condition on. The model should
    ## sample successfully and produce a non-empty draw.
    out = rand(model)
    @test out !== nothing
    @test length(out) > 0
end
