## Compartmental architecture smoke run (branch `arch-compartmental-mtk`).
##
## Exercises four things:
##   1. The Catalyst SEIR network builds.
##   2. The daily stepper runs and conserves total population.
##   3. The Turing joint composer produces a prior-predictive draw.
##   4. A short NUTS sample using the Mooncake adtype completes
##      without erroring.
##
## Intended for hand-running on a developer machine. Not part of the CI
## test suite — that uses lighter wiring in `test/test_compartmental.jl`.

using BVDOutbreakSize
using Random: MersenneTwister, seed!
using Distributions: Gamma

seed!(20260524)

println("== Building Catalyst SEIR network ==")
rn = bvd_seir_network()
println("Species: ", join(string.(BVDOutbreakSize.Catalyst.species(rn)), ", "))

println("\n== Running the daily stepper for 60 days ==")
β, σ, γ, CFR, N = 0.4, 1/6, 1/7, 0.3, 4_392_200.0
full = simulate_seir_daily_full(60;
    β = β, σ = σ, γ = γ, CFR = CFR, N = N,
    S0 = N - 1.0, E0 = 1.0, I0 = 0.0)
println("Day 60 conservation residual = ",
    abs(full.S[end] + full.E[end] + full.I[end] + full.R[end] +
        full.D[end] - N))
println("Cumulative onsets at day 60 = ",
    round(sum(full.onsets); digits = 1))

println("\n== Prior-predictive draw from bvd_compartmental_joint ==")
model = bvd_compartmental_joint(missing, missing, missing)
draw = rand(model)
println("Drew prior sample with ", length(draw), " entries")

println("\n== NUTS smoke run (50 draws, 1 chain, Mooncake AD) ==")
chn = nuts_sample(
    bvd_compartmental_joint(2, 5, 516);
    samples = 50, chains = 1, target_accept = 0.8,
    seed = 20260524, progress = false)
println("NUTS smoke run completed. C_T median = ",
    round(median(vec(Array(chn[:cumulative_cases]))); digits = 0))
