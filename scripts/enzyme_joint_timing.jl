# End-to-end NUTS wall-clock for the full joint model: Mooncake (the
# package default) vs Enzyme reverse. Reports a cold run (includes one-off
# AD compilation, i.e. the cost of a single analysis invocation) and a warm
# run (steady-state sampling once compiled). Single chain via MCMCSerial so
# the comparison is like-for-like and thread-count independent.
#
#   julia -t1 --project=test scripts/enzyme_joint_timing.jl
using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint, load_observations, genetic_seeding_model,
    enzyme_adtype, default_adtype
using Turing
using Turing: DynamicPPL, MCMCSerial, NUTS, sample
using ADTypes: AutoMooncake, AutoEnzyme
using Enzyme
using Mooncake
using LogDensityProblems
using Random

include(joinpath(@__DIR__, "_joint_obs.jl"))

obs = load_observations()
genetic = T -> genetic_seeding_model(T, obs.genetic_tmrca_days;
    tmrca_days_sd = obs.genetic_tmrca_days_sd)
fa = joint_obs(obs)
build() = bvd_joint(obs.exported_cases, fa.deaths, fa.reported,
    fa.export_deaths; fa.kw...,
    first_export_detection_delta = obs.first_export_detection_delta,
    genetic = genetic)

const SAMPLES = 1000
const WARMUP = 1000

fit(adtype) = sample(MersenneTwister(20260518), build(),
    NUTS(WARMUP, 0.95; adtype), MCMCSerial(), SAMPLES, 1;
    initial_params = fill(DynamicPPL.InitFromPrior(), 1), progress = false)

# --- gradient agreement at a prior draw (correctness gate) -------------
m = build()
Random.seed!(20260518)
vi = DynamicPPL.link(DynamicPPL.VarInfo(m), m)
x0 = collect(vi[:])
g(adtype) = last(LogDensityProblems.logdensity_and_gradient(
    DynamicPPL.LogDensityFunction(m, DynamicPPL.getlogjoint, vi;
        adtype = adtype), x0))
gm = g(default_adtype())
ge = g(enzyme_adtype())
println("joint dim: ", length(x0))
println("max |grad Enzyme - grad Mooncake|: ", maximum(abs.(ge .- gm)))

# --- end-to-end timing -------------------------------------------------
backends = [("Mooncake", default_adtype()), ("Enzyme", enzyme_adtype())]
println("\nEnd-to-end NUTS, full joint, ", WARMUP, " warmup + ", SAMPLES,
    " draws, 1 chain (MCMCSerial):")
results = Dict{String, NamedTuple}()
for (name, adt) in backends
    cold = @elapsed fit(adt)          # includes AD compilation
    warm = @elapsed fit(adt)          # compiled; steady-state sampling
    results[name] = (; cold, warm)
    println("  ", rpad(name, 9), " cold ", round(cold, digits = 1),
        " s   warm ", round(warm, digits = 1), " s   (",
        round(SAMPLES / warm, digits = 1), " draws/s warm)")
end

mc = results["Mooncake"]
ez = results["Enzyme"]
println("\nWarm speedup  (Mooncake/Enzyme): ",
    round(mc.warm / ez.warm, digits = 2), "x")
println("Cold speedup  (Mooncake/Enzyme): ",
    round(mc.cold / ez.cold, digits = 2), "x")
