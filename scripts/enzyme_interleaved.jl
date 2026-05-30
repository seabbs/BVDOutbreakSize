# Load-robust end-to-end comparison: alternate Mooncake and Enzyme fits so
# any transient machine load hits both equally, and take the min over reps.
# A non-interleaved benchmark that runs all of one backend then the other
# can be biased when the machine is shared; interleaving + min removes that.
#
#   BVD_WARMUP=200 BVD_SAMPLES=200 BVD_REPS=3 \
#     julia -t1 --project=test scripts/enzyme_interleaved.jl
using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint, load_observations, genetic_seeding_model,
                       enzyme_adtype, default_adtype
using Turing
using Turing: DynamicPPL, MCMCSerial, NUTS, sample
using Enzyme
using Mooncake
using Random
using Statistics: median

include(joinpath(@__DIR__, "_joint_obs.jl"))

obs = load_observations()
genetic = T -> genetic_seeding_model(T, obs.genetic_tmrca_days;
    tmrca_days_sd = obs.genetic_tmrca_days_sd)
fa = joint_obs(obs)
function build()
    bvd_joint(obs.exported_cases, fa.deaths, fa.reported,
        fa.export_deaths; fa.kw...,
        first_export_detection_delta = obs.first_export_detection_delta,
        genetic = genetic)
end

const WARMUP = parse(Int, get(ENV, "BVD_WARMUP", "200"))
const SAMPLES = parse(Int, get(ENV, "BVD_SAMPLES", "200"))
const REPS = parse(Int, get(ENV, "BVD_REPS", "3"))

function fit(adt)
    sample(MersenneTwister(20260518), build(),
        NUTS(WARMUP, 0.95; adtype = adt), MCMCSerial(), SAMPLES, 1;
        initial_params = fill(DynamicPPL.InitFromPrior(), 1), progress = false)
end

backends = [("Mooncake", default_adtype()), ("Enzyme", enzyme_adtype())]

# Compile each once (discard), so timed reps are steady-state.
for (_, adt) in backends
    fit(adt)
end

times = Dict(name => Float64[] for (name, _) in backends)
for rep in 1:REPS
    for (name, adt) in backends            # interleaved within each rep
        t = @elapsed fit(adt)
        push!(times[name], t)
    end
    println("rep ", rep, ": ",
        join(
            [string(n, " ", round(minimum(times[n]); digits = 1), "s")
             for (n, _) in backends], "   "))
end

println("\nInterleaved end-to-end NUTS, full joint, ", WARMUP, " warmup + ",
    SAMPLES, " draws, ", REPS, " reps (min over reps):")
for (name, _) in backends
    ts = times[name]
    println("  ", rpad(name, 9), " min ", round(minimum(ts); digits = 1),
        "s   median ", round(median(ts); digits = 1), "s")
end
mn = minimum(times["Mooncake"])
en = minimum(times["Enzyme"])
println("\nSpeedup (Mooncake min / Enzyme min): ", round(mn / en; digits = 2),
    "x")
