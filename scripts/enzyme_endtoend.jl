# End-to-end NUTS wall-clock for Mooncake vs Enzyme on the joint model,
# plus a check that Enzyme needs runtime activity. Single chain, modest
# draw count: total time includes one-off AD compilation, so it reflects
# the cost of an actual analysis run rather than steady-state gradients.
#
#   julia --project=test scripts/enzyme_endtoend.jl

using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint, load_observations, genetic_seeding_model,
                       nuts_sample
using Turing
using Turing: DynamicPPL
using ADTypes: AutoMooncake, AutoEnzyme
using Enzyme
using Mooncake
using LogDensityProblems
using Random

include(joinpath(@__DIR__, "_joint_obs.jl"))  # joint_obs + _increments

obs = load_observations()
genetic_seeding = T -> genetic_seeding_model(T, obs.genetic_tmrca_days;
    tmrca_days_sd = obs.genetic_tmrca_days_sd)
fit_args = joint_obs(obs)
function build()
    bvd_joint(obs.exported_cases, fit_args.deaths, fit_args.reported,
        fit_args.export_deaths; fit_args.kw...,
        first_export_detection_delta = obs.first_export_detection_delta,
        genetic = genetic_seeding)
end

# --- 1. does Enzyme reverse need runtime activity? ---------------------
println("=== runtime-activity check ===")
vi = DynamicPPL.link(DynamicPPL.VarInfo(build()), build())
x0 = collect(vi[:])
for (name, mode) in [("Enzyme.Reverse (no RTA)", Enzyme.Reverse),
    ("set_runtime_activity(Enzyme.Reverse)",
        Enzyme.set_runtime_activity(Enzyme.Reverse))]
    adt = AutoEnzyme(; mode = mode, function_annotation = Enzyme.Duplicated)
    l = DynamicPPL.LogDensityFunction(build(), DynamicPPL.getlogjoint, vi;
        adtype = adt)
    try
        _, g = LogDensityProblems.logdensity_and_gradient(l, x0)
        println("  ", name, ": OK (", count(isfinite, g), "/", length(g),
            " finite)")
    catch err
        println("  ", name, ": FAILED — ",
            first(split(sprint(showerror, err), '\n')))
    end
end

# --- 2. end-to-end NUTS wall-clock ------------------------------------
samples, chains = 500, 1
adtypes = [
    ("Mooncake", AutoMooncake(; config = Mooncake.Config())),
    ("Enzyme reverse",
        AutoEnzyme(;
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated))
]

println("\n=== end-to-end NUTS (", samples, " draws, ", chains,
    " chain) ===")
for (name, adt) in adtypes
    # cold: includes AD compilation
    cold = @elapsed nuts_sample(build(); samples = samples, chains = chains,
        adtype = adt)
    # warm: second run, AD already compiled this session
    warm = @elapsed nuts_sample(build(); samples = samples, chains = chains,
        adtype = adt)
    println("  ", rpad(name, 16), " cold: ", round(cold, digits = 1),
        " s   warm: ", round(warm, digits = 1), " s")
end
