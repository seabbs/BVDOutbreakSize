# Capture the full EnzymeNonScalarReturn backtrace from joint NUTS so the
# offending term can be pinpointed rather than guessed.
#   julia --project=test scripts/enzyme_joint_debug.jl

using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint, load_observations, genetic_seeding_model,
                       enzyme_adtype
using Turing
using Turing: DynamicPPL, MCMCSerial, NUTS, sample
using Enzyme
using Random

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

try
    rng = MersenneTwister(20260518)
    sample(rng, build(), NUTS(0.8; adtype = enzyme_adtype()), MCMCSerial(),
        60, 1; initial_params = fill(DynamicPPL.InitFromPrior(), 1),
        progress = false)
    println("UNEXPECTED: joint NUTS under Enzyme completed without error")
catch err
    println("CAUGHT: ", typeof(err))
    Base.showerror(stdout, err)
    println()
    println("---- backtrace ----")
    for (i, fr) in enumerate(stacktrace(catch_backtrace()))
        i > 40 && break
        println(fr)
    end
end
