# Isolate where Enzyme works and where it breaks by exercising each
# single-stream composer, the analysis joint, and a couple of incremental
# joints. For every model: compare the Enzyme gradient against Mooncake
# and finite differences at a prior draw, then attempt a short *serial*
# NUTS run (no threads) so trajectory-exploration failures surface
# separately from any threading interaction.
#
#   julia --project=test scripts/enzyme_submodels.jl

using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint, exports_only_model, deaths_only_model,
                       cases_only_model, confirmed_only_model, exports_deaths_only_model,
                       imperial_only_model, load_observations, genetic_seeding_model,
                       enzyme_adtype, default_adtype
using Turing
using Turing: DynamicPPL, MCMCSerial, NUTS, sample
using Enzyme
using Mooncake
using LogDensityProblems
using FiniteDifferences
using FiniteDifferences: central_fdm
using Random

include(joinpath(@__DIR__, "_joint_obs.jl"))

obs = load_observations()
genetic = T -> genetic_seeding_model(T, obs.genetic_tmrca_days;
    tmrca_days_sd = obs.genetic_tmrca_days_sd)
fa = joint_obs(obs)

function joint()
    bvd_joint(obs.exported_cases, fa.deaths, fa.reported,
        fa.export_deaths; fa.kw...,
        first_export_detection_delta = obs.first_export_detection_delta,
        genetic = genetic)
end
function joint_nogen()
    bvd_joint(obs.exported_cases, fa.deaths, fa.reported,
        fa.export_deaths; fa.kw...,
        first_export_detection_delta = obs.first_export_detection_delta)
end

# (label, model-builder). Ordered simplest first so the first failure
# points at the smallest offending component.
models = [
    ("exports_only", () -> exports_only_model(obs.exported_cases)),
    ("deaths_only", () -> deaths_only_model(obs.total_deaths)),
    ("cases_only", () -> cases_only_model(obs.reported_cases)),
    ("confirmed_only",
        () -> confirmed_only_model(obs.confirmed_cases,
            obs.cumulative_tests_analysed)),
    ("exports_deaths_only",
        () -> exports_deaths_only_model(obs.export_deaths_daily)),
    ("imperial_only",
        () -> imperial_only_model(obs.exported_cases, obs.exports_deaths)),
    ("joint_no_genetic", joint_nogen),
    ("joint_full", joint)
]

fdm = central_fdm(5, 1)

# one short serial NUTS attempt; returns (:ok | :error, detail)
function try_nuts(build, adtype; samples = 40)
    try
        rng = MersenneTwister(20260518)
        sample(rng, build(), NUTS(0.8; adtype), MCMCSerial(), samples, 1;
            initial_params = fill(DynamicPPL.InitFromPrior(), 1),
            progress = false)
        return (:ok, "")
    catch err
        return (:error, first(split(sprint(showerror, err), '\n')))
    end
end

println(rpad("model", 20), rpad("dim", 5), rpad("grad relerr(FD)", 18),
    rpad("Enz≈MC", 12), rpad("NUTS MC", 9), "NUTS Enz")
println("-"^85)

for (label, build) in models
    m = build()
    Random.seed!(20260518)
    vi = DynamicPPL.link(DynamicPPL.VarInfo(m), m)
    x0 = collect(vi[:])
    d = length(x0)

    ldf_p = DynamicPPL.LogDensityFunction(m, DynamicPPL.getlogjoint, vi)
    f0(x) = LogDensityProblems.logdensity(ldf_p, x)
    g_fd = first(FiniteDifferences.grad(fdm, f0, x0))

    relerr = "n/a"
    enz_mc = "n/a"
    g_mc = nothing
    try
        l = DynamicPPL.LogDensityFunction(m, DynamicPPL.getlogjoint, vi;
            adtype = default_adtype())
        _, g_mc = LogDensityProblems.logdensity_and_gradient(l, x0)
    catch err
        g_mc = nothing
    end
    try
        l = DynamicPPL.LogDensityFunction(m, DynamicPPL.getlogjoint, vi;
            adtype = enzyme_adtype())
        _, g = LogDensityProblems.logdensity_and_gradient(l, x0)
        re = maximum(abs.(g .- g_fd)) / max(1.0, maximum(abs.(g_fd)))
        relerr = string(round(re; sigdigits = 3))
        if g_mc !== nothing
            enz_mc = string(round(maximum(abs.(g .- g_mc)); sigdigits = 3))
        end
    catch err
        relerr = "GRAD ERR"
    end

    nuts_mc, _ = try_nuts(build, default_adtype())
    nuts_enz, enz_detail = try_nuts(build, enzyme_adtype())

    println(rpad(label, 20), rpad(d, 5), rpad(relerr, 18),
        rpad(enz_mc, 12), rpad(string(nuts_mc), 9), string(nuts_enz))
    if nuts_enz == :error
        println("    └─ Enzyme NUTS error: ", enz_detail[1:min(end, 200)])
    end
end
