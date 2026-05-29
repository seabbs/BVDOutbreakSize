# Compare AD backends (Mooncake vs Enzyme) on the full joint BVD model.
# Checks gradient correctness against finite differences and the Mooncake
# reference, then times repeated gradient evaluations of the unconstrained
# log-density (what NUTS calls each leapfrog step).
#
# Run with the test workspace project (which carries Enzyme + Mooncake +
# FiniteDifferences + LogDensityProblems):
#   julia --project=test scripts/enzyme_benchmark.jl

using BVDOutbreakSize
using BVDOutbreakSize: bvd_joint, load_observations, genetic_seeding_model
using Turing
using Turing: DynamicPPL
using ADTypes: AutoMooncake, AutoEnzyme
using Enzyme
using Mooncake
using LogDensityProblems
using FiniteDifferences
using FiniteDifferences: central_fdm
using Random
using Statistics: median

include(joinpath(@__DIR__, "_joint_obs.jl"))  # joint_obs + _increments

# --- build the full joint model exactly as the analysis fits it --------
obs = load_observations()
genetic_seeding = T -> genetic_seeding_model(T, obs.genetic_tmrca_days;
    tmrca_days_sd = obs.genetic_tmrca_days_sd)
fit_args = joint_obs(obs)
model = bvd_joint(obs.exported_cases, fit_args.deaths, fit_args.reported,
    fit_args.export_deaths; fit_args.kw...,
    first_export_detection_delta = obs.first_export_detection_delta,
    genetic = genetic_seeding)

# Unconstrained (linked) VarInfo: a prior draw gives a finite-density
# starting point in the space NUTS actually samples.
Random.seed!(20260518)
vi = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
x0 = collect(vi[:])
d = length(x0)
println("Model dimension (unconstrained): ", d)

# --- backends ----------------------------------------------------------
backends = [
    ("Mooncake", AutoMooncake(; config = Mooncake.Config())),
    ("Enzyme reverse (RTA + Duplicated)",
        AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated))
]

function ldf(adtype)
    DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint,
        vi; adtype = adtype)
end

# Finite-difference reference on the primal log-density.
ldf_primal = DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint, vi)
f0(x) = LogDensityProblems.logdensity(ldf_primal, x)
println("logdensity at x0: ", f0(x0))
fdm = central_fdm(5, 1)
g_fd = first(FiniteDifferences.grad(fdm, f0, x0))

function timeit(f; warmup = 2, reps = 50)
    for _ in 1:warmup
        f()
    end
    ts = Float64[]
    for _ in 1:reps
        t = time_ns()
        f()
        push!(ts, (time_ns() - t) / 1e6)  # ms
    end
    return (min = minimum(ts), med = median(ts))
end

results = []
for (name, adtype) in backends
    println("\n=== ", name, " ===")
    local l
    try
        l = ldf(adtype)
    catch err
        println("  FAILED to build LogDensityFunction: ", err)
        continue
    end
    local val, g
    try
        val, g = LogDensityProblems.logdensity_and_gradient(l, x0)
    catch err
        println("  FAILED logdensity_and_gradient: ",
            sprint(showerror, err)[1:min(end, 400)])
        continue
    end
    rel_fd = maximum(abs.(g .- g_fd)) /
             max(1.0, maximum(abs.(g_fd)))
    println("  logdensity value: ", val)
    println("  max abs err vs finite-diff: ", maximum(abs.(g .- g_fd)))
    println("  rel err vs finite-diff:     ", rel_fd)
    t = timeit(() -> LogDensityProblems.logdensity_and_gradient(l, x0))
    println("  grad time  min: ", round(t.min, digits = 3),
        " ms   median: ", round(t.med, digits = 3), " ms")
    push!(results, (; name, val, rel_fd, g, t))
end

# --- cross-check Enzyme vs Mooncake gradient agreement -----------------
if length(results) == 2
    g_mooncake = results[1].g
    g_enzyme = results[2].g
    println("\n=== Mooncake vs Enzyme gradient ===")
    println("  max abs diff: ", maximum(abs.(g_mooncake .- g_enzyme)))
    println("  speedup (Mooncake median / Enzyme median): ",
        round(results[1].t.med / results[2].t.med, digits = 2), "x")
end
