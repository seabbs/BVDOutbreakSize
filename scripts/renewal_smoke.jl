## Smoke test for the discrete-time renewal prototype (issue #81).
##
## Demonstrates that `renewal_joint`:
##   1. compiles and draws from the prior (prior-predictive),
##   2. yields a finite Mooncake gradient of its log density,
##   3. runs a short NUTS warmup without erroring.
##
## Run with: julia --project=. scripts/renewal_smoke.jl

using BVDOutbreakSize
using BVDOutbreakSize: renewal_joint, default_adtype
using Turing
using Turing.DynamicPPL: LogDensityFunction, VarInfo
const LDP = Turing.LogDensityProblems
using Random: MersenneTwister

obs = load_observations()

## Grid length = outbreak age in days. Use the genetic TMRCA as a starting
## scale; here a fixed 90-day grid keeps the smoke test fast.
n = 90

println("Building model with n = $n daily steps ...")
model = renewal_joint(
    n,
    obs.exported_cases,
    obs.total_deaths,
    obs.reported_cases,
    obs.exports_deaths;
    tmrca_days    = obs.genetic_tmrca_days,
    tmrca_days_sd = obs.genetic_tmrca_days_sd,
)

## 1. Prior-predictive draw -------------------------------------------
println("\n[1] Prior-predictive draw")
gen = renewal_joint(
    n, missing, missing, missing, missing;
    tmrca_days = obs.genetic_tmrca_days,
    tmrca_days_sd = obs.genetic_tmrca_days_sd,
)
rng = MersenneTwister(20260522)
draw = gen()
println("    C_T (cumulative infections at T) = ", round(draw.C_T))
println("    E[deaths]            = ", round(draw.expected_deaths_T;
                                             digits = 2))
println("    E[reported cases]    = ", round(draw.expected_reports_T;
                                             digits = 2))
println("    E[exports]           = ", round(draw.expected_exports_T;
                                             digits = 3))
println("    E[deaths among exp.] = ",
        round(draw.expected_exports_deaths_T; digits = 4))

## 2. Mooncake gradient -----------------------------------------------
println("\n[2] Mooncake gradient of the joint log density")
ldf = LogDensityFunction(model; adtype = default_adtype())
x0 = VarInfo(model)[:]
println("    parameter dimension = ", length(x0))
val, grad = LDP.logdensity_and_gradient(ldf, x0)
ok = all(isfinite, grad) && any(!iszero, grad)
println("    logdensity = ", round(val; digits = 3))
println("    gradient finite & nonzero: ", ok)
println("    gradient norm = ", round(sqrt(sum(abs2, grad)); digits = 3))
ok || error("Mooncake gradient is not finite/nonzero")

## 3. Short NUTS smoke test -------------------------------------------
println("\n[3] Short NUTS smoke test (50 warmup + 50 sample, 1 chain)")
t = @elapsed chn = sample(
    MersenneTwister(20260522),
    model,
    NUTS(0.8; adtype = default_adtype()),
    100;
    progress = false,
)
println("    completed in ", round(t; digits = 1), " s")
println("    sampled parameters: ", size(chn))

println("\nALL RENEWAL SMOKE CHECKS PASSED")
