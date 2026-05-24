# State-space + particle filter demo driver (issue #48).
#
# Demonstrates the architecture in `src/state_space.jl`:
#
# 1. Prior-predictive draw from the joint composer.
# 2. Continuous-only gradient check (latent path conditioned).
# 3. A short particle-Gibbs smoke test on the latent integer path, with
#    NUTS on the continuous nuisances inside Gibbs.
#
# The full posterior fit is intentionally not attempted; see
# docs/src/proposals/state-space-particle.md for the cost calculation
# and verdict.

using BVDOutbreakSize
using BVDOutbreakSize: state_space_joint, default_adtype, load_observations
using Random: MersenneTwister
import Turing
using Turing: NUTS, PG, Gibbs, sample, MCMCSerial
using Turing.DynamicPPL: LogDensityFunction, VarInfo
using Turing.LogDensityProblems: logdensity_and_gradient
using Statistics: mean, std

const SEED = 20260523

println("=== State-space + PF demo smoke test ===")

obs = load_observations()
n   = 60   # short grid for the smoke test; production would use ~130

# Prior-predictive draw -----------------------------------------------------
println("\n-- Prior-predictive draw --")
rng = MersenneTwister(SEED)
gen = state_space_joint(n, missing, missing, missing, missing)
draw = gen(rng)
println("C_T (cumulative infections, prior draw)      = ", draw.C_T)
println("Expected deaths to cut-off  (prior draw)     = ",
        draw.expected_deaths_T)
println("Expected reports to cut-off (prior draw)     = ",
        draw.expected_reports_T)
println("Expected exports to cut-off (prior draw)     = ",
        draw.expected_exports_T)
println("Expected export deaths to cut-off (prior)    = ",
        draw.expected_exports_deaths_T)
println("Realised infections (first 10 days)          = ",
        draw.infections[1:min(10, n)])

# Continuous-only gradient check -------------------------------------------
println("\n-- Continuous nuisance gradient (latent path conditioned) --")
I_obs = max.(draw.infections, 0)
model_cond = state_space_joint(
    n, obs.exported_cases, obs.total_deaths, obs.reported_cases,
    obs.exports_deaths;
    tmrca_days    = obs.genetic_tmrca_days,
    tmrca_days_sd = obs.genetic_tmrca_days_sd,
    I_obs         = I_obs)
ldf = LogDensityFunction(model_cond; adtype = default_adtype())
x0 = VarInfo(model_cond)[:]
v, g = logdensity_and_gradient(ldf, x0)
println("log-density                = ", v)
println("gradient finite            = ", all(isfinite, g))
println("gradient non-zero entries  = ", count(!iszero, g), " / ",
        length(g))

# Particle Gibbs smoke test -------------------------------------------------
println("\n-- Particle Gibbs smoke test (latent path + nuisances) --")
# PG steps the *full* model (latent integer path and the continuous
# nuisances together). Using PG only as a smoke test demonstrates the
# inference machinery compiles and runs end-to-end on the architecture.
# In production the natural sampler is a Gibbs block
# `Gibbs(:I => PG(N), :nuisances => NUTS())`, exercised separately when
# the latent-variable selection in Turing 0.45's Gibbs API is settled.
# Production particle counts and chain lengths are much higher; see the
# proposal.
n_pg_particles = 50
n_iter         = 10
println("particles = $(n_pg_particles), iterations = $(n_iter)")

model = state_space_joint(
    n, obs.exported_cases, obs.total_deaths, obs.reported_cases,
    obs.exports_deaths;
    tmrca_days    = obs.genetic_tmrca_days,
    tmrca_days_sd = obs.genetic_tmrca_days_sd)

rng = MersenneTwister(SEED)
t0 = time()
chn = sample(rng, model, PG(n_pg_particles), n_iter; progress = false)
dt = time() - t0
println("PG smoke test completed in ", round(dt; digits = 1), " seconds")
println("Drew ", n_iter, " iterations; chain dimensions: ", size(chn))

println("\nDone. See docs/src/proposals/state-space-particle.md for the verdict.")
