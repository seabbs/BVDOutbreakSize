# Speed + correctness comparison for the α-derivative of
# cdf(Gamma(α, θ), x), benchmarked end-to-end through Mooncake's
# reverse-mode AD. Two rrule-equipped wrappers are compared:
#
#   1. `BVDOutbreakSize._gamma_cdf`         — series-based rrule
#      (production, see `src/gamma_cdf.jl`).
#   2. `_gamma_cdf_quad` (loaded from `gamma_cdf_quad.jl`) — QuadGK
#      tail-integral rrule, kept here purely for reference.
#
# Both have identical forward values; the comparison isolates the cost
# (and accuracy) of the α-derivative routine each rrule uses. The
# series implementation is the package's primary path because it is
# typically ~50× faster than adaptive quadrature on this integrand
# while matching FD to ~1e-12 across the stress-tested grid
# (α ∈ [0.001, 100] × z ∈ [0.01, 200]).
#
# Run from the repo root with the test environment:
#
#     julia --project=test test/bench/bench_gamma_cdf_partial.jl

using BenchmarkTools: @benchmark, median
using BVDOutbreakSize
using Distributions: Gamma, cdf
using Mooncake: Mooncake

include(joinpath(@__DIR__, "gamma_cdf_quad.jl"))

# (α, z) test grid, modelled on Stan's grad_reg_inc_gamma_test.cpp plus
# α < 1 cases (NUTS warmup regime) and a deep-tail case.
const CASES = [
    # (α,     z,      label)
    (0.3,  13.04, "small α, deep tail"),
    (0.5,  1.0,   "small α, z ≈ α"),
    (0.5,  5.0,   "small α, z > α"),
    (1.1,  0.2,   "Stan: z < α"),
    (1.1,  2.0,   "Stan: z > α"),
    (2.5,  1.3,   "Stan: z < α"),
    (2.5,  30.0,  "Stan: deep tail"),
    (9.0,  10.0,  "Stan: crossover"),
    (10.0, 9.0,   "Stan: z < α"),
    (25.0, 13.04, "large α, z < α"),
]

const θ_ref = 1.0
const H = 1e-5  # FD step for the reference

# Build Mooncake rules once per (α, θ, x) signature. The signature is
# fixed across CASES so we build at the first case and reuse.
α0, z0 = CASES[1][1], CASES[1][2]
x0     = z0 * θ_ref
rule_series = Mooncake.build_rrule(BVDOutbreakSize._gamma_cdf, α0, θ_ref, x0)
rule_quad   = Mooncake.build_rrule(_gamma_cdf_quad,            α0, θ_ref, x0)

# Warmup
for (α, z, _) in CASES
    x = z * θ_ref
    Mooncake.value_and_gradient!!(rule_series, BVDOutbreakSize._gamma_cdf, α, θ_ref, x)
    Mooncake.value_and_gradient!!(rule_quad,   _gamma_cdf_quad,            α, θ_ref, x)
end

println(rpad("α", 6), rpad("z", 8), rpad("label", 22),
        rpad("series μs", 12), rpad("quad μs", 11),
        rpad("|series-fd|", 13), rpad("|quad-fd|", 12), "speedup")
println(repeat("-", 105))

for (α, z, label) in CASES
    x = z * θ_ref
    fd = (cdf(Gamma(α + H, θ_ref), x) - cdf(Gamma(α - H, θ_ref), x)) / (2H)

    _, grads_s = Mooncake.value_and_gradient!!(
        rule_series, BVDOutbreakSize._gamma_cdf, α, θ_ref, x)
    _, grads_q = Mooncake.value_and_gradient!!(
        rule_quad, _gamma_cdf_quad, α, θ_ref, x)
    dα_series, dα_quad = grads_s[2], grads_q[2]

    bs = @benchmark Mooncake.value_and_gradient!!(
        $rule_series, $BVDOutbreakSize._gamma_cdf, $α, $θ_ref, $x)
    bq = @benchmark Mooncake.value_and_gradient!!(
        $rule_quad, $_gamma_cdf_quad, $α, $θ_ref, $x)
    ts = median(bs).time / 1e3
    tq = median(bq).time / 1e3

    println(rpad(α, 6), rpad(z, 8), rpad(label, 22),
            rpad(round(ts; digits=2), 12),
            rpad(round(tq; digits=2), 11),
            rpad(round(abs(dα_series - fd); sigdigits=2), 13),
            rpad(round(abs(dα_quad - fd); sigdigits=2), 12),
            round(tq / ts; digits=2), "×")
end
