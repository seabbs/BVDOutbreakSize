# Timing comparison of the two `expected_deaths` methods.
#
# The Gamma-distribution analytic form
#
#   expected_deaths(CFR, r, T, ::Gamma)
#
# carries its own Mooncake-compatible reverse-mode rule
# (`_gamma_cdf` in `src/BVDOutbreakSize.jl`); the generic
#
#   expected_deaths(CFR, r, T, ::Any; alg)
#
# integration method is differentiated by Mooncake unrolling tape
# through the 64-point Gauss-Legendre integrator. This script reports
# per-call timings for the forward evaluation and for the
# Mooncake-backed gradient, so the cost of the analytic rule vs.
# Mooncake-through-quadrature can be checked after any change to
# either path.
#
# Run from the repo root with the test environment so BenchmarkTools
# is available:
#
#     julia --project=test test/scripts/bench_expected_deaths.jl

using BenchmarkTools: @benchmark, median
using BVDOutbreakSize: expected_deaths
using Distributions: Gamma
using Mooncake: Mooncake

const CFR, r, T, α, θ = 0.3, 0.05, 30.0, 0.3, 2.6

# Analytic Gamma method (dispatches on ::Gamma)
f_analytic = (CFR, r, T, α, θ) -> expected_deaths(CFR, r, T, Gamma(α, θ))

# Integration method (force the generic 4-arg method via `invoke`)
f_integral = (CFR, r, T, α, θ) -> invoke(expected_deaths,
    Tuple{Any, Any, Any, Any}, CFR, r, T, Gamma(α, θ))

println("Sanity check (values should agree to ~1e-6):")
println("  analytic : ", f_analytic(CFR, r, T, α, θ))
println("  integral : ", f_integral(CFR, r, T, α, θ))

rule_a = Mooncake.build_rrule(f_analytic, CFR, r, T, α, θ)
rule_i = Mooncake.build_rrule(f_integral, CFR, r, T, α, θ)

# `$`-interpolation keeps BenchmarkTools from re-resolving the globals
# each sample. `median` is preferred over `mean` because the GC tail
# from sampling itself biases the latter upward.
b_fwd_a = @benchmark $f_analytic($CFR, $r, $T, $α, $θ)
b_fwd_i = @benchmark $f_integral($CFR, $r, $T, $α, $θ)
b_ad_a  = @benchmark Mooncake.value_and_gradient!!(
    $rule_a, $f_analytic, $CFR, $r, $T, $α, $θ)
b_ad_i  = @benchmark Mooncake.value_and_gradient!!(
    $rule_i, $f_integral, $CFR, $r, $T, $α, $θ)

# Median time in nanoseconds -> microseconds.
μs(b) = round(median(b).time / 1e3; digits = 2)

println()
println("Per-call timings (median, μs):")
println("  forward  analytic : ", μs(b_fwd_a))
println("  forward  integral : ", μs(b_fwd_i))
println("  AD-grad  analytic : ", μs(b_ad_a))
println("  AD-grad  integral : ", μs(b_ad_i))
println()
println("AD-grad ratio (analytic / integral): ",
        round(median(b_ad_a).time / median(b_ad_i).time; digits = 3))
