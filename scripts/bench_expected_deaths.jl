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
# Run from the repo root:
#
#     julia --project=. scripts/bench_expected_deaths.jl

using BVDOutbreakSize: expected_deaths
using Distributions: Gamma
using Mooncake: Mooncake

const CFR, r, T, α, θ = 0.3, 0.05, 30.0, 4.3, 2.6

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

# Warm-up
for _ in 1:5
    Mooncake.value_and_gradient!!(rule_a, f_analytic, CFR, r, T, α, θ)
    Mooncake.value_and_gradient!!(rule_i, f_integral, CFR, r, T, α, θ)
end

const N_FWD = 5_000
const N_AD  = 2_000

t_fwd_a = @elapsed for _ in 1:N_FWD; f_analytic(CFR, r, T, α, θ); end
t_fwd_i = @elapsed for _ in 1:N_FWD; f_integral(CFR, r, T, α, θ); end
t_ad_a  = @elapsed for _ in 1:N_AD
    Mooncake.value_and_gradient!!(rule_a, f_analytic, CFR, r, T, α, θ)
end
t_ad_i  = @elapsed for _ in 1:N_AD
    Mooncake.value_and_gradient!!(rule_i, f_integral, CFR, r, T, α, θ)
end

μs(t, n) = round(t / n * 1e6; digits = 2)

println()
println("Per-call timings (μs):")
println("  forward  analytic : ", μs(t_fwd_a, N_FWD))
println("  forward  integral : ", μs(t_fwd_i, N_FWD))
println("  AD-grad  analytic : ", μs(t_ad_a,  N_AD))
println("  AD-grad  integral : ", μs(t_ad_i,  N_AD))
println()
println("AD-grad ratio (analytic / integral): ",
        round(t_ad_a / t_ad_i; digits = 3))
