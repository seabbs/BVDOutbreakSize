# Reference QuadGK-based rrule for cdf(Gamma(α, θ), x), kept here as
# a test-only comparison target for the series-based primary path in
# `src/gamma_cdf.jl`. Same forward value as the package's `_gamma_cdf`,
# but the α-partial is computed by adaptive Gauss-Kronrod on the
# Meijer-G tail-integral form
#
#     ∂_α P(α, y) = -(1/Γ(α)) ∫_y^∞ t^(α-1) e^(-t) (log t - ψ(α)) dt.
#
# Used by `bench_gamma_cdf_partial.jl` to compare AD-gradient cost
# between the series approach (primary) and adaptive quadrature
# (reference). Not loaded by the package; not run as part of the test
# suite.
#
# Math + integration tail rationale: see the discussion in
# `src/gamma_cdf.jl` and the EpiAware/CensoredDistributions PR #250
# comparison.

using ChainRulesCore: ChainRulesCore, NoTangent
using Distributions: Gamma, cdf, pdf
using Integrals: IntegralProblem, QuadGKJL, solve
using Mooncake: Mooncake
import SpecialFunctions
using SpecialFunctions: digamma

_gamma_cdf_quad(α, θ, x) = cdf(Gamma(α, θ), x)

function _gamma_cdf_quad_partials(α, θ, x)
    R = float(promote_type(typeof(α), typeof(θ), typeof(x)))
    y = x / θ
    y <= zero(y) && return zero(R), zero(R), zero(R)
    f      = pdf(Gamma(α, θ), x)
    df_dx  = f
    df_dθ  = -y * f
    ψα     = digamma(α)
    integrand = (t, _) -> t^(α - 1) * exp(-t) * (log(t) - ψα)
    prob = IntegralProblem(integrand, (float(y), float(Inf)))
    I = solve(prob, QuadGKJL()).u::Float64
    df_dα = -I / SpecialFunctions.gamma(α)
    return df_dα, df_dθ, df_dx
end

function ChainRulesCore.rrule(::typeof(_gamma_cdf_quad),
        α::Real, θ::Real, x::Real)
    y = _gamma_cdf_quad(α, θ, x)
    dα, dθ, dx = _gamma_cdf_quad_partials(α, θ, x)
    function _gamma_cdf_quad_pullback(ȳ)
        return (NoTangent(), dα' * ȳ, dθ' * ȳ, dx' * ȳ)
    end
    return y, _gamma_cdf_quad_pullback
end

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_gamma_cdf_quad), Float64, Float64, Float64}
