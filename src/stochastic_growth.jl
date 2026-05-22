# # Stochastic latent growth (PROTOTYPE, issue #48)
#
# PROTOTYPE. A stochastic latent infection process to replace the
# deterministic `C(s) = exp(r s)` trajectory of the baseline model. This
# file is additive and self-contained: it adds new building-block
# submodels to the package without touching any existing code, so the
# production model is unaffected.
#
# It deliberately exposes the SAME `growth_state` interface as the
# baseline `exponential_growth_model` (the NamedTuple
# `(; τ, r, m, T, C_T, cumulative)`), so the existing observation
# submodels — which only read `growth_state.cumulative`,
# `growth_state.C_T`, `growth_state.r` and `growth_state.T` — consume it
# unchanged. The joint composer that wires it to those observation
# submodels lives in `scripts/prototype_stochastic.jl`, following the
# repo convention that the `@model` observation blocks live outside the
# package (issue #81); only the genuinely new, self-contained growth
# pieces live here so they are importable and testable.
#
# Inference route: a continuous log-Gaussian / linear-noise relaxation
# of the latent log-cumulative incidence, so the model stays
# differentiable and NUTS + Mooncake can sample it directly. No discrete
# latent counts, so no marginalisation or particle filter is needed for
# this tractable version. See docs/proposals/stochastic-latent.md.

"""
    STOCH_GROWTH_KNOTS

Number of latent log-incidence increments (knots) in the stochastic
growth process. The continuous-time trajectory is reconstructed by
interpolating the cumulative log-incidence levels onto a fixed grid of
this many points over `[0, T]`. More knots resolve finer early-phase
fluctuations at higher sampling cost.
"""
const STOCH_GROWTH_KNOTS = 24

"""
$(TYPEDSIGNATURES)

Build a continuous, differentiable cumulative-incidence trajectory
`C(s)` on `[0, T]` from log-incidence levels `logC` on the knot grid
`0 = s_1 < … < s_n = T`. Linear interpolation in `log C` (so `C` stays
positive and smooth), flat outside `[0, T]`. Drop-in replacement for the
deterministic `s -> exp(r * s)`.

Interpolating in log space keeps the trajectory positive under any NUTS
proposal and differentiable everywhere except the measure-zero knot
set, so Mooncake pushes gradients through every downstream integral.
"""
function lna_trajectory(logC::AbstractVector, T::Real)
    n = length(logC)
    return function (s)
        s <= zero(s) && return exp(logC[1])
        s >= T && return exp(logC[n])
        pos = (s / T) * (n - 1)
        i = floor(Int, pos) + 1
        i = min(i, n - 1)
        frac = pos - (i - 1)
        lc = logC[i] + frac * (logC[i + 1] - logC[i])
        return exp(lc)
    end
end

"""
$(TYPEDSIGNATURES)

Reconstruct the log-cumulative-incidence levels on the knot grid from a
growth rate `r`, an outbreak age `T`, a process-noise scale `σ`, and a
vector of `n - 1` standard-Normal increments `z`. Seed `log C(0) = 0`
(a single zoonotic case). Step `Δ = T / (n - 1)`:

```math
\\log C(s_{j+1}) = \\log C(s_j) + r\\,Δ + \\sigma\\,\\sqrt{Δ}\\; z_j.
```

Returned as a vector so it can be both interpolated by
[`lna_trajectory`](@ref) and read at its endpoint for `C(T)`. Separated
out from the submodel so it is unit-testable without Turing.
"""
function lna_logC(r::Real, T::Real, σ::Real, z::AbstractVector)
    n = length(z) + 1
    Δ = T / (n - 1)
    sqrtΔ = sqrt(Δ)
    logC = Vector{typeof(r * Δ + σ * sqrtΔ * zero(eltype(z)))}(undef, n)
    logC[1] = zero(eltype(logC))
    for j in 1:(n - 1)
        logC[j + 1] = logC[j] + r * Δ + σ * sqrtΔ * z[j]
    end
    return logC
end

"""
$(TYPEDSIGNATURES)

Stochastic latent growth submodel (PROTOTYPE, issue #48).

A log-Gaussian / linear-noise relaxation of a stochastic incidence
process. Replaces the deterministic `C(s) = exp(r s)` with a random
log-cumulative trajectory whose mean drift is exponential growth at rate
`r` but which carries early-phase demographic variance. The latent
log-increments `z` are continuous standard Normals, so the submodel is
differentiable and samples under NUTS + Mooncake with no discrete latent
state. `σ = 0` recovers the deterministic baseline exactly, so the
baseline is nested.

The non-centred parameterisation (sample standard-Normal `z`, scale by
`σ` inside the model) avoids the funnel between `σ` and the increments.

## Returns

A NamedTuple with the SAME fields the deterministic growth submodel
exposes — `(; τ, r, m, T, C_T, cumulative)` — so the existing
observation submodels are unchanged. `C_T = C(T)` is read off the
stochastic trajectory endpoint (not `2^m`), and `cumulative` is the
interpolated random trajectory from [`lna_trajectory`](@ref).

A fuller linear-noise approximation would scale per-step variance by
`1/C` (demographic noise loudest at O(1–10) counts); that refinement is
left as aspiration (see the proposal) as it complicates the gradient and
is not needed to demonstrate the route.
"""
@model function stochastic_growth_model(;
        tau_prior   = LogNormal(log(14), 0.4),
        m_prior     = truncated(Normal(7.0, 2.5);
                                lower = 0, upper = 13.0),
        sigma_prior = truncated(Normal(0.0, 0.3); lower = 0),
        n_knots::Integer = STOCH_GROWTH_KNOTS)
    τ ~ tau_prior
    m ~ m_prior
    σ ~ sigma_prior
    r := log(2) / τ
    T := m * τ
    z ~ filldist(Normal(0, 1), n_knots - 1)

    logC = lna_logC(r, T, σ, z)
    cumulative = lna_trajectory(logC, T)
    C_T := exp(logC[n_knots])
    return (; τ, r, m, T, C_T, cumulative)
end

"""
$(TYPEDSIGNATURES)

Earliest-known-onset timing term (PROTOTYPE, issue #48).

The first confirmed symptom-onset date is an "at-or-before" observation
on the outbreak age `T`: the outbreak must be at least `onset_delta`
days old because a case had already had onset by then. Modelled as a
soft one-sided bound, mirroring the genetic TMRCA term
(`genetic_seeding_model`): `onset_delta` is a noisy, right-censored
reading of `T`, contributing

```math
p_\\text{onset}(T) = \\Pr[\\mathrm{Normal}(T, \\sigma_o) \\ge g]
    = \\Phi\\!\\left(\\frac{T - g}{\\sigma_o}\\right), \\qquad
g = \\texttt{onset\\_delta}.
```

`onset_delta` is the elapsed time (days) from the earliest onset to the
cut-off. The SD on the location of the bound, `σ_o`, is not a fixed
constant: it is sampled from `onset_sd_prior`, because it is dominated by
the infection-to-onset (incubation) delay plus onset-date recording
uncertainty, and the project requires every delay to carry prior
uncertainty into the fit rather than be hardcoded. The default prior is
weakly informative, `σ_o ~ Normal⁺(14, 5)` days, spanning plausible BVD
incubation-plus-recording spreads where direct data is thin. Passing
`onset_delta = missing` makes the term a no-op (and skips sampling
`σ_o`, so the term adds no parameters when unused).

This is what makes the earliest onset (24 Apr 2026) *usable*: under the
deterministic single-seed curve the implied first case is detection-
limited and inconsistent with 131 deaths ~24 days later, so it was
discarded. As a censored bound on `T` it just says the outbreak is at
least that old, which the stochastic trajectory can satisfy without
distorting the rest of the curve.
"""
@model function onset_timing_model(T;
        onset_delta::Union{Missing, Real},
        onset_sd_prior = truncated(Normal(14.0, 5.0); lower = 1e-3))
    if !ismissing(onset_delta)
        ## The bound's location SD absorbs the infection-to-onset
        ## (incubation) delay and onset-date recording noise; sample it
        ## so that delay uncertainty propagates, rather than fixing it.
        onset_sd ~ onset_sd_prior
        onset_delta ~ censored(Normal(T, onset_sd); upper = onset_delta)
        return (; onset_delta, onset_sd)
    end
    return (; onset_delta, onset_sd = missing)
end
