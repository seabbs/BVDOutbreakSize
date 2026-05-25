# # Stochastic latent infection process (issue #48)
#
# A stochastic latent infection process to replace the deterministic
# `C(s) = exp(r s)` trajectory of the baseline model. This file is
# additive and self-contained: it adds new building-block submodels to
# the package without touching any existing code, so the production
# model is unaffected.
#
# Design rules followed (cross-project directives):
#
# 1. *Modular submodels.* Every prior — for `τ`, `m`, `σ`, the latent
#    increments, and the incubation delay — lives in its own submodel,
#    so the joint composer takes them as keyword arguments and any one
#    can be swapped without editing model bodies. No `Normal(...)` /
#    `Gamma(...)` constants are baked inside `@model` bodies.
# 2. *Observation distributions are injected* by the composer, not
#    hardcoded here; the helpers in this file only produce expected
#    counts on the stochastic trajectory.
# 3. *All delays are prior-based* (incubation, onset-to-death, detection
#    window): each is a sampled submodel. No fixed generation interval
#    is needed because growth is the sampled rate `r` plus latent
#    increments, not a renewal kernel.
# 4. *No inline using/import here.* All package imports live on the
#    module page `src/BVDOutbreakSize.jl`.
# 5. *Onset staging.* Observations stage as infections → onset →
#    death/report/detection. The LNA trajectory represents latent
#    log-cumulative *infections*; the onset cumulative `C_o(t)` is
#    convolved off it once via [`onset_cumulative`](@ref) and reused by
#    every downstream stream — deaths, reports, exports, and exports-
#    deaths — rather than re-convolving infections to each stream.
# 6. *Non-centred parameterisation* for the latent log-incidence
#    increments: sample standard-Normal `z`, scale by `σ` inside the
#    model, so the funnel between `σ` and the increments is avoided.
# 7. *Single-side censoring only.* The earliest-onset and TMRCA timing
#    terms use `censored(Normal(T, σ); upper = g)` (one-sided). This
#    reduces to a `logcdf` call on the underlying `Normal`, which
#    Mooncake differentiates; two-sided / interval `censored` and
#    `CensoredDistributions` are avoided. The honest cost is that the
#    bound *only* uses the upper-tail information (T ≥ g); any further
#    refinement (e.g. an interval) would need a manually-coded
#    logcdf difference rather than `censored`.

# ---------- Knot grid -------------------------------------------------

"""
    STOCH_GROWTH_KNOTS

Number of latent log-incidence increments (knots) in the stochastic
growth process. The continuous-time trajectory is reconstructed by
interpolating the cumulative log-incidence levels onto a fixed grid of
this many points over `[0, T]`. More knots resolve finer early-phase
fluctuations at higher sampling cost.
"""
const STOCH_GROWTH_KNOTS = 24

# ---------- Trajectory helpers ----------------------------------------

"""
$(TYPEDSIGNATURES)

Build the latent log-incidence levels `logi[1:n]` on the knot grid
from a growth rate `r`, outbreak age `T`, process-noise scale `σ` and
a vector of `n - 1` standard-Normal increments `z`. Seed `log i(0) =
log r` so a deterministic-growth limit (`σ = 0`) reproduces the
exponential-growth incidence rate `i(s) = r · exp(r s)` exactly (since
`dC/ds = r exp(r s)` when `C(s) = exp(r s)`). Step `Δ = T / (n - 1)`:

```math
\\log i(s_{j+1}) = \\log i(s_j) + r\\,Δ
   + \\sigma\\,\\sqrt{Δ}\\; z_j.
```

Working on log-incidence (not log-cumulative) keeps the modelled
incidence rate strictly positive under any sample, so the implied
cumulative trajectory built from it stays monotone — the property we
want of a cumulative-infection process.
"""
function lna_logi(r::Real, T::Real, σ::Real, z::AbstractVector)
    n = length(z) + 1
    Δ = T / (n - 1)
    sqrtΔ = sqrt(Δ)
    logr = log(max(r, eps(typeof(r))))
    logi = Vector{typeof(r * Δ + σ * sqrtΔ * zero(eltype(z)))}(undef, n)
    logi[1] = logr
    for j in 1:(n - 1)
        logi[j + 1] = logi[j] + r * Δ + σ * sqrtΔ * z[j]
    end
    return logi
end

"""
$(TYPEDSIGNATURES)

Build a continuous, differentiable positive infection-incidence
trajectory `i(s)` on `[0, T]` from log-incidence levels `logi` on the
knot grid. Log-linear interpolation between knots (so `i(s) > 0`
everywhere), flat outside `[0, T]`. The natural drop-in for what the
deterministic model would call `r · exp(r s)`.
"""
function lna_incidence(logi::AbstractVector, T::Real)
    n = length(logi)
    return function (s)
        s <= zero(s) && return zero(eltype(logi))
        s >= T && return exp(logi[n])
        pos = (s / T) * (n - 1)
        i = floor(Int, pos) + 1
        i = min(i, n - 1)
        frac = pos - (i - 1)
        lc = logi[i] + frac * (logi[i + 1] - logi[i])
        return exp(lc)
    end
end

"""
$(TYPEDSIGNATURES)

Build the monotone cumulative-infection trajectory `C(s) = ∫_0^s i(u)
du` from log-incidence levels `logi` on the knot grid, by trapezoid
integration on the segments. The returned closure interpolates
linearly between the knot cumulatives for arbitrary `s ∈ [0, T]`. By
construction `C(s)` is non-decreasing because the incidence rate is
positive, so it is a well-defined cumulative-infection process.
"""
function lna_trajectory(logi::AbstractVector, T::Real)
    n = length(logi)
    Δ = T / (n - 1)
    Tt = typeof(exp(zero(eltype(logi))) * Δ)
    C = Vector{Tt}(undef, n)
    C[1] = zero(Tt)
    for j in 1:(n - 1)
        ## Trapezoid on (exp(logi[j]) + exp(logi[j+1])) / 2.
        C[j + 1] = C[j] + (exp(logi[j]) + exp(logi[j + 1])) * Δ / 2
    end
    return function (s)
        s <= zero(s) && return C[1]
        s >= T && return C[n]
        pos = (s / T) * (n - 1)
        i = floor(Int, pos) + 1
        i = min(i, n - 1)
        frac = pos - (i - 1)
        return C[i] + frac * (C[i + 1] - C[i])
    end
end

# ---------- Onset staging ---------------------------------------------

"""
$(TYPEDSIGNATURES)

Cumulative onsets by elapsed time `t`, obtained by convolving the
latent infection incidence `incidence(s) = C'(s)` against the
incubation CDF `F_inc`:

```math
C_o(t) = \\int_0^t i(s)\\,F_\\mathrm{inc}(t - s)\\,ds.
```

This is the *infections → onset* stage of the observation pipeline. It
is computed once from the latent trajectory and reused by every
downstream onset-keyed stream (deaths, reports, detection), so we never
convolve raw infections directly with the downstream delays. `alg`
defaults to [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function onset_cumulative(incidence, F_inc, t::Real;
        alg = CUMULATIVE_INTEGRAL_ALG)
    t <= zero(t) && return zero(t)
    f = let incidence = incidence, F_inc = F_inc, t = t
        s -> incidence(s) * F_inc(t - s)
    end
    return integrate(f, zero(t), t; alg)
end

"""
$(TYPEDSIGNATURES)

Build a callable onset-incidence trajectory `i_o(t)` from the latent
infection incidence and the incubation density `f_inc`:

```math
i_o(t) = \\int_0^t i(s)\\,f_\\mathrm{inc}(t - s)\\,ds.
```

Returned as a closure so it can be passed to the downstream
observation integrals (deaths convolution, exports person-time, etc.)
in place of raw infection incidence. Differentiates through the latent
log-incidence levels and the incubation parameters under Mooncake.
"""
function onset_incidence(incidence, f_inc;
        alg = CUMULATIVE_INTEGRAL_ALG)
    return function (t)
        t <= zero(t) && return zero(t)
        g = let incidence = incidence, f_inc = f_inc, t = t
            s -> incidence(s) * f_inc(t - s)
        end
        return integrate(g, zero(t), t; alg)
    end
end

# ---------- Building-block prior submodels ----------------------------

"""
$(TYPEDSIGNATURES)

Doubling-time prior submodel. Returns `(; τ)`. Default
`τ ~ LogNormal(log(14), 0.4)` matches the baseline
`exponential_growth_model`.
"""
@model function doubling_time_model(;
        tau_prior = LogNormal(log(14), 0.4))
    τ ~ tau_prior
    return (; τ)
end

"""
$(TYPEDSIGNATURES)

Doubling-time multiplier prior submodel. Returns `(; m)`. Default
`m ~ Normal(7, 2.5)` on `(0, 13]` matches the baseline.
"""
@model function multiplier_model(;
        m_prior = truncated(Normal(7.0, 2.5);
                            lower = 0, upper = 13.0))
    m ~ m_prior
    return (; m)
end

"""
$(TYPEDSIGNATURES)

Process-noise prior submodel for the log-Gaussian growth relaxation.
Returns `(; σ)`. Default `σ ~ Normal⁺(0, 0.3)` is weakly informative
and centred at the deterministic limit, so the data must argue for
early-phase variance.
"""
@model function process_noise_model(;
        sigma_prior = truncated(Normal(0.0, 0.3); lower = 0))
    σ ~ sigma_prior
    return (; σ)
end

"""
$(TYPEDSIGNATURES)

Latent-increment prior submodel. Returns `(; z)`, an `n_knots - 1`
vector of standard-Normal increments. Non-centred: `σ` enters
downstream, so the funnel between scale and increments is avoided.
"""
@model function latent_increments_model(;
        n_knots::Integer = STOCH_GROWTH_KNOTS,
        z_prior = Normal(0, 1))
    z ~ filldist(z_prior, n_knots - 1)
    return (; z)
end

"""
$(TYPEDSIGNATURES)

Incubation (infection-to-onset) delay submodel. Returns
`(; α_inc, θ_inc, dist = Gamma(α_inc, θ_inc), F, f)`, where `F(x)` is
the incubation CDF and `f(x)` the density, both as closures. The
parameter names carry the `_inc` suffix to avoid collision with the
onset-to-death submodel that already uses `α`, `θ`. Default priors are
weakly-informative, spanning published Ebola-family incubation periods
(median around 6–10 days, shape > 1): `α_inc ~ Normal⁺(3, 1)`,
`θ_inc ~ Normal⁺(3, 1.5)`. Override `alpha_prior` / `theta_prior` with
any positive-supported distribution.
"""
@model function incubation_model(;
        alpha_prior = truncated(Normal(3.0, 1.0); lower = 1e-3),
        theta_prior = truncated(Normal(3.0, 1.5); lower = 1e-3))
    α_inc ~ alpha_prior
    θ_inc ~ theta_prior
    dist = Gamma(α_inc, θ_inc)
    f = let d = dist; x -> pdf(d, max(x, zero(x))); end
    ## `F` is written as the inner integral of `f`, NOT
    ## `Distributions.cdf(::Gamma, x)`, because Mooncake does not
    ## support the Gamma CDF shape-parameter derivative; the package
    ## uses the same workaround for the onset-to-death convolution
    ## (see `integrate_exports_deaths`). The integral differentiates
    ## through the density alone, which Mooncake handles.
    F = let f = f
        x -> begin
            x <= zero(x) && return zero(x)
            return integrate(f, zero(x), x)
        end
    end
    return (; α_inc, θ_inc, dist, F, f)
end

# ---------- Composed growth submodel ----------------------------------

"""
$(TYPEDSIGNATURES)

Stochastic latent growth submodel (issue #48).

Composes the prior sub-submodels (`tau`, `multiplier`, `process_noise`,
`increments`) and the trajectory helpers into the same `growth_state`
interface as the deterministic baseline. Every prior is passed in as a
submodel so the composer can swap any of them without editing this
body. `σ = 0` recovers the deterministic baseline exactly, so the
baseline is nested.

## Returns

A NamedTuple `(; τ, r, m, T, C_T, cumulative, incidence, logi)`:
the first six fields keep the deterministic submodel's interface;
`incidence` is the positive infection-incidence closure `i(s)` used
by the onset-staging pipeline; `logi` is the vector of latent
log-incidence-rate levels on the knot grid. `C_T` is the monotone
cumulative incidence at `T`, built by trapezoid integration of `i(s)`
on the knot grid, so it is non-negative by construction.
"""
@model function stochastic_growth_model(;
        tau           = doubling_time_model(),
        multiplier    = multiplier_model(),
        process_noise = process_noise_model(),
        increments    = latent_increments_model(),
        n_knots::Integer = STOCH_GROWTH_KNOTS)
    tau_state ~ to_submodel(tau, false)
    mult_state ~ to_submodel(multiplier, false)
    noise_state ~ to_submodel(process_noise, false)
    incs_state ~ to_submodel(increments, false)

    τ = tau_state.τ
    m = mult_state.m
    σ = noise_state.σ
    z = incs_state.z
    r := log(2) / τ
    T := m * τ

    logi = lna_logi(r, T, σ, z)
    incidence  = lna_incidence(logi, T)
    cumulative = lna_trajectory(logi, T)
    C_T := cumulative(T)
    return (; τ, r, m, T, C_T, cumulative, incidence, logi)
end

# ---------- Timing terms (single-side censoring only) ----------------

"""
$(TYPEDSIGNATURES)

Earliest-known-onset timing term (issue #48).

The first confirmed symptom-onset date is an "at-or-before" observation
on the outbreak age `T`: the outbreak must be at least `onset_delta`
days old because a case had already had onset by then. Modelled as a
soft *single-side* censored bound (one-sided is the only `censored`
form Mooncake differentiates, via the underlying `logcdf`):

```math
p_\\text{onset}(T) = \\Pr[\\mathrm{Normal}(T, \\sigma_o) \\ge g]
    = \\Phi\\!\\left(\\frac{T - g}{\\sigma_o}\\right),\\quad
g = \\texttt{onset\\_delta}.
```

The SD `σ_o` absorbs the infection-to-onset (incubation) delay plus
onset-date recording uncertainty, and is sampled from an injected
submodel `onset_sd` (default a weakly-informative `Normal⁺(14, 5)`),
so the delay carries prior uncertainty rather than being fixed.
Passing `onset_delta = missing` makes the term a no-op.

The single-side approximation is honest about the information used:
only the upper-tail probability `P(T ≥ g)` enters. An interval bound
(e.g. censoring also from below if a known absence-of-cases date were
available) would need a manually-coded logcdf difference because the
two-sided `censored` form does not differentiate under Mooncake.
"""
@model function onset_timing_model(T;
        onset_delta::Union{Missing, Real},
        onset_sd = onset_sd_model())
    ismissing(onset_delta) && return (; onset_delta, onset_sd = missing)
    onset_sd_state ~ to_submodel(onset_sd, false)
    σ_o = onset_sd_state.onset_sd
    onset_delta ~ censored(Normal(T, σ_o); upper = onset_delta)
    return (; onset_delta, onset_sd = σ_o)
end

"""
$(TYPEDSIGNATURES)

Prior on the SD of the earliest-onset bound's location.
Returns `(; onset_sd)`. Default `onset_sd ~ Normal⁺(14, 5)` days,
weakly informative on plausible incubation-plus-recording spreads.
"""
@model function onset_sd_model(;
        prior = truncated(Normal(14.0, 5.0); lower = 1e-3))
    onset_sd ~ prior
    return (; onset_sd)
end

"""
$(TYPEDSIGNATURES)

Genetic-TMRCA timing submodel. Single-side censored bound on `T`,
mirroring the package's existing `genetic_seeding_model` but lifted
into this file's modular submodel API so the composer can take it
as a keyword argument and the literal `Normal(...)` does not sit
inline in the composer's body. `tmrca_days_sd` is the genetic-clock
SD (a data-derived input, not a delay distribution, so it is not
sampled).
"""
@model function tmrca_timing_model(T;
        tmrca_days::Union{Missing, Real},
        tmrca_days_sd::Real)
    ismissing(tmrca_days) && return (; tmrca_days, tmrca_days_sd)
    tmrca_days ~ censored(Normal(T, tmrca_days_sd); upper = tmrca_days)
    return (; tmrca_days, tmrca_days_sd)
end
