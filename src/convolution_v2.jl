## Continuous-time explicit-convolution v2 architecture (issue #5).
##
## The current model treats the latent state `C(s) = exp(r·s)` as
## cumulative *onsets* for the deaths convolution but as cumulative
## *infections* for the exports window. This file resolves that
## inconsistency: the latent state is unambiguously cumulative
## infections, and every observation stream builds its expected count
## through explicit delay convolutions starting from infection.
##
## ```
## I(t)       = exp(r·t)                  cumulative infections
## i_inf(t)   = r·exp(r·t)                daily infection incidence
## i_onset(t) = ∫₀ᵗ i_inf(s) f_inc(t−s)ds onset incidence (incubation)
##
## E[exports] = travel · ∫_{T−w}^{T} i_onset(s) ds
## E[deaths]  = CFR · ∫₀^T i_onset(s) F_otd(T−s) ds
## E[reports] = ∫₀^T i_onset(s) F_otr(T−s) ds
## ```
##
## CFR is now unambiguously the fraction of *onsets* that die. The
## deaths and reports expectations are nested convolutions (an onset
## incidence that is itself a convolution, integrated again against a
## delay CDF). To keep the cost near the single-convolution model we
## tabulate `i_onset` once per draw on a grid (`OnsetIncidence`) and
## reuse it across all three observation integrals.
##
## The whole layer reuses the package's Gauss-Legendre `integrate`
## helpers and is AD-compatible with Mooncake: every quantity flows
## through the incubation and onset densities alone, never their CDF
## shape-parameter derivatives.

"""
$(TYPEDSIGNATURES)

Daily infection incidence under exponential growth from a single
zoonotic seed at `s = 0`:

```math
i_{\\text{inf}}(s) = r \\, e^{r s}.
```

The derivative of the cumulative infections `I(s) = exp(r·s)`. Used as
the innermost driver of the onset convolution.
"""
infection_incidence(r, s) = r * exp(r * s)

"""
$(TYPEDSIGNATURES)

Onset (symptom) incidence at outbreak age `t`, the convolution of
infection incidence with the incubation density `f_inc`:

```math
i_{\\text{onset}}(t) = \\int_0^t r\\,e^{r s}\\, f_{\\text{inc}}(t - s)\\, ds.
```

`incub_dist` is the infection-to-onset (incubation) distribution. The
integrand is clustered towards `t`, where the incubation density has
mass, over a window set by the incubation scale (see
[`DELAY_SUPPORT_K`](@ref)), so a short incubation over a long elapsed
time is resolved accurately. Returns zero for `t <= 0`.
"""
function onset_incidence(r, incub_dist, t; alg = DEATH_INTEGRAL_ALG)
    t <= zero(t) && return zero(r * t)
    scale = _delay_scale(incub_dist)
    g = let r = r, incub_dist = incub_dist, t = t
        s -> begin
            d = t - s
            d <= 0 ? zero(r) : infection_incidence(r, s) * pdf(incub_dist, d)
        end
    end
    return integrate(g, zero(t), t, scale; alg)
end

"""
    ONSET_GRID_POINTS

Number of evenly spaced grid points over `[0, T]` used to tabulate the
onset-incidence curve in [`OnsetIncidence`](@ref). The onset curve is
smooth (it is a convolution), so a moderate grid with linear
interpolation tracks it closely while paying the inner incubation
convolution only once per grid node rather than once per outer node of
every observation integral.

Set to 65: the onset curve grows exponentially, so a uniform-grid
linear interpolation has its largest pointwise error in the tail near
`T`, but the observation integrals average that out (cumulative onsets
accurate to <0.1% at 65 points vs a 1025-point reference). Raising it
sharpens the tail at a near-linear build cost; lowering it speeds the
build but coarsens the deaths/reports convolutions, which weight the
steep tail.
"""
const ONSET_GRID_POINTS = 65

"""
$(TYPEDSIGNATURES)

Onset-incidence curve `i_onset(t)` tabulated on an evenly spaced grid
over `[0, T]`, built once per draw and reused across the exports,
deaths and reports observation integrals rather than recomputing the
inner incubation convolution at every outer quadrature node.

Construction evaluates [`onset_incidence`](@ref) at each of `npts`
grid points (`O(npts)` inner convolutions); each subsequent evaluation
is an `O(1)` linear interpolation. This is the precompute that keeps
the nested-convolution cost close to the single-convolution model: the
inner convolution is paid `npts` times per draw instead of
`(outer nodes) × (3 streams)` times.

`r` is the growth rate, `incub_dist` the incubation distribution and
`T` the elapsed time (upper grid bound). Fields:

$(TYPEDFIELDS)
"""
struct OnsetIncidence{R, V}
    "Growth rate `r`."
    r::R
    "Upper grid bound `T` (elapsed time)."
    T::R
    "Grid spacing `T / (npts - 1)`."
    dt::R
    "Tabulated `i_onset` at the grid nodes, index 1 at `t = 0`."
    vals::V
end

function OnsetIncidence(r, incub_dist, T;
        npts::Integer = ONSET_GRID_POINTS, alg = DEATH_INTEGRAL_ALG)
    Tt = float(T)
    dt = Tt / (npts - 1)
    v1 = onset_incidence(r, incub_dist, dt; alg)
    vals = Vector{typeof(v1)}(undef, npts)
    vals[1] = zero(v1)
    vals[2] = v1
    @inbounds for i in 3:npts
        vals[i] = onset_incidence(r, incub_dist, (i - 1) * dt; alg)
    end
    return OnsetIncidence(r, Tt, oftype(Tt, dt), vals)
end

# Linear interpolation of the tabulated onset incidence. Zero outside
# `[0, T]` (no onsets before the seed; the grid stops at the cut-off).
@inline function (oi::OnsetIncidence)(t)
    (t <= zero(t) || t >= oi.T) && return zero(eltype(oi.vals))
    pos  = t / oi.dt
    i    = floor(Int, pos) + 1
    frac = pos - (i - 1)
    return @inbounds oi.vals[i] + frac * (oi.vals[i + 1] - oi.vals[i])
end

"""
$(TYPEDSIGNATURES)

Expected cumulative onsets by `T` under v2, the integral of the onset
incidence:

```math
C_{\\text{onset}}(T) = \\int_0^T i_{\\text{onset}}(s)\\, ds.
```

This is the v2 analogue of the latent `C(T)` reported by the current
model (which equals `exp(r·T)` cumulative *infections*). Onsets lag
infections by the incubation period, so this is slightly below the
cumulative-infection count `exp(r·T)`. Pass an [`OnsetIncidence`](@ref)
to reuse the tabulated curve.
"""
function expected_onsets_v2(oi::OnsetIncidence; alg = CUMULATIVE_INTEGRAL_ALG)
    return integrate(oi, zero(oi.T), oi.T; alg)
end

"""
$(TYPEDSIGNATURES)

Expected detected exports by `T` under v2:

```math
\\mathbb{E}[\\text{exports}] = p \\cdot q \\cdot
    \\int_{T-w}^{T} i_{\\text{onset}}(s)\\, ds,
```

with `p` the Uganda ascertainment fraction, `q` the per-capita travel
rate and `w` the onset-to-detection window. Unlike the current model
the window is applied to *onset* incidence, so `w` is unambiguously
onset-to-detection rather than the incubation-plus-detection mixture of
the current detection window. Clamped strictly positive and finite for
the likelihood. Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_exports_v2(oi::OnsetIncidence, p, q, w;
        alg = CUMULATIVE_INTEGRAL_ALG)
    lo  = max(oi.T - w, zero(oi.T))
    raw = p * q * integrate(oi, lo, oi.T; alg)
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end

"""
$(TYPEDSIGNATURES)

Expected cumulative deaths by `T` under v2, the CFR-weighted
convolution of onset incidence with the onset-to-death CDF `F_otd`:

```math
\\mathbb{E}[\\text{deaths}] = \\mathrm{CFR} \\cdot
    \\int_0^T i_{\\text{onset}}(s)\\, F_{\\text{otd}}(T - s)\\, ds.
```

`death_delay` is the onset-to-death distribution; CFR is now
unambiguously the fraction of onsets that die. The CDF is supplied as a
precomputed [`ExportDeathDelay`](@ref) so the convolution differentiates
through the density alone (the AD backend lacks the Gamma CDF shape
derivative) and the CDF grid is built once per draw. Clamped strictly
positive and finite. Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_deaths_v2(oi::OnsetIncidence, death_delay::ExportDeathDelay,
        CFR; alg = CUMULATIVE_INTEGRAL_ALG)
    T = oi.T
    g = let oi = oi, death_delay = death_delay, T = T
        s -> oi(s) * _cdf_to(death_delay, T - s)
    end
    raw = CFR * integrate(g, zero(T), T; alg)
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end

"""
$(TYPEDSIGNATURES)

Expected cumulative reported cases by `T` under v2, the convolution of
onset incidence with the onset-to-report CDF `F_otr`:

```math
\\mathbb{E}[\\text{reports}] = p \\cdot
    \\int_0^T i_{\\text{onset}}(s)\\, F_{\\text{otr}}(T - s)\\, ds,
```

with `p` the DRC ascertainment fraction. Replaces the current model's
instantaneous `p · C(T)` reporting with a delayed report: a case is
reported only after its onset-to-report delay has elapsed, so recent
infections are not yet fully ascertained. `report_delay` is supplied as
a precomputed [`ExportDeathDelay`](@ref) (any onset-to-event CDF holder)
for AD safety and one-build-per-draw reuse. Clamped strictly positive
and finite. Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_reports_v2(oi::OnsetIncidence, report_delay::ExportDeathDelay,
        p; alg = CUMULATIVE_INTEGRAL_ALG)
    T = oi.T
    g = let oi = oi, report_delay = report_delay, T = T
        s -> oi(s) * _cdf_to(report_delay, T - s)
    end
    raw = p * integrate(g, zero(T), T; alg)
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end
