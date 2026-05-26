module BVDOutbreakSize

using Statistics: quantile, mean, std
using Printf: @sprintf
using TOML
using DataFrames: DataFrame
using DataFramesMeta
using Chain: @chain
using Random: MersenneTwister
using Dates: Date, date2epochdays, epochdays2date
using ADTypes: AutoMooncake
using Mooncake: Mooncake
using ChainRulesCore: ChainRulesCore, NoTangent
using SpecialFunctions: gamma_inc, digamma, loggamma
import SpecialFunctions
using Turing
using Turing.DynamicPPL: InitFromPrior
import FlexiChains
using DocStringExtensions
using Distributions: Distribution, Gamma, cdf, ccdf, mgf, pdf, Poisson, NegativeBinomial
using Integrals: IntegralProblem, GaussLegendre, QuadGKJL, solve
import FastGaussQuadrature
import CairoMakie
import AlgebraOfGraphics as AoG
import PairPlots
using CairoMakie: Figure, Axis, hist!, density!, vlines!, vspan!,
                  lines!, scatter!

export REPORT_SCENARIOS,
       ITURI_POPULATION, ITURI_DAILY_TRAVEL,
       ITURI_DAILY_TRAVEL_SD,
       EXPORTED_CASES, EXPORTS_DEATHS,
       load_observations,
       summary_table, posterior_summary,
       fit_diagnostics, diagnostics_table,
       streams_table, comparison_table,
       nuts_sample, default_adtype,
       DEATH_INTEGRAL_ALG, CUMULATIVE_INTEGRAL_ALG,
       integrate, delay_convolution,
       integrate_cumulative, integrate_exports_deaths,
       expected_exports, expected_exports_deaths,
       ExportDeathDelay, EXPORT_DELAY_GRID_POINTS,
       plot_cumulative_cases, plot_density_overlay, plot_prior_predictive,
       plot_posterior_predictive, plot_posterior_predictive_grid,
       plot_pair, plot_start_date_pair, plot_estimate_comparison,
       plot_cfr_prior,
       predict_no_onward_deaths, plot_no_onward_deaths,
       forecast_reported, forecast_table, plot_forecast,
       forecast_vs_truth, plot_forecast_vs_truth

"""
    REPORT_SCENARIOS

Published point estimates of cumulative cases `C_T` from McCabe et
al. (Imperial College London, 20 May 2026 update), as `(label, value)`
tuples in the order they appear in Tables 1 and 2.
"""
const REPORT_SCENARIOS = [
    ("Method 1 Ituri, w=10 d",   470),
    ("Method 1 Ituri, w=15 d",   313),
    ("Method 1 Ituri, w=20 d",   235),
    ("Method 1 +N. Kivu, w=10",  617),
    ("Method 1 +N. Kivu, w=15",  412),
    ("Method 1 +N. Kivu, w=20",  309),
    ("Method 2 τ=14 d, CFR 26%", 860),
    ("Method 2 τ=14 d, CFR 33%", 678),
    ("Method 2 τ=14 d, CFR 40%", 559),
    ("Method 2 τ= 7 d, CFR 26%", 1386),
    ("Method 2 τ= 7 d, CFR 33%", 1092),
    ("Method 2 τ= 7 d, CFR 40%", 901),
    ("Method 2 τ=21 d, CFR 26%", 730),
    ("Method 2 τ=21 d, CFR 33%", 575),
    ("Method 2 τ=21 d, CFR 40%", 474),
]

"""
    ITURI_POPULATION

Source population for the Ituri Province (McCabe et al., Table 1).
"""
const ITURI_POPULATION   = 4_392_200

"""
    ITURI_DAILY_TRAVEL

Default prior mean for the daily outbound traveller volume from
Ituri Province across seven points of entry.
"""
const ITURI_DAILY_TRAVEL = 1_871

"""
    ITURI_DAILY_TRAVEL_SD

Default prior SD for the daily outbound traveller volume, covering
point-of-entry-to-point-of-entry variation and reporting uncertainty
in the underlying mobility survey.
"""
const ITURI_DAILY_TRAVEL_SD = 200

"""
    EXPORTED_CASES

BVD cases detected in Uganda having travelled from Ituri Province.
"""
const EXPORTED_CASES = 2

"""
    EXPORTS_DEATHS

Deaths recorded in Uganda among the exported BVD cases.
"""
const EXPORTS_DEATHS = 1

# Include reverse rules for the gamma CDF
include("gamma_cdf.jl")

"""
$(TYPEDSIGNATURES)

Load the observation block from `data/observations.toml` and return
it as a `NamedTuple`. Each observation in the TOML is a subtable
with `value = …` and `source = "…"`; this function returns both the
parsed numeric values and a parallel `sources::NamedTuple` of
citation strings so they can be printed alongside the data.

Fields returned:

- `exported_cases::Int`
- `exports_deaths::Int`
- `total_deaths::Int`
- `reported_cases::Int` — DRC suspected cumulative case count.
- `confirmed_cases::Union{Int, Missing}` — DRC laboratory-confirmed
  cumulative case count, the truth-anchor on the latent
  eventually-confirmable pool ``C(T)`` (reported counts are an inflated
  view); `missing` when no `confirmed_cases` block is present.
- `daily_outbound_travellers::Real`
- `daily_outbound_travellers_sd::Real`
- `source_population::Int`
- `genetic_tmrca_days::Union{Real, Missing}` — estimated time to the
  most recent common ancestor (TMRCA) in days before `as_of_date`, a
  soft lower bound on the seeding time `T`; `missing` when no
  `genetic_tmrca` block is present.
- `genetic_tmrca_days_sd::Union{Real, Missing}` — SD (days) on the
  location of that floor; `missing` when absent.
- `genetic_tmrca_alt_days::Union{Real, Missing}` — TMRCA (days before
  `as_of_date`) under the alternative clock rate, for the clock-rate
  sensitivity; `missing` when no `alt_date` is present.
- `genetic_tmrca_alt_days_sd::Union{Real, Missing}` — SD (days) on the
  alternative-clock floor; `missing` when absent.
- `sources::NamedTuple` — citation per loaded field, with the same keys
  as the numeric fields above. Optional fields (`confirmed_cases`,
  `genetic_tmrca`) carry `missing` rather than a citation when absent.
"""
function load_observations(
        path::AbstractString = joinpath(@__DIR__, "..", "data",
                                        "observations.toml"))
    raw = TOML.parsefile(path)
    _val(k) = raw[k]["value"]
    _src(k) = String(raw[k]["source"])
    as_of = String(raw["as_of_date"])
    _gap(d) = date2epochdays(Date(as_of)) - date2epochdays(Date(String(d)))
    ## Days between a recorded event date and the cut-off, used as the
    ## elapsed-time offset for the timing terms. A scalar date gives a
    ## `missing` offset when absent (so its term is a no-op).
    _delta(k) = haskey(raw, k) ? _gap(_val(k)) : missing
    ## Daily export-death series, earliest dated death (index 1) to the
    ## cut-off day (offset 0, kept); empty when no dates are present.
    export_deaths_daily = if haskey(raw, "export_death_dates")
        offs = Int[_gap(d) for d in _val("export_death_dates")]
        isempty(offs) ? Int[] :
            Int[count(==(δ), offs) for δ in maximum(offs):-1:0]
    else
        Int[]
    end
    has_gen = haskey(raw, "genetic_tmrca")
    return (;
        as_of_date                   = as_of,
        exported_cases               = Int(_val("exported_cases")),
        exports_deaths               = Int(_val("exports_deaths")),
        total_deaths                 = Int(_val("total_deaths")),
        reported_cases               = Int(_val("reported_cases")),
        confirmed_cases              = haskey(raw, "confirmed_cases") ?
            Int(_val("confirmed_cases")) : missing,
        daily_outbound_travellers    = float(
            _val("daily_outbound_travellers")),
        daily_outbound_travellers_sd = float(
            _val("daily_outbound_travellers_sd")),
        source_population            = Int(_val("source_population")),
        export_deaths_daily          = export_deaths_daily,
        first_export_detection_delta = _delta("first_export_detection_date"),
        genetic_tmrca_days           = has_gen ?
            _gap(raw["genetic_tmrca"]["date"]) : missing,
        genetic_tmrca_days_sd        = has_gen ?
            float(raw["genetic_tmrca"]["days_sd"]) : missing,
        genetic_tmrca_alt_days       =
            has_gen && haskey(raw["genetic_tmrca"], "alt_date") ?
            _gap(raw["genetic_tmrca"]["alt_date"]) : missing,
        genetic_tmrca_alt_days_sd    =
            has_gen && haskey(raw["genetic_tmrca"], "alt_days_sd") ?
            float(raw["genetic_tmrca"]["alt_days_sd"]) : missing,
        sources = (;
            exported_cases               = _src("exported_cases"),
            exports_deaths               = _src("exports_deaths"),
            total_deaths                 = _src("total_deaths"),
            reported_cases               = _src("reported_cases"),
            confirmed_cases              = haskey(raw, "confirmed_cases") ?
                _src("confirmed_cases") : missing,
            daily_outbound_travellers    = _src("daily_outbound_travellers"),
            daily_outbound_travellers_sd = _src("daily_outbound_travellers_sd"),
            source_population            = _src("source_population"),
            genetic_tmrca                = has_gen ?
                String(raw["genetic_tmrca"]["source"]) : missing,
        ),
    )
end

"""
$(TYPEDSIGNATURES)

Mooncake reverse-mode AD with default `Mooncake.Config()`. Used as
the NUTS `adtype` keyword.
"""
default_adtype() = AutoMooncake(; config = Mooncake.Config())

"""
$(TYPEDSIGNATURES)

NUTS on `model`, parallel chains via `MCMCThreads`. Chains
initialise from the prior to keep the sampler away from the
boundary of constrained variables.
"""
function nuts_sample(model;
        samples::Integer    = 1_000,
        chains::Integer     = 4,
        target_accept::Real = 0.95,
        seed::Integer       = 20260518,
        progress::Bool      = false,
        adtype              = default_adtype())
    rng = MersenneTwister(seed)
    return sample(
        rng,
        model,
        NUTS(target_accept; adtype),
        MCMCThreads(),
        samples, chains;
        initial_params = fill(InitFromPrior(), chains),
        progress       = progress,
    )
end

## --- Shared Gauss-Legendre quadrature -----------------------------------
##
## A single reusable integrator backs every forward integral in the
## package: the gamma onset-to-death convolution for deaths, the
## at-risk person-time integral for exports, the deaths-among-exports
## convolution, and the counterfactual and forecast integrals. Each of
## those was previously a near-duplicate inline integrand; they now all
## route through `integrate` and the typed helpers below.

"""
    DEATH_INTEGRAL_ALG

Gauss-Legendre quadrature scheme (`n = 64`) used for the deaths
onset-to-death convolution, the no-onward-transmission counterfactual,
and the forecast deaths integral.
"""
const DEATH_INTEGRAL_ALG = GaussLegendre(; n = 64)

"""
    CUMULATIVE_INTEGRAL_ALG

Gauss-Legendre quadrature scheme (`n = 32`) used for the at-risk
person-time export integral and the deaths-among-exports convolution
(outer and inner integrals).
"""
const CUMULATIVE_INTEGRAL_ALG = GaussLegendre(; n = 32)

_integrate_kernel(u, p) = p.f(p.halfwidth * (u + 1) + p.lo)

"""
$(TYPEDSIGNATURES)

Integrate a scalar function `f` over `[lo, hi]` by Gauss-Legendre
quadrature. The reference domain `[-1, 1]` is mapped onto `[lo, hi]`,
so any forward integral in the package can share one integrator. Returns
zero when `hi <= lo`. The default scheme is [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function integrate(f, lo, hi; alg = CUMULATIVE_INTEGRAL_ALG)
    hi <= lo && return zero(hi - lo)
    halfwidth = (hi - lo) / 2
    prob = IntegralProblem(_integrate_kernel, (-1.0, 1.0),
                           (; f, halfwidth, lo))
    return halfwidth * solve(prob, alg).u
end

# Change of variables clustering the reference nodes towards `hi`. With
# `d = hi - s` measured back from the upper limit and `v = (u + 1) / 2`,
# the map `d = span · vᵖ` sends the dense end of the reference grid to
# `s = hi` and stretches the sparse end out to `s = lo`. The Jacobian
# `dd/du = span · p · vᵖ⁻¹ / 2` is folded into the integrand so a single
# fixed `solve` covers the whole domain.
_clustered_kernel(u, p) = begin
    v = (u + one(u)) / 2
    d = p.span * v^p.expo
    return p.f(p.hi - d) * (p.span * p.expo * v^(p.expo - one(p.expo)) / 2)
end

"""
$(TYPEDSIGNATURES)

Integrate a scalar function `f` over `[lo, hi]` by Gauss-Legendre
quadrature with the nodes clustered towards `hi`, resolving features of
size `scale` near that limit. Use for onset-to-death convolutions, whose
integrand mass piles up against the cut-off `hi` over a window set by the
sampled delay scale: the uniform [`integrate`](@ref) spread across a wide
`[lo, hi]` would resolve that peak too coarsely. The whole domain is
still covered (no truncation), so a delay wider than `[lo, hi]` is
integrated exactly; when `scale ≥ hi - lo` the map reduces to the uniform
method. Returns zero when `hi <= lo`. The default scheme is
[`DEATH_INTEGRAL_ALG`](@ref).
"""
function integrate(f, lo, hi, scale; alg = DEATH_INTEGRAL_ALG)
    hi <= lo && return zero(hi - lo)
    span = hi - lo
    # `expo` places ~half the nodes within `scale` of `hi`; `expo = 1`
    # (uniform) once the feature is as wide as the domain. Fall back to
    # the uniform rule for a degenerate scale so a non-finite delay
    # moment cannot produce a `NaN` exponent on an AD proposal.
    (isfinite(scale) && scale > zero(scale)) ||
        return integrate(f, lo, hi; alg)
    expo = max(one(span), log(span / scale) / log(oftype(span, 2)))
    prob = IntegralProblem(_clustered_kernel, (-1.0, 1.0),
                           (; f, hi, span, expo))
    return solve(prob, alg).u
end

"""
    DELAY_SUPPORT_K

Number of standard deviations beyond the mean used as the clustering
scale for a delay distribution in the onset-to-death convolution
integrals. `mean + DELAY_SUPPORT_K · std` is the width near the cut-off
over which the clustered [`integrate`](@ref) packs roughly half its
nodes, so the quadrature tracks the delay's scale as it is sampled.
"""
const DELAY_SUPPORT_K = 10

# Clustering scale for a delay distribution, `mean + K·std`: the width of
# the window near the cut-off where the convolution integrand has mass.
# Scales with the sampled `std`, so gradients flow through it on the
# AD-supported parameter path used by the clustered `integrate`.
_delay_scale(dist) = mean(dist) + DELAY_SUPPORT_K * std(dist)

"""
$(TYPEDSIGNATURES)

Delay-convolved cumulative-incidence count by time `T` under
exponential growth at rate `r`:

```math
\\text{scale} \\cdot
    \\int_0^T e^{r s}\\, f(T - s)\\, ds,
```

with `f` the `delay_dist` density (per-event delay from incidence to
the observed event class — e.g. onset-to-death for deaths,
infection-to-report for suspected cases). The integrand returns zero
past `T` so the convolution support is respected. Integrated with the
clustered [`integrate`](@ref) so the quadrature nodes pack near `T`,
where `f(T − s)` has mass, over a window set by the delay scale (see
[`DELAY_SUPPORT_K`](@ref)). Uses [`DEATH_INTEGRAL_ALG`](@ref).
"""
function delay_convolution(scale, r, T, delay_dist; alg = DEATH_INTEGRAL_ALG)
    s_lo = zero(T)
    s_scale = _delay_scale(delay_dist)
    g = let r = r, T = T, delay_dist = delay_dist
        s -> begin
            d = T - s
            d <= 0 ? zero(r) : exp(r * s) * pdf(delay_dist, d)
        end
    end
    return scale * integrate(g, s_lo, T, s_scale; alg)
end

"""
$(TYPEDSIGNATURES)

[`delay_convolution`](@ref) specialised to the Gamma family, using the
analytic closed form

```math
\\text{scale} \\cdot
    \\int_0^T e^{r s}\\, f(T - s)\\, ds
  = \\text{scale} \\cdot e^{r T} \\cdot M(-r) \\cdot F(T (1 + \\theta r)),
```

where ``f`` is the Gamma(α, θ) density, ``M`` its moment-generating
function and ``F`` its CDF. The CDF is routed through
[`_gamma_cdf`](@ref) so Mooncake reverse-mode AD picks up a shape-
parameter gradient (computed internally with ForwardDiff).
"""
function delay_convolution(scale, r, T, delay_dist::Gamma)
    α, θ = delay_dist.α, delay_dist.θ
    return scale * exp(r * T) * mgf(delay_dist, -r) *
           _gamma_cdf(α, θ, T * (1 + θ * r))
end

"""
$(TYPEDSIGNATURES)

Integrate a cumulative-incidence trajectory `cumulative` (a callable
`C(s)`) over `[lo, hi]`. Backs the at-risk person-time export integral
``\\int_{T-w}^{T} C(s)\\, ds``. Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function integrate_cumulative(cumulative, lo, hi; alg = CUMULATIVE_INTEGRAL_ALG)
    return integrate(cumulative, lo, hi; alg)
end

"""
$(TYPEDSIGNATURES)

Deaths-among-exports convolution
``\\int_{lo}^{hi} C(s)\\, F_d(T - s)\\, ds`` with `F_d` the `delay_dist`
onset-to-death CDF. The CDF is itself written as the inner integral of
the density, ``F_d(x) = \\int_0^x f_d(u)\\,du``, so the whole expression
differentiates through the density alone.
Previously, the reverse-mode AD backend did not support the gamma CDF
shape-parameter derivative). Outer and inner integrals both use
[`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function integrate_exports_deaths(cumulative, delay_dist, lo, hi, T;
        alg = CUMULATIVE_INTEGRAL_ALG)
    cdf_to(x) = integrate(u -> pdf(delay_dist, u), zero(x), x; alg)
    g = let cumulative = cumulative, T = T
        s -> cumulative(s) * cdf_to(T - s)
    end
    return integrate(g, lo, hi; alg)
end

"""
$(TYPEDSIGNATURES)

Deaths-among-exports convolution specialised for `Gamma` delay
distributions. Same expression as the generic method —
``\\int_{lo}^{hi} C(s)\\, F_d(T - s)\\, ds`` — but the onset-to-death
CDF is evaluated in closed form via [`_gamma_cdf`](@ref) rather than
re-integrated from the density at each outer quadrature node. Drops one
entire quadrature level (just the outer integral remains) and
sidesteps the singularity that fixed-node Gauss-Legendre hits at the
head-form ``\\int_0^y t^{\\alpha-1} e^{-t} \\, du`` for `α < 1`. The
shape-parameter derivative the AD backend needed for `cdf(::Gamma, ·)`
is supplied by `_gamma_cdf`'s rrule (see `src/gamma_cdf.jl`).
"""
function integrate_exports_deaths(cumulative, delay_dist::Gamma, lo, hi, T;
        alg = CUMULATIVE_INTEGRAL_ALG)
    α, θ = delay_dist.α, delay_dist.θ
    g = let cumulative = cumulative, T = T, α = α, θ = θ
        s -> cumulative(s) * _gamma_cdf(α, θ, T - s)
    end
    return integrate(g, lo, hi; alg)
end

"""
    EXPORT_DELAY_GRID_POINTS

Number of evenly spaced grid points used to precompute the onset-to-death
CDF in [`ExportDeathDelay`](@ref).
"""
const EXPORT_DELAY_GRID_POINTS = 256

"""
$(TYPEDSIGNATURES)

Onset-to-death delay carrying its CDF `F_d` precomputed on an evenly
spaced grid over `[0, gmax]`, built once and reused across every outer
node and bin edge of the deaths-among-exports convolution rather than
re-integrating the density at each (see [`integrate_exports_deaths`](@ref)).
`gmax` must cover the largest delay argument reached, the detection window
`w`. Pass one in place of the raw delay distribution to select this path by
dispatch; the distribution methods are unchanged and remain the reference.

`F_d` is a cumulative trapezoid of the density, so the convolution still
differentiates through the density alone (the AD backend lacks the Gamma
CDF shape derivative). The density is never evaluated at `0`, whose Gamma
shape derivative is `0·log 0 = NaN` under AD; for `f_d(0) = 0` (Gamma
shape > 1) treating it as zero is exact.
"""
struct ExportDeathDelay{D, T}
    dist::D
    gmax::T
    dx::T
    F::Vector{T}
end

function ExportDeathDelay(dist, gmax::Real;
        npts::Integer = EXPORT_DELAY_GRID_POINTS)
    g  = float(gmax)
    dx = g / (npts - 1)
    Tt = typeof(pdf(dist, dx) * dx)
    F  = Vector{Tt}(undef, npts)
    F[1] = zero(Tt)
    prev = zero(Tt)
    @inbounds for i in 2:npts
        fx   = pdf(dist, (i - 1) * dx)
        F[i] = F[i - 1] + (prev + fx) * dx / 2
        prev = fx
    end
    return ExportDeathDelay(dist, g, dx, F)
end

# Linear interpolation of the precomputed CDF: F_d(0) = 0 and flat past
# `gmax` (all delay mass within the window has accumulated by then).
@inline function _cdf_to(ed::ExportDeathDelay, y)
    y <= zero(y)  && return zero(eltype(ed.F))
    y >= ed.gmax  && return @inbounds ed.F[end]
    pos  = y / ed.dx
    i    = floor(Int, pos) + 1
    frac = pos - (i - 1)
    return @inbounds ed.F[i] + frac * (ed.F[i + 1] - ed.F[i])
end

"""
$(TYPEDSIGNATURES)

Deaths-among-exports convolution using a precomputed [`ExportDeathDelay`](@ref):
``\\int_{lo}^{hi} C(s)\\, F_d(T - s)\\, ds`` with `F_d` interpolated off the
grid rather than re-integrated at each node. Mathematically identical to
the distribution method up to the grid resolution.
"""
function integrate_exports_deaths(cumulative, ed::ExportDeathDelay, lo, hi, T;
        alg = CUMULATIVE_INTEGRAL_ALG)
    g = let cumulative = cumulative, T = T, ed = ed
        s -> cumulative(s) * _cdf_to(ed, T - s)
    end
    return integrate(g, lo, hi; alg)
end

"""
$(TYPEDSIGNATURES)

Expected cumulative detected exports by elapsed time `t`, clamped to be
strictly positive and finite:

```math
\\mathbb{E}[\\text{exports}(t)] = p \\cdot q \\cdot
    \\int_{t-w}^{t} C(s)\\, ds,
```

with `cumulative` the trajectory ``C(s)``, `p` the detection
probability, `q` the per-capita travel rate, and `window` the detection
window ``w``. Backs both the exports count likelihood (evaluated at
`t = T`) and the first-export-detection survival term (evaluated at an
earlier `t`). Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_exports(cumulative, p, q, t, window;
        alg = CUMULATIVE_INTEGRAL_ALG)
    window_start = max(t - window, zero(t))
    integral = integrate_cumulative(cumulative, window_start, t; alg)
    raw = p * q * integral
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end

"""
$(TYPEDSIGNATURES)

Expected cumulative deaths among detected exports by elapsed time `t`,
clamped to be strictly positive and finite:

```math
\\mathbb{E}[D_{\\text{uganda}}(t)] = \\mathrm{CFR} \\cdot
    p \\cdot q \\cdot
    \\int_{t-w}^{t} C(s)\\, F_d(t - s)\\, ds,
```

with `cumulative` the trajectory ``C(s)``, `delay_dist` the onset-to-death
distribution (CDF ``F_d``), `p` the detection probability, `q` the
per-capita travel rate, and `window` the detection window ``w``. Backs
the binned-Poisson export-death likelihood, evaluated at the daily bin
edges. Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_exports_deaths(cumulative, delay_dist, CFR, p, q,
        t, window; alg = CUMULATIVE_INTEGRAL_ALG)
    window_start = max(t - window, zero(t))
    integral = integrate_exports_deaths(
        cumulative, delay_dist, window_start, t, t; alg)
    raw = CFR * p * q * integral
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end

_draws(chn, name::Symbol) = vec(Array(chn[name]))

"""
$(TYPEDSIGNATURES)

Return `(lo90, lo60, lo30, hi30, hi60, hi90)` equal-tailed credible
interval endpoints from a vector of draws.
"""
function posterior_summary(xs)
    return (
        lo90 = quantile(xs, 0.05),
        lo60 = quantile(xs, 0.20),
        lo30 = quantile(xs, 0.35),
        hi30 = quantile(xs, 0.65),
        hi60 = quantile(xs, 0.80),
        hi90 = quantile(xs, 0.95),
    )
end

## Human-readable headers for the displayed summary tables. Internal
## column keys stay machine-friendly; this maps them to nice labels at
## the point each table is returned.
const _PRETTY_COLS = Dict(
    "quantity"           => "Quantity",
    "stream"             => "Stream",
    "scenario"           => "Scenario",
    "central_estimate"   => "Central estimate",
    "reported_cases"     => "Reported cases",
    "narrowest_interval" => "Narrowest interval",
    "observed"           => "Observed",
    "within_90"          => "Within 90% PI",
    "lower_90" => "Lower 90%", "lower_60" => "Lower 60%",
    "lower_30" => "Lower 30%", "upper_30" => "Upper 30%",
    "upper_60" => "Upper 60%", "upper_90" => "Upper 90%",
)

_prettify(df::DataFrame) =
    rename(df, [n => get(_PRETTY_COLS, n, n) for n in names(df)])

"""
$(TYPEDSIGNATURES)

`DataFrame` with one row per posterior parameter and the columns
`Quantity, Lower 90%, Lower 60%, Lower 30%, Upper 30%, Upper 60%,
Upper 90%` giving the lower and upper endpoints of the equal-tailed
30%, 60% and 90% credible intervals.
"""
function summary_table(chn, params::AbstractVector{Symbol};
        digits::Integer = 2)
    df = @chain DataFrame(
            quantity = String[],
            lower_90 = Float64[], lower_60 = Float64[],
            lower_30 = Float64[], upper_30 = Float64[],
            upper_60 = Float64[], upper_90 = Float64[],
        ) begin
        let df = _
            for p in params
                s = posterior_summary(_draws(chn, p))
                push!(df, (string(p),
                           round(s.lo90; digits), round(s.lo60; digits),
                           round(s.lo30; digits), round(s.hi30; digits),
                           round(s.hi60; digits), round(s.hi90; digits)))
            end
            df
        end
    end
    return _prettify(df)
end

## --- Fit diagnostics ----------------------------------------------------

# Flat vector of a scalar diagnostic (R-hat or ESS), one entry per
# scalar parameter in a FlexiChains summary.
function _scalar_stats(summary)
    out = Float64[]
    for p in FlexiChains.parameters(summary)
        v = summary[p]
        if v isa Number
            ismissing(v) && continue
            push!(out, Float64(v))
        else
            for x in skipmissing(vec(collect(v)))
                push!(out, Float64(x))
            end
        end
    end
    return out
end

# Number of divergent NUTS transitions recorded in the chain.
function _num_divergences(chn)
    for e in FlexiChains.extras(chn)
        e.name === :numerical_error || continue
        return Int(sum(skipmissing(vec(chn[e]))))
    end
    return 0
end

"""
$(TYPEDSIGNATURES)

NUTS fit-quality summary for one chain: the worst (maximum) R-hat and
the smallest bulk effective sample size across parameters, and the
number of divergent transitions.
"""
function fit_diagnostics(chn)
    rhats = _scalar_stats(FlexiChains.rhat(chn))
    esses = _scalar_stats(FlexiChains.ess(chn; kind = :bulk))
    return (max_rhat     = maximum(rhats),
            min_ess_bulk = minimum(esses),
            n_divergent  = _num_divergences(chn))
end

"""
$(TYPEDSIGNATURES)

`DataFrame` of fit-quality diagnostics with one row per fit. Pass each
fit as `"label" => chain`. Columns `:fit, :max_rhat, :min_ess_bulk,
:divergences`.
"""
function diagnostics_table(fits::Pair{String}...)
    rows = map(fits) do (label, chn)
        d = fit_diagnostics(chn)
        (fit          = label,
         max_rhat     = round(d.max_rhat; digits = 3),
         min_ess_bulk = round(d.min_ess_bulk; digits = 0),
         divergences  = d.n_divergent)
    end
    return DataFrame(rows)
end

"""
$(TYPEDSIGNATURES)

Side-by-side credible intervals for `C_T` from several fits. Pass
each fit as `"label" => draws_vector`.
"""
function streams_table(streams::Pair{String, <:AbstractVector}...;
        digits::Integer = 0)
    rows = map(streams) do (label, draws)
        s = posterior_summary(draws)
        (stream = label,
         lower_90 = round(s.lo90; digits), lower_60 = round(s.lo60; digits),
         lower_30 = round(s.lo30; digits), upper_30 = round(s.hi30; digits),
         upper_60 = round(s.hi60; digits), upper_90 = round(s.hi90; digits))
    end
    return _prettify(DataFrame(rows))
end

"""
$(TYPEDSIGNATURES)

For each published `C_T` scenario, the narrowest joint posterior
credible interval (30, 60 or 90%) that contains it, or "outside
90%".
"""
function comparison_table(C_draws::AbstractVector;
        scenarios = REPORT_SCENARIOS)
    s = posterior_summary(C_draws)
    rows = map(scenarios) do (label, val)
        crI = if s.lo30 <= val <= s.hi30
            "30%"
        elseif s.lo60 <= val <= s.hi60
            "60%"
        elseif s.lo90 <= val <= s.hi90
            "90%"
        else
            "outside 90%"
        end
        (scenario = label, reported_cases = val,
         narrowest_interval = crI)
    end
    return _prettify(DataFrame(rows))
end

"""
$(TYPEDSIGNATURES)

Overlaid posterior densities of `C_T` from one or more fits, built
through AlgebraOfGraphics. The 15 published scenario point estimates
are drawn as faint dashed Makie `vlines` on top of the AoG figure.
"""
function plot_cumulative_cases(
        streams::Pair{String, <:AbstractVector}...;
        scenarios = REPORT_SCENARIOS,
        xmax::Real = 2_500)
    df = @chain DataFrame(
            stream = String[], C_T = Float64[],
        ) begin
        let df = _
            for (label, draws) in streams
                for x in draws
                    0 < x < xmax * 1.05 && push!(df, (label, float(x)))
                end
            end
            df
        end
    end

    spec = AoG.data(df) *
           AoG.mapping(:C_T => "Cumulative cases C_T",
                       color = :stream => "Data stream") *
           AoG.AlgebraOfGraphics.density() *
           AoG.visual(linewidth = 2)
    fg = AoG.draw(spec;
        axis  = (; ylabel = "Posterior density",
                   title  = "Posterior C_T by data stream",
                   limits = ((0, xmax), nothing)),
        figure = (; size = (760, 420)),
    )

    scenario_xs = Float64[val for (_, val) in scenarios if val < xmax]
    isempty(scenario_xs) || vlines!(fg.figure.content[1], scenario_xs;
            color = (:grey, 0.4), linestyle = :dash)
    return fg
end

"""
$(TYPEDSIGNATURES)

Overlaid posterior densities of an arbitrary scalar quantity from one
or more fits, built through AlgebraOfGraphics. Pass each fit as
`"label" => draws`; `xlabel` and `title` set the axis text.
"""
function plot_density_overlay(
        streams::Pair{String, <:AbstractVector}...;
        xlabel::AbstractString = "Value",
        title::AbstractString = "Posterior density")
    df = @chain DataFrame(stream = String[], value = Float64[]) begin
        let df = _
            for (label, draws) in streams
                for x in draws
                    push!(df, (label, float(x)))
                end
            end
            df
        end
    end

    spec = AoG.data(df) *
           AoG.mapping(:value => xlabel, color = :stream => "Fit") *
           AoG.AlgebraOfGraphics.density() *
           AoG.visual(linewidth = 2)
    return AoG.draw(spec;
        axis  = (; ylabel = "Posterior density", title = title),
        figure = (; size = (760, 420)),
    )
end

_panel_pos(pos::Integer) = (1, pos)
_panel_pos(pos::Tuple)   = pos

_panel_exports!(fig, pos, pp, obs; predictive_label = "Posterior") = begin
    r, c = _panel_pos(pos)
    upper = max(20, ceil(Int, quantile(pp, 0.99)))
    ax = Axis(fig[r, c];
        xlabel = "Replicated exported cases",
        ylabel = "$(predictive_label) predictive frequency",
        title  = "Exports (cases)",
        limits = ((0, upper), nothing),
    )
    hist!(ax, pp; bins = 0:1:upper, color = (:steelblue, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

_panel_exports_deaths!(fig, pos, pp, obs;
        predictive_label = "Posterior") = begin
    r, c = _panel_pos(pos)
    upper = max(3, ceil(Int, quantile(pp, 0.995)))
    ax = Axis(fig[r, c];
        xlabel = "Replicated deaths among exports",
        ylabel = "$(predictive_label) predictive frequency",
        title  = "Exports (deaths)",
        limits = ((0, upper), nothing),
    )
    hist!(ax, pp; bins = 0:1:upper, color = (:rebeccapurple, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

_panel_deaths!(fig, pos, pp, obs; predictive_label = "Posterior") = begin
    r, c = _panel_pos(pos)
    upper = max(1.0, quantile(pp, 0.995))
    ax = Axis(fig[r, c];
        xlabel = "Replicated deaths",
        ylabel = "$(predictive_label) predictive frequency",
        title  = "Deaths (DRC)",
        limits = ((0, upper), nothing),
    )
    hist!(ax, pp; bins = range(0, upper; length = 40),
          color = (:firebrick, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

_panel_cases!(fig, pos, pp, obs; predictive_label = "Posterior") = begin
    r, c = _panel_pos(pos)
    upper = max(1.0, quantile(pp, 0.995))
    ax = Axis(fig[r, c];
        xlabel = "Replicated reported cases",
        ylabel = "$(predictive_label) predictive frequency",
        title  = "Reported cases (DRC)",
        limits = ((0, upper), nothing),
    )
    hist!(ax, pp; bins = range(0, upper; length = 40),
          color = (:seagreen, 0.7))
    if obs !== nothing
        vlines!(ax, [obs]; color = :red, linewidth = 2)
    end
    return ax
end

"""
$(TYPEDSIGNATURES)

Posterior predictive histogram with one panel per supplied data
stream. Pass `pp_exports`/`pp_deaths` as `nothing` to suppress
either of the first two panels, and supply `pp_cases` and/or
`pp_exports_deaths` to add the reported-cases and deaths-among-exports
panels. Observed values are drawn as red `vlines`. With four streams
the panels are laid out as a 2×2 grid (exports cases, exports deaths,
DRC deaths, DRC reported cases); fewer streams are placed in a single
row.
"""
function plot_posterior_predictive(
        pp_exports::Union{Nothing, AbstractVector},
        pp_deaths::Union{Nothing, AbstractVector},
        obs_exports::Union{Nothing, Real},
        obs_deaths::Union{Nothing, Real};
        pp_cases::Union{Nothing, AbstractVector}          = nothing,
        obs_cases::Union{Nothing, Real}                   = nothing,
        pp_exports_deaths::Union{Nothing, AbstractVector} = nothing,
        obs_exports_deaths::Union{Nothing, Real}          = nothing,
        predictive_label::AbstractString                  = "Posterior")
    panels = Tuple{Symbol, Any, Any}[]
    pp_exports === nothing ||
        push!(panels, (:exports, pp_exports, obs_exports))
    pp_exports_deaths === nothing ||
        push!(panels, (:exports_deaths, pp_exports_deaths,
                       obs_exports_deaths))
    pp_deaths === nothing ||
        push!(panels, (:deaths,  pp_deaths,  obs_deaths))
    pp_cases === nothing ||
        push!(panels, (:cases,   pp_cases,   obs_cases))

    isempty(panels) && error(
        "plot_posterior_predictive needs at least one stream")

    ncols = length(panels) >= 4 ? 2 : length(panels)
    nrows = cld(length(panels), ncols)
    fig = Figure(; size = (450 * ncols, 380 * nrows))
    for (i, (kind, pp, obs)) in enumerate(panels)
        pos = (cld(i, ncols), mod1(i, ncols))
        if kind === :exports
            _panel_exports!(fig, pos, pp, obs; predictive_label)
        elseif kind === :exports_deaths
            _panel_exports_deaths!(fig, pos, pp, obs; predictive_label)
        elseif kind === :deaths
            _panel_deaths!(fig, pos, pp, obs; predictive_label)
        else
            _panel_cases!(fig, pos, pp, obs; predictive_label)
        end
    end
    return fig
end

"""
$(TYPEDSIGNATURES)

Two-row × four-column comparison of posterior-predictive
distributions. Top row: replicates from the per-stream fits. Bottom
row: replicates from the joint fit, conditioning on all observed
streams. Observed values shown as red vertical lines.

Each `NamedTuple` carries `(; exports, exports_deaths, deaths,
cases)`. Each panel is a histogram of replicated counts; rows share
the same x-axis (the stream's count) so the per-stream and joint
predictives are directly comparable.
"""
function plot_posterior_predictive_grid(;
        individual::NamedTuple,
        joint::NamedTuple,
        observed::NamedTuple,
    )
    fig = Figure(; size = (1600, 640))
    rows = ((:individual, individual, "per-stream fit"),
            (:joint,      joint,      "joint fit"))
    for (i, (_, pp, label)) in enumerate(rows)
        _panel_exports!(fig, (i, 1), pp.exports, observed.exports;
                        predictive_label = label)
        _panel_exports_deaths!(fig, (i, 2), pp.exports_deaths,
                        observed.exports_deaths; predictive_label = label)
        _panel_deaths!(fig, (i, 3),  pp.deaths,  observed.deaths;
                       predictive_label = label)
        _panel_cases!(fig, (i, 4),   pp.cases,   observed.cases;
                      predictive_label = label)
    end
    return fig
end

"""
$(TYPEDSIGNATURES)

Prior predictive variant of `plot_posterior_predictive`, with the
panel labels switched to "Prior".
"""
function plot_prior_predictive(
        pp_exports::Union{Nothing, AbstractVector},
        pp_deaths::Union{Nothing, AbstractVector},
        obs_exports::Union{Nothing, Real},
        obs_deaths::Union{Nothing, Real};
        pp_cases::Union{Nothing, AbstractVector} = nothing,
        obs_cases::Union{Nothing, Real}          = nothing)
    return plot_posterior_predictive(
        pp_exports, pp_deaths, obs_exports, obs_deaths;
        pp_cases, obs_cases, predictive_label = "Prior")
end

"""
$(TYPEDSIGNATURES)

PairPlots.jl corner plot over the named posterior parameters,
thinned by `thin`. Pass `prior` (another chain holding the same
parameters) to overlay the prior as a second series with a legend,
so the data's contribution to each marginal is visible.
"""
function plot_pair(chn, params::AbstractVector{Symbol};
        thin::Integer = 2, prior = nothing)
    _table(c) = DataFrame(
        NamedTuple(p => _draws(c, p) for p in params))[1:thin:end, :]
    post = _table(chn)
    prior === nothing && return PairPlots.pairplot(post)
    colours = CairoMakie.Makie.wong_colors()
    return PairPlots.pairplot(
        PairPlots.Series(post;  label = "Posterior", color = colours[1]),
        PairPlots.Series(_table(prior); label = "Prior",
                         color = colours[2]),
    )
end

"""
$(TYPEDSIGNATURES)

Horizontal point-and-interval comparison of cumulative-case estimates
from several sources. `rows` is a vector of
`(label, central, lower, upper)` tuples, drawn top to bottom with the
central estimate as a point and `[lower, upper]` as a bar. Use it to
place model posteriors next to published point estimates and their
intervals.
"""
function plot_estimate_comparison(
        rows::AbstractVector;
        xlabel::AbstractString = "Cumulative cases C(T)",
        xmax::Union{Nothing, Real} = nothing)
    n       = length(rows)
    labels  = [String(r[1]) for r in rows]
    central = [float(r[2])  for r in rows]
    lo      = [float(r[3])  for r in rows]
    hi      = [float(r[4])  for r in rows]
    top     = isnothing(xmax) ? maximum(hi) * 1.08 : xmax

    fig = Figure(; size = (840, 120 + 46n))
    ax = Axis(fig[1, 1];
        xlabel = xlabel,
        yticks = (collect(1:n), reverse(labels)),
        limits = ((0, top), (0.5, n + 0.5)),
    )
    for i in 1:n
        y = n - i + 1
        lines!(ax, [lo[i], hi[i]], [y, y];
               color = (:steelblue, 0.8), linewidth = 3)
        scatter!(ax, [central[i]], [y];
                 color = :firebrick, markersize = 12)
    end
    return fig
end

"""
$(TYPEDSIGNATURES)

Density of a prior over the case-fatality ratio (CFR) on `[0, 1]`,
plotted on the sub-range `[0, 0.7]`. The CDC central estimate of
55/169 ≈ 0.33 is drawn as a solid vertical rule, and the report's 26%
and 40% scenario bounds as dashed rules, so the prior can be read
against the published CFR scenarios.
"""
function plot_cfr_prior(prior::Distribution)
    colours = CairoMakie.Makie.wong_colors()
    xs = range(0.0, 0.7; length = 400)
    ys = pdf.(Ref(prior), xs)

    fig = Figure(; size = (760, 420))
    ax = Axis(fig[1, 1];
        xlabel = "Case-fatality ratio (CFR)",
        ylabel = "Prior density",
        title  = "Prior over the case-fatality ratio",
        limits = ((0, 0.7), nothing),
    )
    lines!(ax, xs, ys; color = colours[1], linewidth = 2)
    vlines!(ax, [55 / 169]; color = :firebrick, linewidth = 2)
    vlines!(ax, [0.26, 0.40];
            color = (:grey, 0.6), linestyle = :dash, linewidth = 2)
    return fig
end

"""
$(TYPEDSIGNATURES)

One-row, two-panel figure summarising when the outbreak began. The
left panel is the posterior density of the outbreak start date,
obtained by rescaling the days-since-seeding `T` to a calendar date
(`as_of_date` minus `T`). The right panel is the joint `(τ, T)`
posterior pair plot, which is positively correlated: slower growth
(larger `τ`) needs a longer elapsed `T` to reach the same counts.
"""
function plot_start_date_pair(chn;
        as_of_date::AbstractString, thin::Integer = 2)
    T_draws     = _draws(chn, :T)
    cutoff_days = date2epochdays(Date(as_of_date))
    start_days  = cutoff_days .- T_draws

    fig = Figure(; size = (1100, 460))
    ax = Axis(fig[1, 1];
        xlabel = "Outbreak start date",
        ylabel = "Posterior density",
        title  = "Implied start of sustained transmission",
        xticklabelrotation = π / 6,
    )
    density!(ax, start_days; color = (:steelblue, 0.5),
             strokecolor = :steelblue, strokewidth = 2)
    ## Date ticks every four weeks across the posterior range, so the
    ## start date stays readable rather than relying on the default
    ## locator or crowding the axis as the range widens.
    lo = floor(Int, minimum(start_days))
    hi = ceil(Int, maximum(start_days))
    ax.xticks = collect(lo:28:hi)
    ax.xtickformat = vals ->
        [string(epochdays2date(round(Int, v))) for v in vals]

    pair_df = DataFrame(τ = _draws(chn, :τ), T = T_draws)
    PairPlots.pairplot(fig[1, 2], pair_df[1:thin:end, :])
    return fig
end

## --- Future-expected-deaths counterfactual -----------------------------
##
## Lower bound on future deaths under the counterfactual that every
## onward transmission stops at time `T`. The cohort already infected
## by `T` still contributes future expected deaths in the onset-to-death
## tail; per draw,
##
##     ΔD = CFR · ∫_0^T r · exp(r · s) · (1 - F_d(T - s)) ds,
##
## with `F_d` the Gamma(α, θ) CDF. Total projected cumulative deaths
## under the no-onward-transmission counterfactual is `obs_deaths + ΔD`.

function _committed_deaths_one(r, T, α, θ, CFR;
        alg = DEATH_INTEGRAL_ALG)
    delay_dist = Gamma(α, θ)
    scale = _delay_scale(delay_dist)
    g = let r = r, T = T, delay_dist = delay_dist
        s -> r * exp(r * s) * ccdf(delay_dist, T - s)
    end
    return CFR * integrate(g, zero(T), T, scale; alg)
end

"""
    predict_no_onward_deaths(chn; obs_deaths,
                             alg = GaussLegendre(n = 64))

Per-draw projection of cumulative deaths under the counterfactual
that every onward transmission stops at time `T`. Reads `:r, :T, :α,
:θ, :CFR` from the posterior `chn` and integrates

```math
\\Delta D = \\mathrm{CFR} \\cdot \\int_0^T r\\,\\exp(r\\,s)
            \\,\\bigl(1 - F_d(T - s)\\bigr)\\,ds,
```

with `F_d` the Gamma(α, θ) onset-to-death CDF, returning a
`DataFrame` with one row per draw:

- `:delta_deaths`     additional future expected deaths beyond `obs_deaths`
- `:total_projected`  `obs_deaths + delta_deaths`

`obs_deaths` is the number of deaths already observed at time `T`
(e.g. `obs.total_deaths` from the bundled observations). `alg` is the
quadrature scheme used for ΔD; defaults to `GaussLegendre(n = 64)`,
matching the rest of the package.
"""
function predict_no_onward_deaths(chn;
        obs_deaths::Real,
        alg = DEATH_INTEGRAL_ALG)
    r_draws   = _draws(chn, :r)
    T_draws   = _draws(chn, :T)
    α_draws   = _draws(chn, :α)
    θ_draws   = _draws(chn, :θ)
    CFR_draws = _draws(chn, :CFR)

    n = length(r_draws)
    delta = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        delta[i] = _committed_deaths_one(r_draws[i], T_draws[i],
                                         α_draws[i], θ_draws[i],
                                         CFR_draws[i]; alg)
    end
    total = float(obs_deaths) .+ delta
    return DataFrame(delta_deaths = delta, total_projected = total)
end

"""
    plot_no_onward_deaths(df; obs_deaths)

Two-panel density of the no-onward-transmission counterfactual from
[`predict_no_onward_deaths`](@ref). The left panel shows the *still
expected* deaths (`:delta_deaths`, the future deaths in cases already
infected by `T`, net of the `obs_deaths` already observed). The right
panel shows the *projected total* (`:total_projected = obs_deaths +
delta_deaths`) with a dashed black rule at `obs_deaths`. Both are
lower bounds: they assume every onward transmission stops at time `T`.
"""
function plot_no_onward_deaths(df::DataFrame; obs_deaths::Real)
    fig = Figure(; size = (980, 420))

    ax1 = Axis(fig[1, 1];
        xlabel = "Still expected deaths (beyond those already observed)",
        ylabel = "Posterior density",
        title  = "Still expected (future)")
    density!(ax1, df.delta_deaths; color = (:firebrick, 0.5),
             strokecolor = :firebrick, strokewidth = 2)

    ax2 = Axis(fig[1, 2];
        xlabel = "Projected total deaths (no onward transmission)",
        ylabel = "Posterior density",
        title  = "Projected total")
    density!(ax2, df.total_projected; color = (:firebrick, 0.5),
             strokecolor = :firebrick, strokewidth = 2)
    vlines!(ax2, [float(obs_deaths)];
            color = :black, linestyle = :dash, linewidth = 2)

    return fig
end

## --- One-week-ahead forecast --------------------------------------------
##
## Continue the fitted exponential growth `h` days past the cut-off `T`
## and apply the same observation models to forecast the cumulative
## reported cases (DRC), deaths (DRC) and exports (Uganda) by `T + h`,
## plus the new counts expected over the coming week (cumulative at
## `T + h` minus the count observed at `T`). Posterior-predictive: each
## draw produces a replicated integer count, so the intervals include
## both parameter and observation uncertainty.

# Deaths convolution evaluated at horizon `Th = T + h`, sharing the
# package-wide `delay_convolution` integrator.
function _forecast_deaths_mean(r, Th, α, θ, CFR; alg = DEATH_INTEGRAL_ALG)
    return delay_convolution(CFR, r, Th, Gamma(α, θ); alg)
end

# Reported (suspected) cases at horizon `Th`: the truth-anchored BVD
# contribution plus the non-BVD background. The BVD contribution reuses
# `delay_convolution` as a delay-convolved cumulative integrator at unit
# ascertainment to compute `∫₀^{Th} exp(r·s) · f_rep(Th-s) ds`; the
# background contribution `λ_bg · Th` accrues with elapsed time.
function _forecast_cases_mean(r, Th, α_rep, θ_rep, p_drc, λ_bg;
        alg = DEATH_INTEGRAL_ALG)
    conv = delay_convolution(one(p_drc), r, Th, Gamma(α_rep, θ_rep); alg)
    return p_drc * conv + λ_bg * Th
end

# Laboratory-confirmed cases at horizon `Th`: outer quadrature against
# the lab-turnaround Gamma, with the inner reported-cases
# closure `τ -> p_drc · delay_convolution(1, r, τ, f_rep)` evaluated at
# the pushed-back cut-off `Th - u`. Same exact double integral as the
# in-model confirmed likelihood, no moment-match.
function _forecast_confirmed_mean(r, Th, α_rep, θ_rep, α_lab, θ_lab,
        p_drc, s_test; alg = DEATH_INTEGRAL_ALG)
    d_rep = Gamma(α_rep, θ_rep)
    d_lab = Gamma(α_lab, θ_lab)
    bvd_reported_at = let r = r, p_drc = p_drc, d_rep = d_rep, alg = alg
        τ -> p_drc * delay_convolution(one(p_drc), r, τ, d_rep; alg)
    end
    scale = _delay_scale(d_lab)
    g = let bvd_reported_at = bvd_reported_at, d_lab = d_lab, Th = Th
        u -> begin
            d = Th - u
            d <= 0 ? zero(Th) :
                pdf(d_lab, u) * bvd_reported_at(d)
        end
    end
    return s_test * integrate(g, zero(Th), Th, scale; alg)
end

function _nb_rand(rng, k, μ)
    μs = max(μ, eps(typeof(μ)))
    p = clamp(k / (k + μs), eps(typeof(k)), one(k) - eps(typeof(k)))
    return rand(rng, NegativeBinomial(k, p))
end

"""
    forecast_reported(chn; horizon = 7, daily_travellers, source_population,
                      obs_cases, obs_deaths, obs_exports,
                      obs_confirmed = missing, seed = 20260520)

One-week-ahead (default `horizon = 7` days) posterior-predictive
forecast. For each draw, continue exponential growth to `T + horizon`
and apply the observation models, returning a `DataFrame` with one row
per draw and columns:

- `:cases_cum`, `:deaths_cum`, `:exports_cum` — replicated cumulative
  counts reported by `T + horizon`.
- `:cases_new`, `:deaths_new`, `:exports_new` — new counts over the
  coming week (`*_cum` minus the corresponding observed count at `T`,
  floored at zero).
- `:confirmed_cum`, `:confirmed_new` — laboratory-confirmed counterparts
  when the chain carries the lab-turnaround delay (`:α_lab`, `:θ_lab`)
  and `obs_confirmed` is supplied. Otherwise these columns are absent.

DRC reported cases follow the additive expectation
`p_drc · ∫₀^{T+h} exp(r·s) · f_rep(T+h-s) ds + λ_bg · (T+h)`, with
`f_rep = Gamma(α_rep, θ_rep)` for the BVD-driven contribution and
`λ_bg` the per-day non-BVD background. Laboratory-confirmed cases
follow `s_test · p_drc · ∫₀^{T+h} exp(r·s) · f_conf(T+h-s) ds`, with
`f_conf` the moment-matched Gamma of `f_rep ∗ f_lab` and `s_test` the
PCR sensitivity. Exports use `p_uganda · q` with
`q = daily_travellers / source_population`. Assumes growth continues
unchanged over the horizon (no interventions, no saturation).
"""
function forecast_reported(chn;
        horizon::Real          = 7,
        daily_travellers::Real,
        source_population::Real,
        obs_cases::Real,
        obs_deaths::Real,
        obs_exports::Real,
        obs_confirmed::Union{Real, Missing} = missing,
        seed::Integer          = 20260520,
        alg                    = DEATH_INTEGRAL_ALG)
    r     = _draws(chn, :r)
    T     = _draws(chn, :T)
    CFR   = _draws(chn, :CFR)
    α     = _draws(chn, :α)
    θ     = _draws(chn, :θ)
    w     = _draws(chn, :w)
    pr    = _draws(chn, :p_drc)
    pu    = _draws(chn, :p_uganda)
    k     = _draws(chn, :k)
    α_rep = _draws(chn, :α_rep)
    θ_rep = _draws(chn, :θ_rep)
    λ_bg  = _draws(chn, :λ_bg)
    ## Lab-turnaround and PCR sensitivity draws live on the joint chain
    ## only; their absence drops the confirmed-cases columns.
    has_lab = all(haskey_chain(chn, n) for n in (:α_lab, :θ_lab, :s_test)) &&
              obs_confirmed !== missing
    α_lab  = has_lab ? _draws(chn, :α_lab)  : nothing
    θ_lab  = has_lab ? _draws(chn, :θ_lab)  : nothing
    s_test = has_lab ? _draws(chn, :s_test) : nothing

    rng = MersenneTwister(seed)
    n = length(r)
    q = daily_travellers / source_population
    cases_cum     = Vector{Int}(undef, n)
    deaths_cum    = Vector{Int}(undef, n)
    exports_cum   = Vector{Int}(undef, n)
    confirmed_cum = has_lab ? Vector{Int}(undef, n) : nothing

    @inbounds for i in 1:n
        Th = T[i] + horizon
        ## DRC reported cases: p_drc · ∫₀^{T+h} exp(r·s) · f_rep(T+h-s) ds
        ## + λ_bg · (T+h).
        μ_cases = _forecast_cases_mean(r[i], Th, α_rep[i], θ_rep[i],
                                       pr[i], λ_bg[i]; alg)
        cases_cum[i] = _nb_rand(rng, k[i], μ_cases)
        ## DRC deaths: CFR · ∫_0^{T+h} exp(r·s) f(T+h−s) ds.
        μ_deaths = _forecast_deaths_mean(r[i], Th, α[i], θ[i], CFR[i]; alg)
        deaths_cum[i] = _nb_rand(rng, k[i], μ_deaths)
        ## Uganda exports: p_uganda · q · ∫_{T+h−w}^{T+h} C(s) ds (closed
        ## form for exponential growth).
        lo = max(Th - w[i], zero(Th))
        μ_exports = pu[i] * q * (exp(r[i] * Th) - exp(r[i] * lo)) / r[i]
        exports_cum[i] = rand(rng, Poisson(max(μ_exports, eps(μ_exports))))
        if has_lab
            μ_confirmed = _forecast_confirmed_mean(r[i], Th,
                α_rep[i], θ_rep[i], α_lab[i], θ_lab[i], pr[i],
                s_test[i]; alg)
            confirmed_cum[i] = _nb_rand(rng, k[i], μ_confirmed)
        end
    end

    _new(cum, obs) = max.(cum .- round(Int, obs), 0)
    df = DataFrame(
        cases_cum    = cases_cum,
        deaths_cum   = deaths_cum,
        exports_cum  = exports_cum,
        cases_new    = _new(cases_cum,   obs_cases),
        deaths_new   = _new(deaths_cum,  obs_deaths),
        exports_new  = _new(exports_cum, obs_exports),
    )
    if has_lab
        df.confirmed_cum = confirmed_cum
        df.confirmed_new = _new(confirmed_cum, obs_confirmed)
    end
    return df
end

## Best-effort presence check for a chain key across the FlexiChains /
## MCMCChains containers in use. Avoids loading the FlexiChains type
## just to dispatch.
function haskey_chain(chn, name::Symbol)
    try
        chn[name]
        return true
    catch
        return false
    end
end

"""
    forecast_table(fc::DataFrame; digits = 0)

Summarise a [`forecast_reported`](@ref) result into a `DataFrame` with
one row per stream (cases, deaths, exports) and quantity (cumulative
total by `T + 7`, or new this week), reporting the same equal-tailed
30/60/90% credible interval endpoints (`lower_90 … upper_90`) as the
other summary tables.
"""
function forecast_table(fc::DataFrame; digits::Integer = 0)
    _row(label, quantity, draws) = begin
        s = posterior_summary(draws)
        (stream = label, quantity = quantity,
         lower_90 = round(s.lo90; digits), lower_60 = round(s.lo60; digits),
         lower_30 = round(s.lo30; digits), upper_30 = round(s.hi30; digits),
         upper_60 = round(s.hi60; digits), upper_90 = round(s.hi90; digits))
    end
    rows = NamedTuple[]
    for (label, cum, new) in (
            ("DRC reported cases", :cases_cum,   :cases_new),
            ("DRC deaths",         :deaths_cum,  :deaths_new),
            ("Uganda exports",     :exports_cum, :exports_new))
        push!(rows, _row(label, "cumulative by T+7", fc[!, cum]))
        push!(rows, _row(label, "new this week",     fc[!, new]))
    end
    return _prettify(DataFrame(rows))
end

"""
    forecast_vs_truth(fc::DataFrame; cases, deaths, exports, digits = 0)

Validate a [`forecast_reported`](@ref) projection against the counts
that were later observed. `cases`, `deaths` and `exports` are the
observed cumulative DRC reported cases, DRC deaths and Uganda exports at
the forecast target date. Returns a `DataFrame` with one row per stream
giving the observed count, the equal-tailed 30/60/90% predictive
intervals (the same endpoints as the other summary tables), and whether
the observed count falls inside the 90% interval. Use it to forecast
from an earlier data snapshot and check the now-known truth against the
projection.
"""
function forecast_vs_truth(fc::DataFrame;
        cases::Real, deaths::Real, exports::Real, digits::Integer = 0)
    _row(label, draws, obs) = begin
        s  = posterior_summary(draws)
        lo = round(s.lo90; digits)
        hi = round(s.hi90; digits)
        (stream = label, observed = round(obs; digits),
         lower_90 = lo, lower_60 = round(s.lo60; digits),
         lower_30 = round(s.lo30; digits), upper_30 = round(s.hi30; digits),
         upper_60 = round(s.hi60; digits), upper_90 = hi,
         within_90 = lo <= obs <= hi ? "yes" : "no")
    end
    rows = [
        _row("DRC reported cases", fc[!, :cases_cum],   cases),
        _row("DRC deaths",         fc[!, :deaths_cum],  deaths),
        _row("Uganda exports",     fc[!, :exports_cum], exports),
    ]
    return _prettify(DataFrame(rows))
end

"""
    plot_forecast(fc::DataFrame)

Three-panel histogram of the new-this-week forecast counts (cases,
deaths, exports) from [`forecast_reported`](@ref).
"""
function plot_forecast(fc::DataFrame)
    fig = Figure(; size = (1100, 360))
    cols = ((:cases_new,   "New reported cases (DRC)",  :steelblue),
            (:deaths_new,  "New deaths (DRC)",          :firebrick),
            (:exports_new, "New exports (Uganda)",      :seagreen))
    for (i, (col, title, colour)) in enumerate(cols)
        v = fc[!, col]
        upper = max(1.0, quantile(v, 0.995))
        ax = Axis(fig[1, i];
            xlabel = title, ylabel = "Predictive frequency",
            title = "One week ahead", limits = ((0, upper), nothing))
        hist!(ax, v; bins = range(0, upper; length = 30),
              color = (colour, 0.7))
    end
    return fig
end

"""
    plot_forecast_vs_truth(fc::DataFrame; cases, deaths, exports,
                           baseline_cases = 0, baseline_deaths = 0,
                           baseline_exports = 0)

Validation figure for a [`forecast_reported`](@ref) projection, laid out
as a 2×3 grid. The top row shows the cumulative forecast distribution per
stream (DRC reported cases, DRC deaths, Uganda exports); the bottom row
shows the new counts forecast over the horizon, mirroring the
one-week-ahead forecast. Each panel is a histogram with the 90%
predictive interval shaded and the later-observed count drawn as a dashed
black rule. `cases`, `deaths` and `exports` are the observed cumulative
counts; `baseline_*` are the counts at the forecast origin, so the
observed new count is the cumulative truth minus the baseline.
"""
function plot_forecast_vs_truth(fc::DataFrame;
        cases::Real, deaths::Real, exports::Real,
        baseline_cases::Real = 0, baseline_deaths::Real = 0,
        baseline_exports::Real = 0)
    fig = Figure(; size = (1100, 680))
    function panel!(row, col, v, obs, title, colour)
        lo    = quantile(v, 0.05)
        hi    = quantile(v, 0.95)
        upper = max(1.0, quantile(v, 0.995), obs * 1.05)
        ax = Axis(fig[row, col];
            xlabel = title, ylabel = "Predictive frequency",
            limits = ((0, upper), nothing))
        vspan!(ax, lo, hi; color = (colour, 0.15))
        hist!(ax, v; bins = range(0, upper; length = 30),
              color = (colour, 0.7))
        vlines!(ax, [obs]; color = :black, linestyle = :dash, linewidth = 2)
    end
    streams = (
        (:cases_cum,   :cases_new,   "reported cases (DRC)", :steelblue,
         float(cases),   float(cases)   - float(baseline_cases)),
        (:deaths_cum,  :deaths_new,  "deaths (DRC)",         :firebrick,
         float(deaths),  float(deaths)  - float(baseline_deaths)),
        (:exports_cum, :exports_new, "exports (Uganda)",     :seagreen,
         float(exports), float(exports) - float(baseline_exports)))
    for (j, (ccol, ncol, name, colour, obs_cum, obs_new)) in
            enumerate(streams)
        panel!(1, j, fc[!, ccol], obs_cum, "Cumulative $name", colour)
        panel!(2, j, fc[!, ncol], max(obs_new, 0.0), "New $name", colour)
    end
    return fig
end

end # module
