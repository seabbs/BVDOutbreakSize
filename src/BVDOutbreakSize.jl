module BVDOutbreakSize

using Statistics: quantile
using Printf: @sprintf
using TOML
using DataFrames: DataFrame
using DataFramesMeta
using Chain: @chain
using Random: MersenneTwister
using ADTypes: AutoMooncake
using Mooncake: Mooncake
using ChainRulesCore: ChainRulesCore, NoTangent
using SpecialFunctions: gamma_inc, digamma
import SpecialFunctions
using Turing
using Turing.DynamicPPL: InitFromPrior
using MCMCChains: Chains
import MCMCChains
import FlexiChains
using DocStringExtensions
using Distributions: Gamma, cdf, ccdf, mgf, pdf, Poisson, NegativeBinomial
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
import CairoMakie
import AlgebraOfGraphics as AoG
import PairPlots
using CairoMakie: Figure, Axis, hist!, density!, vlines!

export REPORT_SCENARIOS,
       ITURI_POPULATION, ITURI_DAILY_TRAVEL,
       ITURI_DAILY_TRAVEL_SD,
       EXPORTED_CASES, EXPORTS_DEATHS, TOTAL_DEATHS, REPORTED_CASES,
       load_observations,
       summary_table, posterior_summary,
       fit_diagnostics, diagnostics_table,
       streams_table, comparison_table,
       nuts_sample, default_adtype,
       DEATH_INTEGRAL_ALG, CUMULATIVE_INTEGRAL_ALG,
       integrate, expected_deaths,
       integrate_cumulative, integrate_exports_deaths,
       plot_cumulative_cases, plot_prior_predictive,
       plot_posterior_predictive, plot_posterior_predictive_grid,
       plot_pair,
       predict_no_onward_deaths, plot_no_onward_deaths,
       forecast_reported, forecast_table, plot_forecast

"""
    REPORT_SCENARIOS

Published point estimates of cumulative cases `C_T` from McCabe et
al. (Imperial College London, 18 May 2026), as `(label, value)`
tuples in the order they appear in Tables 1 and 2.
"""
const REPORT_SCENARIOS = [
    ("Method 1 Ituri, w=10 d",   470),
    ("Method 1 Ituri, w=15 d",   313),
    ("Method 1 Ituri, w=20 d",   235),
    ("Method 1 +N. Kivu, w=10",  617),
    ("Method 1 +N. Kivu, w=15",  412),
    ("Method 1 +N. Kivu, w=20",  309),
    ("Method 2 τ=14 d, CFR 24%", 626),
    ("Method 2 τ=14 d, CFR 30%", 501),
    ("Method 2 τ=14 d, CFR 40%", 376),
    ("Method 2 τ= 7 d, CFR 24%", 1008),
    ("Method 2 τ= 7 d, CFR 30%", 807),
    ("Method 2 τ= 7 d, CFR 40%", 605),
    ("Method 2 τ=21 d, CFR 24%", 531),
    ("Method 2 τ=21 d, CFR 30%", 425),
    ("Method 2 τ=21 d, CFR 40%", 319),
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

"""
    TOTAL_DEATHS

Suspected BVD deaths reported in DRC, taken from the most recent
Guardian situation report (19 May 2026). Imperial's 18 May 2026
report uses the earlier 16 May 2026 snapshot of 88 deaths.
"""
const TOTAL_DEATHS = 130

"""
    REPORTED_CASES

Suspected BVD cases reported in DRC, taken from the Guardian
situation report (19 May 2026). Used by the ascertainment extension
beyond Imperial Methods 1 and 2.
"""
const REPORTED_CASES = 500

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
- `reported_cases::Int`
- `daily_outbound_travellers::Real`
- `daily_outbound_travellers_sd::Real`
- `source_population::Int`
- `sources::NamedTuple{(:exported_cases, :exports_deaths, :total_deaths,
  :reported_cases, :daily_outbound_travellers,
  :daily_outbound_travellers_sd, :source_population),
  NTuple{7, String}}` — citation per field.
"""
function load_observations(
        path::AbstractString = joinpath(@__DIR__, "..", "data",
                                        "observations.toml"))
    raw = TOML.parsefile(path)
    _val(k) = raw[k]["value"]
    _src(k) = String(raw[k]["source"])
    return (;
        as_of_date                   = String(raw["as_of_date"]),
        exported_cases               = Int(_val("exported_cases")),
        exports_deaths               = Int(_val("exports_deaths")),
        total_deaths                 = Int(_val("total_deaths")),
        reported_cases               = Int(_val("reported_cases")),
        daily_outbound_travellers    = float(
            _val("daily_outbound_travellers")),
        daily_outbound_travellers_sd = float(
            _val("daily_outbound_travellers_sd")),
        source_population            = Int(_val("source_population")),
        sources = (;
            exported_cases               = _src("exported_cases"),
            exports_deaths               = _src("exports_deaths"),
            total_deaths                 = _src("total_deaths"),
            reported_cases               = _src("reported_cases"),
            daily_outbound_travellers    = _src("daily_outbound_travellers"),
            daily_outbound_travellers_sd = _src("daily_outbound_travellers_sd"),
            source_population            = _src("source_population"),
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
        target_accept::Real = 0.9,
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

"""
$(TYPEDSIGNATURES)

Expected cumulative deaths by time `T` from a single seeding case under
exponential growth at rate `r`:

```math
\\mathbb{E}[D_T] = \\mathrm{CFR} \\cdot
    \\int_0^T e^{r s}\\, f(T - s)\\, ds,
```

with `f` the `delay_dist` onset-to-death density. The integrand returns
zero past `T` so the convolution support is respected. Uses
[`DEATH_INTEGRAL_ALG`](@ref).
"""
function expected_deaths(CFR, r, T, delay_dist; alg = DEATH_INTEGRAL_ALG)
    g = let r = r, T = T, delay_dist = delay_dist
        s -> begin
            d = T - s
            d <= 0 ? zero(r) : exp(r * s) * pdf(delay_dist, d)
        end
    end
    return CFR * integrate(g, zero(T), T; alg)
end

## Wrapper around `cdf(Gamma(α, θ), x)` exposed as a 3-arg scalar
## primitive so we can attach a Mooncake-compatible reverse-mode rule.
##
## SpecialFunctions.jl's `gamma_inc(a, x)` ChainRule defines only the
## `x`-partial; the shape (`a`) partial is `@not_implemented`. Mooncake
## inherits that gap and returns `NaN` for any α-gradient flowing
## through `cdf(::Gamma, ::Real)`. The two AD-system workarounds
## (propagating ForwardDiff `Dual` numbers through the implementation;
## relying on a built-in `a`-partial) both fail because
## `SpecialFunctions._gamma_inc` is dispatched only on concrete
## `Float64`/`Float32`/`BigFloat`, so duals never reach it and no
## reverse-mode rule exists.
##
## Instead we attach our own analytic rrule:
##
## * `∂F/∂x  = pdf(Gamma(α, θ), x)`
## * `∂F/∂θ  = -(x/θ) · pdf(Gamma(α, θ), x)`
## * `∂F/∂α  = -ψ(α)·P(α, y) + (1/Γ(α)) · ∫₀^y t^{α-1} e^{-t} log t dt`,
##   with `y = x/θ` and `P` the regularized lower incomplete gamma.
##
## The `α`-integral is evaluated by the package-wide Gauss-Legendre
## scheme, so the cost per gradient is one extra ~32-point quadrature.
## Mooncake picks the rule up via `@from_rrule`. Stan and JAX hand-code
## the same gradient as a primitive in their AD libraries (Moore 1982,
## *Algorithm AS 187*); this is the Julia version.
_gamma_cdf(α, θ, x) = cdf(Gamma(α, θ), x)

function _gamma_cdf_partials(α, θ, x)
    R = float(promote_type(typeof(α), typeof(θ), typeof(x)))
    y = x / θ
    y <= zero(y) && return zero(R), zero(R), zero(R)
    f      = pdf(Gamma(α, θ), x)
    df_dx  = f
    df_dθ  = -y * f
    P      = first(gamma_inc(α, y))
    integrand = t -> t > zero(t) ?
        t^(α - 1) * exp(-t) * log(t) : zero(t)
    I      = integrate(integrand, zero(y), y; alg = CUMULATIVE_INTEGRAL_ALG)
    df_dα  = -digamma(α) * P + I / SpecialFunctions.gamma(α)
    return df_dα, df_dθ, df_dx
end

function ChainRulesCore.rrule(::typeof(_gamma_cdf),
        α::Real, θ::Real, x::Real)
    y = _gamma_cdf(α, θ, x)
    dα, dθ, dx = _gamma_cdf_partials(α, θ, x)
    pullback = let dα = dα, dθ = dθ, dx = dx
        ȳ -> (NoTangent(), ȳ * dα, ȳ * dθ, ȳ * dx)
    end
    return y, pullback
end

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(_gamma_cdf), Float64, Float64, Float64}

"""
$(TYPEDSIGNATURES)

Expected cumulative deaths by time `T` from a single seeding case under
exponential growth at rate `r`, using analytic result specific to the
Gamma distribution:

```math
\\mathbb{E}[D_T] = \\mathrm{CFR} \\cdot
    \\int_0^T e^{r s}\\, f(T - s)\\, ds = \\mathrm{CFR} \\cdot e^{r T} \\cdot M(-r) \\cdot F(T (1 + \\theta r)),
```

where f is the Gamma(α, θ) density, M is its moment-generating function,
and F is its CDF. The CDF is routed through [`_gamma_cdf`](@ref) so
Mooncake reverse-mode AD picks up a shape-parameter gradient (computed
internally with ForwardDiff).
"""
function expected_deaths(CFR, r, T, delay_dist::Gamma)
    α, θ = delay_dist.α, delay_dist.θ
    return CFR * exp(r * T) * mgf(delay_dist, -r) *
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
differentiates through the density alone (the reverse-mode AD backend
does not support the gamma CDF shape-parameter derivative). Outer and
inner integrals both use [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function integrate_exports_deaths(cumulative, delay_dist, lo, hi, T;
        alg = CUMULATIVE_INTEGRAL_ALG)
    cdf_to(x) = integrate(u -> pdf(delay_dist, u), zero(x), x; alg)
    g = let cumulative = cumulative, T = T
        s -> cumulative(s) * cdf_to(T - s)
    end
    return integrate(g, lo, hi; alg)
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

"""
$(TYPEDSIGNATURES)

`DataFrame` with one row per posterior parameter and columns
`:quantity, :lo90, :lo60, :lo30, :hi30, :hi60, :hi90` giving
equal-tailed 30%, 60% and 90% credible intervals.
"""
function summary_table(chn, params::AbstractVector{Symbol};
        digits::Integer = 2)
    @chain DataFrame(
            quantity = String[],
            lo90 = Float64[], lo60 = Float64[], lo30 = Float64[],
            hi30 = Float64[], hi60 = Float64[], hi90 = Float64[],
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
    rhats = _scalar_stats(MCMCChains.rhat(chn))
    esses = _scalar_stats(MCMCChains.ess(chn; kind = :bulk))
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
         lo90 = round(s.lo90; digits), lo60 = round(s.lo60; digits),
         lo30 = round(s.lo30; digits), hi30 = round(s.hi30; digits),
         hi60 = round(s.hi60; digits), hi90 = round(s.hi90; digits))
    end
    return DataFrame(rows)
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
        (scenario = label, reported_C_T = val, narrowest_CrI = crI)
    end
    return DataFrame(rows)
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

_panel_pos(pos::Integer) = (1, pos)
_panel_pos(pos::Tuple)   = pos

_panel_exports!(fig, pos, pp, obs; predictive_label = "Posterior") = begin
    r, c = _panel_pos(pos)
    upper = max(20, ceil(Int, quantile(pp, 0.99)))
    ax = Axis(fig[r, c];
        xlabel = "Replicated exports",
        ylabel = "$(predictive_label) predictive count",
        title  = "Exports (Poisson)",
        limits = ((0, upper), nothing),
    )
    hist!(ax, pp; bins = 0:1:upper, color = (:steelblue, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

_panel_deaths!(fig, pos, pp, obs; predictive_label = "Posterior") = begin
    r, c = _panel_pos(pos)
    upper = max(1.0, quantile(pp, 0.995))
    ax = Axis(fig[r, c];
        xlabel = "Replicated deaths",
        ylabel = "$(predictive_label) predictive count",
        title  = "Deaths (NegBinomial)",
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
        ylabel = "$(predictive_label) predictive count",
        title  = "Reported cases (NegBinomial)",
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
either of the first two panels, and supply `pp_cases` to add the
reported-cases panel. Observed values are drawn as red `vlines`.
"""
function plot_posterior_predictive(
        pp_exports::Union{Nothing, AbstractVector},
        pp_deaths::Union{Nothing, AbstractVector},
        obs_exports::Union{Nothing, Real},
        obs_deaths::Union{Nothing, Real};
        pp_cases::Union{Nothing, AbstractVector} = nothing,
        obs_cases::Union{Nothing, Real}          = nothing,
        predictive_label::AbstractString         = "Posterior")
    panels = Tuple{Symbol, Any, Any}[]
    pp_exports === nothing ||
        push!(panels, (:exports, pp_exports, obs_exports))
    pp_deaths === nothing ||
        push!(panels, (:deaths,  pp_deaths,  obs_deaths))
    pp_cases === nothing ||
        push!(panels, (:cases,   pp_cases,   obs_cases))

    isempty(panels) && error(
        "plot_posterior_predictive needs at least one stream")

    fig = Figure(; size = (450 * length(panels), 380))
    for (i, (kind, pp, obs)) in enumerate(panels)
        if kind === :exports
            _panel_exports!(fig, i, pp, obs; predictive_label)
        elseif kind === :deaths
            _panel_deaths!(fig, i, pp, obs; predictive_label)
        else
            _panel_cases!(fig, i, pp, obs; predictive_label)
        end
    end
    return fig
end

"""
$(TYPEDSIGNATURES)

Two-row × three-column comparison of posterior-predictive
distributions. Top row: replicates from the per-stream fits
(`exports_only`, `deaths_only`, `cases_only`). Bottom row:
replicates from the joint fit, conditioning on all three observed
streams. Observed values shown as red vertical lines.

Each panel is a histogram of replicated counts; rows share the
same x-axis (the stream's count) so the per-stream and joint
predictives are directly comparable.
"""
function plot_posterior_predictive_grid(;
        individual::NamedTuple,   # (; exports, deaths, cases) of pp draws
        joint::NamedTuple,        # (; exports, deaths, cases) of pp draws
        observed::NamedTuple,     # (; exports, deaths, cases) of obs values
    )
    fig = Figure(; size = (1200, 640))
    rows = ((:individual, individual, "per-stream fit"),
            (:joint,      joint,      "joint fit"))
    for (i, (_, pp, label)) in enumerate(rows)
        _panel_exports!(fig, (i, 1), pp.exports, observed.exports;
                        predictive_label = label)
        _panel_deaths!(fig, (i, 2),  pp.deaths,  observed.deaths;
                       predictive_label = label)
        _panel_cases!(fig, (i, 3),   pp.cases,   observed.cases;
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
thinned by `thin`.
"""
function plot_pair(chn, params::AbstractVector{Symbol};
        thin::Integer = 2)
    cols = NamedTuple(p => _draws(chn, p) for p in params)
    df = DataFrame(cols)
    return PairPlots.pairplot(df[1:thin:end, :])
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
    g = let r = r, T = T, delay_dist = delay_dist
        s -> r * exp(r * s) * ccdf(delay_dist, T - s)
    end
    return CFR * integrate(g, zero(T), T; alg)
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
(e.g. `TOTAL_DEATHS` from the bundled observations). `alg` is the
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
# package-wide `expected_deaths` integrator.
function _forecast_deaths_mean(r, Th, α, θ, CFR; alg = DEATH_INTEGRAL_ALG)
    return expected_deaths(CFR, r, Th, Gamma(α, θ); alg)
end

function _nb_rand(rng, k, μ)
    μs = max(μ, eps(typeof(μ)))
    p = clamp(k / (k + μs), eps(typeof(k)), one(k) - eps(typeof(k)))
    return rand(rng, NegativeBinomial(k, p))
end

"""
    forecast_reported(chn; horizon = 7, daily_travellers, source_population,
                      obs_cases, obs_deaths, obs_exports, seed = 20260520)

One-week-ahead (default `horizon = 7` days) posterior-predictive
forecast. For each draw, continue exponential growth to `T + horizon`
and apply the observation models, returning a `DataFrame` with one row
per draw and columns:

- `:cases_cum`, `:deaths_cum`, `:exports_cum` — replicated cumulative
  counts reported by `T + horizon`.
- `:cases_new`, `:deaths_new`, `:exports_new` — new counts over the
  coming week (`*_cum` minus the corresponding observed count at `T`,
  floored at zero).

Reads `:r, :T, :CFR, :α, :θ, :w, :p_drc, :p_uganda, :k` from `chn`. DRC
reported cases use the DRC ascertainment fraction `p_drc`; exports use
`p_uganda · q` with `q = daily_travellers / source_population`. Assumes
growth continues unchanged over the horizon (no interventions, no
saturation).
"""
function forecast_reported(chn;
        horizon::Real          = 7,
        daily_travellers::Real,
        source_population::Real,
        obs_cases::Real,
        obs_deaths::Real,
        obs_exports::Real,
        seed::Integer          = 20260520,
        alg                    = DEATH_INTEGRAL_ALG)
    r   = _draws(chn, :r)
    T   = _draws(chn, :T)
    CFR = _draws(chn, :CFR)
    α   = _draws(chn, :α)
    θ   = _draws(chn, :θ)
    w   = _draws(chn, :w)
    pr  = _draws(chn, :p_drc)
    pu  = _draws(chn, :p_uganda)
    k   = _draws(chn, :k)

    rng = MersenneTwister(seed)
    n = length(r)
    q = daily_travellers / source_population
    cases_cum   = Vector{Int}(undef, n)
    deaths_cum  = Vector{Int}(undef, n)
    exports_cum = Vector{Int}(undef, n)

    @inbounds for i in 1:n
        Th = T[i] + horizon
        ## DRC reported cases: p_drc · C(T+h).
        μ_cases = pr[i] * exp(r[i] * Th)
        cases_cum[i] = _nb_rand(rng, k[i], μ_cases)
        ## DRC deaths: CFR · ∫_0^{T+h} exp(r·s) f(T+h−s) ds.
        μ_deaths = _forecast_deaths_mean(r[i], Th, α[i], θ[i], CFR[i]; alg)
        deaths_cum[i] = _nb_rand(rng, k[i], μ_deaths)
        ## Uganda exports: p_uganda · q · ∫_{T+h−w}^{T+h} C(s) ds (closed
        ## form for exponential growth).
        lo = max(Th - w[i], zero(Th))
        μ_exports = pu[i] * q * (exp(r[i] * Th) - exp(r[i] * lo)) / r[i]
        exports_cum[i] = rand(rng, Poisson(max(μ_exports, eps(μ_exports))))
    end

    _new(cum, obs) = max.(cum .- round(Int, obs), 0)
    return DataFrame(
        cases_cum    = cases_cum,
        deaths_cum   = deaths_cum,
        exports_cum  = exports_cum,
        cases_new    = _new(cases_cum,   obs_cases),
        deaths_new   = _new(deaths_cum,  obs_deaths),
        exports_new  = _new(exports_cum, obs_exports),
    )
end

"""
    forecast_table(fc::DataFrame; digits = 0)

Summarise a [`forecast_reported`](@ref) result into a `DataFrame` with
one row per stream (cases, deaths, exports) and 30/60/90% credible
intervals for the cumulative forecast and the new-this-week count.
"""
function forecast_table(fc::DataFrame; digits::Integer = 0)
    rows = NamedTuple[]
    for (label, cum, new) in (
            ("DRC reported cases", :cases_cum,  :cases_new),
            ("DRC deaths",         :deaths_cum, :deaths_new),
            ("Uganda exports",     :exports_cum, :exports_new))
        sc = posterior_summary(fc[!, cum])
        sn = posterior_summary(fc[!, new])
        push!(rows, (
            stream         = label,
            cum_lo90 = round(sc.lo90; digits), cum_med = round(quantile(fc[!, cum], 0.5); digits),
            cum_hi90 = round(sc.hi90; digits),
            new_lo90 = round(sn.lo90; digits), new_med = round(quantile(fc[!, new], 0.5); digits),
            new_hi90 = round(sn.hi90; digits),
        ))
    end
    return DataFrame(rows)
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
            xlabel = title, ylabel = "Forecast count",
            title = "One week ahead", limits = ((0, upper), nothing))
        hist!(ax, v; bins = range(0, upper; length = 30),
              color = (colour, 0.7))
    end
    return fig
end

end # module
