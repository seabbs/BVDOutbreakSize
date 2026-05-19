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
using Turing
using Turing.DynamicPPL: InitFromPrior
using MCMCChains: Chains
using DocStringExtensions
using Distributions: Gamma, ccdf
using Integrals: IntegralProblem, GaussLegendre, solve
import FastGaussQuadrature
import CairoMakie
import AlgebraOfGraphics as AoG
import PairPlots
using CairoMakie: Figure, Axis, hist!, vlines!

export REPORT_SCENARIOS,
       ITURI_POPULATION, ITURI_DAILY_TRAVEL,
       ITURI_DAILY_TRAVEL_SD,
       EXPORTED_CASES, TOTAL_DEATHS, REPORTED_CASES,
       load_observations,
       summary_table, posterior_summary,
       streams_table, comparison_table,
       nuts_sample, default_adtype,
       plot_cumulative_cases, plot_prior_predictive,
       plot_posterior_predictive, plot_posterior_predictive_grid,
       plot_pair,
       predict_no_onward_deaths, plot_no_onward_deaths

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
- `total_deaths::Int`
- `reported_cases::Int`
- `daily_outbound_travellers::Real`
- `daily_outbound_travellers_sd::Real`
- `source_population::Int`
- `sources::NamedTuple{(:exported_cases, :total_deaths, :reported_cases,
  :daily_outbound_travellers, :daily_outbound_travellers_sd,
  :source_population), NTuple{6, String}}` — citation per field.
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
        total_deaths                 = Int(_val("total_deaths")),
        reported_cases               = Int(_val("reported_cases")),
        daily_outbound_travellers    = float(
            _val("daily_outbound_travellers")),
        daily_outbound_travellers_sd = float(
            _val("daily_outbound_travellers_sd")),
        source_population            = Int(_val("source_population")),
        sources = (;
            exported_cases               = _src("exported_cases"),
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

    scenario_xs = [val for (_, val) in scenarios if val < xmax]
    vlines!(fg.figure.content[1], scenario_xs;
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

## --- Committed-deaths counterfactual ----------------------------------
##
## Lower bound on future deaths under the counterfactual that every
## onward transmission stops at time `T`. The cohort already infected
## by `T` still contributes deaths in the onset-to-death tail; per
## draw,
##
##     ΔD = CFR · ∫_0^T r · exp(r · s) · (1 - F_d(T - s)) ds,
##
## with `F_d` the Gamma(α, θ) CDF. Total projected cumulative deaths
## under the no-onward-transmission counterfactual is `obs_deaths + ΔD`.

const _NO_ONWARD_INTEGRAL_ALG = GaussLegendre(; n = 64)

function _committed_deaths_integrand(u, p)
    s = p.halfwidth * (u + 1)         # u ∈ [-1, 1] → s ∈ [0, T]
    return p.r * exp(p.r * s) * ccdf(p.delay_dist, p.T - s)
end

function _committed_deaths_one(r, T, α, θ, CFR;
        alg = _NO_ONWARD_INTEGRAL_ALG)
    halfwidth = T / 2
    delay_dist = Gamma(α, θ)
    params = (; r, T, halfwidth, delay_dist)
    prob = IntegralProblem(_committed_deaths_integrand,
                           (-1.0, 1.0), params)
    return CFR * halfwidth * solve(prob, alg).u
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

- `:delta_deaths`     additional committed deaths beyond `obs_deaths`
- `:total_projected`  `obs_deaths + delta_deaths`

`obs_deaths` is the number of deaths already observed at time `T`
(e.g. `TOTAL_DEATHS` from the bundled observations). `alg` is the
quadrature scheme used for ΔD; defaults to `GaussLegendre(n = 64)`,
matching the rest of the package.
"""
function predict_no_onward_deaths(chn;
        obs_deaths::Real,
        alg = _NO_ONWARD_INTEGRAL_ALG)
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

AlgebraOfGraphics density of the projected-total distribution from
[`predict_no_onward_deaths`](@ref) with a Makie vertical rule at
`obs_deaths`. The vertical rule marks deaths already observed at
time `T`; the density to its right is the lower-bound projection
under the no-onward-transmission counterfactual.
"""
function plot_no_onward_deaths(df::DataFrame; obs_deaths::Real)
    spec = AoG.data(df) *
           AoG.mapping(:total_projected =>
                       "Projected total deaths (no onward transmission)") *
           AoG.AlgebraOfGraphics.density() *
           AoG.visual(linewidth = 2, color = :firebrick)
    fg = AoG.draw(spec;
        axis = (; ylabel = "Posterior density",
                  title  = "Lower bound under no onward transmission"),
        figure = (; size = (760, 420)),
    )
    vlines!(fg.figure.content[1], [float(obs_deaths)];
            color = :black, linestyle = :dash, linewidth = 2)
    return fg
end

end # module
