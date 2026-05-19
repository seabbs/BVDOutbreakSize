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
using FlexiChains
import CairoMakie
import AlgebraOfGraphics as AoG
import PairPlots
using CairoMakie: Figure, Axis, hist!, vlines!

export REPORT_SCENARIOS,
       load_observations, DEFAULT_OBSERVATIONS_PATH,
       summary_table, posterior_summary,
       streams_table, comparison_table,
       nuts_sample, default_adtype,
       plot_cumulative_cases,
       plot_prior_predictive, plot_posterior_predictive,
       plot_pair

"""
    DEFAULT_OBSERVATIONS_PATH

Path to the bundled `data/observations.toml`. Override
[`load_observations`](@ref) to point at a different file.
"""
const DEFAULT_OBSERVATIONS_PATH = joinpath(@__DIR__, "..", "data",
                                           "observations.toml")

"""
    load_observations(path = DEFAULT_OBSERVATIONS_PATH)

Read the observations TOML and return a `NamedTuple` of named fields:

- `exported_cases::Int`              cases detected in Uganda
- `total_deaths::Int`                suspected BVD deaths in DRC
- `as_of_date::String`               ISO 8601 cut-off
- `source_population_label::String`  source-area label
- `source_population_size::Int`      source-area population
- `daily_outbound_travellers::Int`   mean daily PoE flow

To update for a later sitrep, edit `data/observations.toml` rather
than the model code; the literate walkthrough picks up the new
numbers on the next build.
"""
function load_observations(path::AbstractString = DEFAULT_OBSERVATIONS_PATH)
    cfg = TOML.parsefile(path)
    obs = cfg["observations"]
    pop = cfg["population"]
    return (
        exported_cases            = Int(obs["exported_cases"]),
        total_deaths              = Int(obs["total_deaths"]),
        as_of_date                = String(obs["as_of_date"]),
        source_population_label   = String(pop["source"]),
        source_population_size    = Int(pop["size"]),
        daily_outbound_travellers = Int(pop["daily_outbound_travellers"]),
    )
end

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
    default_adtype()

Mooncake reverse-mode AD with default `Mooncake.Config()`. Used as
the NUTS `adtype` keyword.
"""
default_adtype() = AutoMooncake(; config = Mooncake.Config())

"""
    nuts_sample(model; samples = 1000, chains = 4, target_accept = 0.9,
                seed = 20260518, progress = false, adtype = default_adtype())

NUTS on `model`, four parallel chains via `MCMCThreads`. Chains
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
    posterior_summary(xs)

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
    summary_table(chn, params; digits = 2)

`DataFrame` with one row per posterior parameter and columns
`:quantity, :lo90, :lo60, :lo30, :hi30, :hi60, :hi90` giving
equal-tailed 30%, 60% and 90% credible intervals (no point
estimate).
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
    streams_table(streams::Pair{String, <:AbstractVector}...; digits = 0)

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
    comparison_table(C_draws; scenarios = REPORT_SCENARIOS)

For each published `C_T` scenario, the narrowest joint posterior
credible interval (30, 60 or 90%) that contains it, or "outside 90%".
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
    plot_cumulative_cases(streams...; scenarios = REPORT_SCENARIOS,
                          xmax = 2_500)

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

"""
    plot_predictive(pp_exports, pp_deaths, obs_exports, obs_deaths;
                    kind = "Posterior")

Two-panel predictive histogram (NegBinomial exports; Poisson deaths)
with the observed values drawn in red. `kind` is "Prior" or
"Posterior" and is used in the panel titles.
"""
function plot_predictive(
        pp_exports::AbstractVector, pp_deaths::AbstractVector,
        obs_exports::Real,         obs_deaths::Real;
        kind::AbstractString = "Posterior")
    fig = Figure(; size = (900, 380))

    e_upper = max(20, ceil(Int, quantile(pp_exports, 0.99)))
    ax_e = Axis(fig[1, 1];
        xlabel = "Replicated exports",
        ylabel = "$(kind) predictive count",
        title  = "$(kind) predictive — exports",
        limits = ((0, e_upper), nothing),
    )
    hist!(ax_e, pp_exports; bins = 0:1:e_upper,
          color = (:steelblue, 0.7))
    vlines!(ax_e, [obs_exports]; color = :red, linewidth = 2)

    d_upper = max(20.0, quantile(pp_deaths, 0.995))
    ax_d = Axis(fig[1, 2];
        xlabel = "Replicated deaths",
        ylabel = "$(kind) predictive count",
        title  = "$(kind) predictive — deaths",
        limits = ((0, d_upper), nothing),
    )
    hist!(ax_d, pp_deaths; bins = range(0, d_upper; length = 40),
          color = (:firebrick, 0.7))
    vlines!(ax_d, [obs_deaths]; color = :red, linewidth = 2)

    return fig
end

"Convenience wrapper for `plot_predictive(...; kind = \"Posterior\")`."
plot_posterior_predictive(pp_exports, pp_deaths, obs_exports, obs_deaths) =
    plot_predictive(pp_exports, pp_deaths, obs_exports, obs_deaths;
                    kind = "Posterior")

"Convenience wrapper for `plot_predictive(...; kind = \"Prior\")`."
plot_prior_predictive(pp_exports, pp_deaths, obs_exports, obs_deaths) =
    plot_predictive(pp_exports, pp_deaths, obs_exports, obs_deaths;
                    kind = "Prior")

"""
    plot_pair(chn, params::Vector{Symbol}; thin::Int = 2)

PairPlots.jl corner plot over the named posterior parameters,
thinned by `thin`. Follows the `plot_pair` convention from the
hantavirus realtime work.
"""
function plot_pair(chn, params::AbstractVector{Symbol};
        thin::Integer = 2)
    cols = NamedTuple(p => _draws(chn, p) for p in params)
    df = DataFrame(cols)
    return PairPlots.pairplot(df[1:thin:end, :])
end

end # module
