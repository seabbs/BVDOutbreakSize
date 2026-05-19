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
import CairoMakie
import AlgebraOfGraphics as AoG
import PairPlots
using CairoMakie: Figure, Axis, hist!, vlines!

export REPORT_SCENARIOS,
       ITURI_POPULATION, ITURI_DAILY_TRAVEL,
       ITURI_DAILY_TRAVEL_SD,
       EXPORTED_CASES, TOTAL_DEATHS,
       load_observations,
       summary_table, posterior_summary,
       streams_table, comparison_table,
       nuts_sample, default_adtype,
       plot_cumulative_cases, plot_prior_predictive,
       plot_posterior_predictive, plot_pair

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
$(TYPEDSIGNATURES)

Load the observation block from `data/observations.toml` and return
it as a `NamedTuple`. If `path` is omitted, the bundled TOML file
shipped with the package is used. Expected fields:

- `exported_cases::Int`
- `total_deaths::Int`
- `daily_outbound_travellers::Real`
- `daily_outbound_travellers_sd::Real`
- `source_population::Int`
"""
function load_observations(
        path::AbstractString = joinpath(@__DIR__, "..", "data",
                                        "observations.toml"))
    raw = TOML.parsefile(path)
    return (;
        exported_cases               = Int(raw["exported_cases"]),
        total_deaths                 = Int(raw["total_deaths"]),
        daily_outbound_travellers    = float(
            raw["daily_outbound_travellers"]),
        daily_outbound_travellers_sd = float(
            raw["daily_outbound_travellers_sd"]),
        source_population            = Int(raw["source_population"]),
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

"""
$(TYPEDSIGNATURES)

Two-panel posterior predictive histogram (Poisson exports and
NegBinomial deaths) with the observed values drawn in red.
"""
function plot_posterior_predictive(
        pp_exports::AbstractVector, pp_deaths::AbstractVector,
        obs_exports::Real,         obs_deaths::Real)
    fig = Figure(; size = (900, 380))

    e_upper = max(20, ceil(Int, quantile(pp_exports, 0.99)))
    ax_e = Axis(fig[1, 1];
        xlabel = "Replicated exports",
        ylabel = "Posterior predictive count",
        title  = "Exports (Poisson)",
        limits = ((0, e_upper), nothing),
    )
    hist!(ax_e, pp_exports; bins = 0:1:e_upper,
          color = (:steelblue, 0.7))
    vlines!(ax_e, [obs_exports]; color = :red, linewidth = 2)

    d_upper = max(1.0, quantile(pp_deaths, 0.995))
    ax_d = Axis(fig[1, 2];
        xlabel = "Replicated deaths",
        ylabel = "Posterior predictive count",
        title  = "Deaths (NegBinomial)",
        limits = ((0, d_upper), nothing),
    )
    hist!(ax_d, pp_deaths; bins = range(0, d_upper; length = 40),
          color = (:firebrick, 0.7))
    vlines!(ax_d, [obs_deaths]; color = :red, linewidth = 2)

    return fig
end

"""
$(TYPEDSIGNATURES)

Two-panel prior predictive histogram for replicated exports and
deaths. Same layout as `plot_posterior_predictive` but with the
prior-predictive label.
"""
function plot_prior_predictive(
        pp_exports::AbstractVector, pp_deaths::AbstractVector,
        obs_exports::Real,         obs_deaths::Real)
    fig = plot_posterior_predictive(pp_exports, pp_deaths,
                                    obs_exports, obs_deaths)
    fig.content[1].title[] = "Exports (prior predictive)"
    fig.content[2].title[] = "Deaths (prior predictive)"
    return fig
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

end # module
