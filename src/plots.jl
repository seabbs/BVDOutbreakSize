# All package figures: posterior densities of `C_T`, posterior- and
# prior-predictive panel grids, pair plots, point-and-interval
# comparison, CFR prior, start-date and no-onward-transmission
# densities, and the one-week-ahead forecast figures.

"""
Overlaid posterior densities of `C_T` from one or more fits, built
through AlgebraOfGraphics. The 15 published scenario point estimates
are drawn as faint dashed Makie `vlines` on top of the AoG figure.
"""
function plot_cumulative_cases(
        streams::Pair{String, <:AbstractVector}...;
        scenarios = REPORT_SCENARIOS,
        xmax::Union{Nothing, Real} = nothing)
    upper = isnothing(xmax) ?
            1.05 * maximum(quantile(s.second, 0.995) for s in streams) :
            xmax
    df = @chain DataFrame(
        stream = String[], C_T = Float64[]
    ) begin
        let df = _
            for (label, draws) in streams
                for x in draws
                    0 < x < upper * 1.05 && push!(df, (label, float(x)))
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
        axis = (; ylabel = "Posterior density",
            title = "Posterior C_T by data stream",
            limits = ((0, upper), nothing)),
        figure = (; size = (760, 420))
    )

    scenario_xs = Float64[val for (_, val) in scenarios if val < upper]
    isempty(scenario_xs) || vlines!(fg.figure.content[1], scenario_xs;
        color = (:grey, 0.4), linestyle = :dash)
    return fg
end

"""
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
        axis = (; ylabel = "Posterior density", title = title),
        figure = (; size = (760, 420))
    )
end

_panel_pos(pos::Integer) = (1, pos)
_panel_pos(pos::Tuple) = pos

function _panel_exports!(fig, pos, pp, obs; predictive_label = "Posterior")
    r, c = _panel_pos(pos)
    upper = max(20, ceil(Int, quantile(pp, 0.99)))
    ax = Axis(fig[r, c];
        xlabel = "Replicated exported cases",
        ylabel = "$(predictive_label) predictive frequency",
        title = "Exports (cases)",
        limits = ((0, upper), nothing)
    )
    hist!(ax, pp; bins = 0:1:upper, color = (:steelblue, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

function _panel_exports_deaths!(fig, pos, pp, obs;
        predictive_label = "Posterior")
    r, c = _panel_pos(pos)
    upper = max(3, ceil(Int, quantile(pp, 0.995)))
    ax = Axis(fig[r, c];
        xlabel = "Replicated deaths among exports",
        ylabel = "$(predictive_label) predictive frequency",
        title = "Exports (deaths)",
        limits = ((0, upper), nothing)
    )
    hist!(ax, pp; bins = 0:1:upper, color = (:rebeccapurple, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

function _panel_deaths!(fig, pos, pp, obs; predictive_label = "Posterior")
    r, c = _panel_pos(pos)
    upper = max(1.0, quantile(pp, 0.995))
    ax = Axis(fig[r, c];
        xlabel = "Replicated deaths",
        ylabel = "$(predictive_label) predictive frequency",
        title = "Deaths (DRC)",
        limits = ((0, upper), nothing)
    )
    hist!(ax, pp; bins = range(0, upper; length = 40),
        color = (:firebrick, 0.7))
    vlines!(ax, [obs]; color = :red, linewidth = 2)
    return ax
end

function _panel_cases!(fig, pos, pp, obs; predictive_label = "Posterior")
    r, c = _panel_pos(pos)
    upper = max(1.0, quantile(pp, 0.995))
    ax = Axis(fig[r, c];
        xlabel = "Replicated reported cases",
        ylabel = "$(predictive_label) predictive frequency",
        title = "Reported cases (DRC)",
        limits = ((0, upper), nothing)
    )
    hist!(ax, pp; bins = range(0, upper; length = 40),
        color = (:seagreen, 0.7))
    if obs !== nothing
        vlines!(ax, [obs]; color = :red, linewidth = 2)
    end
    return ax
end

"""
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
        pp_cases::Union{Nothing, AbstractVector} = nothing,
        obs_cases::Union{Nothing, Real} = nothing,
        pp_exports_deaths::Union{Nothing, AbstractVector} = nothing,
        obs_exports_deaths::Union{Nothing, Real} = nothing,
        predictive_label::AbstractString = "Posterior")
    panels = Tuple{Symbol, Any, Any}[]
    pp_exports === nothing ||
        push!(panels, (:exports, pp_exports, obs_exports))
    pp_exports_deaths === nothing ||
        push!(panels, (:exports_deaths, pp_exports_deaths,
            obs_exports_deaths))
    pp_deaths === nothing ||
        push!(panels, (:deaths, pp_deaths, obs_deaths))
    pp_cases === nothing ||
        push!(panels, (:cases, pp_cases, obs_cases))

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
        observed::NamedTuple
)
    fig = Figure(; size = (1600, 640))
    rows = ((:individual, individual, "per-stream fit"),
        (:joint, joint, "joint fit"))
    for (i, (_, pp, label)) in enumerate(rows)
        _panel_exports!(fig, (i, 1), pp.exports, observed.exports;
            predictive_label = label)
        _panel_exports_deaths!(fig, (i, 2), pp.exports_deaths,
            observed.exports_deaths; predictive_label = label)
        _panel_deaths!(fig, (i, 3), pp.deaths, observed.deaths;
            predictive_label = label)
        _panel_cases!(fig, (i, 4), pp.cases, observed.cases;
            predictive_label = label)
    end
    return fig
end

"""
Prior predictive variant of `plot_posterior_predictive`, with the
panel labels switched to "Prior".
"""
function plot_prior_predictive(
        pp_exports::Union{Nothing, AbstractVector},
        pp_deaths::Union{Nothing, AbstractVector},
        obs_exports::Union{Nothing, Real},
        obs_deaths::Union{Nothing, Real};
        pp_cases::Union{Nothing, AbstractVector} = nothing,
        obs_cases::Union{Nothing, Real} = nothing)
    return plot_posterior_predictive(
        pp_exports, pp_deaths, obs_exports, obs_deaths;
        pp_cases, obs_cases, predictive_label = "Prior")
end

"""
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
        PairPlots.Series(post; label = "Posterior", color = colours[1]),
        PairPlots.Series(_table(prior); label = "Prior",
            color = colours[2])
    )
end

"""
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
    n = length(rows)
    labels = [String(r[1]) for r in rows]
    central = [float(r[2]) for r in rows]
    lo = [float(r[3]) for r in rows]
    hi = [float(r[4]) for r in rows]
    top = isnothing(xmax) ? maximum(hi) * 1.08 : xmax

    fig = Figure(; size = (840, 120 + 46n))
    ax = Axis(fig[1, 1];
        xlabel = xlabel,
        yticks = (collect(1:n), reverse(labels)),
        limits = ((0, top), (0.5, n + 0.5))
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
        title = "Prior over the case-fatality ratio",
        limits = ((0, 0.7), nothing)
    )
    lines!(ax, xs, ys; color = colours[1], linewidth = 2)
    vlines!(ax, [55 / 169]; color = :firebrick, linewidth = 2)
    vlines!(ax, [0.26, 0.40];
        color = (:grey, 0.6), linestyle = :dash, linewidth = 2)
    return fig
end

"""
One-row, two-panel figure summarising when the outbreak began. The
left panel is the posterior density of the outbreak start date,
obtained by rescaling the days-since-seeding `T` to a calendar date
(`as_of_date` minus `T`). The right panel is the joint `(τ, T)`
posterior pair plot, which is positively correlated: slower growth
(larger `τ`) needs a longer elapsed `T` to reach the same counts.
"""
function plot_start_date_pair(chn;
        as_of_date::AbstractString, thin::Integer = 2)
    T_draws = _draws(chn, :T)
    cutoff_days = date2epochdays(Date(as_of_date))
    start_days = cutoff_days .- T_draws

    fig = Figure(; size = (1100, 460))
    ax = Axis(fig[1, 1];
        xlabel = "Outbreak start date",
        ylabel = "Posterior density",
        title = "Implied start of sustained transmission",
        xticklabelrotation = π / 6
    )
    density!(ax, start_days; color = (:steelblue, 0.5),
        strokecolor = :steelblue, strokewidth = 2)
    ## Date ticks every four weeks across the posterior range, so the
    ## start date stays readable rather than relying on the default
    ## locator or crowding the axis as the range widens.
    lo = floor(Int, minimum(start_days))
    hi = ceil(Int, maximum(start_days))
    ax.xticks = collect(lo:28:hi)
    ax.xtickformat = vals -> [string(epochdays2date(round(Int, v))) for v in vals]

    pair_df = DataFrame(τ = _draws(chn, :τ), T = T_draws)
    PairPlots.pairplot(fig[1, 2], pair_df[1:thin:end, :])
    return fig
end

"""
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
        title = "Still expected (future)")
    density!(ax1, df.delta_deaths; color = (:firebrick, 0.5),
        strokecolor = :firebrick, strokewidth = 2)

    ax2 = Axis(fig[1, 2];
        xlabel = "Projected total deaths (no onward transmission)",
        ylabel = "Posterior density",
        title = "Projected total")
    density!(ax2, df.total_projected; color = (:firebrick, 0.5),
        strokecolor = :firebrick, strokewidth = 2)
    vlines!(ax2, [float(obs_deaths)];
        color = :black, linestyle = :dash, linewidth = 2)

    return fig
end

"""
Three-panel histogram of the new-this-week forecast counts (cases,
deaths, exports) from [`forecast_reported`](@ref).
"""
function plot_forecast(fc::DataFrame)
    fig = Figure(; size = (1100, 360))
    cols = ((:cases_new, "New reported cases (DRC)", :steelblue),
        (:deaths_new, "New deaths (DRC)", :firebrick),
        (:exports_new, "New exports (Uganda)", :seagreen))
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
        lo = quantile(v, 0.05)
        hi = quantile(v, 0.95)
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
        (:cases_cum, :cases_new, "reported cases (DRC)", :steelblue,
            float(cases), float(cases) - float(baseline_cases)),
        (:deaths_cum, :deaths_new, "deaths (DRC)", :firebrick,
            float(deaths), float(deaths) - float(baseline_deaths)),
        (:exports_cum, :exports_new, "exports (Uganda)", :seagreen,
            float(exports), float(exports) - float(baseline_exports)))
    for (j, (ccol, ncol, name, colour, obs_cum, obs_new)) in
        enumerate(streams)
        panel!(1, j, fc[!, ccol], obs_cum, "Cumulative $name", colour)
        panel!(2, j, fc[!, ncol], max(obs_new, 0.0), "New $name", colour)
    end
    return fig
end
