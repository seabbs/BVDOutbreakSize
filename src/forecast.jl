# One-week-ahead posterior-predictive forecast. Continues the fitted
# renewal trajectory `horizon` days past the cut-off at the current growth
# rate `r` (the daily growth of infections at the cut-off, held constant
# over the short horizon) and scales the fitted expected counts for each
# stream, then replicates an integer count per draw so the intervals carry
# both parameter and observation uncertainty.

function _nb_rand(rng, k, μ)
    μs = max(μ, eps(typeof(μ)))
    p = clamp(k / (k + μs), eps(typeof(k)), one(k) - eps(typeof(k)))
    return rand(rng, NegativeBinomial(k, p))
end

"""
One-week-ahead (default `horizon = 7` days) posterior-predictive
forecast. For each draw, continue the current growth rate `r` over the
horizon and scale the fitted expected counts by `exp(r · horizon)`, then
replicate the cumulative counts. Returns a `DataFrame` with one row per
draw and columns:

- `:cases_cum`, `:deaths_cum`, `:exports_cum` — replicated cumulative
  counts reported by the cut-off plus the horizon.
- `:cases_new`, `:deaths_new`, `:exports_new` — new counts over the
  coming week (`*_cum` minus the corresponding observed count at the
  cut-off, floored at zero).

Reads `:r`, `:expected_reports_T`, `:expected_deaths_T`,
`:expected_exports_T` and `:k` from `chn`. Assumes the current growth rate
continues unchanged over the horizon (no further interventions, no
saturation).
"""
function forecast_reported(chn;
        horizon::Real = 7,
        obs_cases::Real,
        obs_deaths::Real,
        obs_exports::Real,
        seed::Integer = 20260520)
    r = _draws(chn, :r)
    cases_T = _draws(chn, :expected_reports_T)
    deaths_T = _draws(chn, :expected_deaths_T)
    exports_T = _draws(chn, :expected_exports_T)
    k = _draws(chn, :k)

    rng = MersenneTwister(seed)
    n = length(r)
    cases_cum = Vector{Int}(undef, n)
    deaths_cum = Vector{Int}(undef, n)
    exports_cum = Vector{Int}(undef, n)

    @inbounds for i in 1:n
        grow = exp(r[i] * horizon)
        cases_cum[i] = _nb_rand(rng, k[i], cases_T[i] * grow)
        deaths_cum[i] = _nb_rand(rng, k[i], deaths_T[i] * grow)
        μ_exports = exports_T[i] * grow
        exports_cum[i] = rand(rng, Poisson(max(μ_exports, eps(μ_exports))))
    end

    _new(cum, obs) = max.(cum .- round(Int, obs), 0)
    return DataFrame(
        cases_cum = cases_cum,
        deaths_cum = deaths_cum,
        exports_cum = exports_cum,
        cases_new = _new(cases_cum, obs_cases),
        deaths_new = _new(deaths_cum, obs_deaths),
        exports_new = _new(exports_cum, obs_exports)
    )
end

"""
Summarise a [`forecast_reported`](@ref) result into a `DataFrame` with
one row per stream (cases, deaths, exports) and quantity (cumulative
total by the cut-off plus the horizon, or new this week), reporting the
equal-tailed 30/60/90% credible interval endpoints (`lower_90 …
upper_90`) used by the other summary tables.
"""
function forecast_table(fc::DataFrame; digits::Integer = 0)
    _row(label,
        quantity,
        draws) = begin
        s = posterior_summary(draws)
        (stream = label, quantity = quantity,
            lower_90 = round(s.lo90; digits), lower_60 = round(s.lo60; digits),
            lower_30 = round(s.lo30; digits), upper_30 = round(s.hi30; digits),
            upper_60 = round(s.hi60; digits), upper_90 = round(s.hi90; digits))
    end
    rows = NamedTuple[]
    for (label, cum, new) in (
        ("DRC reported cases", :cases_cum, :cases_new),
        ("DRC deaths", :deaths_cum, :deaths_new),
        ("Uganda exports", :exports_cum, :exports_new))
        push!(rows, _row(label, "cumulative by T+7", fc[!, cum]))
        push!(rows, _row(label, "new this week", fc[!, new]))
    end
    return _prettify(DataFrame(rows))
end

"""
Validate a [`forecast_reported`](@ref) projection against the counts that
were later observed. `cases`, `deaths` and `exports` are the observed
cumulative DRC reported cases, DRC deaths and Uganda exports at the
forecast target date. Returns a `DataFrame` with one row per stream giving
the observed count, the equal-tailed 30/60/90% predictive intervals (the
same endpoints as the other summary tables), and whether the observed
count falls inside the 90% interval.
"""
function forecast_vs_truth(fc::DataFrame;
        cases::Real, deaths::Real, exports::Real, digits::Integer = 0)
    _row(label,
        draws,
        obs) = begin
        s = posterior_summary(draws)
        lo = round(s.lo90; digits)
        hi = round(s.hi90; digits)
        (stream = label, observed = round(obs; digits),
            lower_90 = lo, lower_60 = round(s.lo60; digits),
            lower_30 = round(s.lo30; digits), upper_30 = round(s.hi30; digits),
            upper_60 = round(s.hi60; digits), upper_90 = hi,
            within_90 = lo <= obs <= hi ? "yes" : "no")
    end
    rows = [
        _row("DRC reported cases", fc[!, :cases_cum], cases),
        _row("DRC deaths", fc[!, :deaths_cum], deaths),
        _row("Uganda exports", fc[!, :exports_cum], exports)
    ]
    return _prettify(DataFrame(rows))
end

"""
Roll the one-week-ahead forecast across an observed cumulative
trajectory. `targets` is a vector of `(label, horizon_days,
observed_cumulative)` triples: for each, the fitted current growth rate
`r` is projected `horizon_days` past the cut-off and the predicted
cumulative reported cases compared against `observed_cumulative`. Returns
a `DataFrame` with one row per target giving the horizon, the observed
count, the equal-tailed 30/60/90% predictive intervals, and whether the
observed count falls inside the 90% interval. Unlike
[`forecast_vs_truth`](@ref), which scores only the endpoint, this scores
the whole observed trajectory across the horizon. Reads `:r`,
`:expected_reports_T` and `:k` from `chn`.
"""
function forecast_vs_truth_trajectory(
        chn; targets::AbstractVector,
        seed::Integer = 20260520)
    r = _draws(chn, :r)
    cases_T = _draws(chn, :expected_reports_T)
    k = _draws(chn, :k)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for (label, horizon, obs) in targets
        grow = exp.(r .* horizon)
        cases_cum = [_nb_rand(rng, k[i], cases_T[i] * grow[i])
                     for i in eachindex(r)]
        s = posterior_summary(cases_cum)
        lo = round(s.lo90)
        hi = round(s.hi90)
        push!(rows,
            (label = label, horizon_days = horizon,
                observed = round(obs),
                lower_90 = lo, lower_60 = round(s.lo60),
                lower_30 = round(s.lo30), upper_30 = round(s.hi30),
                upper_60 = round(s.hi60), upper_90 = hi,
                within_90 = lo <= obs <= hi ? "yes" : "no"))
    end
    return _prettify(DataFrame(rows))
end
