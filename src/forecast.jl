# One-week-ahead posterior-predictive forecast. Continues the fitted
# exponential growth `h` days past the cut-off `T` and applies the same
# observation models to forecast the cumulative reported cases (DRC),
# deaths (DRC) and exports (Uganda) by `T + h`, plus the new counts
# expected over the coming week. Each draw produces a replicated integer
# count so the intervals include both parameter and observation
# uncertainty.

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
