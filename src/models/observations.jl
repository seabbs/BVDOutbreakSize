# Observation submodels: each ties one data stream to the shared daily
# onset incidence from the generating infection process. A stream
# convolves the onsets with its own sampled onset-to-event delay, scales
# by the relevant ascertainment / CFR / positivity / travel factor, and
# reads the cumulative expected count off the daily series at each vintage
# day, fitting the between-vintage increments (and the cut-off total).
# Convolutions replace the integral model's continuous onset-to-event
# integrals while preserving the v1.3.0 per-vintage time-series
# likelihoods.

"""
NaN / Inf-safe `NegativeBinomial` constructor parameterised by mean `μ`
and dispersion `k`, with clamping on the success probability so extreme
NUTS proposals during warmup do not trip the distribution domain check.
Shared by the count-stream observation submodels.
"""
function safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

"""
Cumulative of a daily series `daily` read off at the vintage day indices
`days` (1-based into the grid). Each entry is the running sum up to and
including that day, giving the modelled cumulative count at each vintage.
Days beyond the series length are clamped to the final cumulative. Pure
and AD-transparent; the output element type follows `daily`.
"""
function cumulative_at(daily::AbstractVector, days::AbstractVector{<:Integer})
    cum = cumsum(daily)
    n = length(cum)
    return [cum[clamp(d, 1, n)] for d in days]
end

"""
Add the per-vintage cumulative-history likelihood for one stream. Given
the modelled cumulative `expected_cum` at each vintage day and the
observed cumulative `counts`, scores the between-vintage increments with
NegativeBinomials sharing the dispersion `k` (one per vintage). The
observed increment for vintage `i` is `counts[i] - counts[i-1]` (the first
is `counts[1]`). The counts are fixed data rather than a sampled quantity,
so the likelihood is added with `@addlogprob!` (a continuous function of
`expected_cum` that differentiates cleanly under Mooncake) rather than a
discrete `~`, which the NUTS model check would otherwise reject. A no-op
when `counts` is empty, so the caller doubles as a predictive generator.
Returns the modelled increments for reuse.
"""
@model function vintage_increments_model(expected_cum::AbstractVector,
        counts::AbstractVector{<:Integer}, k::Real)
    increments = diff(vcat(zero(eltype(expected_cum)), expected_cum))
    obs_increments = diff(vcat(zero(eltype(counts)), counts))
    if !isempty(obs_increments)
        lp = sum(logpdf(safe_nbinomial(k, safe_rate(increments[i])),
                     obs_increments[i]) for i in eachindex(obs_increments))
        @addlogprob! lp
    end
    return (; increments, obs_increments)
end

"""
DRC suspected-deaths likelihood, per-vintage time series. Convolves the
daily onsets with the sampled onset-to-death delay, scales by the CFR, and
reads the modelled cumulative deaths at each vintage day off the daily
series, fitting the between-vintage increments with a NegativeBinomial
sharing the surveillance dispersion `k` ([`surveillance_dispersion_model`]
(@ref)) and the cut-off total likewise. Samples the onset-to-death delay
and the CFR via injected submodels. The onset-to-death prior is centred on
the Bayesian BDBV line-list reanalysis (mean 11.2 d, SD 5.4 d; the
`bdbv-linelist-analysis` submodule), the same source the integral model
used. Returns the cut-off expected count, the daily death series, the
onset-to-death PMF and the CFR for reuse by [`exports_deaths_model`](@ref).
"""
@model function deaths_model(
        deaths_history,
        total_deaths::Union{Missing, Integer},
        onsets::AbstractVector, k::Real;
        cfr = cfr_model(),
        onset_to_death = censored_delay_model(60;
            mean_prior = truncated(Normal(11.2, 2.0); lower = 1),
            sd_prior = truncated(Normal(5.4, 1.5); lower = 1)))
    cfr_state ~ to_submodel(cfr)
    od_state ~ to_submodel(onset_to_death)
    CFR = cfr_state.CFR
    deaths_daily = CFR .* convolve_delay(onsets, od_state.pmf)

    expected_cum = cumulative_at(deaths_daily, deaths_history.days)
    _inc ~ to_submodel(vintage_increments_model(expected_cum,
            deaths_history.counts, k), false)

    raw_total = sum(deaths_daily)
    expected_deaths_T := safe_rate(raw_total)
    total_deaths ~ safe_nbinomial(k, expected_deaths_T)

    return (; CFR, od_pmf = od_state.pmf, deaths_daily, expected_deaths_T)
end

"""
DRC reported-cases ascertainment likelihood, per-vintage time series.
Convolves the daily onsets with the sampled onset-to-report delay, scales
by the DRC ascertainment fraction `p_drc`, and reads the modelled
cumulative reported cases at each vintage day off the daily series,
fitting the increments with a NegativeBinomial sharing `k`. The
onset-to-report delay is injected, defaulting to a weakly-informative
prior on the onset-to-notification delay (mean 4.5 d, SD 3.6 d),
consistent with Ebola surveillance reporting delays.
"""
@model function reported_cases_model(
        reported_history,
        reported_cases::Union{Missing, Integer},
        onsets::AbstractVector, k::Real, p_drc::Real;
        onset_to_report = censored_delay_model(30;
            mean_prior = truncated(Normal(4.5, 1.5); lower = 1),
            sd_prior = truncated(Normal(3.6, 1.2); lower = 1)))
    report_state ~ to_submodel(onset_to_report)
    reports_daily = p_drc .* convolve_delay(onsets, report_state.pmf)

    expected_cum = cumulative_at(reports_daily, reported_history.days)
    _inc ~ to_submodel(vintage_increments_model(expected_cum,
            reported_history.counts, k), false)

    raw_total = sum(reports_daily)
    expected_reports := safe_rate(raw_total)
    reported_cases ~ safe_nbinomial(k, expected_reports)

    return (; p_drc, reports_daily, expected_reports)
end

"""
DRC confirmed-cases likelihood, per-vintage time series. Convolves the
daily onsets with the sampled laboratory-confirmation delay, scales by the
test positivity (the fraction of suspected cases laboratory confirmed),
and reads the modelled cumulative confirmed cases at each vintage day off
the daily series, fitting the increments with a NegativeBinomial sharing
`k`. Samples the lab delay and the positivity via injected submodels. The
lab-confirmation delay defaults to a weakly-informative prior (mean 6.0 d,
SD 4.0 d).
"""
@model function confirmed_cases_model(
        confirmed_history,
        confirmed_cases::Union{Missing, Integer},
        onsets::AbstractVector, k::Real;
        positivity = test_positivity_model(),
        onset_to_confirm = censored_delay_model(30;
            mean_prior = truncated(Normal(6.0, 2.0); lower = 1),
            sd_prior = truncated(Normal(4.0, 1.5); lower = 1)))
    pos_state ~ to_submodel(positivity)
    lab_state ~ to_submodel(onset_to_confirm)
    positivity_val = pos_state.positivity
    confirmed_daily = positivity_val .* convolve_delay(onsets, lab_state.pmf)

    expected_cum = cumulative_at(confirmed_daily, confirmed_history.days)
    _inc ~ to_submodel(vintage_increments_model(expected_cum,
            confirmed_history.counts, k), false)

    raw_total = sum(confirmed_daily)
    expected_confirmed := safe_rate(raw_total)
    confirmed_cases ~ safe_nbinomial(k, expected_confirmed)

    return (; positivity = positivity_val, confirmed_daily,
        expected_confirmed)
end

"""
Uganda exports likelihood (geographic spread). Builds the export onset
incidence `p_uganda · q · onsets` (with `q = daily_travellers /
source_population` the per-capita travel rate) and convolves it with the
sampled onset-to-detection delay, replacing the integral model's
detection-window survival term with a convolved set of delays. Sums to the
expected detected exports by the cut-off, fitted with Poisson (Uganda's
stream is small). Samples the traveller volume and the onset-to-detection
delay via injected submodels. The onset-to-detection prior is centred on
the Ebola onset-to-hospitalisation delay (mean 5.0 d, SD 4.7 d; WHO Ebola
Response Team 2014, NEJM), used here as the delay from symptom onset to
detection at a point of entry abroad. Returns the expected count, the
detection-timed series and the export onsets for reuse by
[`exports_deaths_model`](@ref).
"""
@model function exports_model(
        exported_cases::Union{Missing, Integer},
        onsets::AbstractVector, p_uganda::Real;
        source_population::Real = ITURI_POPULATION,
        traveller = traveller_volume_model(),
        onset_to_detection = censored_delay_model(30;
            mean_prior = truncated(Normal(5.0, 2.0); lower = 1),
            sd_prior = truncated(Normal(4.7, 1.5); lower = 1)))
    travel_state ~ to_submodel(traveller)
    daily_travellers = travel_state.daily_travellers
    q = daily_travellers / source_population

    export_onsets = p_uganda .* q .* onsets
    detect_state ~ to_submodel(onset_to_detection)
    detect_daily = convolve_delay(export_onsets, detect_state.pmf)

    raw_exports = sum(detect_daily)
    expected_exports_T := safe_rate(raw_exports)
    exported_cases ~ Poisson(expected_exports_T)

    return (; p_uganda, daily_travellers, export_onsets,
        expected_exports = expected_exports_T)
end

"""
Deaths-among-detected-exports likelihood. Convolves the export onsets
(timed from onset, the same staging as detection) with the onset-to-death
PMF shared from [`deaths_model`](@ref), scales by the CFR, and sums to the
expected cumulative export deaths by the cut-off, fitted with Poisson.
"""
@model function exports_deaths_model(
        exports_deaths::Union{Missing, Integer},
        export_onsets::AbstractVector, CFR::Real,
        od_pmf::AbstractVector)
    series = CFR .* convolve_delay(export_onsets, od_pmf)

    raw = sum(series)
    expected_exports_deaths_T := safe_rate(raw)
    exports_deaths ~ Poisson(expected_exports_deaths_T)

    return (; expected_exports_deaths_T, series)
end
