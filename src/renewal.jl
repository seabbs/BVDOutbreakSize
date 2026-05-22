## Discrete-time renewal architecture (prototype, issue #81).
##
## A candidate redesign that replaces the deterministic continuous-time
## cumulative-incidence trajectory C(s) = exp(r·s) with a daily renewal
## process driven by a time-varying reproduction number R_t. Latent daily
## infections are convolved through realistic delays to build expected
## daily then cumulative counts for each observed stream.
##
## This file is package code (proper functions and Turing submodels), kept
## separate from the existing exponential-growth model so both can coexist.
## Names are suffixed or namespaced to avoid clashing with the exported
## exponential-growth helpers.

using Turing.DynamicPPL: @addlogprob!

## Names used below (`@model`, `to_submodel`, `Normal`, `Gamma`,
## `truncated`, `Poisson`, `Beta`, `logit`, `logistic`, `normlogcdf`,
## `safe_nbinomial`, the `ITURI_*` constants) are already imported by the
## enclosing module, into which this file is `include`d.

## --- Delay discretisation (AD-safe) -------------------------------------
##
## The owner's preferred route was CensoredDistributions.jl for the
## double-interval-censored delay PMFs. Its analytical primary-censored
## CDF for a Gamma routes through `HypergeometricFunctions.pFqweniger`,
## which contains a try/catch and so cannot be reverse-differentiated by
## Mooncake; `force_numeric = true` instead hits QuadGK's `cachedrule`,
## which Mooncake also rejects. `interval_censored` alone differentiates
## but returns a NaN shape-parameter gradient because the Gamma CDF shape
## derivative is unstable under Mooncake. We therefore discretise by hand
## here, integrating the density over each daily bin and forcing the
## density at zero to zero (exact for a Gamma with shape > 1, where the
## origin's `0·log 0` shape derivative would otherwise be NaN). This is the
## same trick the package already uses in `ExportDeathDelay`, and it gives
## fully finite Mooncake gradients (see docs/proposals/discrete-renewal.md).

"""
$(TYPEDSIGNATURES)

Density of `dist` at `x`, with the value at the origin forced to zero. For
a Gamma with shape `> 1` the density at zero is genuinely zero; forcing it
avoids the `0·log 0 = NaN` shape-parameter derivative that Mooncake would
otherwise produce when differentiating `pdf(Gamma, 0)`.
"""
@inline function _safe_pdf(dist, x)
    z = zero(pdf(dist, oneunit(x)))
    return x <= zero(x) ? z : pdf(dist, x)
end

"""
$(TYPEDSIGNATURES)

Discretise a continuous delay `dist` to a daily probability mass function
over lags `0, 1, …, nmax`. Each bin mass is the trapezoidal integral of the
density across `[d, d + 1]`, then the vector is renormalised to sum to one
over the truncated support. Differentiates cleanly under Mooncake because
it touches the density only (never the Gamma CDF) and never evaluates the
density at the origin (see [`_safe_pdf`](@ref)).

Returns a `Vector` whose element type follows the delay parameters, so it
is AD-transparent inside a Turing model.
"""
function discretise_delay(dist, nmax::Integer)
    edges = 0:1:(nmax + 1)
    dens = [_safe_pdf(dist, float(e)) for e in edges]
    pmf = [(dens[i] + dens[i + 1]) / 2 for i in 1:(nmax + 1)]
    return pmf ./ sum(pmf)
end

## NaN/Inf-safe positive rate. The renewal recursion can transiently
## overflow on extreme NUTS warmup proposals (large R_t compounding),
## giving a non-finite expected count; `max(x, eps)` would propagate a NaN
## (max(NaN, eps) = NaN) and trip the Poisson/NegBinomial domain check.
@inline function _safe_rate(x)
    return isfinite(x) ? max(x, eps(typeof(x))) : eps(typeof(x))
end

## --- Renewal recursion --------------------------------------------------

"""
$(TYPEDSIGNATURES)

Daily latent infections from the renewal equation

```math
I_t = R_t \\sum_{s \\ge 1} I_{t-s}\\, g_s,
```

with generation-interval PMF `g` (indexed from lag 1) and per-day
reproduction numbers `Rt` (length `n`). The process is seeded by `I0`
infections placed on day 1, representing the single zoonotic introduction;
days before the seed contribute nothing. Returns the length-`n` infection
trajectory. Type-stable and AD-transparent: the output element type is
promoted from `Rt`, `g` and `I0`.
"""
function renewal_infections(Rt::AbstractVector, g::AbstractVector,
        I0::Real)
    n = length(Rt)
    Tp = promote_type(eltype(Rt), eltype(g), typeof(I0))
    I = zeros(Tp, n)
    @inbounds I[1] = Tp(I0)
    @inbounds for t in 2:n
        force = zero(Tp)
        kmax = min(t - 1, length(g))
        for s in 1:kmax
            force += I[t - s] * g[s]
        end
        I[t] = Rt[t] * force
    end
    return I
end

"""
$(TYPEDSIGNATURES)

Convolve a daily infection (or onset) trajectory `x` with a delay PMF
`delay` (indexed from lag 0), returning the expected daily counts of the
delayed event on the same daily grid. Entry `t` sums `x[t-d] · delay[d+1]`
over lags `d` that stay in range. Used to map infections to onsets, onsets
to deaths, onsets to reports and onsets to detection.
"""
function convolve_delay(x::AbstractVector, delay::AbstractVector)
    n = length(x)
    Tp = promote_type(eltype(x), eltype(delay))
    y = zeros(Tp, n)
    @inbounds for t in 1:n
        acc = zero(Tp)
        dmax = min(t - 1, length(delay) - 1)
        for d in 0:dmax
            acc += x[t - d] * delay[d + 1]
        end
        y[t] = acc
    end
    return y
end

## --- Building-block submodels -------------------------------------------

"""
$(TYPEDSIGNATURES)

Day positions of the weekly log-R knots over an `n`-day grid. Knot 1 sits
on day 1 and the last knot on day `n`, with intermediate knots every
`week` days, so a knot is always pinned to each end of the grid. Returns an
integer vector of day indices of length `cld(n - 1, week) + 1`.
"""
function weekly_knot_days(n::Integer; week::Integer = 7)
    n <= 1 && return [1]
    days = collect(1:week:n)
    days[end] == n || push!(days, n)
    return days
end

"""
$(TYPEDSIGNATURES)

Linearly interpolate weekly log-R knot values `log_R_knots` (placed on the
day indices `knot_days`) onto the full daily grid `1:n`, returning the
length-`n` daily log-R vector. Piecewise-linear on the log scale: each day
is a convex combination of its bracketing knots, so the daily series bends
only at the weekly knots and is otherwise straight. Type-stable and
AD-transparent (the output element type follows the knot values).
"""
function interpolate_knots(log_R_knots::AbstractVector,
        knot_days::AbstractVector{<:Integer}, n::Integer)
    Tp = eltype(log_R_knots)
    out = Vector{Tp}(undef, n)
    nb = length(knot_days)
    @inbounds for t in 1:n
        ## Bracketing knots b and b+1 with t in [knot_days[b], knot_days[b+1]].
        b = 1
        while b < nb - 1 && t > knot_days[b + 1]
            b += 1
        end
        d0 = knot_days[b]
        d1 = knot_days[b + 1]
        frac = d1 == d0 ? zero(Tp) : Tp(t - d0) / Tp(d1 - d0)
        out[t] = log_R_knots[b] + frac * (log_R_knots[b + 1] - log_R_knots[b])
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Weekly piecewise-linear log-scale reproduction number over `n` days. Knots
on log R sit at weekly spacing (see [`weekly_knot_days`](@ref)) and follow a
weekly Gaussian random walk, `log_R[1] ~ Normal(log(R0_mean), 1)` then
`log_R[b] ~ Normal(log_R[b-1], σ_rw)`, with `σ_rw` a tight half-Normal
(matching the hantavirus analysis). Daily log-R is the linear interpolation
between knots (see [`interpolate_knots`](@ref)); `Rt = exp.(log_Rt)`.

This regularises toward roughly constant transmission and cuts the latent
R_t dimension by about `week`-fold versus a daily walk, so a handful of
aggregate counts are not pinning a free value every day. Returns
`(; Rt, log_R, knot_days, sigma_rw)`.
"""
@model function rt_walk_model(n::Integer;
        week::Integer = 7,
        log_r0_prior = Normal(log(2.0), 1.0),
        sigma_prior  = truncated(Normal(0, 0.5); lower = 0))
    knot_days = weekly_knot_days(n; week)
    nb = length(knot_days)
    sigma_rw ~ sigma_prior
    log_R = Vector{Real}(undef, nb)
    log_R[1] ~ log_r0_prior
    for b in 2:nb
        log_R[b] ~ Normal(log_R[b - 1], sigma_rw)
    end
    log_Rt = interpolate_knots(log_R, knot_days, n)
    Rt = exp.(log_Rt)
    return (; Rt, log_R, knot_days, sigma_rw)
end

"""
$(TYPEDSIGNATURES)

Generation-interval PMF over lags `0, 1, …, nmax`, discretised from a
Gamma with mean `gi_mean` and SD `gi_sd` (fixed delays by default, since a
handful of aggregate counts cannot identify the generation interval). Lag 0
is dropped and the remainder renormalised, so an infectee is always
infected strictly after its infector. Returns `(; g)`.
"""
@model function generation_interval_model(nmax::Integer;
        gi_mean::Real = 12.0, gi_sd::Real = 6.0)
    shape = (gi_mean / gi_sd)^2
    scale = gi_sd^2 / gi_mean
    raw = discretise_delay(Gamma(shape, scale), nmax)
    g = raw[2:end] ./ sum(raw[2:end])   # drop lag 0, renormalise
    return (; g)
end

"""
$(TYPEDSIGNATURES)

Onset-to-death delay as a daily PMF over lags `0…nmax`, discretised from a
Gamma with truncated-Normal priors on shape and scale anchored on the
Isiro 2012 reanalysis (matching the exponential-growth model). Returns
`(; pmf, dist, alpha, theta)`.
"""
@model function delay_pmf_model(nmax::Integer;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    dist = Gamma(α, θ)
    return (; pmf = discretise_delay(dist, nmax), dist, alpha = α, theta = θ)
end

## --- Composer: discrete renewal joint -----------------------------------

"""
$(TYPEDSIGNATURES)

Prototype joint model for the discrete-time renewal architecture. Runs the
renewal process on a daily grid of length `n` (day `n` is the data cut-off
`T`), maps latent infections through incubation and the onset-to-death,
onset-to-report and onset-to-detection delays, and ties the four observed
streams plus the genetic TMRCA bound to the latent trajectory. Built to
prove the architecture compiles, draws from the prior and differentiates
under Mooncake; the likelihoods are kept deliberately simple.

Arguments mirror the data the package loads. Any count may be passed as
`missing` to drop that stream (so the composer doubles as a
prior-predictive generator). `tmrca_days` is the soft lower bound on the
outbreak age; pass `missing` to drop it.
"""
@model function renewal_joint(
        n::Integer,
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer},
        exports_deaths::Union{Missing, Integer};
        tmrca_days::Union{Missing, Real}     = missing,
        tmrca_days_sd::Real                  = 20.0,
        source_population::Real              = ITURI_POPULATION,
        daily_travel_mean::Real             = ITURI_DAILY_TRAVEL,
        daily_travel_sd::Real               = ITURI_DAILY_TRAVEL_SD,
        delay_nmax::Integer                  = 60,
        gi_nmax::Integer                     = 40,
        incubation_nmax::Integer             = 30,
        rt              = rt_walk_model,
        gi              = generation_interval_model,
        onset_to_death  = delay_pmf_model)

    ## Latent process ----------------------------------------------------
    rt_state ~ to_submodel(rt(n), false)
    gi_state ~ to_submodel(gi(gi_nmax), false)
    Rt = rt_state.Rt
    g  = gi_state.g

    I0 ~ truncated(Normal(1.0, 1.0); lower = 0)
    infections = renewal_infections(Rt, g, I0)

    ## Incubation: infection -> onset (fixed Gamma, mean 7 d).
    incub = discretise_delay(Gamma((7 / 4)^2, 4^2 / 7), incubation_nmax)
    onsets = convolve_delay(infections, incub)

    cum_infections = cumsum(infections)
    C_T := cum_infections[n]

    ## Shared nuisance parameters ---------------------------------------
    CFR ~ Beta(6.6, 13.4)
    inv_sqrt_k ~ truncated(Normal(0.6, 0.2); lower = 0)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))

    μ_logit  ~ Normal(logit(0.25), 1.0)
    τ_logit  ~ truncated(Normal(0, 0.5); lower = 1e-4)
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)

    daily_travellers ~ truncated(Normal(daily_travel_mean, daily_travel_sd);
                                 lower = 0)
    q = daily_travellers / source_population

    ## Stream 2: DRC suspected deaths -----------------------------------
    od_state ~ to_submodel(onset_to_death(delay_nmax), false)
    deaths_daily = CFR .* convolve_delay(onsets, od_state.pmf)
    expected_deaths_T := _safe_rate(sum(deaths_daily))
    if !ismissing(total_deaths)
        total_deaths ~ safe_nbinomial(k, expected_deaths_T)
    end

    ## Stream 3: DRC reported suspected cases ---------------------------
    ## Onsets ascertained at p_drc; cumulative to the cut-off.
    report_pmf = discretise_delay(Gamma((5 / 3)^2, 3^2 / 5), incubation_nmax)
    reports_daily = p_drc .* convolve_delay(onsets, report_pmf)
    expected_reports_T := _safe_rate(sum(reports_daily))
    if !ismissing(reported_cases)
        reported_cases ~ safe_nbinomial(k, expected_reports_T)
    end

    ## Stream 1: exported cases detected in Uganda ----------------------
    ## A case's onsets cross the border and are ascertained at rate
    ## p_uganda · q; `export_onsets` is the onset incidence among detected
    ## exports, timed at onset. The detection event is then this convolved
    ## with the onset-to-detection delay (timed at detection).
    export_onsets = p_uganda .* q .* onsets
    detect_pmf = discretise_delay(Gamma((10 / 4)^2, 4^2 / 10), incubation_nmax)
    detect_daily = convolve_delay(export_onsets, detect_pmf)
    expected_exports_T := _safe_rate(sum(detect_daily))
    if !ismissing(exported_cases)
        exported_cases ~ Poisson(expected_exports_T)
    end

    ## Stream 4: deaths among exported cases ----------------------------
    ## Deaths among detected exports, timed from the same export *onsets* by
    ## the onset-to-death delay (both detection and death are measured from
    ## onset, so death is convolved against `export_onsets`, not against the
    ## detection-timed series).
    export_deaths_daily = CFR .* convolve_delay(export_onsets, od_state.pmf)
    expected_exports_deaths_T := _safe_rate(sum(export_deaths_daily))
    if !ismissing(exports_deaths)
        exports_deaths ~ Poisson(expected_exports_deaths_T)
    end

    ## Genetic TMRCA soft lower bound on the outbreak age ----------------
    ## The TMRCA is a right-censored noisy reading of the seeding time:
    ## adding sequences only pushes it older, so we learn the outbreak is
    ## at least `tmrca_days` old. Here the outbreak age is `n` days (the
    ## grid length), so the bound contributes Φ((n − g)/σ).
    if !ismissing(tmrca_days)
        @addlogprob! normlogcdf((n - tmrca_days) / tmrca_days_sd)
    end

    return (; C_T, Rt, expected_deaths_T, expected_reports_T,
              expected_exports_T, expected_exports_deaths_T)
end
