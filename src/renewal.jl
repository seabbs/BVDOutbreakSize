## Discrete-time renewal architecture (issue #81).
##
## A candidate redesign that replaces the deterministic continuous-time
## cumulative-incidence trajectory C(s) = exp(r·s) with a daily renewal
## process driven by a time-varying reproduction number R_t. Latent daily
## infections are convolved through realistic delays to build expected
## daily then cumulative counts for each observed stream.
##
## Structured to mirror the main model: every parameter family lives in its
## own building-block submodel, the observation streams have per-stream
## submodels that take state and parameters by injection, and the composer
## just wires them together.
##
## All package dependencies are brought in once on the module page
## (src/BVDOutbreakSize.jl). Names used below (`@model`, `to_submodel`,
## `@addlogprob!`, `Normal`, `LogNormal`, `Gamma`, `truncated`, `Poisson`,
## `Beta`, `logit`, `logistic`, `normlogcdf`, `safe_nbinomial`, the
## `ITURI_*` constants, and the `CensoredDistributions` API) are already in
## scope in the enclosing module, into which this file is `include`d.

## --- Delay discretisation -----------------------------------------------
##
## Two routes are available:
##
##   (a) the exact double-interval-censored PMF for a *LogNormal* primary
##       via `CensoredDistributions.double_interval_censored`. LogNormal's
##       CDF differentiates cleanly under Mooncake, so this is the
##       Mooncake-safe primary path for the delays where the data are too
##       thin to pin a shape and scale (generation interval, incubation,
##       onset-to-report, onset-to-detection); see `double_censored_pmf`.
##
##   (b) a manual trapezoidal density-integration with f(0)=0 for *Gamma*
##       primaries (`discretise_delay`). The Gamma analytical primary-
##       censored CDF in CensoredDistributions routes through
##       `HypergeometricFunctions.pFqweniger`, which contains a `try/catch`
##       that Mooncake rejects; `force_numeric` hits QuadGK's `cachedrule`,
##       also undifferentiable. `interval_censored` alone differentiates
##       but returns a NaN Gamma-CDF shape-parameter gradient. The manual
##       trapezoid is an approximation (mean is biased low by roughly half
##       a day at the BVD prior centre, total-variation ≲ 0.06 vs the
##       exact double-censored PMF) but it differentiates cleanly and is
##       used here only for the onset-to-death prior, which is anchored on
##       a Gamma reanalysis. See docs/proposals/discrete-renewal.md for the
##       bias characterisation and recommendation.

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
over lags `0, 1, …, nmax` by trapezoidal density integration. Touches the
density only (never the CDF) and forces the density at the origin to zero,
so it differentiates cleanly under Mooncake for both Gamma shape and scale.

This is an *approximation* to the exact double-interval-censored PMF: at
the BVD onset-to-death prior centre it under-estimates the mean by roughly
half a day with total variation ≲ 0.05 vs the exact PMF. Use it for Gamma
delays where the analytical double-censored CDF would be undifferentiable
under Mooncake. For LogNormal delays use [`double_censored_pmf`](@ref)
instead, which evaluates the exact double-interval-censored PMF and is
also Mooncake-safe.

Returns a `Vector` whose element type follows the delay parameters, so it
is AD-transparent inside a Turing model.
"""
function discretise_delay(dist, nmax::Integer)
    edges = 0:1:(nmax + 1)
    dens = [_safe_pdf(dist, float(e)) for e in edges]
    pmf = [(dens[i] + dens[i + 1]) / 2 for i in 1:(nmax + 1)]
    return pmf ./ sum(pmf)
end

"""
$(TYPEDSIGNATURES)

Exact daily probability mass function for `dist` over lags `0, 1, …, nmax`
under standard double-interval censoring (uniform primary event over a
one-day exposure window, then unit interval censoring of the secondary
event). Uses `CensoredDistributions.double_interval_censored` and is the
no-approximation analogue of [`discretise_delay`](@ref). Differentiates
cleanly under Mooncake for `LogNormal` primaries (the LogNormal CDF is
AD-safe); for `Gamma` primaries it would route through
`HypergeometricFunctions.pFqweniger` (try/catch) and fail under Mooncake,
so call this only with LogNormal-parameterised delays.
"""
function double_censored_pmf(dist, nmax::Integer)
    dic = double_interval_censored(dist; interval = 1.0,
        upper = float(nmax))
    raw = [pdf(dic, float(d)) for d in 0:nmax]
    s = sum(raw)
    ## Extreme NUTS warmup proposals can push the LogNormal parameters into
    ## a range where the quadrature gives NaN/zero. Fall back to a uniform
    ## PMF so the downstream convolution stays finite (the likelihood will
    ## reject the proposal via its low logprob, but the gradient evaluates).
    if !isfinite(s) || s <= zero(s)
        z = zero(pdf(dist, oneunit(float(nmax))))
        return fill(one(z) / (nmax + 1), nmax + 1)
    end
    return raw ./ s
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
weekly Gaussian random walk in **non-centred cumulative-sum** form:
standard-normal innovations `z[b] ~ Normal(0, 1)` are scaled by `sigma_rw`
and accumulated, `log_R = log_R[1] .+ sigma_rw .* cumsum([0, z...])`. This
avoids the funnel geometry of the centred recursion and lets NUTS take
larger steps, matching the non-centred ascertainment block in the main
model. Daily log-R is the linear interpolation between knots
(see [`interpolate_knots`](@ref)); `Rt = exp.(log_Rt)`.

Cuts the latent R_t dimension by about `week`-fold versus a daily walk and
regularises toward roughly constant transmission, so a handful of aggregate
counts are not pinning a free value every day. Returns
`(; Rt, log_R, knot_days, sigma_rw, log_R0)`.
"""
@model function rt_walk_model(n::Integer;
        week::Integer = 7,
        log_r0_prior = Normal(log(2.0), 1.0),
        sigma_prior  = truncated(Normal(0, 0.5); lower = 0))
    knot_days = weekly_knot_days(n; week)
    nb = length(knot_days)
    log_R0   ~ log_r0_prior
    sigma_rw ~ sigma_prior
    ## Non-centred cumulative-sum form (mirrors the main model's
    ## non-centred ascertainment): innovations z_b are standard normal,
    ## the centred walk log_R[b] = log_R[b-1] + sigma_rw * z_b is recovered
    ## as the cumulative sum, but NUTS sees the iid z scale rather than
    ## the funnel of (sigma_rw, log_R).
    z ~ filldist(Normal(0, 1), nb - 1)
    steps  = sigma_rw .* z
    log_R  = log_R0 .+ vcat(zero(log_R0), cumsum(steps))
    log_Rt = interpolate_knots(log_R, knot_days, n)
    Rt = exp.(log_Rt)
    return (; Rt, log_R, knot_days, sigma_rw, log_R0)
end

## --- Delay submodels (priors only; no hardcoded distributions) ----------

"""
$(TYPEDSIGNATURES)

Generic LogNormal delay submodel parameterised by mean and SD. The mean
and SD are sampled from weakly-informative priors and converted to the
LogNormal log-mean/log-SD by moment matching. Uses the
**exact double-interval-censored PMF** via [`double_censored_pmf`](@ref):
under Mooncake the LogNormal CDF differentiates cleanly, so no manual
trapezoidal discretisation is needed. Returns
`(; pmf, dist, delay_mean, delay_sd)`. This is the principal path for
delays where the data cannot pin a shape and scale; the priors carry the
delay rather than fixing it.
"""
@model function delay_lognormal_meansd_model(nmax::Integer;
        mean_prior, sd_prior)
    delay_mean ~ mean_prior
    delay_sd   ~ sd_prior
    ## LogNormal moment matching: var = mean^2 * (exp(σ^2) - 1). Clamp the
    ## moment-match inputs through `_safe_rate` (returns eps for any
    ## non-finite input) so a NaN-prone NUTS warmup proposal cannot push
    ## σ = sqrt(log1p(·)) into NaN territory and trip the LogNormal domain
    ## check; `max(NaN, eps)` would otherwise propagate the NaN.
    m  = _safe_rate(delay_mean)
    s  = _safe_rate(delay_sd)
    σ2 = log1p((s / m)^2)
    μ  = log(m) - σ2 / 2
    dist = LogNormal(μ, sqrt(σ2))
    return (; pmf = double_censored_pmf(dist, nmax), dist,
              delay_mean, delay_sd)
end

"""
$(TYPEDSIGNATURES)

Gamma delay submodel parameterised by mean and SD (kept for back-compat
with earlier renewal tests). Uses the manual trapezoidal
[`discretise_delay`](@ref) because the Gamma analytical primary-censored
CDF is undifferentiable under Mooncake; this is an approximation (see
[`discretise_delay`](@ref) docstring). Prefer
[`delay_lognormal_meansd_model`](@ref) for new code.
"""
@model function delay_meansd_model(nmax::Integer;
        mean_prior, sd_prior)
    delay_mean ~ mean_prior
    delay_sd   ~ sd_prior
    shape = (delay_mean / delay_sd)^2
    scale = delay_sd^2 / delay_mean
    dist = Gamma(shape, scale)
    return (; pmf = discretise_delay(dist, nmax), dist,
              delay_mean, delay_sd)
end

"""
$(TYPEDSIGNATURES)

Generation-interval submodel. The mean and SD of the generation interval
are sampled from weakly-informative priors (mean centred on 12 d, SD on
6 d, both truncated at one day), so the generation time is estimated
rather than fixed (the data are thin, so the priors carry it). Uses a
LogNormal primary with the exact double-interval-censored PMF. The lag-0
bin is dropped and the remainder renormalised, so an infectee is always
infected strictly after its infector. Returns `(; g, gi_mean, gi_sd)`.
"""
@model function generation_interval_model(nmax::Integer;
        mean_prior = truncated(Normal(12.0, 3.0); lower = 1),
        sd_prior   = truncated(Normal(6.0, 2.0); lower = 1))
    d ~ to_submodel(
        delay_lognormal_meansd_model(nmax; mean_prior, sd_prior))
    g = d.pmf[2:end] ./ sum(d.pmf[2:end])   # drop lag 0, renormalise
    return (; g, gi_mean = d.delay_mean, gi_sd = d.delay_sd)
end

"""
$(TYPEDSIGNATURES)

Onset-to-death delay as a daily PMF over lags `0…nmax`, discretised from a
Gamma with truncated-Normal priors on shape and scale anchored on the
Isiro 2012 reanalysis (matching the exponential-growth model). Uses the
manual [`discretise_delay`](@ref) because Mooncake cannot differentiate
the Gamma analytical double-censored CDF (see that docstring for the
bias). Returns `(; pmf, dist, alpha, theta)`.
"""
@model function delay_pmf_model(nmax::Integer;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    dist = Gamma(α, θ)
    return (; pmf = discretise_delay(dist, nmax), dist, alpha = α, theta = θ)
end

## --- Other building-block submodels (priors only) -----------------------

"""
$(TYPEDSIGNATURES)

Seeding submodel: zoonotic seed `I0` placed on day 1 of the grid. Defaults
to a truncated Normal centred on a single introduction; the prior is
injectable.
"""
@model function seed_model(;
        i0_prior = truncated(Normal(1.0, 1.0); lower = 0))
    I0 ~ i0_prior
    return (; I0)
end

"""
$(TYPEDSIGNATURES)

Case-fatality ratio submodel (renewal-side mirror of the main model's
`cfr_model`). Default prior matches: `Beta(6.6, 13.4)` (mean ≈ 0.33).
"""
@model function cfr_renewal_model(; cfr_prior = Beta(6.6, 13.4))
    CFR ~ cfr_prior
    return (; CFR)
end

"""
$(TYPEDSIGNATURES)

Surveillance dispersion `k` for the DRC deaths and reported-cases
likelihoods. Sampled on the `1/sqrt(k)` scale (matches the main model's
`surveillance_dispersion_model`); `inv_sqrt_k_prior` is injectable.
"""
@model function surveillance_dispersion_renewal(;
        inv_sqrt_k_prior = truncated(Normal(0.6, 0.2); lower = 0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

"""
$(TYPEDSIGNATURES)

Pooled DRC and Uganda ascertainment, non-centred logit Normal hierarchy
(mirrors `pooled_ascertainment_model` in the main model). All priors are
injectable.
"""
@model function pooled_ascertainment_renewal(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0, 0.5); lower = 1e-4))
    μ_logit ~ mu_prior
    τ_logit ~ tau_prior
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

"""
$(TYPEDSIGNATURES)

Daily traveller volume between Ituri and Uganda (mirrors
`traveller_volume_model` in the main model). Prior centre and SD
injectable.
"""
@model function traveller_volume_renewal(;
        mean::Real = ITURI_DAILY_TRAVEL,
        sd::Real   = ITURI_DAILY_TRAVEL_SD)
    daily_travellers ~ truncated(Normal(mean, sd); lower = 0)
    return (; daily_travellers)
end

## --- Latent-state submodels ---------------------------------------------

"""
$(TYPEDSIGNATURES)

Renewal-process latent submodel. Samples the reproduction number
trajectory, the generation interval and the seed via injected submodels,
then runs the discrete renewal recursion. Returns the daily infections
together with the latent components.
"""
@model function renewal_process_model(n::Integer;
        rt        = rt_walk_model,
        gi        = generation_interval_model,
        seed      = seed_model,
        gi_nmax::Integer = 40)
    rt_state ~ to_submodel(rt(n), false)
    gi_state ~ to_submodel(gi(gi_nmax), false)
    seed_state ~ to_submodel(seed(), false)
    Rt = rt_state.Rt
    g  = gi_state.g
    infections = renewal_infections(Rt, g, seed_state.I0)
    return (; infections, Rt, g, I0 = seed_state.I0)
end

"""
$(TYPEDSIGNATURES)

Onset-incidence submodel: convolve renewal infections with the sampled
incubation PMF to get the daily symptom-onset incidence. Computed *once*
per draw and reused by every downstream observation stream (deaths,
reports, detection), so the staging is explicit: infections → onsets →
each observed event. The incubation submodel is injected.
"""
@model function onset_incidence_model(infections::AbstractVector;
        incubation = (nmax) -> delay_lognormal_meansd_model(nmax;
            mean_prior = truncated(Normal(7.0, 2.0); lower = 1),
            sd_prior   = truncated(Normal(4.0, 1.5); lower = 1)),
        incubation_nmax::Integer = 30)
    inc_state ~ to_submodel(incubation(incubation_nmax))
    onsets = convolve_delay(infections, inc_state.pmf)
    return (; onsets, incubation_pmf = inc_state.pmf,
              incubation_mean = inc_state.delay_mean,
              incubation_sd   = inc_state.delay_sd)
end

## --- Observation submodels (per stream) ---------------------------------

"""
$(TYPEDSIGNATURES)

DRC suspected-deaths observation submodel. Takes the daily onset incidence
and the shared dispersion `k` as inputs and samples the onset-to-death
delay and the CFR via injected submodels. The observation distribution is
injected too (`obs_dist`, defaulting to [`safe_nbinomial`](@ref)). Returns
`(; expected_deaths_T, deaths_daily)`.
"""
@model function deaths_obs_model(
        total_deaths::Union{Missing, Integer},
        onsets::AbstractVector, k::Real;
        cfr             = cfr_renewal_model,
        onset_to_death  = delay_pmf_model,
        delay_nmax::Integer = 60,
        obs_dist        = safe_nbinomial)
    cfr_state ~ to_submodel(cfr(), false)
    od_state  ~ to_submodel(onset_to_death(delay_nmax), false)
    deaths_daily = cfr_state.CFR .* convolve_delay(onsets, od_state.pmf)
    expected_deaths_T = _safe_rate(sum(deaths_daily))
    if !ismissing(total_deaths)
        total_deaths ~ obs_dist(k, _safe_rate(expected_deaths_T))
    end
    return (; expected_deaths_T, deaths_daily,
              od_pmf = od_state.pmf, CFR = cfr_state.CFR)
end

"""
$(TYPEDSIGNATURES)

DRC reported-cases observation submodel. The onset-to-report delay is
injected (default LogNormal mean/SD with weakly-informative priors), the
ascertainment fraction `p_drc` and the shared dispersion `k` come in as
arguments, and the observation distribution `obs_dist` is injectable.
"""
@model function reports_obs_model(
        reported_cases::Union{Missing, Integer},
        onsets::AbstractVector, k::Real, p_drc::Real;
        onset_to_report = (nmax) -> delay_lognormal_meansd_model(nmax;
            mean_prior = truncated(Normal(5.0, 2.0); lower = 1),
            sd_prior   = truncated(Normal(3.0, 1.5); lower = 1)),
        report_nmax::Integer = 30,
        obs_dist        = safe_nbinomial)
    report_state ~ to_submodel(onset_to_report(report_nmax))
    reports_daily = p_drc .* convolve_delay(onsets, report_state.pmf)
    expected_reports_T = _safe_rate(sum(reports_daily))
    if !ismissing(reported_cases)
        reported_cases ~ obs_dist(k, _safe_rate(expected_reports_T))
    end
    return (; expected_reports_T, reports_daily)
end

"""
$(TYPEDSIGNATURES)

Uganda exports observation submodel. Builds the export onset incidence
(`p_uganda · q · onsets`) and convolves with the injected onset-to-
detection delay. Returns the detection-timed export series alongside the
expected total. Uganda's two streams are small, so the observation
distribution defaults to Poisson; pass `obs_dist` to inject another.
"""
@model function exports_obs_model(
        exported_cases::Union{Missing, Integer},
        onsets::AbstractVector, p_uganda::Real, q::Real;
        onset_to_detection = (nmax) -> delay_lognormal_meansd_model(nmax;
            mean_prior = truncated(Normal(10.0, 3.0); lower = 1),
            sd_prior   = truncated(Normal(4.0, 1.5); lower = 1)),
        detection_nmax::Integer = 30,
        obs_dist        = (μ -> Poisson(μ)))
    export_onsets = p_uganda .* q .* onsets
    detect_state ~ to_submodel(onset_to_detection(detection_nmax))
    detect_daily = convolve_delay(export_onsets, detect_state.pmf)
    expected_exports_T = _safe_rate(sum(detect_daily))
    if !ismissing(exported_cases)
        ## `obs_dist` may construct an unsafe distribution (e.g. Poisson)
        ## from a borderline rate; `_safe_rate` already guarantees a finite
        ## positive value, but guard once more in case a custom obs_dist
        ## injects a NaN-prone parameterisation.
        exported_cases ~ obs_dist(_safe_rate(expected_exports_T))
    end
    return (; expected_exports_T, detect_daily, export_onsets)
end

"""
$(TYPEDSIGNATURES)

Deaths among detected exports. Convolves the export onsets (passed in
explicitly, not the detection-timed series) with the onset-to-death PMF,
so detection and death are both timed from onset. The observation
distribution is injectable (Poisson by default).
"""
@model function exports_deaths_obs_model(
        exports_deaths::Union{Missing, Integer},
        export_onsets::AbstractVector, CFR::Real,
        od_pmf::AbstractVector;
        obs_dist = (μ -> Poisson(μ)))
    series = CFR .* convolve_delay(export_onsets, od_pmf)
    expected_exports_deaths_T = _safe_rate(sum(series))
    if !ismissing(exports_deaths)
        exports_deaths ~ obs_dist(_safe_rate(expected_exports_deaths_T))
    end
    return (; expected_exports_deaths_T, series)
end

"""
$(TYPEDSIGNATURES)

Genetic TMRCA soft lower bound on the outbreak age. The TMRCA is a
right-censored noisy reading of the outbreak age (adding sequences only
pushes it older), so we contribute the log of the upper-tail probability
`Φ((n − g)/σ)`. Passing `tmrca_days = missing` makes the submodel a no-op.
"""
@model function tmrca_bound_model(n::Integer;
        tmrca_days::Union{Missing, Real} = missing,
        tmrca_days_sd::Real              = 20.0)
    if !ismissing(tmrca_days)
        @addlogprob! normlogcdf((n - tmrca_days) / tmrca_days_sd)
    end
    return (;)
end

## --- Composer: discrete renewal joint -----------------------------------

"""
$(TYPEDSIGNATURES)

Joint model for the discrete-time renewal architecture. Runs the renewal
process on a daily grid of length `n` (day `n` is the data cut-off `T`),
stages it explicitly via the onset-incidence submodel, then ties each
observed stream plus the genetic TMRCA bound to the staged onsets through
a per-stream observation submodel.

Composition mirrors the main model: every parameter family and every
likelihood is a submodel injected via keyword arguments, so priors,
delays, and observation distributions can all be swapped without editing
the composer body. Any count may be passed as `missing` to drop that
stream (so the composer doubles as a prior-predictive generator).
`tmrca_days` is the soft lower bound on the outbreak age; pass `missing`
to drop it.

Submodel arguments
------------------

- `renewal`         — latent renewal process (Rt, GI, seed).
- `onset_incidence` — infection→onset staging.
- `cfr`, `dispersion`, `ascertainment`, `traveller` — shared nuisance
  blocks (priors injectable).
- `deaths_obs`, `reports_obs`, `exports_obs`, `exports_deaths_obs` —
  per-stream observation submodels; each accepts its own delay submodel
  and observation distribution.
- `tmrca`           — genetic TMRCA soft bound (no-op when `tmrca_days`
  is `missing`).
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
        renewal            = renewal_process_model,
        onset_incidence    = onset_incidence_model,
        cfr                = cfr_renewal_model,
        dispersion         = surveillance_dispersion_renewal,
        ascertainment      = pooled_ascertainment_renewal,
        traveller          = traveller_volume_renewal,
        deaths_obs         = deaths_obs_model,
        reports_obs        = reports_obs_model,
        exports_obs        = exports_obs_model,
        exports_deaths_obs = exports_deaths_obs_model,
        tmrca              = tmrca_bound_model)

    ## Latent process and onset staging ---------------------------------
    renewal_state ~ to_submodel(renewal(n), false)
    onset_state   ~ to_submodel(
        onset_incidence(renewal_state.infections), false)
    infections = renewal_state.infections
    onsets     = onset_state.onsets

    cum_infections = cumsum(infections)
    C_T := cum_infections[n]

    ## Shared nuisance parameters ---------------------------------------
    disp_state  ~ to_submodel(dispersion(), false)
    asc_state   ~ to_submodel(ascertainment(), false)
    travel_state ~ to_submodel(traveller(), false)
    k        = disp_state.k
    p_drc    = asc_state.p_drc
    p_uganda = asc_state.p_uganda
    q = travel_state.daily_travellers / source_population

    ## Observation submodels -------------------------------------------
    deaths_state ~ to_submodel(
        deaths_obs(total_deaths, onsets, k), false)
    reports_state ~ to_submodel(
        reports_obs(reported_cases, onsets, k, p_drc))
    exports_state ~ to_submodel(
        exports_obs(exported_cases, onsets, p_uganda, q))
    exports_deaths_state ~ to_submodel(
        exports_deaths_obs(exports_deaths,
            exports_state.export_onsets,
            deaths_state.CFR,
            deaths_state.od_pmf), false)

    ## Genetic TMRCA soft bound -----------------------------------------
    tmrca_state ~ to_submodel(
        tmrca(n; tmrca_days, tmrca_days_sd), false)

    expected_deaths_T         := deaths_state.expected_deaths_T
    expected_reports_T        := reports_state.expected_reports_T
    expected_exports_T        := exports_state.expected_exports_T
    expected_exports_deaths_T := exports_deaths_state.expected_exports_deaths_T

    return (; C_T, Rt = renewal_state.Rt, infections, onsets,
              expected_deaths_T, expected_reports_T,
              expected_exports_T, expected_exports_deaths_T)
end
