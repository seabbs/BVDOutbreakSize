## State-space stochastic infection process architecture (issue #48).
##
## A candidate redesign in which the latent daily infection counts follow
## the EXACT discrete branching-process step: I_t | I_{<t}, R_t ~
## NegativeBinomial(mean = λ_t, dispersion = φ) with λ_t the renewal mean
## sum_s I_{t-s} g_s scaled by R_t. The Gaussian relaxation in
## `src/stochastic_growth.jl` (LNA route, issue #48 sibling) is replaced
## here by the exact discrete latent and is intended to be sampled by
## particle-Gibbs over the latent path. Continuous nuisance parameters
## stay differentiable so NUTS + Mooncake handles them under Gibbs.
##
## Additive: this file is included by `src/BVDOutbreakSize.jl` but adds
## no new dependencies and leaves the production model unchanged. Names
## used below (`@model`, `to_submodel`, `Normal`, `Gamma`, `truncated`,
## `Poisson`, `NegativeBinomial`, `Beta`, `logit`, `logistic`,
## `normlogcdf`, `safe_nbinomial`, the `ITURI_*` constants) are already
## imported by the enclosing module (no inline `using`/`import` here).

using Turing.DynamicPPL: @addlogprob!

## --- AD-safe delay discretisation ---------------------------------------
##
## The package's preferred route, CensoredDistributions.jl, is not
## Mooncake-differentiable (the analytical primary-censored CDF routes
## through `HypergeometricFunctions.pFqweniger`, which contains a
## try/catch; `force_numeric` hits `QuadGK.cachedrule`, also rejected).
## We hand-discretise by trapezoidal integration of the density over each
## daily bin, with the density at the origin forced to zero (exact for
## Gamma shape > 1, where 0·log 0 would otherwise produce a NaN
## shape-parameter gradient). Same trick as `ExportDeathDelay` and the
## renewal architecture. The cost is a small bias from omitting the second
## (forward) censoring — a few percent on the central mass — documented
## in docs/src/proposals/state-space-particle.md.

"""
$(TYPEDSIGNATURES)

Density of `dist` at `x`, with the value at the origin forced to zero.
For a Gamma with shape `> 1` the density at zero is genuinely zero;
forcing it avoids the `0·log 0 = NaN` shape-parameter derivative that
Mooncake would otherwise produce when differentiating `pdf(Gamma, 0)`.
"""
@inline function _safe_pdf_ss(dist, x)
    z = zero(pdf(dist, oneunit(x)))
    return x <= zero(x) ? z : pdf(dist, x)
end

"""
$(TYPEDSIGNATURES)

Discretise a continuous delay `dist` to a daily PMF over lags
`0, 1, …, nmax`. Each bin mass is the trapezoidal integral of the
density across `[d, d + 1]`, then the vector is renormalised to sum to
one over the truncated support. Differentiates cleanly under Mooncake
because it touches the density only (never the Gamma CDF) and never
evaluates the density at the origin (see [`_safe_pdf_ss`](@ref)).
"""
function discretise_delay_ss(dist, nmax::Integer)
    edges = 0:1:(nmax + 1)
    dens = [_safe_pdf_ss(dist, float(e)) for e in edges]
    pmf = [(dens[i] + dens[i + 1]) / 2 for i in 1:(nmax + 1)]
    return pmf ./ sum(pmf)
end

"""
$(TYPEDSIGNATURES)

Convolve a daily trajectory `x` with a delay PMF `delay` (indexed from
lag 0), returning expected daily counts of the delayed event on the
same daily grid. Entry `t` sums `x[t-d] · delay[d+1]` over lags `d`
that stay in range. Used to map infections to onsets, and onsets to
deaths / reports / detections.
"""
function convolve_delay_ss(x::AbstractVector, delay::AbstractVector)
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

## --- Renewal-mean kernel -----------------------------------------------

"""
$(TYPEDSIGNATURES)

Renewal mean for day `t` from a partial infection trajectory `I` (with
`I[s]` defined for `s = 1:(t-1)`), generation-interval PMF `g` (indexed
from lag 1) and per-day reproduction numbers `Rt`:

```math
λ_t = R_t \\sum_{s = 1}^{\\min(t-1, |g|)} I_{t-s}\\, g_s.
```

Returned as a non-negative scalar; the AD-safe `_safe_rate` clamp keeps
the downstream NegativeBinomial parameterisation in domain on extreme
warmup proposals.
"""
function nb_branching_step(I::AbstractVector, g::AbstractVector,
        Rt::AbstractVector, t::Integer)
    Tp = promote_type(eltype(I), eltype(g), eltype(Rt))
    force = zero(Tp)
    kmax = min(t - 1, length(g))
    @inbounds for s in 1:kmax
        force += I[t - s] * g[s]
    end
    return _safe_rate_ss(Rt[t] * force)
end

@inline function _safe_rate_ss(x)
    return isfinite(x) ? max(x, eps(typeof(x))) : eps(typeof(x))
end

## --- Weekly piecewise-linear R_t ---------------------------------------

"""
$(TYPEDSIGNATURES)

Day positions of the weekly log-R knots over an `n`-day grid. Knot 1
sits on day 1 and the last knot on day `n`, with intermediate knots
every `week` days, so a knot is pinned to each end of the grid.
"""
function weekly_knot_days_ss(n::Integer; week::Integer = 7)
    n <= 1 && return [1]
    days = collect(1:week:n)
    days[end] == n || push!(days, n)
    return days
end

"""
$(TYPEDSIGNATURES)

Linearly interpolate weekly log-R knot values `log_R_knots` (on day
indices `knot_days`) onto the full daily grid `1:n`. Piecewise-linear
on the log scale: each day is a convex combination of its bracketing
knots, so the daily series bends only at the weekly knots.
"""
function interpolate_knots_ss(log_R_knots::AbstractVector,
        knot_days::AbstractVector{<:Integer}, n::Integer)
    Tp = eltype(log_R_knots)
    out = Vector{Tp}(undef, n)
    nb = length(knot_days)
    @inbounds for t in 1:n
        b = 1
        while b < nb - 1 && t > knot_days[b + 1]
            b += 1
        end
        d0 = knot_days[b]
        d1 = knot_days[b + 1]
        frac = d1 == d0 ? zero(Tp) :
            Tp(t - d0) / Tp(d1 - d0)
        out[t] = log_R_knots[b] +
            frac * (log_R_knots[b + 1] - log_R_knots[b])
    end
    return out
end

## --- Building-block submodels ------------------------------------------
##
## Mirrors the baseline/renewal pattern: every prior is a submodel that
## owns its own parameters and returns a NamedTuple. The composer pulls
## these in by `to_submodel` and threads their outputs through the
## observation kernels.

"""
$(TYPEDSIGNATURES)

Weekly piecewise-linear log-scale reproduction number over `n` days.
Non-centred parameterisation: a `log_R₀ ~ log_r0_prior`, then weekly
standard-Normal innovations `z_b ~ Normal(0, 1)` scaled by
`σ_rw ~ sigma_prior`. Daily `log_Rt` is the linear interpolation
between knots and `Rt = exp.(log_Rt)`.
"""
@model function rt_walk_model_ss(n::Integer;
        week::Integer = 7,
        log_r0_prior = Normal(log(2.0), 1.0),
        sigma_prior  = truncated(Normal(0, 0.5); lower = 0))
    knot_days = weekly_knot_days_ss(n; week)
    nb = length(knot_days)
    sigma_rw ~ sigma_prior
    log_R0 ~ log_r0_prior
    z_rw ~ filldist(Normal(0, 1), nb - 1)
    ## Non-centred cumulative walk in a typed vector: log_R[b] =
    ## log_R0 + σ_rw * (z_1 + … + z_{b-1}). The explicit element type
    ## (promoted from the sampled scalars and the innovation vector)
    ## keeps Libtask's IR transformation happy under PG, which rejects
    ## the abstract `Vector{Real}` allocation pattern.
    Tp = promote_type(typeof(log_R0), typeof(sigma_rw), eltype(z_rw))
    log_R = Vector{Tp}(undef, nb)
    log_R[1] = log_R0
    cumz = zero(Tp)
    for b in 2:nb
        cumz += z_rw[b - 1]
        log_R[b] = log_R0 + sigma_rw * cumz
    end
    log_Rt = interpolate_knots_ss(log_R, knot_days, n)
    Rt = exp.(log_Rt)
    return (; Rt, log_R, knot_days, sigma_rw)
end

"""
$(TYPEDSIGNATURES)

Generation-interval PMF over lags `0, 1, …, nmax`, sampled from a
Gamma with truncated-Normal priors on its mean and SD. Lag 0 is
dropped (an infectee is always infected strictly after its infector)
and the remainder renormalised.
"""
@model function generation_interval_model_ss(nmax::Integer;
        mean_prior = truncated(Normal(12.0, 3.0); lower = 1e-3),
        sd_prior   = truncated(Normal(6.0, 2.0); lower = 1e-3))
    gi_mean ~ mean_prior
    gi_sd   ~ sd_prior
    shape = (gi_mean / gi_sd)^2
    scale = gi_sd^2 / gi_mean
    raw = discretise_delay_ss(Gamma(shape, scale), nmax)
    tail = raw[2:end] ./ sum(raw[2:end])
    return (; g = tail, gi_mean, gi_sd)
end

"""
$(TYPEDSIGNATURES)

Incubation delay PMF (infection-to-onset), sampled Gamma with priors
on its mean and SD.
"""
@model function incubation_model_ss(nmax::Integer;
        mean_prior = truncated(Normal(7.0, 2.0); lower = 1e-3),
        sd_prior   = truncated(Normal(4.0, 1.5); lower = 1e-3))
    incub_mean ~ mean_prior
    incub_sd   ~ sd_prior
    shape = (incub_mean / incub_sd)^2
    scale = incub_sd^2 / incub_mean
    pmf = discretise_delay_ss(Gamma(shape, scale), nmax)
    return (; pmf, incub_mean, incub_sd)
end

"""
$(TYPEDSIGNATURES)

Onset-to-death delay PMF, sampled Gamma anchored on the Isiro 2012
reanalysis (matches the baseline `delay_model`).
"""
@model function onset_to_death_model_ss(nmax::Integer;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    dist = Gamma(α, θ)
    return (; pmf = discretise_delay_ss(dist, nmax),
              dist, alpha = α, theta = θ)
end

"""
$(TYPEDSIGNATURES)

Onset-to-report delay PMF (case ascertainment timing). Defaults to a
short Gamma centred on five days, with priors so the delay carries
uncertainty into the fit.
"""
@model function onset_to_report_model_ss(nmax::Integer;
        mean_prior = truncated(Normal(5.0, 1.5); lower = 1e-3),
        sd_prior   = truncated(Normal(3.0, 1.0); lower = 1e-3))
    rep_mean ~ mean_prior
    rep_sd   ~ sd_prior
    shape = (rep_mean / rep_sd)^2
    scale = rep_sd^2 / rep_mean
    pmf = discretise_delay_ss(Gamma(shape, scale), nmax)
    return (; pmf, rep_mean, rep_sd)
end

"""
$(TYPEDSIGNATURES)

Onset-to-detection delay PMF (border / hospital detection timing on
Uganda exports). Defaults to a moderate Gamma centred on ten days.
"""
@model function onset_to_detect_model_ss(nmax::Integer;
        mean_prior = truncated(Normal(10.0, 3.0); lower = 1e-3),
        sd_prior   = truncated(Normal(4.0, 1.5); lower = 1e-3))
    det_mean ~ mean_prior
    det_sd   ~ sd_prior
    shape = (det_mean / det_sd)^2
    scale = det_sd^2 / det_mean
    pmf = discretise_delay_ss(Gamma(shape, scale), nmax)
    return (; pmf, det_mean, det_sd)
end

"""
$(TYPEDSIGNATURES)

CFR submodel — same prior as the baseline `cfr_model` so the two are
interchangeable.
"""
@model function cfr_model_ss(; cfr_prior = Beta(6.6, 13.4))
    CFR ~ cfr_prior
    return (; CFR)
end

"""
$(TYPEDSIGNATURES)

NegativeBinomial dispersion submodel parameterised on the `1/√k`
scale, matching the baseline `surveillance_dispersion_model`.
"""
@model function surveillance_dispersion_model_ss(;
        inv_sqrt_k_prior = truncated(Normal(0.6, 0.2); lower = 0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

"""
$(TYPEDSIGNATURES)

Pooled DRC / Uganda ascertainment, matching the baseline
`pooled_ascertainment_model` non-centred form.
"""
@model function pooled_ascertainment_model_ss(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0, 0.5); lower = 1e-4))
    μ_logit  ~ mu_prior
    τ_logit  ~ tau_prior
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    p_drc    := logistic(μ_logit + τ_logit * z_drc)
    p_uganda := logistic(μ_logit + τ_logit * z_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

"""
$(TYPEDSIGNATURES)

Daily traveller volume, matching the baseline
`traveller_volume_model`.
"""
@model function traveller_volume_model_ss(;
        mean::Real = ITURI_DAILY_TRAVEL,
        sd::Real   = ITURI_DAILY_TRAVEL_SD)
    daily_travellers ~ truncated(Normal(mean, sd); lower = 0)
    return (; daily_travellers)
end

"""
$(TYPEDSIGNATURES)

Seed-size submodel: the initial integer infection count `I[1]` on day
one of the grid. A small primary cluster prior, so the single-seed
baseline is the mode and modest clusters are admissible.
"""
@model function seed_size_model_ss(;
        prior = truncated(Normal(1.0, 1.0); lower = 1e-3))
    I0 ~ prior
    return (; I0)
end

"""
$(TYPEDSIGNATURES)

Offspring overdispersion `φ` for the NB-branching step. A wide
half-Normal on `1/√φ` lets the prior cover both heavy-tailed (small
`φ`) and near-Poisson (large `φ`) limits.
"""
@model function offspring_dispersion_model_ss(;
        inv_sqrt_phi_prior = truncated(Normal(0.3, 0.3); lower = 1e-3))
    inv_sqrt_phi ~ inv_sqrt_phi_prior
    φ := 1.0 / (inv_sqrt_phi^2 + eps(typeof(inv_sqrt_phi)))
    return (; φ, inv_sqrt_phi)
end

## --- Composer ----------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Joint composer for the state-space architecture (issue #48).

Latent state: a discrete daily infection trajectory `I[1:n]`, with
`I[1]` drawn from a seed-size submodel and each subsequent day drawn
from the exact NB-branching step

```math
I_t \\mid I_{<t}, R_t \\sim
    \\mathrm{NegativeBinomial}(\\mu_t = λ_t, \\phi).
```

The continuous nuisances (the weekly log-`R_t` knots, the delays,
`CFR`, ascertainment, dispersion, traveller volume, offspring `φ`) are
sampled by submodel and stay differentiable, so a NUTS-inside-Gibbs
update can sample them with Mooncake gradients while a particle Gibbs
block updates the latent path. See
docs/src/proposals/state-space-particle.md.

The four observation streams condition on cumulative totals through to
the cut-off day `n`. Any `missing` count drops the corresponding
likelihood so the composer also serves as a prior-predictive
generator.

`I_obs` (optional) conditions on a fixed integer latent path,
collapsing the latent block to a deterministic chain so the remaining
continuous nuisance block is fully differentiable. Used in tests to
check gradient finiteness without running the particle filter.

`tmrca_days` adds the censored upper-tail soft lower bound on the
outbreak age (here `T = n`), mirroring `genetic_seeding_model`. Pass
`missing` to drop it.
"""
@model function state_space_joint(
        n::Integer,
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer},
        exports_deaths::Union{Missing, Integer};
        tmrca_days::Union{Missing, Real}      = missing,
        tmrca_days_sd::Real                   = 20.0,
        source_population::Real               = ITURI_POPULATION,
        delay_nmax::Integer                   = 60,
        gi_nmax::Integer                      = 40,
        incub_nmax::Integer                   = 30,
        report_nmax::Integer                  = 40,
        detect_nmax::Integer                  = 30,
        rt                = rt_walk_model_ss,
        gi                = generation_interval_model_ss,
        incubation        = incubation_model_ss,
        onset_to_death    = onset_to_death_model_ss,
        onset_to_report   = onset_to_report_model_ss,
        onset_to_detect   = onset_to_detect_model_ss,
        cfr               = cfr_model_ss(),
        dispersion        = surveillance_dispersion_model_ss(),
        ascertainment     = pooled_ascertainment_model_ss(),
        traveller         = traveller_volume_model_ss(),
        seed              = seed_size_model_ss(),
        offspring         = offspring_dispersion_model_ss(),
        I_obs::Union{Nothing, AbstractVector{<:Integer}} = nothing)

    ## Continuous nuisance block (NUTS-friendly) ------------------------
    rt_state    ~ to_submodel(rt(n), false)
    gi_state    ~ to_submodel(gi(gi_nmax), false)
    incub_state ~ to_submodel(incubation(incub_nmax), false)
    od_state    ~ to_submodel(onset_to_death(delay_nmax), false)
    rep_state   ~ to_submodel(onset_to_report(report_nmax), false)
    det_state   ~ to_submodel(onset_to_detect(detect_nmax), false)
    cfr_state   ~ to_submodel(cfr, false)
    disp_state  ~ to_submodel(dispersion, false)
    asc_state   ~ to_submodel(ascertainment, false)
    trav_state  ~ to_submodel(traveller, false)
    seed_state  ~ to_submodel(seed, false)
    off_state   ~ to_submodel(offspring, false)

    Rt        = rt_state.Rt
    g         = gi_state.g
    CFR       = cfr_state.CFR
    k         = disp_state.k
    p_drc     = asc_state.p_drc
    p_uganda  = asc_state.p_uganda
    φ         = off_state.φ
    q         = trav_state.daily_travellers / source_population

    ## Latent discrete infection path ----------------------------------
    if I_obs === nothing
        ## Day 1: round the continuous seed to the nearest non-negative
        ## integer through a Poisson, so the latent path is integer-
        ## valued and PG can step it. The seed prior carries the
        ## small-cluster intent.
        I = Vector{Int}(undef, n)
        I[1] ~ Poisson(_safe_rate_ss(seed_state.I0))
        for t in 2:n
            λt = nb_branching_step(I, g, Rt, t)
            ## NegativeBinomial(mean = μ, dispersion = φ) using the
            ## safe-NB success-probability clamp from the baseline.
            I[t] ~ safe_nbinomial_ss(φ, λt)
        end
    else
        ## Conditioning on a fixed integer latent path: the path is
        ## observed, so PG has nothing to do and the rest of the model
        ## reduces to the continuous nuisance block.
        I = I_obs
        for t in 2:n
            λt = nb_branching_step(float.(I), g, Rt, t)
            Turing.@addlogprob! logpdf(safe_nbinomial_ss(φ, λt), I[t])
        end
    end

    ## Onset staging — infections to onsets, once ----------------------
    onsets = convolve_delay_ss(float.(I), incub_state.pmf)
    cum_infections = cumsum(float.(I))
    C_T := cum_infections[n]

    ## Stream 2: DRC suspected deaths (cumulative) ---------------------
    deaths_daily = CFR .* convolve_delay_ss(onsets, od_state.pmf)
    expected_deaths_T := _safe_rate_ss(sum(deaths_daily))
    if !ismissing(total_deaths)
        total_deaths ~ safe_nbinomial_ss(k, expected_deaths_T)
    end

    ## Stream 3: DRC reported suspected cases (cumulative) -------------
    reports_daily = p_drc .* convolve_delay_ss(onsets, rep_state.pmf)
    expected_reports_T := _safe_rate_ss(sum(reports_daily))
    if !ismissing(reported_cases)
        reported_cases ~ safe_nbinomial_ss(k, expected_reports_T)
    end

    ## Stream 1: exported cases detected in Uganda --------------------
    export_onsets = p_uganda .* q .* onsets
    detect_daily = convolve_delay_ss(export_onsets, det_state.pmf)
    expected_exports_T := _safe_rate_ss(sum(detect_daily))
    if !ismissing(exported_cases)
        exported_cases ~ Poisson(expected_exports_T)
    end

    ## Stream 4: deaths among exported cases --------------------------
    export_deaths_daily = CFR .* convolve_delay_ss(
        export_onsets, od_state.pmf)
    expected_exports_deaths_T := _safe_rate_ss(sum(export_deaths_daily))
    if !ismissing(exports_deaths)
        exports_deaths ~ Poisson(expected_exports_deaths_T)
    end

    ## Genetic TMRCA soft lower bound on the outbreak age `n` ---------
    if !ismissing(tmrca_days)
        @addlogprob! normlogcdf((n - tmrca_days) / tmrca_days_sd)
    end

    return (; C_T, Rt, expected_deaths_T, expected_reports_T,
              expected_exports_T, expected_exports_deaths_T,
              infections = I)
end

"""
$(TYPEDSIGNATURES)

NaN-safe NegativeBinomial parameterised by dispersion `k` and mean
`μ`. Mirrors the baseline `safe_nbinomial` in `docs/examples/analysis.jl`;
the package-internal copy here lets the state-space file stand alone.
Extreme NUTS warmup proposals can push `μ` non-finite; the clamp keeps
the success probability in `(eps, 1 - eps)` so the distribution
domain check does not trip.
"""
function safe_nbinomial_ss(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end
