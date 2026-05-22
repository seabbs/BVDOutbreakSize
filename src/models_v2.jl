## Turing submodels for the convolution-v2 architecture (issue #5).
##
## These mirror the structure of the current model in
## `docs/examples/analysis.jl` (building-block submodels, observation
## submodels, a joint composer) but build every expected count from the
## v2 explicit-convolution layer in `convolution_v2.jl`. They are added
## alongside the current package code without changing it; the current
## analysis keeps using its own inline submodels.
##
## The growth, CFR, traveller-volume, dispersion and ascertainment
## priors are unchanged from the current model and are not re-defined
## here. Two pieces are new: an incubation (infection-to-onset) delay,
## and an onset-to-report delay. Both carry weakly-supported priors
## (see `docs/proposals/convolution-v2.md`).

"""
$(TYPEDSIGNATURES)

Exponential-growth building block for v2. Samples the doubling time
`τ` and the doubling-multiplier `m = T/τ` (the same near-orthogonal
parameterisation as the current model) and returns the growth rate
`r`, elapsed time `T`, and cumulative *infections* `I_T = exp(r·T)`.
Under v2 the latent state is infections, so `I_T` is the cumulative
infection count; cumulative onsets are slightly lower and are computed
downstream from the onset-incidence convolution.
"""
@model function growth_v2(;
        tau_prior = LogNormal(log(14), 0.4),
        m_prior   = truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0))
    τ ~ tau_prior
    m ~ m_prior
    r   := log(2) / τ
    T   := m * τ
    I_T := 2.0 ^ m
    return (; τ, r, m, T, I_T)
end

"""
$(TYPEDSIGNATURES)

Incubation (infection-to-onset) delay building block for v2, gamma
distributed. The Imperial report cites a 6–11 day incubation range
across three studies but no Bayesian posterior exists, so the prior is
constructed from that narrative range: a `Gamma`-mean-anchored
truncated-Normal prior on the shape and scale centred to give a mean
near 8 d and SD near 2.4 d. Both parameters are weakly identified by
the data and will be prior-dominated.

```math
\\alpha_{\\text{inc}} \\sim \\mathrm{Normal}^{+}(11, 3), \\qquad
\\theta_{\\text{inc}} \\sim \\mathrm{Normal}^{+}(0.74, 0.25),
```

giving `Gamma(α, θ)` a mean ≈ 8 d (95% range covering roughly the
6–11 d span the report cites).
"""
@model function incubation_v2(;
        alpha_prior = truncated(Normal(11.0, 3.0); lower = 1.0),
        theta_prior = truncated(Normal(0.74, 0.25); lower = 1e-3))
    α_inc ~ alpha_prior
    θ_inc ~ theta_prior
    return (; α_inc, θ_inc, dist = Gamma(α_inc, θ_inc))
end

"""
$(TYPEDSIGNATURES)

Onset-to-death delay building block for v2, gamma distributed,
unchanged from the current model. Anchored on the bdbv-linelist
Bayesian reanalysis of the Isiro 2012 line list:

```math
\\alpha \\sim \\mathrm{Normal}^{+}(4.3, 1.22), \\qquad
\\theta \\sim \\mathrm{Normal}^{+}(2.6, 0.82).
```
"""
@model function onset_to_death_v2(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

"""
$(TYPEDSIGNATURES)

Onset-to-report delay building block for v2, gamma distributed. The
bdbv-linelist Isiro 2012 onset→notification estimate is 19.7 d
(13.7–30.1), but Charniga 2024 flags a 30-day-cap truncation bias in
the Rosello point estimate, so this prior is used with a strong
caveat. Centred to give a mean near 18 d.

```math
\\alpha_{\\text{otr}} \\sim \\mathrm{Normal}^{+}(4, 1.5), \\qquad
\\theta_{\\text{otr}} \\sim \\mathrm{Normal}^{+}(4.5, 1.5).
```
"""
@model function onset_to_report_v2(;
        alpha_prior = truncated(Normal(4.0, 1.5); lower = 1.0),
        theta_prior = truncated(Normal(4.5, 1.5); lower = 1e-3))
    α_otr ~ alpha_prior
    θ_otr ~ theta_prior
    return (; α_otr, θ_otr, dist = Gamma(α_otr, θ_otr))
end

# NaN/Inf-safe NegBinomial(mean μ, dispersion k); duplicated from the
# analysis so the v2 model is self-contained as package code.
function _safe_nbinomial_v2(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

"""
$(TYPEDSIGNATURES)

v2 exports observation submodel. Takes a precomputed
[`OnsetIncidence`](@ref) `oi`, the Uganda ascertainment `p_uganda`, the
per-capita travel rate `q` and the onset-to-detection window `w`, and
ties the exported-case count to the latent state through
[`expected_exports_v2`](@ref) with a Poisson likelihood.
"""
@model function exports_obs_v2(
        exported_cases::Union{Missing, Integer},
        oi, p_uganda::Real, q::Real, w::Real)
    μ_e := expected_exports_v2(oi, p_uganda, q, w)
    exported_cases ~ Poisson(μ_e)
    return (; expected_exports = μ_e)
end

"""
$(TYPEDSIGNATURES)

v2 deaths observation submodel. Builds the onset-to-death CDF holder
once (covering `[0, T]`), forms the CFR-weighted onset-to-death
convolution through [`expected_deaths_v2`](@ref), and applies the
shared-dispersion NegBinomial likelihood. `CFR` is the fraction of
onsets that die.
"""
@model function deaths_obs_v2(
        total_deaths::Union{Missing, Integer},
        oi, death_dist, CFR::Real, k::Real)
    death_delay = ExportDeathDelay(death_dist, oi.T)
    μ_d := expected_deaths_v2(oi, death_delay, CFR)
    total_deaths ~ _safe_nbinomial_v2(k, μ_d)
    return (; expected_deaths = μ_d)
end

"""
$(TYPEDSIGNATURES)

v2 reported-cases observation submodel. Builds the onset-to-report CDF
holder once, forms the delayed-report convolution through
[`expected_reports_v2`](@ref), and applies the shared-dispersion
NegBinomial likelihood. Unlike the current model the reporting is
delayed: recent infections are not yet fully ascertained.
"""
@model function cases_obs_v2(
        reported_cases::Union{Missing, Integer},
        oi, report_dist, p_drc::Real, k::Real)
    report_delay = ExportDeathDelay(report_dist, oi.T)
    μ_c := expected_reports_v2(oi, report_delay, p_drc)
    reported_cases ~ _safe_nbinomial_v2(k, μ_c)
    return (; expected_reports = μ_c)
end

"""
$(TYPEDSIGNATURES)

Joint convolution-v2 composer over all four data streams plus the
genetic TMRCA bound. Samples the v2 building blocks (growth,
incubation, onset-to-death delay, onset-to-report delay, CFR,
traveller volume, dispersion, pooled ascertainment), tabulates the
onset-incidence curve once per draw with [`OnsetIncidence`](@ref), and
threads it into the three observation submodels so the nested
convolution is paid once.

Any stream argument may be `missing` to drop it (so the composer
doubles as a prior/posterior-predictive generator). The genetic
seeding term reuses the current model's `censored`-Normal soft bound
on `T`; pass `tmrca_days = missing` to disable it.

The submodel keyword arguments default to the v2 building blocks but
are injectable so a single-stream variant or a sensitivity analysis can
swap a prior without editing this composer. `growth`, `cfr`,
`dispersion`, `ascertainment` and `traveller` reuse the current
model's building blocks (passed in by the caller) since their priors
are unchanged under v2.
"""
@model function bvd_joint_v2(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer},
        growth, cfr, dispersion, ascertainment, traveller;
        incubation = incubation_v2(),
        death      = onset_to_death_v2(),
        report     = onset_to_report_v2(),
        window_prior = truncated(Normal(7.0, 3.0); lower = 0),
        tmrca_days::Union{Missing, Real} = missing,
        tmrca_days_sd::Real = 20.0,
        source_population::Real = ITURI_POPULATION)

    growth_state ~ to_submodel(growth, false)
    r = growth_state.r
    T = growth_state.T

    ## Genetic TMRCA soft lower bound on T (same construction as the
    ## current model): observing the molecular-clock read at the upper
    ## censoring point of Normal(T, σ) contributes Φ((T − g)/σ).
    if !ismissing(tmrca_days)
        tmrca_days ~ censored(Normal(T, tmrca_days_sd); upper = tmrca_days)
    end

    incub_state  ~ to_submodel(incubation, false)
    death_state  ~ to_submodel(death, false)
    report_state ~ to_submodel(report, false)
    cfr_state    ~ to_submodel(cfr, false)
    disp_state   ~ to_submodel(dispersion, false)
    asc_state    ~ to_submodel(ascertainment, false)
    travel_state ~ to_submodel(traveller, false)

    CFR      = cfr_state.CFR
    k        = disp_state.k
    p_drc    = asc_state.p_drc
    p_uganda = asc_state.p_uganda
    q        = travel_state.daily_travellers / source_population
    w ~ window_prior

    ## Tabulate the onset-incidence curve once, then reuse it across all
    ## three observation integrals (the precompute that bounds the
    ## nested-convolution cost).
    oi = OnsetIncidence(r, incub_state.dist, T)
    ## `I_T` (cumulative infections) is already recorded by the growth
    ## submodel; only the new onset total needs a deterministic here.
    onsets_T := expected_onsets_v2(oi)

    exports_state ~ to_submodel(
        exports_obs_v2(exported_cases, oi, p_uganda, q, w), false)
    deaths_state ~ to_submodel(
        deaths_obs_v2(total_deaths, oi, death_state.dist, CFR, k), false)
    cases_state ~ to_submodel(
        cases_obs_v2(reported_cases, oi, report_state.dist, p_drc, k), false)

    return (; I_T = growth_state.I_T, onsets_T,
            expected_exports = exports_state.expected_exports,
            expected_deaths  = deaths_state.expected_deaths,
            expected_reports = cases_state.expected_reports)
end
