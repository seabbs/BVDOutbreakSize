## Turing submodels for the explicit-convolution architecture (issue
## #5).
##
## These mirror the structure of the current model in
## `docs/examples/analysis.jl` (building-block submodels, observation
## submodels, a joint composer) but build every expected count from
## the explicit-convolution forward layer in `explicit_convolution.jl`.
## They are added alongside the current package code without changing
## it; the current analysis keeps using its own inline submodels.
##
## The growth, CFR, traveller-volume, dispersion and ascertainment
## priors are unchanged from the current model and are not re-defined
## here. Two pieces are new: an incubation (infection-to-onset) delay,
## and an onset-to-report delay. Both carry weakly-supported priors
## (see the "Explicit convolution" entry under "Redesign proposals"
## in the docs).
##
## Design invariants (project-owner directives):
##   1. EVERY prior is a submodel. No literal Normal/Gamma/etc.
##      constants are buried in a model body — each is lifted into its
##      own `@model` submodel that the composer accepts as an
##      injectable keyword. Tests in
##      `test_explicit_convolution_models.jl` enforce this.
##   2. EVERY delay is specified via a prior. No delay distribution
##      and no generation time is a fixed constant. The four time
##      scales consumed by the observation expectations — incubation
##      (`infection_to_onset_delay_model`), onset-to-death
##      (`onset_to_death_delay_model`), onset-to-report
##      (`onset_to_report_delay_model`) and onset-to-detection
##      (`onset_to_detection_window_model`) — each sample their
##      parameter(s) from a prior in their own submodel. The growth
##      rate `r` is itself sampled via `τ` and `m` from
##      `exponential_growth_explicit`, so the timescale of spread is
##      not a fixed input either.
##   3. EVERY observation distribution is injected (with a sensible
##      default), so callers can swap Poisson↔NegBinomial or use the
##      `safe_nbinomial` pattern without editing the observation
##      submodels.
##   4. The onset-incidence curve is built ONCE per draw via
##      `OnsetIncidence` and reused across the three observation
##      integrals; the inner incubation convolution is never repeated.

# NaN/Inf-safe NegBinomial(mean μ, dispersion k); duplicated from the
# analysis so the explicit-convolution model is self-contained as
# package code.
function _safe_nbinomial_explicit(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

# Default observation-distribution constructors. Take the expected
# count `μ` and (for the NegBinomial case) the shared dispersion `k`,
# and return a Distribution. Injected into the observation submodels
# so callers can swap them without editing the submodel.
_default_poisson_obs(μ) = Poisson(max(μ, eps(typeof(μ))))
_default_nbinomial_obs(k, μ) = _safe_nbinomial_explicit(k, μ)

"""
$(TYPEDSIGNATURES)

Exponential-growth building block for the explicit-convolution model.
Samples the doubling time `τ` and the doubling-multiplier `m = T/τ`
(the same near-orthogonal parameterisation as the current model) and
returns the growth rate `r`, elapsed time `T`, and cumulative
*infections* `I_T = exp(r·T)`.  In this architecture the latent state
is infections, so `I_T` is the cumulative infection count; cumulative
onsets are slightly lower and are computed downstream from the
onset-incidence convolution.
"""
@model function exponential_growth_explicit(;
        tau_prior = LogNormal(log(14), 0.4),
        m_prior   = truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0))
    τ ~ tau_prior
    m ~ m_prior
    r := log(2) / τ
    T := m * τ
    ## `I_T = exp(r·T) = 2^m` is recorded by the composer as a shared
    ## deterministic so the chain has it regardless of which growth
    ## submodel is injected.
    return (; τ, r, m, T)
end

"""
$(TYPEDSIGNATURES)

Infection-to-onset (incubation) delay building block, gamma
distributed. The Imperial report cites a 6–11 day incubation range
across three studies but no Bayesian posterior exists, so the prior
is constructed from that narrative range: a `Gamma`-mean-anchored
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
@model function infection_to_onset_delay_model(;
        alpha_prior = truncated(Normal(11.0, 3.0); lower = 1.0),
        theta_prior = truncated(Normal(0.74, 0.25); lower = 1e-3))
    α_inc ~ alpha_prior
    θ_inc ~ theta_prior
    return (; α_inc, θ_inc, dist = Gamma(α_inc, θ_inc))
end

"""
$(TYPEDSIGNATURES)

Onset-to-death delay building block, gamma distributed. Anchored on
the bdbv-linelist Bayesian reanalysis of the Isiro 2012 line list:

```math
\\alpha \\sim \\mathrm{Normal}^{+}(4.3, 1.22), \\qquad
\\theta \\sim \\mathrm{Normal}^{+}(2.6, 0.82).
```

The lower bounds (`α ≥ 1`, `θ ≥ 10^{-3}`) match
`infection_to_onset_delay_model` and `onset_to_report_delay_model` so
an extreme NUTS proposal cannot push the sampled `Gamma(α, θ)` toward
the `α → 0` density spike, where AD on the density itself becomes
ill-behaved.
"""
@model function onset_to_death_delay_model(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 1.0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 1e-3))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

"""
$(TYPEDSIGNATURES)

Onset-to-report delay building block, gamma distributed. The
bdbv-linelist Isiro 2012 onset→notification estimate is 19.7 d
(13.7–30.1), but Charniga 2024 flags a 30-day-cap truncation bias in
the Rosello point estimate, so this prior is used with a strong
caveat. Centred to give a mean near 18 d.

```math
\\alpha_{\\text{otr}} \\sim \\mathrm{Normal}^{+}(4, 1.5), \\qquad
\\theta_{\\text{otr}} \\sim \\mathrm{Normal}^{+}(4.5, 1.5).
```
"""
@model function onset_to_report_delay_model(;
        alpha_prior = truncated(Normal(4.0, 1.5); lower = 1.0),
        theta_prior = truncated(Normal(4.5, 1.5); lower = 1e-3))
    α_otr ~ alpha_prior
    θ_otr ~ theta_prior
    return (; α_otr, θ_otr, dist = Gamma(α_otr, θ_otr))
end

"""
$(TYPEDSIGNATURES)

Onset-to-detection window building block. With incubation modelled
separately, `w` is unambiguously onset-to-detection (the current
model's window mixes incubation and detection together). The prior is
centred at 7 d with SD 3 d:

```math
w \\sim \\mathrm{Normal}^{+}(7, 3).
```

Lifted into its own submodel so the composer accepts it the same way
as the other delays, satisfying the "every prior is a submodel"
invariant.
"""
@model function onset_to_detection_window_model(;
        window_prior = truncated(Normal(7.0, 3.0); lower = 0))
    w ~ window_prior
    return (; w)
end

"""
$(TYPEDSIGNATURES)

Genetic TMRCA soft lower-bound building block. Same construction as
the current model's `genetic_seeding_model`: observing the
molecular-clock TMRCA read `tmrca_days` at the upper censoring point
of `Normal(T, σ)` contributes a one-sided likelihood
``\\Phi((T - g)/\\sigma)``. Lifted into its own submodel so the
composer can inject or drop it the same way as the other priors;
passing `nothing` to the composer's `genetic` keyword disables the
term entirely.

`T` is the elapsed-time deterministic from
[`exponential_growth_explicit`](@ref); `tmrca_days` and
`tmrca_days_sd` come from the observation block.
"""
@model function genetic_seeding_bound_model(T, tmrca_days::Real;
        tmrca_days_sd::Real = 20.0)
    tmrca_days ~ censored(Normal(T, tmrca_days_sd); upper = tmrca_days)
    return (; tmrca_days, tmrca_days_sd)
end

"""
$(TYPEDSIGNATURES)

Exports observation submodel for the explicit-convolution model.
Takes a precomputed [`OnsetIncidence`](@ref) `oi`, the Uganda
ascertainment `p_uganda`, the per-capita travel rate `q` and the
onset-to-detection window `w`, and ties the exported-case count to
the latent state through [`expected_exports_onset_staged`](@ref). The
observation distribution is injected via the `obs` keyword: it must
accept the expected count `μ_e` and return a Distribution. Defaults
to a Poisson, matching the small-detection-probability regime of the
export count.
"""
@model function exports_onset_staged_obs(
        exported_cases::Union{Missing, Integer},
        oi, p_uganda::Real, q::Real, w::Real;
        obs = _default_poisson_obs)
    μ_e := expected_exports_onset_staged(oi, p_uganda, q, w)
    exported_cases ~ obs(μ_e)
    return (; expected_exports = μ_e)
end

"""
$(TYPEDSIGNATURES)

Deaths observation submodel for the explicit-convolution model.
Builds the onset-to-death CDF holder once (covering `[0, T]`), forms
the CFR-weighted onset-to-death convolution through
[`expected_deaths_onset_staged`](@ref), and applies an injectable
observation distribution. `obs` takes the shared dispersion `k` and
the expected count `μ_d` and returns a Distribution; the default uses
the NaN-safe NegBinomial that the rest of the package shares. `CFR`
is the fraction of onsets that die.
"""
@model function deaths_onset_staged_obs(
        total_deaths::Union{Missing, Integer},
        oi, death_dist, CFR::Real, k::Real;
        obs = _default_nbinomial_obs)
    death_delay = ExportDeathDelay(death_dist, oi.T)
    μ_d := expected_deaths_onset_staged(oi, death_delay, CFR)
    total_deaths ~ obs(k, μ_d)
    return (; expected_deaths = μ_d)
end

"""
$(TYPEDSIGNATURES)

Reported-cases observation submodel for the explicit-convolution
model. Builds the onset-to-report CDF holder once, forms the
delayed-report convolution through
[`expected_reports_onset_staged`](@ref), and applies an injectable
observation distribution. Defaults to the same shared-dispersion
NegBinomial as the deaths stream. Unlike the current model the
reporting is delayed: recent infections are not yet fully ascertained.
"""
@model function cases_onset_staged_obs(
        reported_cases::Union{Missing, Integer},
        oi, report_dist, p_drc::Real, k::Real;
        obs = _default_nbinomial_obs)
    report_delay = ExportDeathDelay(report_dist, oi.T)
    μ_c := expected_reports_onset_staged(oi, report_delay, p_drc)
    reported_cases ~ obs(k, μ_c)
    return (; expected_reports = μ_c)
end

"""
$(TYPEDSIGNATURES)

Joint composer for the explicit-convolution architecture over all
four data streams plus the genetic TMRCA bound. Samples the building
blocks (growth, incubation, onset-to-death delay, onset-to-report
delay, onset-to-detection window, CFR, traveller volume, dispersion,
pooled ascertainment), tabulates the onset-incidence curve once per
draw with [`OnsetIncidence`](@ref), and threads it into the three
observation submodels so the nested convolution is paid once.

Any stream argument may be `missing` to drop it (so the composer
doubles as a prior/posterior-predictive generator). Pass
`tmrca_days = missing` to disable the genetic-seeding term.

The submodel keyword arguments default to the explicit-convolution
building blocks but are injectable so a single-stream variant or a
sensitivity analysis can swap a prior without editing this composer:

| Keyword | Default | What it controls |
|---|---|---|
| `growth` | (required) | growth rate / elapsed time |
| `incubation` | `infection_to_onset_delay_model()` | infection→onset delay |
| `death` | `onset_to_death_delay_model()` | onset→death delay |
| `report` | `onset_to_report_delay_model()` | onset→report delay |
| `window` | `onset_to_detection_window_model()` | onset→detection window |
| `cfr` | (required) | case-fatality ratio |
| `dispersion` | (required) | surveillance dispersion |
| `ascertainment` | (required) | pooled ascertainment |
| `traveller` | (required) | daily traveller volume |
| `exports_obs` | Poisson | exports likelihood |
| `deaths_obs` | safe NegBinomial | deaths likelihood |
| `cases_obs` | safe NegBinomial | reports likelihood |
| `genetic` | `genetic_seeding_bound_model` | TMRCA term; `nothing` to drop |
"""
@model function bvd_joint_explicit_convolution(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer},
        growth, cfr, dispersion, ascertainment, traveller;
        incubation = infection_to_onset_delay_model(),
        death      = onset_to_death_delay_model(),
        report     = onset_to_report_delay_model(),
        window     = onset_to_detection_window_model(),
        genetic    = genetic_seeding_bound_model,
        exports_obs = _default_poisson_obs,
        deaths_obs  = _default_nbinomial_obs,
        cases_obs   = _default_nbinomial_obs,
        tmrca_days::Union{Missing, Real} = missing,
        tmrca_days_sd::Real = 20.0,
        source_population::Real = ITURI_POPULATION)

    growth_state ~ to_submodel(growth, false)
    r = growth_state.r
    T = growth_state.T

    ## Genetic TMRCA soft lower bound on `T`: lifted into its own
    ## submodel so it follows the same submodel-injection pattern as
    ## the other priors. Pass `genetic = nothing` (or `tmrca_days =
    ## missing`) to drop the term.
    if genetic !== nothing && !ismissing(tmrca_days)
        genetic_state ~ to_submodel(
            genetic(T, tmrca_days; tmrca_days_sd = tmrca_days_sd), false)
    end

    incub_state  ~ to_submodel(incubation, false)
    death_state  ~ to_submodel(death, false)
    report_state ~ to_submodel(report, false)
    window_state ~ to_submodel(window, false)
    cfr_state    ~ to_submodel(cfr, false)
    disp_state   ~ to_submodel(dispersion, false)
    asc_state    ~ to_submodel(ascertainment, false)
    travel_state ~ to_submodel(traveller, false)

    CFR      = cfr_state.CFR
    k        = disp_state.k
    p_drc    = asc_state.p_drc
    p_uganda = asc_state.p_uganda
    q        = travel_state.daily_travellers / source_population
    w        = window_state.w

    ## Tabulate the onset-incidence curve once, then reuse it across
    ## all three observation integrals (the precompute that bounds the
    ## nested-convolution cost). Built ONCE per draw — the deaths,
    ## reports and exports submodels all receive the same `oi`.
    oi = OnsetIncidence(r, incub_state.dist, T)
    ## Cumulative infections and cumulative onsets at the cut-off,
    ## tracked as deterministics so they appear in the chain
    ## regardless of which growth submodel the caller injects (the
    ## composer takes ownership of `I_T` so the baseline
    ## `exponential_growth_model` from analysis.jl can be injected
    ## directly without it needing to expose a matching field).
    I_T      := exp(r * T)
    onsets_T := expected_onsets_staged(oi)

    exports_state ~ to_submodel(
        exports_onset_staged_obs(exported_cases, oi, p_uganda, q, w;
            obs = exports_obs), false)
    deaths_state ~ to_submodel(
        deaths_onset_staged_obs(total_deaths, oi, death_state.dist,
            CFR, k; obs = deaths_obs), false)
    cases_state ~ to_submodel(
        cases_onset_staged_obs(reported_cases, oi, report_state.dist,
            p_drc, k; obs = cases_obs), false)

    return (; I_T, onsets_T,
            expected_exports = exports_state.expected_exports,
            expected_deaths  = deaths_state.expected_deaths,
            expected_reports = cases_state.expected_reports)
end
