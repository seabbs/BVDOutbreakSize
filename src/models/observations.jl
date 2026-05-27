# Observation submodels: each ties one data stream to the latent
# cumulative case count `C(T)`. They consume the growth state and any
# shared nuisance parameters (dispersion, ascertainment) from the
# composer above, introduce only the priors they need on top, and add
# a single likelihood term.

"""
NaN / Inf-safe `NegativeBinomial` constructor parameterised by mean
`μ` and dispersion `k`, with clamping on the success probability so
extreme NUTS proposals during warmup do not trip the distribution
domain check. Shared by [`deaths_model`](@ref),
[`reported_cases_model`](@ref) and [`confirmed_cases_model`](@ref).
"""
function safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

"""
Exports likelihood (Method 1, geographic spread). Couples the
observed exported-case count to `C(T)` through the at-risk
person-time integral over the detection window, with a Poisson
likelihood. Samples the detection window and traveller volume
submodels internally.
"""
@model function exports_model(
        exported_cases::Union{Missing, Integer},
        growth_state, p_uganda::Real;
        source_population::Real = ITURI_POPULATION,
        window = detection_window_model(),
        traveller = traveller_volume_model())
    cumulative = growth_state.cumulative
    T = growth_state.T

    window_state ~ to_submodel(window, false)
    w = window_state.w

    travel_state ~ to_submodel(traveller, false)
    daily_travellers = travel_state.daily_travellers

    q = daily_travellers / source_population
    expected_exports_T := expected_exports(cumulative, p_uganda, q, T, w)

    exported_cases ~ Poisson(expected_exports_T)

    return (; w, daily_travellers, p_uganda,
        expected_exports = expected_exports_T)
end

"""
Deaths likelihood (Method 2, back-calculation from deaths). Couples
the observed cumulative deaths to `C(T)` through the CFR-weighted
gamma convolution, with a NegativeBinomial likelihood sharing the
dispersion `k` of [`surveillance_dispersion_model`](@ref). Samples
the [`delay_model`](@ref) and [`cfr_model`](@ref) submodels
internally.
"""
@model function deaths_model(
        total_deaths::Union{Missing, Integer},
        growth_state, k::Real;
        delay = delay_model(),
        cfr = cfr_model())
    C_T = growth_state.C_T
    r = growth_state.r
    T = growth_state.T

    delay_state ~ to_submodel(delay, false)
    cfr_state ~ to_submodel(cfr, false)

    CFR = cfr_state.CFR

    ## NaN-safe clamp: extreme NUTS proposals during warmup can push
    ## the expected count to NaN / Inf.
    raw_deaths = delay_convolution(CFR, r, T, delay_state.dist)
    expected_deaths_T := isfinite(raw_deaths) ?
                         max(raw_deaths, eps(typeof(raw_deaths))) :
                         eps(typeof(raw_deaths))

    total_deaths ~ safe_nbinomial(k, expected_deaths_T)

    return (; CFR, delay_dist = delay_state.dist, expected_deaths_T)
end

"""
Reported (suspected) cases likelihood. Couples the observed DRC
suspected-case count to `C(T)` as the sum of (i) a BVD-driven
contribution `p_drc · ∫₀^T exp(r·s) · f_rep(T-s) ds` using the
onset-to-report delay [`report_delay_model`](@ref) and (ii) a non-BVD
background `λ_bg · T`, with `λ_bg` supplied by
[`test_positivity_model`](@ref). The NegativeBinomial likelihood shares
`k` with [`deaths_model`](@ref) and [`confirmed_cases_model`](@ref).
Exposes the derived per-suspected positivity `μ_BVD / μ_cases` as a
diagnostic and the BVD-suspected trajectory `bvd_reported_at` as a
callable so the confirmed-cases submodel can re-use it without
recomputing the convolution.
"""
@model function reported_cases_model(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real;
        report_delay = report_delay_model(),
        test_positivity = test_positivity_model())
    report_state ~ to_submodel(report_delay, false)
    test_positivity_state ~ to_submodel(test_positivity, false)
    λ_bg = test_positivity_state.λ_bg
    τ_test = test_positivity_state.τ_test
    f_rep = report_state.dist
    r = growth_state.r
    T = growth_state.T

    ## BVD-suspected cumulative as a function of elapsed time τ:
    ## reuses `delay_convolution` (i.e. ∫₀^τ exp(r·s)·f_rep(τ-s) ds)
    ## under the unit-ascertainment convention. The confirmed-cases
    ## submodel takes this closure directly and integrates it against
    ## f_lab — no separate f_conf kernel needed.
    bvd_reported_at = let r = r, p_drc = p_drc, f_rep = f_rep
        τ -> p_drc * delay_convolution(one(p_drc), r, τ, f_rep)
    end

    μ_BVD_raw = bvd_reported_at(T)
    μ_bg_raw = λ_bg * T
    μ_BVD := isfinite(μ_BVD_raw) ?
             max(μ_BVD_raw, eps(typeof(μ_BVD_raw))) :
             eps(typeof(μ_BVD_raw))
    μ_bg := isfinite(μ_bg_raw) ?
            max(μ_bg_raw, eps(typeof(μ_bg_raw))) :
            eps(typeof(μ_bg_raw))
    expected_reports := μ_BVD + μ_bg
    ## Implied per-suspected positivity at the cut-off. Exposed as a
    ## derived quantity so it can be checked against the sitrep figure.
    positivity := μ_BVD / expected_reports

    reported_cases ~ safe_nbinomial(k, expected_reports)

    return (; p_drc, λ_bg, τ_test,
        expected_reports, reported_cases,
        μ_BVD, μ_bg, positivity,
        bvd_reported_at,
        report_delay_dist = f_rep)
end

"""
Laboratory pipeline likelihood. Models two coupled observations
covering both the testing-volume gate and the per-test positivity
contrast:

- `tests_analysed` (`Cumul échantillons analysés`): observed cumulative
  count of suspected samples whose lab processing has completed by the
  cut-off. NegativeBinomial mean
  ``\\tau\\,(\\text{BVD}_\\text{tested} + \\text{bg}_\\text{tested})``,
  with
  ``\\text{BVD}_\\text{tested} = \\int_0^T \\text{BVD}_\\text{reported}(u)\\, f_\\text{lab}(T-u)\\,du``
  and
  ``\\text{bg}_\\text{tested} = \\lambda_\\text{bg}\\,\\int_0^T F_\\text{lab}(T-u)\\,du``.
  The lab-delay CDF ``F_\\text{lab}`` only counts samples that have
  finished processing by `T`, so right-truncation is handled by the
  integration limits — no fudge needed.
- `confirmed_cases` (`Cumul positifs`): observed positives among the
  analysed samples. Binomial likelihood conditional on
  `tests_analysed` with positivity probability
  ``s\\,\\text{BVD}_\\text{tested}/(\\text{BVD}_\\text{tested}+\\text{bg}_\\text{tested})``
  (PCR sensitivity times the BVD fraction in the tested pool;
  specificity is assumed perfect, so non-BVD samples never test
  positive).

Takes the BVD-suspected trajectory `bvd_reported_at` and the testing
fraction `τ_test` / background rate `λ_bg` directly from
[`reported_cases_model`](@ref) so the same posterior trajectory drives
both streams.

If `tests_analysed` is missing, falls back to the cumulative-count
NegBinomial form on `confirmed_cases` alone (mean
``s\\cdot\\tau\\,\\text{BVD}_\\text{tested}``); if both are missing,
the model is purely prior-predictive.
"""
@model function confirmed_cases_model(
        confirmed_cases::Union{Missing, Integer},
        tests_analysed::Union{Missing, Integer},
        bvd_reported_at, growth_state, k::Real,
        λ_bg::Real, τ_test::Real;
        lab_delay = lab_delay_model(),
        test_sensitivity = test_sensitivity_model())
    lab_state ~ to_submodel(lab_delay, false)
    sensitivity_state ~ to_submodel(test_sensitivity, false)
    f_lab = lab_state.dist
    s_test = sensitivity_state.s_test
    T = growth_state.T

    ## BVD samples that have completed lab processing by T (before
    ## the τ gate): same kernel as the previous confirmed integrand,
    ## ∫₀^T BVD_reported(u) · f_lab(T-u) du.
    raw_bvd_tested = τ_test * delay_convolution(bvd_reported_at, T, f_lab)
    BVD_tested := isfinite(raw_bvd_tested) ?
                  max(raw_bvd_tested, eps(typeof(raw_bvd_tested))) :
                  eps(typeof(raw_bvd_tested))

    ## Non-BVD samples that have completed lab processing by T:
    ## constant arrival rate λ_bg per day to the suspected pool,
    ## convolved against the lab-delay CDF F_lab so right-truncation
    ## is absorbed in the integration. τ gates the same sampling
    ## fraction as the BVD branch.
    bg_integrand = let f_lab = f_lab, T = T
        u -> cdf(f_lab, T - u)
    end
    bg_integral = integrate(bg_integrand, zero(T), T,
        _delay_scale(f_lab); alg = DEATH_INTEGRAL_ALG)
    raw_bg_tested = τ_test * λ_bg * bg_integral
    bg_tested := isfinite(raw_bg_tested) ?
                 max(raw_bg_tested, eps(typeof(raw_bg_tested))) :
                 eps(typeof(raw_bg_tested))

    expected_tested := BVD_tested + bg_tested
    ## Per-test positivity: s × BVD fraction in the tested pool.
    ## Exposed for direct comparison against the sitrep `Taux de
    ## positivité` figure.
    p_positive := s_test * BVD_tested / expected_tested

    ## Testing-volume likelihood. Skipped when no tests-analysed
    ## observation is supplied — falls through to the cumulative
    ## confirmed NegBinomial below for prior-predictive callers.
    if !ismissing(tests_analysed)
        tests_analysed ~ safe_nbinomial(k, expected_tested)
        ## Confirmed-given-tested Binomial. `tests_analysed` is a data
        ## integer here (not a sampled latent), so the discrete-trial
        ## count of the Binomial is fixed by the observation.
        confirmed_cases ~ Binomial(tests_analysed, p_positive)
    else
        ## No tests-analysed data: use the original cumulative-count
        ## form on confirmed alone. Means scale by τ relative to the
        ## pre-test-positivity form; equivalent up to that factor.
        expected_confirmed := s_test * BVD_tested
        confirmed_cases ~ safe_nbinomial(k, expected_confirmed)
    end

    return (; expected_tested, p_positive,
        BVD_tested, bg_tested, s_test, τ_test,
        lab_delay_dist = f_lab)
end

"""
Time-resolved deaths-among-exports likelihood. Models the dated
Uganda export deaths as an inhomogeneous Poisson process: a
continuous survival term over the pre-death stretch followed by
per-day Poisson counts. The detection window and traveller volume
are supplied by [`exports_model`](@ref) so the two Uganda-side
likelihoods share person-time.
"""
@model function exports_deaths_model(
        export_deaths_daily::AbstractVector,
        growth_state, CFR::Real, delay_dist, p_uganda::Real;
        pre_start_deaths::Union{Missing, Integer} = 0,
        window::Real,
        daily_travellers::Real,
        source_population::Real = ITURI_POPULATION)
    cumulative = growth_state.cumulative
    T = growth_state.T
    q = daily_travellers / source_population
    n = length(export_deaths_daily)   # days from earliest death to cut-off

    ## Precompute the onset-to-death CDF once and reuse it across every
    ## bin edge below (`T - s ≤ window` over the domain; see
    ## `ExportDeathDelay`).
    delay = ExportDeathDelay(delay_dist, window)
    Λ(t) = expected_exports_deaths(
        cumulative, delay, CFR, p_uganda, q, t, window)

    ## Pre-death zero stretch as one Poisson observed at 0; `missing`
    ## generates it for predictive checks (see equation (20)).
    pre = T - n > zero(T) ? Λ(T - n) : zero(T)
    pre_start_deaths ~ Poisson(max(pre, zero(pre)))

    ## Carry the upper edge forward so each Λ is evaluated once.
    λlo = pre
    for i in 1:n
        λhi = Λ(T - n + i)
        μ_day = max(λhi - λlo, eps(typeof(λhi)))
        export_deaths_daily[i] ~ Poisson(μ_day)
        λlo = λhi
    end

    return (;)
end

"""
First-export-detection timing survival term. Adds a one-sided
`Pr(no export detected before t1)` Poisson observation at zero,
matching the at-risk export person-time intensity. Passing
`delta = missing` makes the submodel a no-op.
"""
@model function exports_detection_timing_model(
        growth_state, p_uganda::Real;
        delta::Union{Missing, Real},
        pre_detection_exports::Union{Missing, Integer} = 0,
        window::Real,
        daily_travellers::Real,
        source_population::Real = ITURI_POPULATION)
    if !ismissing(delta)
        cumulative = growth_state.cumulative
        T = growth_state.T
        t1 = T - delta
        q = daily_travellers / source_population
        survived_exports := t1 <= zero(T) ? zero(T) :
                            expected_exports(cumulative, p_uganda, q, t1, window)
        ## No detection before t1 as a Poisson observed at 0; `missing`
        ## generates it for predictive checks (see equation (22)).
        pre_detection_exports ~ Poisson(max(survived_exports, zero(T)))
    end

    return (;)
end
