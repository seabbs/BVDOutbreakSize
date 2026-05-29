# Joint composer models: build the full generative model for each
# analysis by sampling the shared priors once and routing them into
# the relevant observation submodels. Single-stream composers are
# provided for the four count-based streams; [`bvd_joint`](@ref)
# conditions on all streams simultaneously.

"""
Exports-only composer (Method 1 analogue). Samples growth and
ascertainment, then conditions on the exports likelihood only. See
[`exports_model`](@ref).
"""
@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        growth = exponential_growth_model(),
        exports = exports_model,
        ascertainment = pooled_ascertainment_model())
    growth_state ~ to_submodel(growth, false)
    asc_state ~ to_submodel(ascertainment, false)

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state, asc_state.p_uganda), false)

    cumulative_cases := growth_state.C_T
end

"""
Deaths-only composer (Method 2 analogue). Samples growth and
dispersion, then conditions on the deaths likelihood only. See
[`deaths_model`](@ref).
"""
@model function deaths_only_model(
        total_deaths::Union{Missing, Integer};
        growth = exponential_growth_model(),
        deaths = deaths_model,
        dispersion = surveillance_dispersion_model())
    growth_state ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    k = dispersion_state.k

    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end

"""
Cases-only composer (ascertainment extension). Samples growth,
dispersion and pooled ascertainment, then conditions on the
reported-cases likelihood. See [`reported_cases_model`](@ref).
"""
@model function cases_only_model(
        reported_cases::Union{Missing, Integer};
        growth = exponential_growth_model(),
        reported_cases_submodel = reported_cases_model,
        dispersion = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model())
    growth_state ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state ~ to_submodel(ascertainment, false)
    k = dispersion_state.k

    reported_state ~ to_submodel(
        reported_cases_submodel(
            reported_cases, growth_state, k, asc_state.p_drc), false)

    cumulative_cases := growth_state.C_T
end

"""
Confirmed-and-tested-only composer (laboratory pipeline in isolation).
Samples growth, dispersion, pooled ascertainment and the report-delay
and test-positivity blocks, builds the BVD-suspected trajectory
directly, then conditions on the laboratory likelihood only. Unlike a
full reported-cases fit it does **not** instantiate the reported-count
likelihood: sampling the discrete suspected count would introduce a
latent NUTS cannot move, and only the confirmed/tested pair carries
information here. See [`confirmed_cases_model`](@ref). Use it for the
per-stream comparison against the joint fit.
"""
@model function confirmed_only_model(
        confirmed_cases::Union{Missing, Integer},
        cumulative_tests_analysed::Union{Missing, Integer} = missing;
        growth = exponential_growth_model(),
        confirmed = confirmed_cases_model,
        dispersion = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model(),
        test_positivity = test_positivity_model(),
        report_delay = report_delay_model(),
        lab_delay = lab_delay_model(),
        test_sensitivity = test_sensitivity_model())
    growth_state ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state ~ to_submodel(ascertainment, false)
    report_state ~ to_submodel(report_delay, false)
    test_positivity_state ~ to_submodel(test_positivity, false)
    k = dispersion_state.k
    p_drc = asc_state.p_drc
    λ_bg = test_positivity_state.λ_bg
    τ_test = test_positivity_state.τ_test
    f_rep = report_state.dist
    r = growth_state.r

    ## BVD-suspected cumulative trajectory, mirroring the closure in
    ## reported_cases_model so the confirmed submodel integrates the same
    ## convolution against the lab-delay kernel.
    bvd_reported_at = let r = r, p_drc = p_drc, f_rep = f_rep
        u -> p_drc * delay_convolution(one(p_drc), r, u, f_rep)
    end

    confirmed_state ~ to_submodel(
        confirmed(confirmed_cases, cumulative_tests_analysed,
            bvd_reported_at, growth_state, k, λ_bg, τ_test;
            lab_delay = lab_delay,
            test_sensitivity = test_sensitivity), false)

    cumulative_cases := growth_state.C_T
end

"""
Deaths-among-exports-only composer. Samples growth, delay, CFR,
detection window, traveller volume and ascertainment, then conditions
on the dated export-deaths likelihood. See
[`exports_deaths_model`](@ref).
"""
@model function exports_deaths_only_model(
        export_deaths_daily::AbstractVector;
        growth = exponential_growth_model(),
        delay = delay_model(),
        cfr = cfr_model(),
        window = detection_window_model(),
        traveller = traveller_volume_model(),
        exports_deaths_model = exports_deaths_model,
        ascertainment = pooled_ascertainment_model(),
        source_population::Real = ITURI_POPULATION,
        pre_start_deaths::Union{Missing, Integer} = 0)
    growth_state ~ to_submodel(growth, false)
    delay_state ~ to_submodel(delay, false)
    cfr_state ~ to_submodel(cfr, false)
    window_state ~ to_submodel(window, false)
    asc_state ~ to_submodel(ascertainment, false)

    travel_state ~ to_submodel(traveller, false)
    daily_travellers = travel_state.daily_travellers

    exports_deaths_state ~ to_submodel(
        exports_deaths_model(export_deaths_daily, growth_state,
            cfr_state.CFR, delay_state.dist, asc_state.p_uganda;
            pre_start_deaths = pre_start_deaths,
            window = window_state.w,
            daily_travellers = daily_travellers,
            source_population = source_population),
        false)

    cumulative_cases := growth_state.C_T
end

"""
Joint composer over all data streams. Conditions on exports, deaths,
reported cases, dated deaths-among-exports, optional
export-detection timing and optional genetic seeding bound. Each
stream argument may be passed as `missing` to drop it, so the same
model doubles as a generator for prior- and posterior-predictive
checks.
"""
@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer} = missing,
        export_deaths_daily::AbstractVector = Int[];
        confirmed_cases::Union{Missing, Integer} = missing,
        cumulative_tests_analysed::Union{Missing, Integer} = missing,
        predict_confirmed::Bool = false,
        growth = exponential_growth_model(),
        exports = exports_model,
        deaths = deaths_model,
        reported_cases_submodel = reported_cases_model,
        confirmed = confirmed_cases_model,
        exports_deaths_model = exports_deaths_model,
        exports_detection_timing = exports_detection_timing_model,
        dispersion = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model(),
        test_positivity = test_positivity_model(),
        report_delay = report_delay_model(),
        lab_delay = lab_delay_model(),
        test_sensitivity = test_sensitivity_model(),
        genetic = nothing,
        source_population::Real = ITURI_POPULATION,
        pre_start_deaths::Union{Missing, Integer} = 0,
        pre_detection_exports::Union{Missing, Integer} = 0,
        first_export_detection_delta::Union{Missing, Real} = missing)
    growth_state ~ to_submodel(growth, false)
    if genetic !== nothing
        genetic_state ~ to_submodel(genetic(growth_state.T), false)
    end
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state ~ to_submodel(ascertainment, false)
    k = dispersion_state.k
    p_drc = asc_state.p_drc
    p_uganda = asc_state.p_uganda

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state, p_uganda), false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)
    reported_state ~ to_submodel(
        reported_cases_submodel(reported_cases, growth_state, k, p_drc;
            report_delay = report_delay,
            test_positivity = test_positivity), false)
    ## Include the confirmed submodel when there is data to condition
    ## on or when the caller explicitly opts in to predictive sampling
    ## (`predict_confirmed = true`). Older snapshot fits with no
    ## confirmed data should leave it off so NUTS does not attempt to
    ## sample the discrete `tests_analysed` and `confirmed_cases`
    ## variables.
    if confirmed_cases !== missing || predict_confirmed
        confirmed_state ~ to_submodel(
            confirmed(confirmed_cases, cumulative_tests_analysed,
                reported_state.bvd_reported_at, growth_state, k,
                reported_state.λ_bg, reported_state.τ_test;
                lab_delay = lab_delay,
                test_sensitivity = test_sensitivity),
            false)
    end
    exports_deaths_state ~ to_submodel(
        exports_deaths_model(export_deaths_daily, growth_state,
            deaths_state.CFR, deaths_state.delay_dist, p_uganda;
            pre_start_deaths = pre_start_deaths,
            window = exports_state.w,
            daily_travellers = exports_state.daily_travellers,
            source_population = source_population),
        false)
    detection_timing_state ~ to_submodel(
        exports_detection_timing(growth_state, p_uganda;
            delta = first_export_detection_delta,
            pre_detection_exports = pre_detection_exports,
            window = exports_state.w,
            daily_travellers = exports_state.daily_travellers,
            source_population = source_population),
        false)

    cumulative_cases := growth_state.C_T
end

"""
McCabe et al. reimplementation composer: exports and deaths only,
mirroring the joint configuration in the Imperial report (no
reported-cases or deaths-among-exports likelihood). Passing
`missing` for `exported_cases` reduces to a pure Method 2 (deaths-
only) fit.
"""
@model function imperial_only_model(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer};
        growth = exponential_growth_model(),
        exports = exports_model,
        deaths = deaths_model,
        dispersion = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model())
    growth_state ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state ~ to_submodel(ascertainment, false)
    k = dispersion_state.k
    p_uganda = asc_state.p_uganda

    if !ismissing(exported_cases)
        exports_state ~ to_submodel(
            exports(exported_cases, growth_state, p_uganda), false)
    end
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end
