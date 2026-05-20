# Experiment: does the first-export-death survival term move the T / r
# posterior? Fits the joint model with and without the term and compares.
# Reconstructs the submodel stack from docs/examples/analysis.jl (the
# @model blocks live in the literate doc) so this runs without the full
# doc-build pipeline. Reduced draws for speed.

using BVDOutbreakSize
using BVDOutbreakSize: expected_deaths, expected_exports,
                       expected_exports_deaths,
                       integrate_cumulative, load_observations,
                       nuts_sample, posterior_summary
using Turing: Turing, @model, to_submodel, Poisson
using Distributions: Gamma, Normal, Beta, LogNormal, truncated
using StatsFuns: logit, logistic

@model function growth_m(; tau_prior = LogNormal(log(14), 0.4),
        m_prior = truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0))
    τ ~ tau_prior
    m ~ m_prior
    r := log(2) / τ
    T := m * τ
    C_T := 2.0^m
    cumulative = s -> exp(r * s)
    return (; τ, r, m, T, C_T, cumulative)
end

@model function delay_m(; alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

@model cfr_m() = begin CFR ~ Beta(6.0, 14.0); return (; CFR) end

@model function window_m(; window_prior = truncated(Normal(15.0, 5.0); lower = 0))
    w ~ window_prior
    return (; w)
end

@model function traveller_m(; mean = ITURI_DAILY_TRAVEL, sd = ITURI_DAILY_TRAVEL_SD)
    daily_travellers ~ truncated(Normal(mean, sd); lower = 0)
    return (; daily_travellers)
end

@model function dispersion_m(; inv_sqrt_k_prior = truncated(Normal(0, 1); lower = 0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

@model function pooled_m(; mu_prior = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0, 0.5); lower = 1e-4))
    μ_logit ~ mu_prior
    τ_logit ~ tau_prior
    logit_p_drc    ~ Normal(μ_logit, τ_logit)
    logit_p_uganda ~ Normal(μ_logit, τ_logit)
    p_drc := logistic(logit_p_drc)
    p_uganda := logistic(logit_p_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

safe_nb(k, μ) = begin
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ? clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    Turing.NegativeBinomial(k, p)
end

@model function exports_m(exported_cases, growth_state, p_uganda;
        source_population = ITURI_POPULATION,
        window = window_m(), traveller = traveller_m())
    cumulative = growth_state.cumulative
    T = growth_state.T
    window_state ~ to_submodel(window, false); w = window_state.w
    travel_state ~ to_submodel(traveller, false)
    daily_travellers = travel_state.daily_travellers
    integral := integrate_cumulative(cumulative, max(T - w, zero(T)), T)
    μ_e := max(p_uganda * (daily_travellers / source_population) * integral,
             eps(typeof(daily_travellers * one(T) * p_uganda)))
    exported_cases ~ Poisson(μ_e)
    return (; w, daily_travellers, p_uganda)
end

@model function deaths_m(total_deaths, growth_state, k;
        delay = delay_m(), cfr = cfr_m())
    r = growth_state.r; T = growth_state.T
    delay_state ~ to_submodel(delay, false)
    cfr_state ~ to_submodel(cfr, false); CFR = cfr_state.CFR
    raw = expected_deaths(CFR, r, T, delay_state.dist)
    μ_d := isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
    total_deaths ~ safe_nb(k, μ_d)
    return (; CFR, delay_dist = delay_state.dist)
end

@model function cases_m(reported_cases, growth_state, k, p_drc)
    raw = p_drc * growth_state.C_T
    μ_c := isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
    reported_cases ~ safe_nb(k, μ_c)
    return (; p_drc)
end

@model function xd_m(exports_deaths, growth_state, CFR, delay_dist, p_uganda;
        window, daily_travellers, source_population = ITURI_POPULATION)
    q = daily_travellers / source_population
    μ = expected_exports_deaths(growth_state.cumulative, delay_dist, CFR,
                                p_uganda, q, growth_state.T, window)
    md := μ
    exports_deaths ~ Poisson(md)
    return (;)
end

@model function timing_m(growth_state, CFR, delay_dist, p_uganda;
        delta, window, daily_travellers, source_population = ITURI_POPULATION)
    if !ismissing(delta)
        T = growth_state.T; t1 = T - delta
        q = daily_travellers / source_population
        Λ := t1 <= zero(T) ? zero(T) :
            expected_exports_deaths(growth_state.cumulative, delay_dist, CFR,
                                    p_uganda, q, t1, window)
        Turing.@addlogprob! -Λ
    end
    return (;)
end

@model function detect_timing_m(growth_state, p_uganda;
        delta, window, daily_travellers, source_population = ITURI_POPULATION)
    if !ismissing(delta)
        T = growth_state.T; t1 = T - delta
        q = daily_travellers / source_population
        Λe := t1 <= zero(T) ? zero(T) :
            expected_exports(growth_state.cumulative, p_uganda, q, t1, window)
        Turing.@addlogprob! -Λe
    end
    return (;)
end

@model function joint(ec, td, rc, xd; death_delta = missing,
        detect_delta = missing, source_population = ITURI_POPULATION)
    growth_state ~ to_submodel(growth_m(), false)
    disp_state ~ to_submodel(dispersion_m(), false); k = disp_state.k
    asc_state ~ to_submodel(pooled_m(), false)
    p_drc = asc_state.p_drc; p_uganda = asc_state.p_uganda
    exports_state ~ to_submodel(exports_m(ec, growth_state, p_uganda), false)
    deaths_state ~ to_submodel(deaths_m(td, growth_state, k), false)
    cases_state ~ to_submodel(cases_m(rc, growth_state, k, p_drc), false)
    xd_state ~ to_submodel(xd_m(xd, growth_state, deaths_state.CFR,
        deaths_state.delay_dist, p_uganda; window = exports_state.w,
        daily_travellers = exports_state.daily_travellers,
        source_population), false)
    timing_state ~ to_submodel(timing_m(growth_state, deaths_state.CFR,
        deaths_state.delay_dist, p_uganda; delta = death_delta,
        window = exports_state.w,
        daily_travellers = exports_state.daily_travellers,
        source_population), false)
    detect_state ~ to_submodel(detect_timing_m(growth_state, p_uganda;
        delta = detect_delta, window = exports_state.w,
        daily_travellers = exports_state.daily_travellers,
        source_population), false)
    cumulative_cases := growth_state.C_T
end

import Statistics
const SAMPLES = 800
const CHAINS  = 2
o = load_observations()

if get(ENV, "SMOKE", "0") == "1"
    using Turing: Prior, sample
    import MCMCChains
    c = sample(joint(o.exported_cases, o.total_deaths, o.reported_cases,
                     o.exports_deaths; death_delta = o.first_export_death_delta,
                     detect_delta = o.first_export_detection_delta),
               Prior(), 20; chain_type = MCMCChains.Chains, progress = false)
    println("SMOKE OK: drew ", length(vec(Array(c[:T]))), " T values; ",
            "Λ=", :Λ in keys(c), " Λe=", :Λe in keys(c))
    exit(0)
end
println("death_delta=", o.first_export_death_delta,
        " detect_delta=", o.first_export_detection_delta)

fit(; death_delta = missing, detect_delta = missing) = nuts_sample(
    joint(o.exported_cases, o.total_deaths, o.reported_cases,
          o.exports_deaths; death_delta, detect_delta);
    samples = SAMPLES, chains = CHAINS)

variants = (
    ("baseline",   fit()),
    ("+death",     fit(death_delta = o.first_export_death_delta)),
    ("+detection", fit(detect_delta = o.first_export_detection_delta)),
    ("+both",      fit(death_delta = o.first_export_death_delta,
                       detect_delta = o.first_export_detection_delta)),
)

function show_row(label, chn, sym)
    d = vec(Array(chn[sym]))
    s = posterior_summary(d)
    println(rpad(label, 12), rpad(string(sym), 18),
            "median=", rpad(round(Statistics.median(d); digits = 3), 9),
            "90% CI [", round(s.lo90; digits = 2), ", ",
            round(s.hi90; digits = 2), "]")
end

println("\n=== Posterior comparison (", SAMPLES, "x", CHAINS, " draws) ===")
for sym in (:T, :r, :cumulative_cases)
    for (label, chn) in variants
        show_row(label, chn, sym)
    end
    println()
end
