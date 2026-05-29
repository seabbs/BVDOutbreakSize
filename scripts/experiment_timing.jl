# Before/after comparison for the export-death modelling change.
#
# "Before": export deaths enter as a single total count,
#   exports_deaths ~ Poisson(Λ_D(T)), with no timing information.
# "After": the time-resolved binned-Poisson model in this PR (a
#   continuous survival weight for the pre-death stretch plus a per-day
#   Poisson to the cut-off), plus the first-export-detection survival.
#
# To avoid prior drift, the real submodels are included from
# docs/examples/analysis.jl (everything up to the first fit), so growth,
# delay, CFR, window, traveller, ascertainment and dispersion priors are
# identical across the two fits; only the export-death likelihood differs.

using BVDOutbreakSize
using Turing: Turing, @model, to_submodel, Poisson, @varname
using Statistics: median

const REPO = pkgdir(BVDOutbreakSize)
let src = read(joinpath(REPO, "docs", "examples", "analysis.jl"), String)
    cut = first(findfirst("prior_chn = sample", src)) - 1
    include_string(Main, src[1:cut], "analysis_defs.jl")
end

## "Before": single-count export-deaths likelihood at the cut-off.
@model function count_xd_model(
        exports_deaths, growth_state, CFR, delay_dist, p_uganda;
        window, daily_travellers, source_population = ITURI_POPULATION)
    q = daily_travellers / source_population
    μ := expected_exports_deaths(growth_state.cumulative, delay_dist, CFR,
        p_uganda, q, growth_state.T, window)
    exports_deaths ~ Poisson(μ)
    return (;)
end

@model function count_joint(ec, td, rc, xd_total;
        source_population = ITURI_POPULATION)
    growth_state ~ to_submodel(exponential_growth_model(), false)
    dispersion_state ~ to_submodel(surveillance_dispersion_model(), false)
    asc_state ~ to_submodel(pooled_ascertainment_model(), false)
    k = dispersion_state.k
    p_drc = asc_state.p_drc
    p_uganda = asc_state.p_uganda
    exports_state ~ to_submodel(exports_model(ec, growth_state, p_uganda), false)
    deaths_state ~ to_submodel(deaths_model(td, growth_state, k), false)
    cases_state ~ to_submodel(
        reported_cases_model(rc, growth_state, k, p_drc), false)
    xd_state ~ to_submodel(
        count_xd_model(xd_total, growth_state,
            deaths_state.CFR, deaths_state.delay_dist, p_uganda;
            window = exports_state.w,
            daily_travellers = exports_state.daily_travellers,
            source_population),
        false)
    cumulative_cases := growth_state.C_T
end

before = count_joint(obs.exported_cases, obs.total_deaths,
    obs.reported_cases, obs.exports_deaths)
after = bvd_joint(obs.exported_cases, obs.total_deaths, obs.reported_cases,
    obs.export_deaths_daily;
    first_export_detection_delta =
    obs.first_export_detection_delta)

const SAMPLES = 1_000
const CHAINS = 4
chn_before = nuts_sample(before; samples = SAMPLES, chains = CHAINS)
chn_after = nuts_sample(after; samples = SAMPLES, chains = CHAINS)

function show_row(label, chn, sym)
    d = vec(Array(chn[sym]))
    s = posterior_summary(d)
    println(rpad(label, 8), rpad(string(sym), 18),
        "median=", rpad(round(median(d); digits = 3), 9),
        "90% CI [", round(s.lo90; digits = 2), ", ",
        round(s.hi90; digits = 2), "]")
end

println("\n=== Before (count) vs after (binned timing): ",
    SAMPLES, "x", CHAINS, " ===")
for sym in (:T, :r, :cumulative_cases)
    show_row("before", chn_before, sym)
    show_row("after", chn_after, sym)
    println()
end
