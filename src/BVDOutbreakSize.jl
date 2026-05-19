module BVDOutbreakSize

using Statistics: median, quantile
using Printf: @sprintf

export ReportScenarios, REPORT_SCENARIOS,
       PosteriorSummary, summarise, format_summary,
       compare_to_report, print_comparison

"""
    REPORT_SCENARIOS

The published point estimates of cumulative cases `C_T` from
McCabe et al. (Imperial, 18 May 2026), as a vector of
`(label, value)` tuples in the order they appear in Tables 1 and 2.
"""
const REPORT_SCENARIOS = [
    ("Method 1 Ituri, w=10 d",   470),
    ("Method 1 Ituri, w=15 d",   313),
    ("Method 1 Ituri, w=20 d",   235),
    ("Method 1 +N. Kivu, w=10",  617),
    ("Method 1 +N. Kivu, w=15",  412),
    ("Method 1 +N. Kivu, w=20",  309),
    ("Method 2 τ=14 d, CFR 24%", 626),
    ("Method 2 τ=14 d, CFR 30%", 501),
    ("Method 2 τ=14 d, CFR 40%", 376),
    ("Method 2 τ= 7 d, CFR 24%", 1008),
    ("Method 2 τ= 7 d, CFR 30%", 807),
    ("Method 2 τ= 7 d, CFR 40%", 605),
    ("Method 2 τ=21 d, CFR 24%", 531),
    ("Method 2 τ=21 d, CFR 30%", 425),
    ("Method 2 τ=21 d, CFR 40%", 319),
]

"""
    PosteriorSummary

Median and equal-tailed credible interval for a posterior sample.
"""
struct PosteriorSummary
    median::Float64
    lo::Float64
    hi::Float64
end

"""
    summarise(xs; q = (0.025, 0.5, 0.975))

Return a `PosteriorSummary` from a vector of posterior draws.
"""
function summarise(xs; q = (0.025, 0.5, 0.975))
    return PosteriorSummary(quantile(xs, q[2]),
                            quantile(xs, q[1]),
                            quantile(xs, q[3]))
end

"""
    format_summary(label, s; digits = 2)

One-line "label: median X 95% CrI (lo, hi)" string for printing.
"""
function format_summary(label, s::PosteriorSummary; digits::Integer = 2)
    fmt = x -> @sprintf("%.*f", digits, x)
    return string(rpad(label, 18), "median=", fmt(s.median),
                  "  95% CrI=(", fmt(s.lo), ", ", fmt(s.hi), ")")
end

"""
    compare_to_report(c_summary; scenarios = REPORT_SCENARIOS)

For each reported point estimate, return `(label, value, inside)` where
`inside` is true when `value` falls inside the joint posterior 95% CrI
for `C_T`.
"""
function compare_to_report(c_summary::PosteriorSummary;
        scenarios = REPORT_SCENARIOS)
    return [(label, val, c_summary.lo <= val <= c_summary.hi)
            for (label, val) in scenarios]
end

"""
    print_comparison(c_summary; io = stdout, scenarios = REPORT_SCENARIOS)

Print a side-by-side table of reported point estimates and whether each
falls inside the joint posterior 95% CrI for `C_T`.
"""
function print_comparison(c_summary::PosteriorSummary;
        io = stdout, scenarios = REPORT_SCENARIOS)
    println(io, "Joint posterior C_T: median ", round(c_summary.median; digits = 0),
            "  95% CrI=(", round(c_summary.lo; digits = 0),
            ", ", round(c_summary.hi; digits = 0), ")")
    println(io, rpad("Reported scenario", 28), rpad("C_T", 8), "Inside 95% CrI?")
    for (label, val, inside) in compare_to_report(c_summary; scenarios)
        marker = inside ? "yes" : "no"
        println(io, rpad(label, 28), rpad(string(val), 8), marker)
    end
end

end # module
