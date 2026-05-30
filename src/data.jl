# Observation data loading from a TOML manifest. The renewal model takes
# the cut-off grid length, the per-stream cumulative totals, and the
# per-vintage histories with their day grids; this loader returns them in
# one named tuple together with the first WHO situation-report offset used
# as the intervention breakpoint.

"""
Load the BVD observation manifest from `path` (a TOML file). Returns a
named tuple with the cut-off grid length `n`, the cut-off and seeding
dates, the per-stream cumulative totals, the per-vintage histories with
their day grids, the genetic TMRCA bound, and the first WHO
situation-report offset `who_first_sitrep_days` (days from that report to
the cut-off, inclusive). The intervention breakpoint grid day is
`n - who_first_sitrep_days`.
"""
function load_observations(
        path::AbstractString = joinpath(@__DIR__, "..", "data",
        "observations.toml"))
    raw = TOML.parsefile(path)
    cutoff = Date(raw["cutoff_date"])
    seeding = Date(raw["seeding_date"])
    n = Int(date2epochdays(cutoff) - date2epochdays(seeding)) + 1
    who_first = Date(raw["who_first_sitrep_date"])
    who_first_sitrep_days = Int(date2epochdays(cutoff) - date2epochdays(who_first)) + 1
    streams = raw["streams"]
    histories = get(raw, "histories", Dict{String, Any}())
    function history(key)
        h = get(histories, key, nothing)
        h === nothing && return (; days = Int[], counts = Int[])
        return (; days = Int.(h["days"]), counts = Int.(h["counts"]))
    end
    return (; n, cutoff, seeding,
        exported_cases = Int(streams["exported_cases"]),
        total_deaths = Int(streams["total_deaths"]),
        reported_cases = Int(streams["reported_cases"]),
        confirmed_cases = Int(streams["confirmed_cases"]),
        exports_deaths = Int(streams["exports_deaths"]),
        deaths_history = history("deaths"),
        reported_history = history("reported"),
        confirmed_history = history("confirmed"),
        lab_history = history("lab"),
        tmrca_days = get(raw, "tmrca_days", missing),
        who_first_sitrep_days)
end
