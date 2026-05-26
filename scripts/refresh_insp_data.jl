#!/usr/bin/env julia
#
# Refresh DRC suspected case and death totals from the INRB-UMIE
# transcription of the INSP situation reports
# (https://github.com/INRB-UMIE/Ebola_DRC_2026), and print the values
# in TOML form ready to drop into `data/observations.toml`.
#
# Usage:
#
#   julia --project=. scripts/refresh_insp_data.jl
#   julia --project=. scripts/refresh_insp_data.jl 2026-05-23  # pin a cut-off
#
# With no argument the cut-off is the latest sitrep date in the file.
# Pass an ISO date to pin a specific cut-off (useful when the latest
# vintage has downward revisions, as 2026-05-24 did for Goma and Katwa).
#
# Reads the raw CSVs into a DataFrame, drops "ND" entries (incomplete
# zone backfill) rather than treating them as zero, and sums across
# all reporting health zones for each sitrep date.

using CSV
using DataFrames
using Dates
using Downloads

const BASE_URL = "https://raw.githubusercontent.com/INRB-UMIE/" *
                 "Ebola_DRC_2026/main/data/insp_sitrep/processed"
const CASES_FILE  = "insp_sitrep__cumulative_suspected_cases__daily.csv"
const DEATHS_FILE = "insp_sitrep__cumulative_suspected_deaths__daily.csv"

function load_insp(file::AbstractString)
    url = "$BASE_URL/$file"
    tmp = Downloads.download(url)
    df = CSV.read(tmp, DataFrame; missingstring = ["ND"])
    value_col = setdiff(names(df), ["nom", "date"])[1]
    rename!(df, value_col => :value)
    df.date = Date.(df.date)
    return df
end

function trajectory(df::DataFrame)
    grouped = groupby(dropmissing(df, :value), :date)
    out = combine(grouped, :value => sum => :total,
                  nrow => :n_zones_reporting)
    sort!(out, :date)
    return out
end

function format_trajectory(rows::DataFrame, cut_off::Date)
    kept = filter(:date => <=(cut_off), rows)
    dates = ["\"$(d)\"" for d in kept.date]
    vals  = string.(kept.total)
    return (dates_str = "[" * join(dates, ", ") * "]",
            values_str = "[" * join(vals, ", ") * "]",
            kept = kept)
end

cut_off = length(ARGS) >= 1 ? Date(ARGS[1]) : Date(0)

cases  = trajectory(load_insp(CASES_FILE))
deaths = trajectory(load_insp(DEATHS_FILE))

cut_off == Date(0) && (cut_off = min(maximum(cases.date),
                                     maximum(deaths.date)))

cases_at = filter(:date => ==(cut_off), cases)
deaths_at = filter(:date => ==(cut_off), deaths)
isempty(cases_at)  && error("no cases vintage on $cut_off")
isempty(deaths_at) && error("no deaths vintage on $cut_off")

println("Cut-off: $cut_off")
println()
println("Cases trajectory (date, total, n_zones_reporting):")
show(stdout, MIME("text/plain"), cases); println()
println()
println("Deaths trajectory (date, total, n_zones_reporting):")
show(stdout, MIME("text/plain"), deaths); println()
println()

cases_fmt = format_trajectory(cases, cut_off)

println("--- TOML snippet for data/observations.toml ---")
println()
println("as_of_date = \"$cut_off\"")
println()
println("[total_deaths]")
println("value  = $(deaths_at.total[1])")
println()
println("[reported_cases]")
println("value  = $(cases_at.total[1])")
println()
println("[reported_case_history]")
println("dates  = $(cases_fmt.dates_str)")
println("values = $(cases_fmt.values_str)")
