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
# Reads the raw CSVs, drops "ND" entries (incomplete zone backfill)
# rather than treating them as zero, and sums across all reporting
# health zones for each sitrep date.

using CSV
using Chain
using DataFrames
using DataFramesMeta
using Dates
using Downloads

const BASE_URL = "https://raw.githubusercontent.com/INRB-UMIE/" *
                 "Ebola_DRC_2026/main/data/insp_sitrep/processed"
const CASES_FILE = "insp_sitrep__cumulative_suspected_cases__daily.csv"
const DEATHS_FILE = "insp_sitrep__cumulative_suspected_deaths__daily.csv"

function trajectory(file)
    df = CSV.read(Downloads.download("$BASE_URL/$file"),
        DataFrame; missingstring = ["ND"])
    value_col = names(df)[3]
    @chain df begin
        @rename :value = $value_col
        @transform :date = Date.(:date)
        @rsubset !ismissing(:value)
        @groupby :date
        @combine :total=sum(:value) :n_zones=length(:value)
        @orderby :date
    end
end

cases = trajectory(CASES_FILE)
deaths = trajectory(DEATHS_FILE)

cut_off = length(ARGS) >= 1 ? Date(ARGS[1]) :
          min(maximum(cases.date), maximum(deaths.date))

cases_at = @subset cases :date .== cut_off
deaths_at = @subset deaths :date .== cut_off
isempty(cases_at) && error("no cases vintage on $cut_off")
isempty(deaths_at) && error("no deaths vintage on $cut_off")

cases_kept = @subset cases :date .<= cut_off

println("Cut-off: $cut_off")
println()
println("Cases trajectory (date, total, n_zones):")
show(stdout, MIME("text/plain"), cases);
println()
println()
println("Deaths trajectory (date, total, n_zones):")
show(stdout, MIME("text/plain"), deaths);
println()
println()
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
println("dates  = [", join(("\"$(d)\"" for d in cases_kept.date), ", "), "]")
println("values = [", join(cases_kept.total, ", "), "]")
