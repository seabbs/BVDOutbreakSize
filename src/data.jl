# Observation loader: reads `data/observations.toml` and assembles
# the parsed counts, date offsets and citation strings into a
# `NamedTuple` consumed by the analysis pipeline.

"""
$(TYPEDSIGNATURES)

Load the observation block from `data/observations.toml` and return
it as a `NamedTuple`. Each observation in the TOML is a subtable
with `value = …` and `source = "…"`; this function returns both the
parsed numeric values and a parallel `sources::NamedTuple` of
citation strings so they can be printed alongside the data.

Fields returned:

- `exported_cases::Int`
- `exports_deaths::Int`
- `total_deaths::Int`
- `reported_cases::Int`
- `daily_outbound_travellers::Real`
- `daily_outbound_travellers_sd::Real`
- `source_population::Int`
- `genetic_tmrca_days::Union{Real, Missing}` — estimated time to the
  most recent common ancestor (TMRCA) in days before `as_of_date`, a
  soft lower bound on the seeding time `T`; `missing` when no
  `genetic_tmrca` block is present.
- `genetic_tmrca_days_sd::Union{Real, Missing}` — SD (days) on the
  location of that floor; `missing` when absent.
- `genetic_tmrca_alt_days::Union{Real, Missing}` — TMRCA (days before
  `as_of_date`) under the alternative clock rate, for the clock-rate
  sensitivity; `missing` when no `alt_date` is present.
- `genetic_tmrca_alt_days_sd::Union{Real, Missing}` — SD (days) on the
  alternative-clock floor; `missing` when absent.
- `sources::NamedTuple{(:exported_cases, :exports_deaths, :total_deaths,
  :reported_cases, :daily_outbound_travellers,
  :daily_outbound_travellers_sd, :source_population, :genetic_tmrca),
  NTuple{8, String}}` — citation per field.
"""
function load_observations(
        path::AbstractString = joinpath(@__DIR__, "..", "data",
                                        "observations.toml"))
    raw = TOML.parsefile(path)
    _val(k) = raw[k]["value"]
    _src(k) = String(raw[k]["source"])
    as_of = String(raw["as_of_date"])
    _gap(d) = date2epochdays(Date(as_of)) - date2epochdays(Date(String(d)))
    ## Days between a recorded event date and the cut-off, used as the
    ## elapsed-time offset for the timing terms. A scalar date gives a
    ## `missing` offset when absent (so its term is a no-op).
    _delta(k) = haskey(raw, k) ? _gap(_val(k)) : missing
    ## Daily export-death series, earliest dated death (index 1) to the
    ## cut-off day (offset 0, kept); empty when no dates are present.
    export_deaths_daily = if haskey(raw, "export_death_dates")
        offs = Int[_gap(d) for d in _val("export_death_dates")]
        isempty(offs) ? Int[] :
            Int[count(==(δ), offs) for δ in maximum(offs):-1:0]
    else
        Int[]
    end
    has_gen = haskey(raw, "genetic_tmrca")
    return (;
        as_of_date                   = as_of,
        exported_cases               = Int(_val("exported_cases")),
        exports_deaths               = Int(_val("exports_deaths")),
        total_deaths                 = Int(_val("total_deaths")),
        reported_cases               = Int(_val("reported_cases")),
        daily_outbound_travellers    = float(
            _val("daily_outbound_travellers")),
        daily_outbound_travellers_sd = float(
            _val("daily_outbound_travellers_sd")),
        source_population            = Int(_val("source_population")),
        export_deaths_daily          = export_deaths_daily,
        first_export_detection_delta = _delta("first_export_detection_date"),
        genetic_tmrca_days           = has_gen ?
            _gap(raw["genetic_tmrca"]["date"]) : missing,
        genetic_tmrca_days_sd        = has_gen ?
            float(raw["genetic_tmrca"]["days_sd"]) : missing,
        genetic_tmrca_alt_days       =
            has_gen && haskey(raw["genetic_tmrca"], "alt_date") ?
            _gap(raw["genetic_tmrca"]["alt_date"]) : missing,
        genetic_tmrca_alt_days_sd    =
            has_gen && haskey(raw["genetic_tmrca"], "alt_days_sd") ?
            float(raw["genetic_tmrca"]["alt_days_sd"]) : missing,
        sources = (;
            exported_cases               = _src("exported_cases"),
            exports_deaths               = _src("exports_deaths"),
            total_deaths                 = _src("total_deaths"),
            reported_cases               = _src("reported_cases"),
            daily_outbound_travellers    = _src("daily_outbound_travellers"),
            daily_outbound_travellers_sd = _src("daily_outbound_travellers_sd"),
            source_population            = _src("source_population"),
            genetic_tmrca                = has_gen ?
                String(raw["genetic_tmrca"]["source"]) : missing,
        ),
    )
end
