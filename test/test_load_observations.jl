## Tests for load_observations: returns the documented fields from
## the bundled data/observations.toml file.

@testset "load_observations returns the documented fields" begin
    obs = load_observations()
    @test obs isa NamedTuple
    @test obs.exported_cases isa Integer
    @test obs.exports_deaths isa Integer
    @test obs.total_deaths isa Integer
    @test obs.reported_cases isa Integer
    @test obs.source_population isa Integer
    @test obs.daily_outbound_travellers isa Real
    @test obs.daily_outbound_travellers_sd isa Real
    @test obs.genetic_tmrca_days isa Real
    @test obs.genetic_tmrca_days_sd isa Real

    @test obs.exported_cases >= 0
    @test obs.exports_deaths >= 0
    @test obs.total_deaths >= 0
    @test obs.reported_cases >= 0
    @test obs.daily_outbound_travellers > 0
    @test obs.daily_outbound_travellers_sd > 0
    @test obs.source_population > 0
    @test obs.genetic_tmrca_days > 0
    @test obs.genetic_tmrca_days_sd > 0

    @test obs.sources isa NamedTuple
    @test obs.sources.exported_cases isa String
    @test obs.sources.exports_deaths isa String
    @test obs.sources.total_deaths isa String
    @test obs.sources.reported_cases isa String
    @test obs.sources.daily_outbound_travellers isa String
    @test obs.sources.daily_outbound_travellers_sd isa String
    @test obs.sources.source_population isa String
    @test obs.sources.genetic_tmrca isa String

    @test !isempty(obs.sources.exported_cases)
    @test !isempty(obs.sources.genetic_tmrca)
end

## Export deaths are loaded as a daily series from the earliest dated
## death to the cut-off (entry i = day at offset n-i+1). Exercise the
## construction on synthetic dates rather than asserting the bundled
## data's value, and check the date-absent path returns an empty vector.
function _write_obs(io; as_of, death_dates = nothing)
    write(io, "as_of_date = \"$as_of\"\n")
    death_dates === nothing || begin
        quoted = join(("\"$d\"" for d in death_dates), ", ")
        write(io, "[export_death_dates]\nvalue = [$quoted]\n",
              "source = \"x\"\n")
    end
    for k in ("exported_cases", "exports_deaths", "total_deaths",
              "reported_cases", "daily_outbound_travellers",
              "daily_outbound_travellers_sd", "source_population")
        write(io, "[$k]\nvalue = 1\nsource = \"x\"\n")
    end
end

@testset "export_deaths_daily is a daily series to the cut-off" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(io -> _write_obs(io; as_of = "2026-05-18",
                              death_dates = ["2026-05-04", "2026-05-14"]),
             path, "w")
        daily = load_observations(path).export_deaths_daily
        ## Offsets 14 (2026-05-04) and 4 (2026-05-14); earliest = 14, so
        ## the series spans offsets 14..0 (length 15), with one death at
        ## index 1 (offset 14) and one at index 11 (offset 4).
        @test length(daily) == 15
        @test sum(daily) == 2
        @test daily[1] == 1
        @test daily[11] == 1
    end
end

@testset "export_deaths_daily is empty when no dates are present" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(io -> _write_obs(io; as_of = "2026-05-18"), path, "w")
        @test load_observations(path).export_deaths_daily == Int[]
    end
end
