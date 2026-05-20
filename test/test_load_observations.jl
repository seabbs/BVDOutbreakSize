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

    @test obs.exported_cases >= 0
    @test obs.exports_deaths >= 0
    @test obs.total_deaths >= 0
    @test obs.reported_cases >= 0
    @test obs.daily_outbound_travellers > 0
    @test obs.daily_outbound_travellers_sd > 0
    @test obs.source_population > 0

    @test obs.sources isa NamedTuple
    @test obs.sources.exported_cases isa String
    @test obs.sources.exports_deaths isa String
    @test obs.sources.total_deaths isa String
    @test obs.sources.reported_cases isa String
    @test obs.sources.daily_outbound_travellers isa String
    @test obs.sources.daily_outbound_travellers_sd isa String
    @test obs.sources.source_population isa String

    @test !isempty(obs.sources.exported_cases)
end

## The first-export-death offset is computed at load time as the number
## of days between the recorded death date and the cut-off. Exercise the
## calculation on synthetic dates rather than asserting the bundled
## data's value, and check the date-absent path leaves it `missing`.
function _write_obs(io; as_of, death_date = nothing)
    write(io, "as_of_date = \"$as_of\"\n")
    death_date === nothing ||
        write(io, "[first_export_death_date]\nvalue = \"$death_date\"\n",
              "source = \"x\"\n")
    for k in ("exported_cases", "exports_deaths", "total_deaths",
              "reported_cases", "daily_outbound_travellers",
              "daily_outbound_travellers_sd", "source_population")
        write(io, "[$k]\nvalue = 1\nsource = \"x\"\n")
    end
end

@testset "first_export_death_delta is days from death date to cut-off" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(io -> _write_obs(io; as_of = "2026-05-18",
                              death_date = "2026-05-04"), path, "w")
        @test load_observations(path).first_export_death_delta == 14
    end
end

@testset "first_export_death_delta is missing when the date is absent" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(io -> _write_obs(io; as_of = "2026-05-18"), path, "w")
        @test ismissing(load_observations(path).first_export_death_delta)
    end
end
