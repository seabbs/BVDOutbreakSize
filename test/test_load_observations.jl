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

    ## First-export-death offset: days from the recorded death date to
    ## the cut-off. The bundled data has death 2026-05-14, cut-off
    ## 2026-05-18, so the offset is 4 days.
    @test obs.first_export_death_delta == 4
end

@testset "first_export_death_delta is missing when the date is absent" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(path, "w") do io
            write(io, """
                as_of_date = "2026-05-18"
                [exported_cases]
                value = 2
                source = "x"
                [exports_deaths]
                value = 1
                source = "x"
                [total_deaths]
                value = 131
                source = "x"
                [reported_cases]
                value = 516
                source = "x"
                [daily_outbound_travellers]
                value = 1871
                source = "x"
                [daily_outbound_travellers_sd]
                value = 200
                source = "x"
                [source_population]
                value = 4392200
                source = "x"
                """)
        end
        obs = load_observations(path)
        @test ismissing(obs.first_export_death_delta)
    end
end
