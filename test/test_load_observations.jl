## Tests for load_observations: returns the documented fields from
## the bundled data/observations.toml file.

@testset "load_observations returns the documented fields" begin
    obs = load_observations()
    @test obs isa NamedTuple
    @test obs.exported_cases isa Integer
    @test obs.total_deaths isa Integer
    @test obs.source_population isa Integer
    @test obs.daily_outbound_travellers isa Real
    @test obs.daily_outbound_travellers_sd isa Real

    @test obs.exported_cases >= 0
    @test obs.total_deaths >= 0
    @test obs.daily_outbound_travellers > 0
    @test obs.daily_outbound_travellers_sd > 0
    @test obs.source_population > 0
end
