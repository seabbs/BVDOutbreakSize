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
    @test obs.genetic_tmrca_alt_days isa Real
    @test obs.genetic_tmrca_alt_days_sd isa Real

    @test obs.exported_cases >= 0
    @test obs.exports_deaths >= 0
    @test obs.total_deaths >= 0
    @test obs.reported_cases >= 0
    @test obs.daily_outbound_travellers > 0
    @test obs.daily_outbound_travellers_sd > 0
    @test obs.source_population > 0
    @test obs.genetic_tmrca_days > 0
    @test obs.genetic_tmrca_days_sd > 0
    @test obs.genetic_tmrca_alt_days > 0
    @test obs.genetic_tmrca_alt_days_sd > 0
    ## The alternative (faster) clock dates the TMRCA more recently, so
    ## fewer days before the cut-off than the baseline estimate.
    @test obs.genetic_tmrca_alt_days < obs.genetic_tmrca_days

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

## The reported-case trajectory is loaded as `(offset, count)` tuples,
## earliest first, with the cut-off vintage at offset 0. Build the data
## file with a history block and check the offsets and ordering.
function _write_obs_history(io; as_of, dates, values)
    write(io, "as_of_date = \"$as_of\"\n")
    qd = join(("\"$d\"" for d in dates), ", ")
    qv = join(string.(values), ", ")
    write(io, "[reported_case_history]\ndates = [$qd]\n",
          "values = [$qv]\nsource = \"x\"\n")
    for k in ("exported_cases", "exports_deaths", "total_deaths",
              "reported_cases", "daily_outbound_travellers",
              "daily_outbound_travellers_sd", "source_population")
        write(io, "[$k]\nvalue = 1\nsource = \"x\"\n")
    end
end

@testset "reported_case_trajectory maps dates to offsets" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(io -> _write_obs_history(io; as_of = "2026-05-18",
                                      dates = ["2026-05-16", "2026-05-18"],
                                      values = [336, 516]), path, "w")
        traj = load_observations(path).reported_case_trajectory
        ## Earliest first: 16 May is offset 2, 18 May (cut-off) offset 0.
        @test traj == [(2, 336), (0, 516)]
    end
end

@testset "reported_case_trajectory is empty when no history present" begin
    mktempdir() do dir
        path = joinpath(dir, "obs.toml")
        open(io -> _write_obs(io; as_of = "2026-05-18"), path, "w")
        @test load_observations(path).reported_case_trajectory ==
            Tuple{Int, Int}[]
    end
end

@testset "bundled observations expose the documented trajectory" begin
    traj = load_observations().reported_case_trajectory
    @test traj == [(2, 336), (0, 516)]
end
