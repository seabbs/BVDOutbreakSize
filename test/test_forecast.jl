## Smoke tests for the one-week-ahead forecast. Builds a tiny
## synthetic chain carrying the parameters `forecast_reported` reads,
## then checks the returned DataFrame contract.

@testsnippet ForecastFixtures begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: bvd_joint

    ## A prior draw from the real joint model exposes every parameter
    ## name (:r, :T, :CFR, :α, :θ, :w, :p_drc, :p_uganda, :k) that
    ## `forecast_reported` reads.
    _forecast_chain(n) = sample(bvd_joint(missing, missing), Prior(), n;
        chain_type = FlexiChains.VNChain, progress = false)
end

@testitem "forecast_reported returns the documented columns" tags=[:slow] setup=[ForecastFixtures] begin
    using DataFrames: DataFrame, nrow
    using BVDOutbreakSize: forecast_reported

    chn = _forecast_chain(200)

    fc = forecast_reported(chn;
        horizon           = 7,
        daily_travellers  = 1871,
        source_population = 4_392_200,
        obs_cases         = 514,
        obs_deaths        = 136,
        obs_exports       = 2)

    @test fc isa DataFrame
    @test nrow(fc) == 200
    cols = [:cases_cum, :deaths_cum, :exports_cum,
            :cases_new, :deaths_new, :exports_new]
    @test all(c -> c in propertynames(fc), cols)
    ## Counts are non-negative integers.
    @test all(fc.cases_cum   .>= 0)
    @test all(fc.deaths_cum  .>= 0)
    @test all(fc.exports_cum .>= 0)
    @test all(fc.cases_new   .>= 0)
    ## New-this-week cannot exceed the cumulative forecast.
    @test all(fc.cases_new  .<= fc.cases_cum)
    @test all(fc.deaths_new .<= fc.deaths_cum)
end

@testitem "forecast_table and plot_forecast" tags=[:slow] setup=[ForecastFixtures, HeadlessMakie] begin
    using DataFrames: DataFrame, nrow
    using BVDOutbreakSize: forecast_reported, forecast_table, plot_forecast

    chn = _forecast_chain(200)
    fc = forecast_reported(chn;
        horizon = 7, daily_travellers = 1871,
        source_population = 4_392_200,
        obs_cases = 514, obs_deaths = 136, obs_exports = 2)

    tbl = forecast_table(fc)
    @test tbl isa DataFrame
    ## Three streams × two quantities (cumulative, new this week).
    @test nrow(tbl) == 6
    @test names(tbl) ==
          ["Stream", "Quantity", "Lower 90%", "Lower 60%", "Lower 30%",
           "Upper 30%", "Upper 60%", "Upper 90%"]
    @test Set(tbl[!, "Quantity"]) ==
          Set(["cumulative by T+7", "new this week"])

    fig = plot_forecast(fc)
    @test fig !== nothing
end

@testitem "forecast_vs_truth compares cumulative forecast to observed" tags=[:slow] setup=[ForecastFixtures, HeadlessMakie] begin
    using DataFrames: DataFrame, nrow
    using BVDOutbreakSize: forecast_reported, forecast_vs_truth, plot_forecast_vs_truth

    chn = _forecast_chain(200)
    fc = forecast_reported(chn;
        horizon = 7, daily_travellers = 1871,
        source_population = 4_392_200,
        obs_cases = 514, obs_deaths = 136, obs_exports = 2)

    tbl = forecast_vs_truth(fc;
        cases = 600, deaths = 150, exports = 3)

    @test tbl isa DataFrame
    ## One row per stream (cases, deaths, exports).
    @test nrow(tbl) == 3
    @test names(tbl) ==
          ["Stream", "Observed", "Lower 90%", "Lower 60%", "Lower 30%",
           "Upper 30%", "Upper 60%", "Upper 90%", "Within 90% PI"]
    @test Set(tbl[!, "Stream"]) ==
          Set(["DRC reported cases", "DRC deaths", "Uganda exports"])

    ## Coverage flag agrees with the reported interval endpoints.
    for row in eachrow(tbl)
        covered = row["Lower 90%"] <= row.Observed <= row["Upper 90%"]
        @test row["Within 90% PI"] == (covered ? "yes" : "no")
    end

    fig = plot_forecast_vs_truth(fc; cases = 600, deaths = 150, exports = 3)
    @test fig !== nothing
end
