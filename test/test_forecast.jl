## Smoke tests for the one-week-ahead forecast. Builds a tiny
## synthetic chain carrying the parameters `forecast_reported` reads,
## then checks the returned DataFrame contract.

using DataFrames: DataFrame, nrow
using Distributions: Normal, Gamma, Beta, truncated
using Turing: Turing, @model, sample, Prior
import FlexiChains

@model function _forecast_test()
    r          ~ truncated(Normal(0.05, 0.01); lower = 1e-3)
    T          ~ truncated(Normal(100.0, 10.0); lower = 1.0)
    CFR        ~ Beta(6.0, 14.0)
    α          ~ truncated(Normal(4.3, 0.5); lower = 0.5)
    θ          ~ truncated(Normal(2.6, 0.3); lower = 0.2)
    w          ~ truncated(Normal(15.0, 2.0); lower = 1.0)
    p_drc      ~ Beta(2.0, 6.0)
    p_uganda   ~ Beta(2.0, 6.0)
    inv_sqrt_k ~ truncated(Normal(0.0, 1.0); lower = 1e-3)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return nothing
end

@testset "forecast_reported returns the documented columns" begin
    chn = sample(_forecast_test(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)

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

@testset "forecast_table and plot_forecast" begin
    chn = sample(_forecast_test(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
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

@testset "forecast_vs_truth compares cumulative forecast to observed" begin
    chn = sample(_forecast_test(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
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
          ["Stream", "Observed", "Central estimate",
           "Lower 90%", "Upper 90%", "Within 90% PI"]
    @test Set(tbl[!, "Stream"]) ==
          Set(["DRC reported cases", "DRC deaths", "Uganda exports"])

    ## Coverage flag agrees with the reported interval endpoints.
    for row in eachrow(tbl)
        covered = row["Lower 90%"] <= row.Observed <= row["Upper 90%"]
        @test row["Within 90% PI"] == (covered ? "yes" : "no")
    end
end
