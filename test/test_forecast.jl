## Smoke tests for the one-week-ahead forecast. Builds a tiny
## synthetic chain carrying the parameters `forecast_reported` reads,
## then checks the returned DataFrame contract.

@testsnippet ForecastFixtures begin
    using Turing: Turing, @model, sample, Prior
    using Distributions: Beta, Normal, truncated
    import FlexiChains
    using BVDOutbreakSize: bvd_joint

    ## Synthetic prior carrying every parameter name that
    ## `forecast_reported` reads. `include_lab = true` adds the
    ## lab-turnaround delay and PCR sensitivity draws so the
    ## confirmed-cases columns are populated.
    @model function _forecast_test(; include_lab::Bool = false)
        r ~ truncated(Normal(0.05, 0.01); lower = 1e-3)
        T ~ truncated(Normal(100.0, 10.0); lower = 1.0)
        CFR ~ Beta(6.0, 14.0)
        α ~ truncated(Normal(4.3, 0.5); lower = 0.5)
        θ ~ truncated(Normal(2.6, 0.3); lower = 0.2)
        w ~ truncated(Normal(15.0, 2.0); lower = 1.0)
        p_drc ~ Beta(2.0, 6.0)
        p_uganda ~ Beta(2.0, 6.0)
        inv_sqrt_k ~ truncated(Normal(0.0, 1.0); lower = 1e-3)
        k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
        α_rep ~ truncated(Normal(4.0, 0.5); lower = 0.5)
        θ_rep ~ truncated(Normal(3.0, 0.3); lower = 0.2)
        λ_bg ~ truncated(Normal(0.0, 10.0); lower = 0)
        if include_lab
            α_lab ~ truncated(Normal(2.0, 0.5); lower = 0.5)
            θ_lab ~ truncated(Normal(1.5, 0.3); lower = 0.2)
            s_test ~ Beta(15.0, 2.0)
        end
        return nothing
    end

    _forecast_chain(n;
        include_lab::Bool = false) = sample(_forecast_test(; include_lab), Prior(), n;
        chain_type = FlexiChains.VNChain, progress = false)
end

@testitem "forecast_reported returns the documented columns" tags=[:slow] setup=[ForecastFixtures] begin
    using DataFrames: DataFrame, nrow
    using BVDOutbreakSize: forecast_reported

    chn=_forecast_chain(200)

    fc=forecast_reported(chn;
        horizon = 7,
        daily_travellers = 1871,
        source_population = 4_392_200,
        obs_cases = 514,
        obs_deaths = 136,
        obs_exports = 2)

    @test fc isa DataFrame
    @test nrow(fc) == 200
    cols=[:cases_cum, :deaths_cum, :exports_cum,
        :cases_new, :deaths_new, :exports_new]
    @test all(c -> c in propertynames(fc), cols)
    ## Counts are non-negative integers.
    @test all(fc.cases_cum .>= 0)
    @test all(fc.deaths_cum .>= 0)
    @test all(fc.exports_cum .>= 0)
    @test all(fc.cases_new .>= 0)
    ## New-this-week cannot exceed the cumulative forecast.
    @test all(fc.cases_new .<= fc.cases_cum)
    @test all(fc.deaths_new .<= fc.deaths_cum)
    ## No lab-turnaround draws in this fixture → no confirmed columns.
    @test !(:confirmed_cum in propertynames(fc))
end

@testitem "forecast_reported adds confirmed columns when lab delay sampled" tags=[:slow] setup=[ForecastFixtures] begin
    using DataFrames: DataFrame
    using BVDOutbreakSize: forecast_reported

    chn=_forecast_chain(200; include_lab = true)
    fc=forecast_reported(chn;
        horizon = 7, daily_travellers = 1871,
        source_population = 4_392_200,
        obs_cases = 514, obs_deaths = 136, obs_exports = 2,
        obs_confirmed = 33)

    @test :confirmed_cum in propertynames(fc)
    @test :confirmed_new in propertynames(fc)
    @test all(fc.confirmed_cum .>= 0)
    @test all(fc.confirmed_new .>= 0)
    @test all(fc.confirmed_new .<= fc.confirmed_cum)
end

@testitem "forecast_table and plot_forecast" tags=[:slow] setup=[
    ForecastFixtures, HeadlessMakie] begin
    using DataFrames: DataFrame, nrow
    using BVDOutbreakSize: forecast_reported, forecast_table, plot_forecast

    chn=_forecast_chain(200)
    fc=forecast_reported(chn;
        horizon = 7, daily_travellers = 1871,
        source_population = 4_392_200,
        obs_cases = 514, obs_deaths = 136, obs_exports = 2)

    tbl=forecast_table(fc)
    @test tbl isa DataFrame
    ## Three streams × two quantities (cumulative, new this week).
    @test nrow(tbl) == 6
    @test names(tbl) ==
          ["Stream", "Quantity", "Lower 90%", "Lower 60%", "Lower 30%",
        "Upper 30%", "Upper 60%", "Upper 90%"]
    @test Set(tbl[!, "Quantity"]) ==
          Set(["cumulative by T+7", "new this week"])

    fig=plot_forecast(fc)
    @test fig !== nothing
end

@testitem "forecast_vs_truth compares cumulative forecast to observed" tags=[:slow] setup=[
    ForecastFixtures, HeadlessMakie] begin
    using DataFrames: DataFrame, nrow
    using BVDOutbreakSize: forecast_reported, forecast_vs_truth, plot_forecast_vs_truth

    chn=_forecast_chain(200)
    fc=forecast_reported(chn;
        horizon = 7, daily_travellers = 1871,
        source_population = 4_392_200,
        obs_cases = 514, obs_deaths = 136, obs_exports = 2)

    tbl=forecast_vs_truth(fc;
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
        covered=row["Lower 90%"]<=row.Observed<=row["Upper 90%"]
        @test row["Within 90% PI"] == (covered ? "yes" : "no")
    end

    fig=plot_forecast_vs_truth(fc; cases = 600, deaths = 150, exports = 3)
    @test fig !== nothing
end
