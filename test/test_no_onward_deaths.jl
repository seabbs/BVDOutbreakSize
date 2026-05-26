## Smoke tests for predict_no_onward_deaths and plot_no_onward_deaths.
## A tiny synthetic Turing model produces a chain with the required
## parameter names (:r, :T, :α, :θ, :CFR). We check the DataFrame
## return contract and basic sanity (non-negative, finite, and
## total_projected = obs_deaths + delta_deaths).

@testitem "predict_no_onward_deaths returns the documented columns" tags=[:slow] begin
    using DataFrames: DataFrame, nrow
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: deaths_only_model, predict_no_onward_deaths

    chn = sample(deaths_only_model(missing), Prior(), 100;
                 chain_type = FlexiChains.VNChain, progress = false)
    obs_deaths = 88
    df = predict_no_onward_deaths(chn; obs_deaths = obs_deaths)

    @test df isa DataFrame
    @test sort(names(df)) == sort(["delta_deaths", "total_projected"])
    @test nrow(df) == 100

    @test all(isfinite, df.delta_deaths)
    @test all(isfinite, df.total_projected)
    @test all(df.delta_deaths .>= 0)
    @test all(df.total_projected .>= obs_deaths)
    @test maximum(abs.(df.total_projected .- (obs_deaths .+ df.delta_deaths))) < 1e-8
end

@testitem "plot_no_onward_deaths returns a renderable figure-grid" tags=[:slow] setup=[HeadlessMakie] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: deaths_only_model,
                           predict_no_onward_deaths, plot_no_onward_deaths

    chn = sample(deaths_only_model(missing), Prior(), 80;
                 chain_type = FlexiChains.VNChain, progress = false)
    df = predict_no_onward_deaths(chn; obs_deaths = 50)
    fg = plot_no_onward_deaths(df; obs_deaths = 50)
    @test fg !== nothing
    @test fg isa CairoMakie.Makie.Figure
end
