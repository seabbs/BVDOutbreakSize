## Smoke tests for predict_no_onward_deaths and plot_no_onward_deaths.
## A tiny synthetic Turing model produces a chain with the required
## parameter names (:r, :T, :α, :θ, :CFR). We check the DataFrame
## return contract and basic sanity (non-negative, finite, and
## total_projected = obs_deaths + delta_deaths).

using BVDOutbreakSize: predict_no_onward_deaths, plot_no_onward_deaths
using Distributions: Gamma, Beta, truncated
using DataFrames: DataFrame, nrow

@model function _no_onward_synthetic()
    r   ~ truncated(Normal(0.05, 0.02); lower = 1e-3)
    T   ~ truncated(Normal(60.0, 10.0); lower = 14.0, upper = 180.0)
    α   ~ truncated(Normal(4.3, 1.0);   lower = 0.5)
    θ   ~ truncated(Normal(2.6, 0.6);   lower = 0.2)
    CFR ~ Beta(6.0, 14.0)
end

@testset "predict_no_onward_deaths returns the documented columns" begin
    chn = sample(_no_onward_synthetic(), Prior(), 100;
                 chain_type = MCMCChains.Chains, progress = false)
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

@testset "plot_no_onward_deaths returns a renderable figure-grid" begin
    chn = sample(_no_onward_synthetic(), Prior(), 80;
                 chain_type = MCMCChains.Chains, progress = false)
    df = predict_no_onward_deaths(chn; obs_deaths = 50)
    fg = plot_no_onward_deaths(df; obs_deaths = 50)
    @test fg !== nothing
    @test fg isa CairoMakie.Makie.Figure
end
