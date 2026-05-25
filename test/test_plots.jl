## Smoke tests for the three plotting functions. We do not compare
## pixels — we only check that each call returns a renderable object
## without throwing. CairoMakie is activated headless in runtests.jl.

@model function _plot_model()
    a ~ Normal(0.0, 1.0)
    b ~ Normal(2.0, 0.5)
end

@testset "plot_cumulative_cases returns a figure-grid" begin
    rng = MersenneTwister(4)
    a = randn(rng, 300) .* 50 .+ 400
    b = randn(rng, 300) .* 80 .+ 600
    fg = plot_cumulative_cases("fit A" => a, "fit B" => b; xmax = 1_500)
    @test fg !== nothing
    # AlgebraOfGraphics.draw returns a FigureGrid wrapping a Makie Figure.
    @test fg.figure isa CairoMakie.Makie.Figure
end

@testset "plot_density_overlay returns a figure-grid" begin
    rng = MersenneTwister(14)
    a = randn(rng, 300) .* 5 .+ 50
    b = randn(rng, 300) .* 5 .+ 40
    fg = plot_density_overlay("fit A" => a, "fit B" => b;
        xlabel = "Seeding time", title = "by clock rate")
    @test fg !== nothing
    @test fg.figure isa CairoMakie.Makie.Figure
end

@testset "plot_posterior_predictive returns a Makie figure" begin
    rng = MersenneTwister(5)
    pp_exports = rand(rng, 0:10, 500)
    pp_deaths = rand(rng, 0:5, 500)
    fig = plot_posterior_predictive(pp_exports, pp_deaths, 3, 1)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_posterior_predictive lays out four streams" begin
    rng = MersenneTwister(15)
    fig = plot_posterior_predictive(
        rand(rng, 0:10, 400), rand(rng, 0:60, 400), 3, 40;
        pp_cases = rand(rng, 0:30, 400), obs_cases = 20,
        pp_exports_deaths = rand(rng, 0:3, 400), obs_exports_deaths = 1)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_prior_predictive returns a Makie figure" begin
    rng = MersenneTwister(6)
    pp_exports = rand(rng, 0:10, 500)
    pp_deaths = rand(rng, 0:5, 500)
    fig = plot_prior_predictive(pp_exports, pp_deaths, 3, 1)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_posterior_predictive_grid lays out four columns" begin
    rng = MersenneTwister(17)
    streams = (; exports        = rand(rng, 0:10, 300),
                 exports_deaths = rand(rng, 0:3, 300),
                 deaths         = rand(rng, 0:60, 300),
                 cases          = rand(rng, 0:30, 300))
    observed = (; exports = 2, exports_deaths = 1,
                  deaths = 40, cases = 20)
    fig = BVDOutbreakSize.plot_posterior_predictive_grid(;
        individual = streams, joint = streams, observed = observed)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_pair returns a renderable object" begin
    chn = sample(_plot_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    obj = plot_pair(chn, [:a, :b]; thin = 4)
    @test obj !== nothing
end

@testset "plot_pair overlays a prior series" begin
    chn = sample(_plot_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    obj = plot_pair(chn, [:a, :b]; thin = 4, prior = chn)
    @test obj !== nothing
end

@testset "plot_estimate_comparison returns a Makie figure" begin
    rows = [
        ("Source A", 313, 39, 870),
        ("Source B", 501, 402, 612),
        ("Our model", 240, 150, 400),
    ]
    fig = plot_estimate_comparison(rows)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_start_date_pair returns a Makie figure" begin
    rng = MersenneTwister(16)
    n = 200
    vals = hcat(abs.(randn(rng, n)) .+ 7, abs.(randn(rng, n)) .* 30)
    chn = FlexiChains.FlexiChain{Symbol}(n, 1, Dict(
        FlexiChains.Parameter(:τ) => reshape(vals[:, 1], n, 1),
        FlexiChains.Parameter(:T) => reshape(vals[:, 2], n, 1)))
    fig = plot_start_date_pair(chn; as_of_date = "2026-05-20")
    @test fig isa CairoMakie.Makie.Figure
end
