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

@testset "plot_posterior_predictive returns a Makie figure" begin
    rng = MersenneTwister(5)
    pp_exports = rand(rng, 0:10, 500)
    pp_deaths = rand(rng, 0:5, 500)
    fig = plot_posterior_predictive(pp_exports, pp_deaths, 3, 1)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_prior_predictive returns a Makie figure" begin
    rng = MersenneTwister(6)
    pp_exports = rand(rng, 0:10, 500)
    pp_deaths = rand(rng, 0:5, 500)
    fig = plot_prior_predictive(pp_exports, pp_deaths, 3, 1)
    @test fig isa CairoMakie.Makie.Figure
end

@testset "plot_pair returns a renderable object" begin
    chn = sample(_plot_model(), Prior(), 200;
                 chain_type = MCMCChains.Chains, progress = false)
    obj = plot_pair(chn, [:a, :b]; thin = 4)
    @test obj !== nothing
end
