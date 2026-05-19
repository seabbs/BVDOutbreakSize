## End-to-end smoke test for nuts_sample on a trivial one-parameter
## model. 50 draws × 2 chains is enough to exercise the wiring without
## blowing the CI budget. We accept whichever sample container Turing
## returns by default (MCMCChains.Chains or FlexiChains.VNChain) and
## check shape + finite draws.

@model function _nuts_model()
    x ~ Normal(0.0, 1.0)
end

@testset "nuts_sample returns a sample container with finite draws" begin
    chn = nuts_sample(_nuts_model(); samples = 50, chains = 2)
    @test chn !== nothing

    # Extract the draws of `x` in a way that works for both the
    # MCMCChains.Chains and FlexiChains.VNChain return types.
    xs = vec(Array(chn[:x]))
    @test length(xs) == 50 * 2
    @test all(isfinite, xs)
end
