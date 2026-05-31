## Smoke tests for the confirmed-cases stream composer.
## Exercises the real `confirmed_only_model` from `src/models/joint.jl`.

@testitem "confirmed_only_model prior draws are finite and non-negative" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: confirmed_only_model

    chn = sample(
        confirmed_only_model(40, missing),
        Prior(), 100;
        chain_type = FlexiChains.VNChain, progress = false
    )

    C_T = vec(Array(chn[:C_T]))
    @test length(C_T) == 100
    @test all(isfinite, C_T)
    @test all(C_T .> 0)
end

@testitem "confirmed_only_model conditioned on an observation stays positive" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: confirmed_only_model

    chn = sample(
        confirmed_only_model(40, 20),
        Prior(), 100;
        chain_type = FlexiChains.VNChain, progress = false
    )
    C_T = vec(Array(chn[:C_T]))
    @test length(C_T) == 100
    @test all(isfinite, C_T)
    @test all(C_T .> 0)
end
