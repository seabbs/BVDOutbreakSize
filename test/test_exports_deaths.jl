## Smoke tests for the deaths-among-exports likelihood exercised
## through `exports_deaths_only_model` from `src/models/joint.jl`.

@testitem "exports_deaths_only prior draws produce non-negative counts" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: exports_deaths_only_model

    chn = sample(
        exports_deaths_only_model(40, missing),
        Prior(), 100;
        chain_type = FlexiChains.VNChain, progress = false
    )

    C_T = vec(Array(chn[:C_T]))
    @test length(C_T) == 100
    @test all(isfinite, C_T)
    @test all(C_T .> 0)
end

@testitem "exports_deaths_only conditioned on zero observation stays positive" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: exports_deaths_only_model

    chn = sample(
        exports_deaths_only_model(40, 0),
        Prior(), 100;
        chain_type = FlexiChains.VNChain, progress = false
    )
    C_T = vec(Array(chn[:C_T]))
    @test length(C_T) == 100
    @test all(isfinite, C_T)
    @test all(C_T .> 0)
end
