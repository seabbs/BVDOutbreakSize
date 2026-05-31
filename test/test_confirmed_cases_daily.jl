## Smoke tests for the per-vintage confirmed-cases likelihood exercised
## through `confirmed_only_model` with a history `(; days, counts)`.

@testitem "confirmed_cases: prior draws are finite and non-negative" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: confirmed_only_model

    history = (; days = [13, 18, 23], counts = [9, 17, 27])
    chn = sample(
        confirmed_only_model(23, missing; confirmed_history = history),
        Prior(), 100;
        chain_type = FlexiChains.VNChain, progress = false
    )

    C_T = vec(Array(chn[:C_T]))
    @test length(C_T) == 100
    @test all(isfinite, C_T)
    @test all(C_T .> 0)
end

@testitem "confirmed_cases: tiny fit with history and total stays positive" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: confirmed_only_model

    history = (; days = [13, 18, 23], counts = [9, 17, 27])
    chn = sample(
        confirmed_only_model(23, 27; confirmed_history = history),
        Prior(), 100;
        chain_type = FlexiChains.VNChain, progress = false
    )
    C_T = vec(Array(chn[:C_T]))
    @test length(C_T) == 100
    @test all(isfinite, C_T)
    @test all(C_T .> 0)
end
