## Smoke tests for the cases ascertainment likelihood. Exercises the
## real `cases_only_model` from `src/models/joint.jl`.

@testitem "cases_model prior draws finite reported_cases" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: cases_only_model
    chn = sample(cases_only_model(missing), Prior(), 200;
        chain_type = FlexiChains.VNChain, progress = false)
    rc = vec(Array(chn[:reported_cases]))
    @test length(rc) == 200
    @test all(isfinite, rc)
    @test all(rc .>= 0)

    p_drc = vec(Array(chn[:p_drc]))
    @test all(0 .<= p_drc .<= 1)
end

@testitem "cases_only_model fits a tiny observation" tags=[:slow] begin
    using Turing: sample, Prior
    import FlexiChains
    using BVDOutbreakSize: cases_only_model
    chn = sample(cases_only_model(50), Prior(), 200;
        chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end
