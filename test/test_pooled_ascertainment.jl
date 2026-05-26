## Smoke tests for the pooled DRC / Uganda ascertainment submodel.
## Exercises the real `pooled_ascertainment_model` from
## `src/models/priors.jl`.

@testsnippet PooledFixtures begin
    using Turing: @model, to_submodel
    using BVDOutbreakSize: pooled_ascertainment_model

    @model function _pooled_test_compose()
        asc ~ to_submodel(pooled_ascertainment_model(), false)
        p_drc_outer    := asc.p_drc
        p_uganda_outer := asc.p_uganda
        return asc
    end
end

@testitem "pooled_ascertainment prior draws produce p ∈ (0, 1)" tags=[:slow] setup=[PooledFixtures] begin
    using Turing: sample, Prior
    import FlexiChains
    chn = sample(pooled_ascertainment_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    p_drc    = vec(Array(chn[:p_drc]))
    p_uganda = vec(Array(chn[:p_uganda]))
    τ_logit  = vec(Array(chn[:τ_logit]))
    @test length(p_drc) == 200
    @test length(p_uganda) == 200
    @test all(0 .< p_drc .< 1)
    @test all(0 .< p_uganda .< 1)
    @test all(τ_logit .>= 0)
    @test all(isfinite, p_drc)
    @test all(isfinite, p_uganda)
end

@testitem "pooled_ascertainment composes via to_submodel" tags=[:slow] setup=[PooledFixtures] begin
    using Turing: sample, Prior
    import FlexiChains
    chn = sample(_pooled_test_compose(), Prior(), 100;
                 chain_type = FlexiChains.VNChain, progress = false)
    p_drc    = vec(Array(chn[:p_drc_outer]))
    p_uganda = vec(Array(chn[:p_uganda_outer]))
    @test length(p_drc) == 100
    @test all(0 .< p_drc .< 1)
    @test all(0 .< p_uganda .< 1)
end
