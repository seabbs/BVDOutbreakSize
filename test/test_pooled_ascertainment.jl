## Smoke tests for the pooled DRC / Uganda ascertainment submodel.
## The `@model` blocks live in the literate walkthrough, so we
## recreate the minimal set here to keep the tests self-contained
## and avoid a dependency on the doc-build pipeline.

@model function _pooled_test(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0.0, 0.5); lower = 1e-4))
    μ_logit  ~ mu_prior
    τ_logit  ~ tau_prior
    logit_p_drc    ~ Normal(μ_logit, τ_logit)
    logit_p_uganda ~ Normal(μ_logit, τ_logit)
    p_drc    := logistic(logit_p_drc)
    p_uganda := logistic(logit_p_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

@model function _pooled_test_compose()
    asc ~ to_submodel(_pooled_test(), false)
    p_drc_outer    := asc.p_drc
    p_uganda_outer := asc.p_uganda
    return asc
end

@testset "pooled_ascertainment prior draws produce p ∈ (0, 1)" begin
    chn = sample(_pooled_test(), Prior(), 200;
                 chain_type = MCMCChains.Chains, progress = false)
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

@testset "pooled_ascertainment composes via to_submodel" begin
    chn = sample(_pooled_test_compose(), Prior(), 100;
                 chain_type = MCMCChains.Chains, progress = false)
    p_drc    = vec(Array(chn[:p_drc_outer]))
    p_uganda = vec(Array(chn[:p_uganda_outer]))
    @test length(p_drc) == 100
    @test all(0 .< p_drc .< 1)
    @test all(0 .< p_uganda .< 1)
end
