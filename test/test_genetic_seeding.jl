## Tests for the genetic seeding submodel. The `@model` block lives in
## the literate walkthrough, so we recreate it here to keep the tests
## self-contained. The submodel encodes the molecular-clock TMRCA as an
## upper-censored, noisy reading of the seeding time `T`: a soft one-
## sided bound, flat above the TMRCA `g` and decaying below it.

using Distributions: Normal, logcdf, censored, truncated
using Turing: Turing, @model, sample, Prior, to_submodel, logjoint

@model function _genetic_seeding(T, tmrca_days::Real; width::Real)
    tmrca_days ~ censored(Normal(T, width); upper = tmrca_days)
    return (; tmrca_days, width)
end

@testset "genetic_seeding adds the soft lower-bound log density" begin
    ## g = 80 days is the molecular-clock TMRCA: the root-to-tip genetic
    ## distance of the first sequenced genomes divided by a 1.2e-3
    ## substitutions/site/year clock rate. It is taken here as ~80 days
    ## before the 18 May 2026 data cut-off (TMRCA date 2026-02-27), not
    ## relative to the run date. This approximates the TMRCA as measured
    ## from the cut-off rather than from the individual sequence sampling
    ## dates (6-16 May 2026); since those fall within ~2 weeks of the
    ## cut-off the approximation is small. width = 20 days is the SD of
    ## the one-sided soft bound.
    g, width = 80.0, 20.0
    at(T) = logjoint(_genetic_seeding(T, g; width), (;))

    ## Matches the analytic one-sided (censored) likelihood P(read ≥ g).
    @test at(112.0) ≈ logcdf(Normal(0.0, width), 112.0 - g)
    @test at(56.0) ≈ logcdf(Normal(0.0, width), 56.0 - g)

    ## Monotone increasing in T: earlier seeding is penalised more.
    @test at(40.0) < at(80.0) < at(160.0)

    ## Flat well above the bound, steep decay well below it.
    @test at(200.0) - at(160.0) < at(40.0) - at(20.0)
end

@model function _seed_growth(genetic)
    m ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
    τ := 14.0
    T := m * τ
    if genetic !== nothing
        genetic_state ~ to_submodel(genetic(T), false)
    end
    return (; T)
end

@testset "genetic_seeding composes into a growth model via to_submodel" begin
    seed = T -> _genetic_seeding(T, 80.0; width = 20.0)
    chn = sample(_seed_growth(seed), Prior(), 50; progress = false)
    T_draws = vec(Array(chn[:T]))
    @test length(T_draws) == 50
    @test all(isfinite, T_draws)
end
