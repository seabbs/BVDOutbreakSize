## Tests for the genetic seeding submodel. The `@model` block lives in
## the literate walkthrough, so we recreate it here to keep the tests
## self-contained. The submodel adds a one-sided soft lower bound on the
## seeding time `T` via `@addlogprob!`, leaving `T` unpenalised above the
## genetic TMRCA bound and decaying below it.

using Distributions: Normal, logcdf, truncated
using Turing: Turing, @model, @addlogprob!, sample, Prior, to_submodel,
              logjoint

@model function _genetic_seeding(T; lower_days::Real, width::Real)
    @addlogprob! logcdf(Normal(0.0, width), T - lower_days)
    return (; lower_days, width)
end

@testset "genetic_seeding adds the soft lower-bound log density" begin
    lower, width = 80.0, 20.0
    at(T) = logjoint(_genetic_seeding(T; lower_days = lower, width), (;))

    ## Matches the analytic one-sided penalty.
    @test at(112.0) ≈ logcdf(Normal(0.0, width), 112.0 - lower)
    @test at(56.0) ≈ logcdf(Normal(0.0, width), 56.0 - lower)

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
    seed = T -> _genetic_seeding(T; lower_days = 80.0, width = 20.0)
    chn = sample(_seed_growth(seed), Prior(), 50; progress = false)
    T_draws = vec(Array(chn[:T]))
    @test length(T_draws) == 50
    @test all(isfinite, T_draws)
end
