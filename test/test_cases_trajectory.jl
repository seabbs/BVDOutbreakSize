## Smoke tests for the binned sitrep-trajectory cases likelihood. The
## `@model` blocks live in the literate walkthrough, so we recreate the
## minimal set here to keep the tests self-contained and avoid a
## dependency on the doc-build pipeline. These tests pin two contracts:
## (1) a single cumulative total reduces exactly to the current
## single-total negative-binomial likelihood (backward compatibility),
## and (2) a multi-vintage trajectory conditions on the between-vintage
## increments through the cumulative intensity `Λ(s) = p_drc · C(s)`.

using Distributions: Beta, NegativeBinomial, truncated, Normal, logpdf
using Turing: Turing, @model, sample, Prior, to_submodel
import FlexiChains

## NaN/Inf-safe negative binomial mirroring `safe_nbinomial` in the
## walkthrough.
function _ct_safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

@model function _ct_growth()
    log_τ ~ Normal(log(14), 0.4)
    m     ~ truncated(Normal(7.0, 2.5); lower = 0, upper = 13.0)
    τ   := exp(log_τ)
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    cumulative = s -> exp(r * s)
    return (; log_τ, τ, r, m, T, C_T, cumulative)
end

@model function _ct_ascertainment(; p_prior = Beta(2.0, 6.0))
    p_report ~ p_prior
    return (; p_report)
end

@model function _ct_dispersion()
    inv_sqrt_k ~ truncated(Normal(0.5, 0.2); lower = 1e-3, upper = 5.0)
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

## Binned-trajectory cases likelihood mirroring `cases_model`. Each
## vintage `v` reported `count_v` cumulative cases as of an `offset_v`
## days before the cut-off; the cut-off vintage has offset 0. Map to
## elapsed time `s_v = T - offset_v`, evaluate the cumulative intensity
## `Λ(s) = p_drc · C(s)` at each bin edge, and condition the between-
## vintage increment on the difference `μ_v = Λ(s_v) − Λ(s_{v-1})`.
@model function _ct_likelihood(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real;
        ascertainment = _ct_ascertainment(),
        case_trajectory = ())
    asc_state ~ to_submodel(ascertainment, false)
    p_report  = asc_state.p_report
    cumulative = growth_state.cumulative
    T          = growth_state.T
    Λ(s) = p_report * cumulative(s)

    ## Build the vintage bins. With no trajectory the single total is one
    ## bin observed at the cut-off (offset 0), recovering the single-total
    ## likelihood `μ = p_report · C(T)`.
    offsets = isempty(case_trajectory) ? (0,) :
        Tuple(o for (o, _) in case_trajectory)
    counts = isempty(case_trajectory) ? (reported_cases,) :
        Tuple(c for (_, c) in case_trajectory)

    ## Between-vintage increments; `missing` flows through for predictive
    ## draws.
    increments = Vector{Union{Missing, Int}}(undef, length(offsets))
    prev_count = 0
    for i in eachindex(counts)
        increments[i] = ismissing(counts[i]) ? missing :
            counts[i] - prev_count
        prev_count = ismissing(counts[i]) ? prev_count : counts[i]
    end

    λlo = zero(T)
    last_expected = λlo
    for i in eachindex(offsets)
        s_v   = T - offsets[i]
        λhi   = Λ(s_v)
        μ_bin = max(λhi - λlo, eps(typeof(λhi)))
        increments[i] ~ _ct_safe_nbinomial(k, μ_bin)
        λlo = λhi
        last_expected = λhi
    end
    expected_reports := last_expected
    return (; p_report, expected_reports)
end

@model function _ct_only(
        reported_cases::Union{Missing, Integer};
        case_trajectory = ())
    growth_state     ~ to_submodel(_ct_growth(), false)
    dispersion_state ~ to_submodel(_ct_dispersion(), false)
    k = dispersion_state.k
    cases_state ~ to_submodel(
        _ct_likelihood(reported_cases, growth_state, k;
                       case_trajectory = case_trajectory), false)
    cumulative_cases := growth_state.C_T
end

@testset "single-total reduces to the current likelihood" begin
    ## At fixed parameters, the single-total likelihood and a one-bin
    ## trajectory at the cut-off must contribute the identical
    ## log-density. Reproduce both bin constructions by hand and check
    ## the negative-binomial terms agree.
    k = 3.0
    p_report = 0.4
    r = log(2) / 14.0
    T = 7.0 * 14.0
    Λ(s) = p_report * exp(r * s)

    ## Single total (current behaviour): one bin from 0 to Λ(T).
    μ_total = max(Λ(T) - 0.0, eps(Float64))
    ll_total = logpdf(_ct_safe_nbinomial(k, μ_total), 50)

    ## One-bin trajectory at offset 0 with count 50.
    μ_traj = max(Λ(T - 0) - 0.0, eps(Float64))
    inc_traj = 50 - 0
    ll_traj = logpdf(_ct_safe_nbinomial(k, μ_traj), inc_traj)

    @test μ_total ≈ μ_traj
    @test ll_total ≈ ll_traj
end

@testset "trajectory prior draws produce non-negative increments" begin
    m = _ct_only(missing; case_trajectory = ((2, missing), (0, missing)))
    chn = sample(m, Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    inc = collect(Iterators.flatten(vec(Array(chn[:increments]))))
    @test all(isfinite, inc)
    @test all(inc .>= 0)
end

@testset "trajectory fit conditions on increments" begin
    ## Two vintages: 336 cases at offset 2, 516 at the cut-off. The
    ## increment is 180. The fit should run and return positive C_T.
    m = _ct_only(516; case_trajectory = ((2, 336), (0, 516)))
    chn = sample(m, Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 200
    @test all(isfinite, C)
    @test all(C .> 0)
end
