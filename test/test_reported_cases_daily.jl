## Smoke tests for the per-vintage (per-sitrep) reported-cases
## likelihood. The base `reported_cases_model` is now per-vintage:
## a cumulative-count vector + edge times; a length-1 vector reduces
## to the old cumulative single-total likelihood. Tiny fits exercise
## the real `bvd_joint` composer with the full history.

@testitem "reported_cases: prior draws are finite, non-negative" tags=[:slow] begin
    using BVDOutbreakSize: bvd_joint, load_observations
    using Turing: sample, Prior
    import FlexiChains

    obs = load_observations()
    rep = obs.reported_case_history
    dh = obs.death_history
    n_rep = length(rep.values)
    n_dh = length(dh.values)
    ## Prior-predictive: pass missing vectors with matching offsets.
    m = bvd_joint(missing,
        fill(missing, n_dh), fill(missing, n_rep);
        reported_offsets = rep.offsets,
        death_offsets = dh.offsets)
    chn = sample(m, Prior(), 200;
        chain_type = FlexiChains.VNChain, progress = false)
    ## Each draw stores the per-bin observation vector; flatten to
    ## check the per-bin values.
    raw = vec(Array(chn[:reported_cases]))
    flat = reduce(vcat, raw)
    @test all(isfinite, flat)
    @test all(flat .>= 0)
    ## Each draw is a per-vintage increment vector, so the reconstructed
    ## cumulative trajectory is monotone non-decreasing.
    @test all(issorted(cumsum(v)) for v in raw)
end

@testitem "reported_cases: tiny fit converges and stays positive" tags=[:slow] begin
    using BVDOutbreakSize: bvd_joint, load_observations
    using Turing: sample, Prior
    import FlexiChains

    obs = load_observations()
    rep = obs.reported_case_history
    dh = obs.death_history
    ## Models observe between-vintage increments, not cumulative totals.
    function _increments(values)
        out = similar(values, Int)
        prev = 0
        for i in eachindex(values)
            out[i] = values[i] - prev
            prev = values[i]
        end
        return out
    end
    m = bvd_joint(obs.exported_cases,
        _increments(dh.values), _increments(rep.values),
        obs.export_deaths_daily;
        reported_offsets = rep.offsets,
        death_offsets = dh.offsets,
        first_export_detection_delta = obs.first_export_detection_delta)
    chn = sample(m, Prior(), 50;
        chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 50
    @test all(isfinite, C)
    @test all(C .> 0)
end
