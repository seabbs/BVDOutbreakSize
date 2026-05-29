## Smoke tests for the per-vintage (per-sitrep) confirmed-cases
## likelihood. The base `confirmed_cases_model` is now per-vintage:
## a cumulative-count vector + edge times; a length-1 vector reduces
## to the cumulative single-total likelihood. Tiny fits exercise the
## real `bvd_joint` composer with the full history.

@testitem "confirmed_cases: prior draws are finite, non-negative" tags=[:slow] begin
    using BVDOutbreakSize: bvd_joint, load_observations
    using Turing: sample, Prior
    import FlexiChains

    obs = load_observations()
    rep = obs.reported_case_history
    conf = obs.confirmed_case_history
    dh = obs.death_history
    n_rep = length(rep.values)
    n_conf = length(conf.values)
    n_dh = length(dh.values)
    ## Prior-predictive: pass missing vectors with matching offsets.
    m = bvd_joint(missing,
        fill(missing, n_dh), fill(missing, n_rep);
        reported_offsets = rep.offsets,
        death_offsets = dh.offsets,
        confirmed_cases = fill(missing, n_conf),
        confirmed_offsets = conf.offsets)
    chn = sample(m, Prior(), 200;
        chain_type = FlexiChains.VNChain, progress = false)
    raw = vec(Array(chn[:confirmed_cases]))
    flat = reduce(vcat, raw)
    @test all(isfinite, flat)
    @test all(flat .>= 0)
    totals = vec(Array(chn[:confirmed_cases_total]))
    sums = [sum(v) for v in raw]
    @test all(totals .== sums)
end

@testitem "confirmed_cases: tiny fit stays positive" tags=[:slow] begin
    using BVDOutbreakSize: bvd_joint, load_observations
    using Turing: sample, Prior
    import FlexiChains

    obs = load_observations()
    rep = obs.reported_case_history
    conf = obs.confirmed_case_history
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
        confirmed_cases = _increments(conf.values),
        confirmed_offsets = conf.offsets,
        tests_analysed = obs.cumulative_tests_analysed,
        tests_offset = 0,
        first_export_detection_delta = obs.first_export_detection_delta)
    chn = sample(m, Prior(), 50;
        chain_type = FlexiChains.VNChain, progress = false)
    C = vec(Array(chn[:cumulative_cases]))
    @test length(C) == 50
    @test all(isfinite, C)
    @test all(C .> 0)
end

@testitem "DailyBVDTrajectory matches per-edge delay_convolution" tags=[:slow] begin
    using BVDOutbreakSize: DailyBVDTrajectory, delay_convolution, integrate
    using Distributions: Gamma, pdf, mean, std

    r = 0.1
    f_rep = Gamma(4.0, 3.0)
    f_lab = Gamma(2.0, 1.5)
    T_max = 30.0
    ## Unit-ascertainment precomputation; per-bin ascertainment is
    ## applied by the daily likelihood on the *returned* increment, so
    ## the struct itself is independent of `p_drc`.
    d = DailyBVDTrajectory(T_max, r, f_rep)
    edges = [10.0, 15.0, 20.0, 25.0, 30.0]
    fast = delay_convolution(d, edges, f_lab)

    ## Reference: evaluate I_lab,0(T) explicitly at each bin edge by
    ## running a fresh quadrature against the (Gamma-analytic)
    ## unit-ascertainment μ_BVD,0 curve.
    function _ref(T)
        bvd = s -> delay_convolution(1.0, r, s, f_rep)
        g = let bvd = bvd, T = T, f_lab = f_lab
            s -> begin
                d = T - s
                d <= 0 ? zero(T) : bvd(s) * pdf(f_lab, d)
            end
        end
        s_scale = mean(f_lab) + 10 * std(f_lab)
        return integrate(g, zero(T), T, s_scale)
    end
    ref = [_ref(T) for T in edges]
    ## The struct uses a uniform-map quadrature, the reference uses the
    ## clustered map: small differences are expected, so a percent-level
    ## tolerance is plenty for the smoke test.
    @test all(isfinite, fast)
    @test all(fast .> 0)
    @test maximum(abs.(fast .- ref) ./ ref) < 0.05
end
