@testset "discretise_delay normalises and is non-negative" begin
    pmf = BVDOutbreakSize.discretise_delay(
        Distributions.Gamma(4.3, 2.6), 40)
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1e-12)
    ## Density at the origin is forced to zero (shape > 1), so the lag-0
    ## bin is the trapezoid of [0, f(1)] only.
    @test pmf[1] < pmf[2]
end

@testset "double_censored_pmf is normalised for a LogNormal primary" begin
    pmf = BVDOutbreakSize.double_censored_pmf(LogNormal(log(7.0), 0.5), 30)
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1e-12)
end

@testset "delay_lognormal_meansd_model samples a normalised PMF" begin
    m = BVDOutbreakSize.delay_lognormal_meansd_model(40;
        mean_prior = truncated(Normal(7.0, 2.0); lower = 1),
        sd_prior   = truncated(Normal(4.0, 1.5); lower = 1))
    st = m()
    @test st.delay_mean > 0
    @test st.delay_sd > 0
    @test all(>=(0), st.pmf)
    @test isapprox(sum(st.pmf), 1.0; atol = 1e-12)
    @test st.dist isa LogNormal     # exact double-censored PMF path
end

@testset "delay_meansd_model samples a normalised PMF from priors" begin
    m = BVDOutbreakSize.delay_meansd_model(40;
        mean_prior = truncated(Normal(7.0, 2.0); lower = 1),
        sd_prior   = truncated(Normal(4.0, 1.5); lower = 1))
    st = m()
    @test st.delay_mean > 0          # sampled, not fixed
    @test st.delay_sd > 0
    @test all(>=(0), st.pmf)
    @test isapprox(sum(st.pmf), 1.0; atol = 1e-12)
    ## Two independent draws differ (the delay is sampled from a prior).
    st2 = m()
    @test (st.delay_mean, st.delay_sd) != (st2.delay_mean, st2.delay_sd)
end

@testset "generation_interval_model samples mean/SD (no fixed GT)" begin
    m = BVDOutbreakSize.generation_interval_model(40)
    st = m()
    @test st.gi_mean > 0             # generation time is estimated
    @test st.gi_sd > 0
    @test all(>=(0), st.g)
    @test isapprox(sum(st.g), 1.0; atol = 1e-12)
    ## Lag 0 dropped: an infectee is infected strictly after its infector.
    @test length(st.g) == 40
end

@testset "renewal_infections seeds and grows under R > 1" begin
    g = [0.0, 0.5, 0.3, 0.2]
    Rt = fill(2.0, 10)
    I = BVDOutbreakSize.renewal_infections(Rt, g, 1.0)
    @test I[1] == 1.0
    @test all(>=(0), I)
    @test I[end] > I[2]              # sustained growth
    ## R = 0 after the seed gives no secondary infections.
    I0only = BVDOutbreakSize.renewal_infections(zeros(5), g, 3.0)
    @test I0only[1] == 3.0
    @test all(==(0), I0only[2:end])
end

@testset "convolve_delay conserves mass for a normalised delay" begin
    x = [1.0, 2.0, 3.0, 4.0]
    delay = [0.2, 0.5, 0.3]         # sums to 1
    y = BVDOutbreakSize.convolve_delay(x, delay)
    @test length(y) == length(x)
    ## A point mass delay at lag 0 returns the input unchanged.
    y0 = BVDOutbreakSize.convolve_delay(x, [1.0])
    @test y0 == x
end

@testset "weekly_knot_days pins both ends at weekly spacing" begin
    days = BVDOutbreakSize.weekly_knot_days(90; week = 7)
    @test days[1] == 1
    @test days[end] == 90            # last knot pinned to the grid end
    @test issorted(days)
    @test all(diff(days)[1:(end - 1)] .== 7)
    ## Far fewer knots than days: the ~7x dimension cut.
    @test length(days) < 90 / 5
    ## A grid that lands exactly on a weekly multiple needs no extra knot.
    @test BVDOutbreakSize.weekly_knot_days(15; week = 7) == [1, 8, 15]
end

@testset "interpolate_knots is piecewise-linear and hits the knots" begin
    knots = [0.0, 2.0, -1.0]
    knot_days = [1, 5, 9]
    daily = BVDOutbreakSize.interpolate_knots(knots, knot_days, 9)
    @test length(daily) == 9
    ## Exact at the knot days.
    @test daily[1] == 0.0
    @test daily[5] == 2.0
    @test daily[9] == -1.0
    ## Midpoint of the first segment is the average of its two knots.
    @test daily[3] ≈ 1.0
    ## Linear within a segment: constant first difference.
    seg1 = diff(daily[1:5])
    @test all(≈(seg1[1]), seg1)
end

@testset "rt_walk_model returns a weekly-knot daily Rt" begin
    m = BVDOutbreakSize.rt_walk_model(90; week = 7)
    st = m()
    @test length(st.Rt) == 90
    @test all(>(0), st.Rt)
    @test length(st.log_R) == length(st.knot_days)
    @test st.knot_days == BVDOutbreakSize.weekly_knot_days(90; week = 7)
end

@testset "onset_incidence_model stages infections → onsets" begin
    infections = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    m = BVDOutbreakSize.onset_incidence_model(infections;
        incubation_nmax = 15)
    st = m()
    @test length(st.onsets) == length(infections)
    @test all(>=(0), st.onsets)
    @test st.incubation_mean > 0
    @test st.incubation_sd > 0
    @test isapprox(sum(st.incubation_pmf), 1.0; atol = 1e-12)
end

@testset "deaths_obs_model wires CFR × onset⊛onset-to-death" begin
    onsets = collect(1.0:10.0)
    m = BVDOutbreakSize.deaths_obs_model(missing, onsets, 5.0;
        delay_nmax = 30)
    st = m()
    @test st.expected_deaths_T >= 0
    @test isfinite(st.expected_deaths_T)
    @test length(st.deaths_daily) == length(onsets)
    @test 0 < st.CFR < 1
end

@testset "renewal_joint prior-predictive draw is finite" begin
    gen = BVDOutbreakSize.renewal_joint(
        60, missing, missing, missing, missing)
    draw = gen()
    @test isfinite(draw.C_T)
    @test draw.C_T >= 0
    @test isfinite(draw.expected_deaths_T)
    @test isfinite(draw.expected_exports_T)
    @test all(isfinite, draw.Rt)
end

@testset "renewal_joint Mooncake gradient is finite" begin
    model = BVDOutbreakSize.renewal_joint(60, 2, 131, 516, 1;
        tmrca_days = 80.0, tmrca_days_sd = 20.0)
    ldf = Turing.DynamicPPL.LogDensityFunction(
        model; adtype = default_adtype())
    x0 = Turing.DynamicPPL.VarInfo(model)[:]
    v, g = Turing.LogDensityProblems.logdensity_and_gradient(ldf, x0)
    @test isfinite(v)
    @test all(isfinite, g)
    @test any(!iszero, g)
end
