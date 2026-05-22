@testset "discretise_delay normalises and is non-negative" begin
    pmf = BVDOutbreakSize.discretise_delay(
        Distributions.Gamma(4.3, 2.6), 40)
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1e-12)
    ## Density at the origin is forced to zero (shape > 1), so the lag-0
    ## bin is the trapezoid of [0, f(1)] only.
    @test pmf[1] < pmf[2]
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
