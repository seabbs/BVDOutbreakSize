@testitem "lab pipeline daily likelihood runs and is finite" begin
    using BVDOutbreakSize: confirmed_only_model
    using Turing: logjoint
    using Random: MersenneTwister

    ## The confirmed-cases daily likelihood is exercised end to end through
    ## confirmed_only_model, which draws the shared report kernel and runs
    ## the confirmed + tests-analysed streams over the per-vintage history.
    m = confirmed_only_model(40, 8;
        confirmed_history = (; days = [20, 40], counts = [3, 8]),
        lab_history = (; days = [20, 40], counts = [5, 8]),
        tests_analysed = 8)
    lp = logjoint(m, rand(MersenneTwister(1), m))
    @test isfinite(lp)
end
