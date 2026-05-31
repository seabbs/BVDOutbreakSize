@testitem "lab pipeline daily likelihood runs and is finite" begin
    using BVDOutbreakSize: confirmed_cases_model, reported_cases_model,
                           infection_model, onset_incidence_model
    using Turing: logjoint
    using Random: MersenneTwister

    n = 40
    inf = infection_model(n)
    draw = rand(MersenneTwister(1), inf)
    onsets = draw.infections

    onset_state = rand(MersenneTwister(2),
        onset_incidence_model(onsets))
    daily_onsets = onset_state.onsets

    rep = reported_cases_model(
        (; days = Int[], counts = Int[]), missing, daily_onsets, 5.0, 0.3)
    rep_state = rand(MersenneTwister(3), rep)

    m = confirmed_cases_model(
        (; days = [20, 40], counts = [3, 8]),
        8, daily_onsets, 5.0, 0.3, rep_state.λ_bg, rep_state.τ_test,
        rep_state.bvd_reports_daily;
        lab_history = (; days = [20, 40], counts = [5, 8]),
        tests_analysed = 8)
    lp = logjoint(m, rand(MersenneTwister(4), m))
    @test isfinite(lp)
end
