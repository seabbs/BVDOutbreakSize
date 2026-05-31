@testitem "confirmed_cases_model exposes lab-pipeline positivity" begin
    using BVDOutbreakSize: confirmed_cases_model, reported_cases_model,
                           infection_model, onset_incidence_model
    using Random: MersenneTwister

    n = 40
    inf = infection_model(n)
    onsets = rand(MersenneTwister(1), inf).infections

    onset_pmf = onset_incidence_model(onsets)
    onset_state = rand(MersenneTwister(2), onset_pmf)
    daily_onsets = onset_state.onsets

    # Draw the shared report kernel / background / testing fraction.
    rep = reported_cases_model(
        (; days = Int[], counts = Int[]), missing, daily_onsets, 5.0, 0.3)
    rep_state = rand(MersenneTwister(3), rep)

    m = confirmed_cases_model(
        (; days = [20, 40], counts = [3, 8]),
        8, daily_onsets, 5.0, 0.3, rep_state.λ_bg, rep_state.τ_test,
        rep_state.bvd_reports_daily;
        lab_history = (; days = [20, 40], counts = [5, 8]),
        tests_analysed = 8)
    draw = rand(MersenneTwister(4), m)
    @test haskey(draw, :p_positive)
    @test 0 <= draw.p_positive <= 1
    @test draw.expected_confirmed >= 0
    @test draw.expected_tested >= 0
end
