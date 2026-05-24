## Tests for the explicit-convolution forward layer (issue #5): the
## onset convolution accuracy, the tabulated onset curve, and the
## three observation expectations. The onset convolution is pinned
## against an adaptive QuadGK reference; the observation helpers
## against direct re-integration.

using BVDOutbreakSize: infection_incidence, onset_incidence,
                       OnsetIncidence, expected_onsets_staged,
                       expected_exports_onset_staged,
                       expected_deaths_onset_staged,
                       expected_reports_onset_staged,
                       ExportDeathDelay, integrate
using Distributions: Gamma, pdf, cdf
using Integrals: IntegralProblem, QuadGKJL, solve

# Reference onset incidence by adaptive quadrature over [0, t].
function _onset_ref(r, incub, t)
    f(s, p) = (t - s) > 0 ? r * exp(r * s) * pdf(incub, t - s) : 0.0
    prob = IntegralProblem(f, (0.0, t), nothing)
    return solve(prob, QuadGKJL(); reltol = 1e-10, abstol = 1e-12).u
end

@testset "infection_incidence is the derivative of exp(r·s)" begin
    r = log(2) / 14
    @test infection_incidence(r, 0.0) ≈ r
    @test infection_incidence(r, 10.0) ≈ r * exp(r * 10.0)
end

@testset "onset_incidence matches an adaptive reference" begin
    r = log(2) / 14
    incub = Gamma(11.0, 0.74)
    for t in (15.0, 40.0, 90.0)
        @test isapprox(onset_incidence(r, incub, t),
                       _onset_ref(r, incub, t); rtol = 1e-4)
    end
    @test onset_incidence(r, incub, 0.0) == 0.0
    @test onset_incidence(r, incub, -5.0) == 0.0
end

@testset "OnsetIncidence interpolates and integrates accurately" begin
    r = log(2) / 14
    incub = Gamma(11.0, 0.74)
    T = 120.0
    oi = OnsetIncidence(r, incub, T)
    @test oi(0.0) == 0.0
    @test oi(T) == 0.0          # flat-zero outside the grid
    @test oi(-1.0) == 0.0
    # Cumulative onsets accurate vs a fine-grid reference.
    fine = expected_onsets_staged(OnsetIncidence(r, incub, T;
                                                 npts = 1025))
    @test isapprox(expected_onsets_staged(oi), fine; rtol = 5e-3)
    # Onsets lag infections, so cumulative onsets < cumulative
    # infections.
    @test expected_onsets_staged(oi) < exp(r * T)
end

@testset "expected_exports_onset_staged matches direct integral" begin
    r = log(2) / 14
    incub = Gamma(11.0, 0.74)
    T, w, p, q = 90.0, 7.0, 0.3, 1871 / 4_392_200
    oi = OnsetIncidence(r, incub, T)
    ref = p * q * integrate(oi, max(T - w, 0.0), T)
    @test isapprox(expected_exports_onset_staged(oi, p, q, w), ref;
                   rtol = 1e-8)
    @test expected_exports_onset_staged(oi, p, q, w) > 0
end

@testset "expected_deaths_onset_staged is CFR · onset⊗death-CDF" begin
    r = log(2) / 14
    incub = Gamma(11.0, 0.74)
    death = Gamma(4.3, 2.6)
    T, CFR = 90.0, 0.33
    oi = OnsetIncidence(r, incub, T)
    dd = ExportDeathDelay(death, T)
    # Direct convolution of the onset curve with the death CDF.
    g = s -> oi(s) * cdf(death, T - s)
    ref = CFR * integrate(g, 0.0, T)
    @test isapprox(expected_deaths_onset_staged(oi, dd, CFR), ref;
                   rtol = 5e-3)
    @test expected_deaths_onset_staged(oi, dd, CFR) > 0
end

@testset "expected_reports_onset_staged is delayed reporting" begin
    r = log(2) / 14
    incub = Gamma(11.0, 0.74)
    report = Gamma(4.0, 4.5)
    T, p = 90.0, 0.25
    oi = OnsetIncidence(r, incub, T)
    rd = ExportDeathDelay(report, T)
    g = s -> oi(s) * cdf(report, T - s)
    ref = p * integrate(g, 0.0, T)
    @test isapprox(expected_reports_onset_staged(oi, rd, p), ref;
                   rtol = 5e-3)
    # Delayed reporting is below the instantaneous p·C(T) of the
    # current model, since recent infections are not yet fully
    # reported.
    @test expected_reports_onset_staged(oi, rd, p) < p * exp(r * T)
end
