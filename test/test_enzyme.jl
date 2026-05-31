## Tests for the Enzyme AD extension (`ext/BVDOutbreakSizeEnzymeExt.jl`).
## Loading Enzyme activates `enzyme_adtype()` and the EnzymeRules for the
## gamma CDF. The gradient of a composer's unconstrained log-density under
## Enzyme must match Mooncake (the package default) and, through it,
## finite differences. Tagged `:slow` for the one-off Enzyme compilation.

@testitem "enzyme_adtype is plain-reverse AutoEnzyme" tags=[:slow, :ad] begin
    using ADTypes: AutoEnzyme
    using Enzyme
    using BVDOutbreakSize: enzyme_adtype
    ad = enzyme_adtype()
    @test ad isa AutoEnzyme
    ## Duplicated closure annotation is a type parameter; the mode is
    ## plain reverse (runtime activity dropped now that `integrate`
    ## is type-stable).
    @test ad isa AutoEnzyme{<:Any, Enzyme.Duplicated}
    @test ad.mode === Enzyme.Reverse
end

@testitem "Enzyme gradient matches Mooncake on a single-stream model" tags=[:slow, :ad] begin
    using Enzyme
    using Mooncake
    using Turing: DynamicPPL
    using LogDensityProblems: logdensity_and_gradient
    using Random: seed!
    using BVDOutbreakSize: exports_only_model, default_adtype, enzyme_adtype

    model = exports_only_model(3)
    seed!(20260518)
    vi = DynamicPPL.link(DynamicPPL.VarInfo(model), model)
    x0 = collect(vi[:])

    f_moon = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint, vi; adtype = default_adtype())
    f_enz = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint, vi; adtype = enzyme_adtype())
    grad(f, x) = last(logdensity_and_gradient(f, x))

    ## Check the prior-mode draw plus two perturbed points so the
    ## runtime-activity-off config is exercised away from the mode.
    using Random: MersenneTwister
    pts = [x0]
    for k in 1:2
        rng = MersenneTwister(20260518 + k)
        push!(pts, x0 .+ 0.5 .* randn(rng, length(x0)))
    end
    for x in pts
        @test grad(f_enz, x) ≈ grad(f_moon, x) rtol=1e-6
    end
end
