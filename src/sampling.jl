# Sampler glue: the package-default AD type and the NUTS driver used
# to fit every Turing model.

"""
Mooncake reverse-mode AD with default `Mooncake.Config()`. Used as
the NUTS `adtype` keyword.
"""
default_adtype() = AutoMooncake(; config = Mooncake.Config())

"""
Enzyme reverse-mode AD with runtime activity and `Duplicated` function
annotation, the configuration the joint model's analytical gamma-CDF
rule needs (see `ext/BVDOutbreakSizeEnzymeExt.jl`). Returns an
`ADTypes.AutoEnzyme`; pass to `nuts_sample(model; adtype = ...)`.

`enzyme_adtype` is a stub; loading Enzyme (`using Enzyme`) activates
the method via `BVDOutbreakSizeEnzymeExt`. Calling `enzyme_adtype`
without Enzyme loaded raises a `MethodError`.
"""
function enzyme_adtype end

"""
NUTS on `model`, parallel chains via `MCMCThreads`. Chains
initialise from the prior (`InitFromPrior()`) to keep the sampler
in regions with reasonable physical interpretation. Pass `init =
Turing.DynamicPPL.InitFromUniform()` to fall back to unconstrained
uniform initialisation.
"""
function nuts_sample(model;
        samples::Integer = 1_000,
        chains::Integer = 4,
        target_accept::Real = 0.9,
        seed::Integer = 20260518,
        progress::Bool = false,
        adtype = default_adtype(),
        init = InitFromPrior())
    rng = MersenneTwister(seed)
    return sample(
        rng,
        model,
        NUTS(target_accept; adtype),
        MCMCThreads(),
        samples, chains;
        initial_params = fill(init, chains),
        progress = progress
    )
end
