# Sampler glue: the package-default AD type and the NUTS driver used
# to fit every Turing model.

"""
Mooncake reverse-mode AD with default `Mooncake.Config()`. Used as
the NUTS `adtype` keyword.
"""
default_adtype() = AutoMooncake(; config = Mooncake.Config())

"""
NUTS on `model`, parallel chains via `MCMCThreads`. Chains
initialise from a uniform `[-2, 2]` window in the unconstrained
parameter space (`InitFromUniform()`), which guarantees a finite
initial log-density and gradient for any joint prior, then NUTS
adapts away from there during warmup. Pass `init =
InitFromPrior()` to fall back to prior-draw initialisation.
"""
function nuts_sample(model;
        samples::Integer = 1_000,
        chains::Integer = 4,
        target_accept::Real = 0.95,
        seed::Integer = 20260518,
        progress::Bool = false,
        adtype = default_adtype(),
        init = InitFromUniform())
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
