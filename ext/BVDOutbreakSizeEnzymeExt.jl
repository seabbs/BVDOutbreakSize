module BVDOutbreakSizeEnzymeExt

import BVDOutbreakSize
using BVDOutbreakSize: _gamma_cdf, _gamma_cdf_partials
using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules
using SpecialFunctions: gamma, digamma

# Reverse-mode Enzyme with runtime activity (so per-value activity is
# resolved through the quadrature and distribution constructors) and a
# `Duplicated` function annotation (so the closure over the observed data
# is differentiated, not treated as read-only). This is the config the
# `_gamma_cdf` / `gamma` rules below are validated against.
function BVDOutbreakSize.enzyme_adtype()
    return AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Duplicated)
end

# `EnzymeRules.@easy_rule` expands into both the reverse-mode
# (`augmented_primal` / `reverse`) and forward-mode (`forward`) rules for
# `_gamma_cdf`. The analytical (dα, dθ, dx) come from `_gamma_cdf_partials`
# in `src/gamma_cdf.jl`, the same helper used by the ChainRules rrule that
# Mooncake/ReverseDiff pick up. Routing `_gamma_cdf` through this rule
# avoids Enzyme differentiating `SpecialFunctions.gamma_inc` directly,
# which it cannot lower (recursive series + DomainError branches), and
# which the `@import_rrule` lift gets wrong on the shape partial.
EnzymeRules.@easy_rule(_gamma_cdf(α::Real, θ::Real, x::Real),
    @setup(_p=_gamma_cdf_partials(α, θ, x),
        dα=_p[1],
        dθ=_p[2],
        dx=_p[3],),
    (dα, dθ, dx))

# Rule for `SpecialFunctions.gamma`, derivative `d/dx Γ(x) = Γ(x) ψ(x)`
# (`Ω` binds to the primal `Γ(x)`). Enzyme's own `EnzymeSpecialFunctionsExt`
# ships no `gamma` rule and instead mis-lowers `gamma(x)` to the `loggamma`
# known-op, returning `ψ(x)` — wrong by a factor of `Γ(x)`. Distribution
# normalising constants (e.g. `Gamma`, `Beta`, `NegativeBinomial`) reach
# `gamma` outside the `_gamma_cdf` rule, so without this their gradients are
# corrupted.
EnzymeRules.@easy_rule(gamma(x::Real), (Ω * digamma(x),))

end
