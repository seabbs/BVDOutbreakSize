module BVDOutbreakSizeEnzymeExt

import BVDOutbreakSize
using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules
import SpecialFunctions

# Reverse-mode Enzyme with runtime activity (so per-value activity is
# resolved through the distribution constructors) and a `Duplicated`
# function annotation (so the closure over the observed data is
# differentiated, not treated as read-only). This is the config the
# `gamma` rule below is validated against. Opt-in alternative to the
# default Mooncake backend; differentiating the full renewal joint under
# Enzyme is still work in progress.
function BVDOutbreakSize.enzyme_adtype()
    return AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Duplicated)
end

# Rule for `SpecialFunctions.gamma`, derivative `d/dx Γ(x) = Γ(x) ψ(x)`
# (`Ω` binds to the primal `Γ(x)`). Enzyme's own `EnzymeSpecialFunctionsExt`
# ships no `gamma` rule and instead mis-lowers `gamma(x)` to the `loggamma`
# known-op, returning `ψ(x)` — wrong by a factor of `Γ(x)`. The Beta and
# NegativeBinomial normalising constants in the renewal observation
# submodels reach `gamma`, so without this their gradients are corrupted.
EnzymeRules.@easy_rule(SpecialFunctions.gamma(x::Real),
    (Ω * SpecialFunctions.digamma(x),))

end
