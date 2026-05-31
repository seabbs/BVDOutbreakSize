# BVDOutbreakSize Lean proofs

This directory contains a Lean 4/Lake project for the continuous mathematical
claims behind the deterministic numerical layer in `src/BVDOutbreakSize.jl` and
`src/gamma_cdf.jl`.

The proofs verify algebraic identities, endpoint/order facts, clamp behavior,
and parameter conversions used by the Julia code. They do not verify the
`Float64` implementation, fixed-order Gauss-Legendre error bounds, random
number generation, Mooncake internals, or the exact contents of posterior
chains. The Gamma special-function derivatives, Kummer series, and closed-form
integral identities are collected as named assumptions in
`BvdProofs/Foundations.lean`; downstream modules prove the package formulas
from those assumptions.

Run:

```sh
lake build
make axioms
```

`make axioms` prints the trusted assumptions used by the headline theorems.
