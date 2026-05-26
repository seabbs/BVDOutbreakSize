import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
*Source: `Foundations.lean`*

## Shared analytic vocabulary and trusted assumptions

Most claims in `BVDOutbreakSize.jl` are algebraic identities once the
continuous integral or special-function fact has been stated precisely. This
module keeps those analytic facts explicit and named, so the rest of the proof
tree can be checked without hiding which facts are trusted.

The axioms here are intentionally narrow:

* exact exponential integrals used by forecast/export formulas;
* the Gamma-convolution closed form used by `expected_deaths(::Gamma)`;
* Gamma CDF partial derivatives and Kummer expansion used by `_gamma_cdf`;
* exact substitution identities for the two quadrature maps.

They are mathematical specifications, not claims about `Float64` arithmetic or
fixed-order quadrature error.
-/

noncomputable section

open scoped Interval

namespace BvdProofs

/-- A continuous-incidence trajectory `C(t)`. -/
abbrev Incidence := ℝ → ℝ

/-- A one-dimensional probability density. -/
abbrev Density := ℝ → ℝ

/-- A one-dimensional cumulative distribution function. -/
abbrev CDF := ℝ → ℝ

/-- The Gamma density with shape `α` and scale `θ`. -/
axiom gammaPDF : ℝ → ℝ → Density

/-- The Gamma CDF with shape `α` and scale `θ`. -/
axiom gammaCDF : ℝ → ℝ → CDF

/-- The Gamma moment-generating function, parameterized by shape and scale. -/
axiom gammaMGF : ℝ → ℝ → ℝ → ℝ

/-- Regularized lower incomplete Gamma `P(α,z)`. -/
axiom regularizedLowerGamma : ℝ → ℝ → ℝ

/-- Digamma function `ψ`. -/
axiom digamma : ℝ → ℝ

/-- The log-weighted lower-Gamma integral used in the shape derivative. -/
axiom lowerGammaLogIntegral : ℝ → ℝ → ℝ

/-- The Kummer-series sum `Σ z^n / Γ(α+n+1)`. -/
axiom kummerPSeries : ℝ → ℝ → ℝ

/-- The Kummer-series sum with digamma factors. -/
axiom kummerDigammaSeries : ℝ → ℝ → ℝ

/-- CDF is the density integral for arguments on the nonnegative support. -/
axiom cdf_eq_density_integral
    (F : CDF) (f : Density) :
    (∀ x, F x = ∫ u in (0 : ℝ)..x, f u) →
      ∀ x, F x = ∫ u in (0 : ℝ)..x, f u

/-- Exact affine substitution from `[lo, hi]` to `[-1, 1]`. -/
axiom affine_interval_integral
    (f : ℝ → ℝ) (lo hi : ℝ) :
    lo < hi →
      (∫ s in lo..hi, f s) =
        ((hi - lo) / 2) *
          ∫ u in (-1 : ℝ)..(1 : ℝ), f (((hi - lo) / 2) * (u + 1) + lo)

/-- Exact clustered substitution used to cluster nodes toward the upper limit.

`expo` is real-valued in the Julia implementation. The theorem is stated over
the open-domain analytic map; endpoint singularities for `expo < 1` are part of
the standard improper-substitution theorem captured by this assumption. -/
axiom clustered_interval_integral
    (f : ℝ → ℝ) (lo hi expo : ℝ) :
    lo < hi → 0 < expo →
      (∫ s in lo..hi, f s) =
        ∫ u in (-1 : ℝ)..(1 : ℝ),
          let v := (u + 1) / 2
          let span := hi - lo
          let d := span * v ^ expo
          f (hi - d) * (span * expo * v ^ (expo - 1) / 2)

/-- Exact integral of an exponential trajectory for nonzero growth. -/
axiom exp_interval_integral_nonzero
    (r lo hi : ℝ) :
    r ≠ 0 →
      (∫ s in lo..hi, Real.exp (r * s)) =
        (Real.exp (r * hi) - Real.exp (r * lo)) / r

/-- Exact integral of a constant trajectory, the zero-growth limit. -/
axiom exp_interval_integral_zero
    (lo hi : ℝ) :
      (∫ s in lo..hi, Real.exp ((0 : ℝ) * s)) = hi - lo

/-- Gamma convolution closed form used by `expected_deaths(::Gamma)`.

The hypothesis `0 < 1 + θ*r` is the moment/CDF-domain condition for the
shifted Gamma term. -/
axiom gamma_death_convolution
    (α θ r T : ℝ) :
    0 < α → 0 < θ → 0 ≤ T → 0 < 1 + θ * r →
      (∫ s in (0 : ℝ)..T,
          Real.exp (r * s) * gammaPDF α θ (T - s)) =
        Real.exp (r * T) * gammaMGF α θ (-r) *
          gammaCDF α θ (T * (1 + θ * r))

/-- Gamma CDF is regularized lower Gamma at the scaled argument. -/
axiom gammaCDF_eq_regularized
    (α θ x : ℝ) :
    0 < α → 0 < θ → 0 ≤ x →
      gammaCDF α θ x = regularizedLowerGamma α (x / θ)

/-- Partial derivative of the Gamma CDF with respect to `x`. -/
axiom gammaCDF_hasDerivAt_x
    (α θ x : ℝ) :
    0 < α → 0 < θ → 0 < x →
      HasDerivAt (fun y => gammaCDF α θ y) (gammaPDF α θ x) x

/-- Partial derivative of the Gamma CDF with respect to the scale `θ`. -/
axiom gammaCDF_hasDerivAt_theta
    (α θ x : ℝ) :
    0 < α → 0 < θ → 0 < x →
      HasDerivAt (fun η => gammaCDF α η x)
        (-(x / θ) * gammaPDF α θ x) θ

/-- Partial derivative of regularized lower Gamma with respect to shape. -/
axiom regularizedLowerGamma_hasDerivAt_alpha
    (α z : ℝ) :
    0 < α → 0 < z →
      HasDerivAt (fun a => regularizedLowerGamma a z)
        (-digamma α * regularizedLowerGamma α z + lowerGammaLogIntegral α z) α

/-- Kummer expansion for the regularized lower Gamma. -/
axiom regularizedLowerGamma_kummer
    (α z : ℝ) :
    0 < α → 0 < z →
      regularizedLowerGamma α z =
        z ^ α * Real.exp (-z) * kummerPSeries α z

/-- Shape derivative from the differentiated Kummer expansion. -/
axiom regularizedLowerGamma_kummer_derivative
    (α z : ℝ) :
    0 < α → 0 < z →
      HasDerivAt (fun a => regularizedLowerGamma a z)
        (Real.log z * regularizedLowerGamma α z -
          z ^ α * Real.exp (-z) * kummerDigammaSeries α z) α

/-- Digamma recurrence used by the iterative series implementation. -/
axiom digamma_succ
    (x : ℝ) :
    x ≠ 0 → digamma (x + 1) = digamma x + 1 / x

end BvdProofs
