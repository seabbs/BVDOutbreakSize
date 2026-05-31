import BvdProofs.Expectations

/-!
*Source: `GammaCDF.lean`*

## Gamma CDF derivative and rrule claims

`src/gamma_cdf.jl` attaches a ChainRules/Mooncake reverse rule to a
three-argument Gamma CDF wrapper. This module states the analytic partials and
proves the scalar pullback algebra used by the rule.
-/

noncomputable section

namespace BvdProofs

/-- Shape derivative formula from the integral representation. -/
def gammaShapePartial (α z : ℝ) : ℝ :=
  -digamma α * regularizedLowerGamma α z + lowerGammaLogIntegral α z

/-- Shape derivative formula from the differentiated Kummer expansion. -/
def gammaShapePartialKummer (α z : ℝ) : ℝ :=
  Real.log z * regularizedLowerGamma α z -
    z ^ α * Real.exp (-z) * kummerDigammaSeries α z

/-- Scale derivative formula under shape-scale parameterization. -/
def gammaScalePartial (α θ x : ℝ) : ℝ :=
  -(x / θ) * gammaPDF α θ x

/-- Argument derivative of a CDF is its density. -/
def gammaArgumentPartial (α θ x : ℝ) : ℝ :=
  gammaPDF α θ x

theorem gammaCDF_x_partial
    {α θ x : ℝ} (hα : 0 < α) (hθ : 0 < θ) (hx : 0 < x) :
    HasDerivAt (fun y => gammaCDF α θ y)
      (gammaArgumentPartial α θ x) x := by
  simpa [gammaArgumentPartial] using
    gammaCDF_hasDerivAt_x α θ x hα hθ hx

theorem gammaCDF_theta_partial
    {α θ x : ℝ} (hα : 0 < α) (hθ : 0 < θ) (hx : 0 < x) :
    HasDerivAt (fun η => gammaCDF α η x)
      (gammaScalePartial α θ x) θ := by
  simpa [gammaScalePartial] using
    gammaCDF_hasDerivAt_theta α θ x hα hθ hx

theorem regularizedLowerGamma_alpha_partial
    {α z : ℝ} (hα : 0 < α) (hz : 0 < z) :
    HasDerivAt (fun a => regularizedLowerGamma a z)
      (gammaShapePartial α z) α := by
  simpa [gammaShapePartial] using
    regularizedLowerGamma_hasDerivAt_alpha α z hα hz

theorem regularizedLowerGamma_alpha_partial_kummer
    {α z : ℝ} (hα : 0 < α) (hz : 0 < z) :
    HasDerivAt (fun a => regularizedLowerGamma a z)
      (gammaShapePartialKummer α z) α := by
  simpa [gammaShapePartialKummer] using
    regularizedLowerGamma_kummer_derivative α z hα hz

theorem regularizedLowerGamma_kummer_formula
    {α z : ℝ} (hα : 0 < α) (hz : 0 < z) :
    regularizedLowerGamma α z =
      z ^ α * Real.exp (-z) * kummerPSeries α z := by
  exact regularizedLowerGamma_kummer α z hα hz

/-- Scalar pullback returned by the ChainRules rrule. -/
def gammaCdfPullback
    (α θ x seed : ℝ) : ℝ × ℝ × ℝ :=
  (seed * gammaShapePartialKummer α (x / θ),
   seed * gammaScalePartial α θ x,
   seed * gammaArgumentPartial α θ x)

theorem gammaCdfPullback_components
    (α θ x seed : ℝ) :
    gammaCdfPullback α θ x seed =
      (seed * gammaShapePartialKummer α (x / θ),
       seed * (-(x / θ) * gammaPDF α θ x),
       seed * gammaPDF α θ x) := by
  rfl

theorem digamma_recurrence_for_series_step
    {α : ℝ} {n : ℕ} (h : α + n ≠ 0) :
    digamma (α + n + 1) = digamma (α + n) + 1 / (α + n) := by
  simpa [add_assoc] using digamma_succ (α + n) h

end BvdProofs
