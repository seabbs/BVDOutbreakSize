import BvdProofs.Expectations

/-!
*Source: `Interpolation.lean`*

## Cached CDF grid, trapezoid accumulation, and linear interpolation

`ExportDeathDelay` precomputes a cumulative trapezoid approximation to the
delay CDF and then linearly interpolates it. These are approximation claims,
not exact equalities to the analytic CDF. The exact Lean results below cover
the deterministic algebra and clamp behavior of that approximation layer.
-/

noncomputable section

namespace BvdProofs

/-- One cumulative-trapezoid update. -/
def trapezoidStep (Fprev fprev fcur dx : ℝ) : ℝ :=
  Fprev + (fprev + fcur) * dx / 2

theorem trapezoidStep_increment
    (Fprev fprev fcur dx : ℝ) :
    trapezoidStep Fprev fprev fcur dx - Fprev =
      (fprev + fcur) * dx / 2 := by
  simp [trapezoidStep]

theorem trapezoidStep_monotone
    {Fprev fprev fcur dx : ℝ}
    (hfprev : 0 ≤ fprev) (hfcur : 0 ≤ fcur) (hdx : 0 ≤ dx) :
    Fprev ≤ trapezoidStep Fprev fprev fcur dx := by
  dsimp [trapezoidStep]
  have hprod : 0 ≤ (fprev + fcur) * dx / 2 := by positivity
  linarith

/-- Linear interpolation in a grid cell. `frac=0` gives the left endpoint,
`frac=1` gives the right endpoint. -/
def linearInterp (left right frac : ℝ) : ℝ :=
  left + frac * (right - left)

theorem linearInterp_left (left right : ℝ) :
    linearInterp left right 0 = left := by
  simp [linearInterp]

theorem linearInterp_right (left right : ℝ) :
    linearInterp left right 1 = right := by
  simp [linearInterp]

theorem linearInterp_convex_form (left right frac : ℝ) :
    linearInterp left right frac =
      (1 - frac) * left + frac * right := by
  simp [linearInterp]
  ring

theorem linearInterp_between
    {left right frac : ℝ}
    (hle : left ≤ right) (h0 : 0 ≤ frac) (h1 : frac ≤ 1) :
    left ≤ linearInterp left right frac ∧
      linearInterp left right frac ≤ right := by
  constructor
  · dsimp [linearInterp]
    have hdiff : 0 ≤ right - left := by linarith
    have hterm : 0 ≤ frac * (right - left) := mul_nonneg h0 hdiff
    linarith
  · rw [linearInterp_convex_form]
    have hleft : (1 - frac) * left ≤ (1 - frac) * right := by
      exact mul_le_mul_of_nonneg_left hle (by linarith)
    have hright : frac * right ≤ frac * right := le_rfl
    have hsum : (1 - frac) * left + frac * right ≤
        (1 - frac) * right + frac * right := add_le_add hleft hright
    calc
      (1 - frac) * left + frac * right
          ≤ (1 - frac) * right + frac * right := hsum
      _ = right := by ring

/-- The branch structure of `_cdf_to`: zero before support, flat after `gmax`,
and otherwise the cell interpolant. -/
def cachedCDF (gmax last : ℝ) (interp : ℝ → ℝ) (y : ℝ) : ℝ :=
  if y ≤ 0 then 0 else if gmax ≤ y then last else interp y

theorem cachedCDF_nonpositive
    (gmax last : ℝ) (interp : ℝ → ℝ) {y : ℝ} (hy : y ≤ 0) :
    cachedCDF gmax last interp y = 0 := by
  simp [cachedCDF, hy]

theorem cachedCDF_past_gmax
    {gmax last y : ℝ} (interp : ℝ → ℝ)
    (hy0 : ¬ y ≤ 0) (hyg : gmax ≤ y) :
    cachedCDF gmax last interp y = last := by
  simp [cachedCDF, hy0, hyg]

theorem cachedCDF_inside
    {gmax last y : ℝ} (interp : ℝ → ℝ)
    (hy0 : ¬ y ≤ 0) (hyg : ¬ gmax ≤ y) :
    cachedCDF gmax last interp y = interp y := by
  simp [cachedCDF, hy0, hyg]

end BvdProofs
