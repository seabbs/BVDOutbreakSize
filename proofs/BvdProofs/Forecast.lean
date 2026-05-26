import BvdProofs.GammaCDF

/-!
*Source: `Forecast.lean`*

## Forecast and count-distribution algebra

The forecast path reuses the expected-deaths formula, integrates exponential
growth over the export detection window in closed form, and converts a desired
negative-binomial mean to the `Distributions.jl` success-probability parameter.
-/

noncomputable section

open scoped Interval

namespace BvdProofs

/-- Closed-form export mean for exponential incidence over `[lo, hi]`. -/
def forecastExportsClosed (p q r lo hi : ℝ) : ℝ :=
  p * q * (Real.exp (r * hi) - Real.exp (r * lo)) / r

/-- Zero-growth limit of the export mean. -/
def forecastExportsZeroGrowth (p q lo hi : ℝ) : ℝ :=
  p * q * (hi - lo)

theorem exponential_export_integral_closed
    {p q r lo hi : ℝ} (hr : r ≠ 0) :
    p * q * (∫ s in lo..hi, Real.exp (r * s)) =
      forecastExportsClosed p q r lo hi := by
  rw [exp_interval_integral_nonzero r lo hi hr]
  simp [forecastExportsClosed]
  ring

theorem exponential_export_integral_zero_growth
    (p q lo hi : ℝ) :
    p * q * (∫ s in lo..hi, Real.exp ((0 : ℝ) * s)) =
      forecastExportsZeroGrowth p q lo hi := by
  rw [exp_interval_integral_zero lo hi]
  rfl

theorem forecast_deaths_reuses_expectedDeathsGamma
    (CFR r Th α θ : ℝ) :
    expectedDeathsGamma CFR r Th α θ =
      CFR * Real.exp (r * Th) * gammaMGF α θ (-r) *
        gammaCDF α θ (Th * (1 + θ * r)) := by
  rfl

/-- Success-probability parameter for a negative-binomial count with mean `μ`
and dispersion/shape `k`, under the failures-before-`k`-successes convention. -/
def nbSuccessProb (k μ : ℝ) : ℝ := k / (k + μ)

theorem nbSuccessProb_pos {k μ : ℝ}
    (hk : 0 < k) (hμ : 0 ≤ μ) :
    0 < nbSuccessProb k μ := by
  dsimp [nbSuccessProb]
  positivity

theorem nbSuccessProb_lt_one {k μ : ℝ}
    (hk : 0 < k) (hμ : 0 < μ) :
    nbSuccessProb k μ < 1 := by
  dsimp [nbSuccessProb]
  have hden : 0 < k + μ := by positivity
  rw [div_lt_one hden]
  linarith

theorem negativeBinomial_mean_from_prob {k μ : ℝ}
    (hk : 0 < k) (hμ : 0 < μ) :
    k * (1 - nbSuccessProb k μ) / nbSuccessProb k μ = μ := by
  dsimp [nbSuccessProb]
  have hden : k + μ ≠ 0 := by positivity
  have hk_ne : k ≠ 0 := ne_of_gt hk
  field_simp [hden, hk_ne]
  linarith

/-- New reported counts are floored at zero. -/
def newCount (cum obs : ℝ) : ℝ := max (cum - obs) 0

theorem newCount_nonnegative (cum obs : ℝ) :
    0 ≤ newCount cum obs := by
  exact le_max_right (cum - obs) 0

theorem newCount_eq_difference_when_ge {cum obs : ℝ}
    (h : obs ≤ cum) :
    newCount cum obs = cum - obs := by
  dsimp [newCount]
  exact max_eq_left (by linarith)

end BvdProofs
