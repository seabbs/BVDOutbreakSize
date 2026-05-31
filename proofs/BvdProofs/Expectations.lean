import BvdProofs.Quadrature

/-!
*Source: `Expectations.lean`*

## Expected-death and export convolution claims

This module gives Lean names to the continuous expectations documented in
`BVDOutbreakSize.jl`. The theorems below prove the package formulas from these
definitions and from the named Gamma-convolution assumption.
-/

noncomputable section

open scoped Interval

namespace BvdProofs

/-- Cumulative deaths from a single seed under exponential growth and delay density. -/
def expectedDeaths (CFR r T : ℝ) (density : Density) : ℝ :=
  CFR * ∫ s in (0 : ℝ)..T, Real.exp (r * s) * density (T - s)

/-- Expected deaths using a Gamma onset-to-death delay. -/
def expectedDeathsGamma (CFR r T α θ : ℝ) : ℝ :=
  CFR * Real.exp (r * T) * gammaMGF α θ (-r) *
    gammaCDF α θ (T * (1 + θ * r))

theorem expectedDeaths_eq_convolution
    (CFR r T : ℝ) (density : Density) :
    expectedDeaths CFR r T density =
      CFR * ∫ s in (0 : ℝ)..T, Real.exp (r * s) * density (T - s) := by
  rfl

theorem expectedDeathsGamma_eq_closed_form
    {CFR r T α θ : ℝ}
    (hα : 0 < α) (hθ : 0 < θ) (hT : 0 ≤ T) (hdom : 0 < 1 + θ * r) :
    expectedDeaths CFR r T (gammaPDF α θ) =
      expectedDeathsGamma CFR r T α θ := by
  unfold expectedDeaths expectedDeathsGamma
  rw [gamma_death_convolution α θ r T hα hθ hT hdom]
  ring

/-- CDF built as an integral of its density, the generic AD-friendly path. -/
def cdfFromDensity (density : Density) : CDF :=
  fun x => ∫ u in (0 : ℝ)..x, density u

theorem cdfFromDensity_eq_integral (density : Density) (x : ℝ) :
    cdfFromDensity density x = ∫ u in (0 : ℝ)..x, density u := by
  rfl

/-- Deaths-among-exports convolution with an arbitrary delay CDF. -/
def exportsDeathsIntegral (C : Incidence) (F : CDF) (lo hi T : ℝ) : ℝ :=
  ∫ s in lo..hi, C s * F (T - s)

theorem exportsDeathsIntegral_with_density_cdf
    (C : Incidence) (density : Density) (lo hi T : ℝ) :
    exportsDeathsIntegral C (cdfFromDensity density) lo hi T =
      ∫ s in lo..hi, C s * (∫ u in (0 : ℝ)..(T - s), density u) := by
  rfl

theorem exportsDeathsIntegral_gamma
    (C : Incidence) (lo hi T α θ : ℝ) :
    exportsDeathsIntegral C (gammaCDF α θ) lo hi T =
      ∫ s in lo..hi, C s * gammaCDF α θ (T - s) := by
  rfl

/-- Detection-window lower endpoint. -/
def windowStart (t window : ℝ) : ℝ := max (t - window) 0

theorem windowStart_nonneg (t window : ℝ) :
    0 ≤ windowStart t window := by
  exact le_max_right (t - window) 0

theorem windowStart_le_t {t window : ℝ}
    (ht : 0 ≤ t) (hw : 0 ≤ window) :
    windowStart t window ≤ t := by
  dsimp [windowStart]
  exact max_le (by linarith) ht

/-- Expected detected exports over the detection window. -/
def expectedExports (C : Incidence) (p q t window : ℝ) : ℝ :=
  p * q * ∫ s in windowStart t window..t, C s

theorem expectedExports_eq_window_integral
    (C : Incidence) (p q t window : ℝ) :
    expectedExports C p q t window =
      p * q * ∫ s in windowStart t window..t, C s := by
  rfl

/-- Expected deaths among detected exports over the detection window. -/
def expectedExportDeaths
    (C : Incidence) (F : CDF) (CFR p q t window : ℝ) : ℝ :=
  CFR * p * q * exportsDeathsIntegral C F (windowStart t window) t t

theorem expectedExportDeaths_eq_window_convolution
    (C : Incidence) (F : CDF) (CFR p q t window : ℝ) :
    expectedExportDeaths C F CFR p q t window =
      CFR * p * q *
        ∫ s in windowStart t window..t, C s * F (t - s) := by
  rfl

/-- Future deaths among infections that already occurred by `T`. -/
def committedDeaths (CFR r T : ℝ) (F : CDF) : ℝ :=
  CFR * ∫ s in (0 : ℝ)..T, r * Real.exp (r * s) * (1 - F (T - s))

theorem committedDeaths_eq_counterfactual_integral
    (CFR r T : ℝ) (F : CDF) :
    committedDeaths CFR r T F =
      CFR * ∫ s in (0 : ℝ)..T, r * Real.exp (r * s) * (1 - F (T - s)) := by
  rfl

/-- Projected total under no-onward transmission. -/
def totalProjectedDeaths (obsDeaths deltaDeaths : ℝ) : ℝ :=
  obsDeaths + deltaDeaths

theorem totalProjectedDeaths_eq_obs_plus_delta
    (obsDeaths deltaDeaths : ℝ) :
    totalProjectedDeaths obsDeaths deltaDeaths = obsDeaths + deltaDeaths := by
  rfl

end BvdProofs
