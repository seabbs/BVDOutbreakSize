import BvdProofs.Foundations

/-!
*Source: `Quadrature.lean`*

## Gauss-Legendre change-of-variable claims

The Julia code integrates on `[-1,1]` and maps those nodes to the target
interval. This file proves the endpoint and order facts for the maps, and
connects the kernels to the exact substitution assumptions in
`Foundations.lean`.
-/

noncomputable section

open scoped Interval

namespace BvdProofs

/-- Half the interval width used as the affine Jacobian. -/
def halfwidth (lo hi : ℝ) : ℝ := (hi - lo) / 2

/-- The affine map from Gauss-Legendre reference nodes to `[lo, hi]`. -/
def affineNode (lo hi u : ℝ) : ℝ := halfwidth lo hi * (u + 1) + lo

/-- Reference-coordinate normalization for the clustered map. -/
def clusterV (u : ℝ) : ℝ := (u + 1) / 2

/-- Distance back from the upper endpoint in the clustered map. -/
def clusterDistance (span expo u : ℝ) : ℝ := span * clusterV u ^ expo

/-- Target point for clustered quadrature nodes. -/
def clusterPoint (lo hi expo u : ℝ) : ℝ :=
  hi - clusterDistance (hi - lo) expo u

/-- Positive Jacobian `dd/du`; the sign flip is handled by integrating in distance. -/
def clusterJacobian (span expo u : ℝ) : ℝ :=
  span * expo * clusterV u ^ (expo - 1) / 2

theorem halfwidth_pos {lo hi : ℝ} (h : lo < hi) : 0 < halfwidth lo hi := by
  dsimp [halfwidth]
  linarith

theorem affineNode_neg_one (lo hi : ℝ) :
    affineNode lo hi (-1) = lo := by
  simp [affineNode, halfwidth]

theorem affineNode_one (lo hi : ℝ) :
    affineNode lo hi 1 = hi := by
  simp [affineNode, halfwidth]
  ring

theorem affineNode_midpoint (lo hi : ℝ) :
    affineNode lo hi 0 = (lo + hi) / 2 := by
  simp [affineNode, halfwidth]
  ring

theorem affineNode_sub_lo (lo hi u : ℝ) :
    affineNode lo hi u - lo = halfwidth lo hi * (u + 1) := by
  simp [affineNode]

theorem affineNode_in_interval {lo hi u : ℝ}
    (hlohi : lo ≤ hi) (hu_lower : -1 ≤ u) (hu_upper : u ≤ 1) :
    lo ≤ affineNode lo hi u ∧ affineNode lo hi u ≤ hi := by
  constructor
  · dsimp [affineNode, halfwidth]
    have hwidth : 0 ≤ (hi - lo) / 2 := by linarith
    have hu : 0 ≤ u + 1 := by linarith
    nlinarith [mul_nonneg hwidth hu]
  · dsimp [affineNode, halfwidth]
    have hwidth : 0 ≤ (hi - lo) / 2 := by linarith
    have hu : u + 1 ≤ 2 := by linarith
    have hmul : ((hi - lo) / 2) * (u + 1) ≤ ((hi - lo) / 2) * 2 :=
      mul_le_mul_of_nonneg_left hu hwidth
    nlinarith

theorem affine_kernel_matches_integral
    (f : ℝ → ℝ) {lo hi : ℝ} (hlohi : lo < hi) :
    (∫ s in lo..hi, f s) =
      halfwidth lo hi *
        ∫ u in (-1 : ℝ)..(1 : ℝ), f (affineNode lo hi u) := by
  simpa [halfwidth, affineNode] using affine_interval_integral f lo hi hlohi

theorem clusterV_neg_one : clusterV (-1) = 0 := by
  simp [clusterV]

theorem clusterV_one : clusterV 1 = 1 := by
  simp [clusterV]

theorem clusterV_in_unit {u : ℝ}
    (hu_lower : -1 ≤ u) (hu_upper : u ≤ 1) :
    0 ≤ clusterV u ∧ clusterV u ≤ 1 := by
  constructor <;> dsimp [clusterV] <;> linarith

theorem clusterPoint_one (lo hi expo : ℝ) :
    clusterPoint lo hi expo 1 = lo := by
  simp [clusterPoint, clusterDistance, clusterV]

theorem clusterPoint_neg_one {lo hi expo : ℝ} (hexpo : 0 < expo) :
    clusterPoint lo hi expo (-1) = hi := by
  have hzero : (0 : ℝ) ^ expo = 0 := Real.zero_rpow (ne_of_gt hexpo)
  simp [clusterPoint, clusterDistance, clusterV, hzero]

theorem clusterJacobian_nonneg {span expo u : ℝ}
    (hspan : 0 ≤ span) (hexpo : 0 ≤ expo)
    (hu : 0 ≤ clusterV u ^ (expo - 1)) :
    0 ≤ clusterJacobian span expo u := by
  dsimp [clusterJacobian]
  positivity

theorem clustered_kernel_matches_integral
    (f : ℝ → ℝ) {lo hi expo : ℝ}
    (hlohi : lo < hi) (hexpo : 0 < expo) :
    (∫ s in lo..hi, f s) =
      ∫ u in (-1 : ℝ)..(1 : ℝ),
        f (clusterPoint lo hi expo u) *
          clusterJacobian (hi - lo) expo u := by
  simpa [clusterPoint, clusterDistance, clusterJacobian, clusterV]
    using clustered_interval_integral f lo hi expo hlohi hexpo

end BvdProofs
