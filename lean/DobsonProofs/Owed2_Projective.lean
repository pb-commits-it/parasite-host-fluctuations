import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Tactic

/-!
The pathwise contraction calculation in A-ring, Step 2. The normalization
and its derivative are actual functions of the matrix/operator. This is a
supporting step: construction of an adapted norm, stochastic moment bounds,
and uniform bounds on four noise derivatives are not asserted here.
-/
namespace Dobson.Owed2_Projective
open Set Filter
open scoped Topology

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

noncomputable def normalize (A : V →L[ℝ] V) (w : V →L[ℝ] ℝ) (y : V) : V :=
  (w (A y))⁻¹ • A y

noncomputable def jacobian (A : V →L[ℝ] V) (w : V →L[ℝ] ℝ) (y : V) : V →L[ℝ] V :=
  (w (A y))⁻¹ • (A - (w.comp A).smulRight (normalize A w y))

/-- Derivative in the state variable of the actual normalized update. -/
theorem normalize_hasFDerivAt (A : V →L[ℝ] V) (w : V →L[ℝ] ℝ) (y : V)
    (hG : w (A y) ≠ 0) : HasFDerivAt (normalize A w) (jacobian A w y) y := by
  have h := ((hasDerivAt_inv hG).comp_hasFDerivAt y
    (w.comp A).hasFDerivAt).smul A.hasFDerivAt
  convert h using 1
  · rfl
  · ext z
    simp [jacobian, normalize, smul_sub, smul_smul, pow_two]
    module

/-- The tangent hyperplane is preserved by the actual derivative. -/
theorem jacobian_transverse (A : V →L[ℝ] V) (w : V →L[ℝ] ℝ) (y z : V)
    (hG : w (A y) ≠ 0) : w (jacobian A w y z) = 0 := by
  simp [jacobian, normalize, map_sub, map_smul, smul_eq_mul, hG]

lemma exp_dilation (c η : ℝ) (hc : 1 ≤ c) :
    1 + c * (Real.exp η - 1) ≤ Real.exp (c * η) := by
  have hc0 : 0 < c := by linarith
  have h := convexOn_exp.2 (mem_univ (0:ℝ)) (mem_univ (c*η))
    (show 0 ≤ 1-1/c by rw [sub_nonneg]; exact (div_le_one hc0).mpr hc)
    (by positivity : 0 ≤ 1/c) (by ring : 1-1/c+1/c=1)
  have hi : (1-1/c)*(0:ℝ)+(1/c)*(c*η)=η := by field_simp; ring
  simp only [smul_eq_mul, hi, Real.exp_zero, mul_one] at h
  have hh := mul_le_mul_of_nonneg_left h hc0.le
  field_simp at hh
  nlinarith

/-- Exact manuscript exponential absorption, with the printed c4. -/
theorem exponential_absorption (β c η : ℝ) (hβ : 0 < β) (hη : 0 ≤ η) :
    Real.exp η * (β+c*(Real.exp η-1)) ≤
      β*Real.exp ((1+max 1 (c/β))*η) := by
  let d := max 1 (c/β)
  have hd : 1 ≤ d := le_max_left _ _
  have hcb : c ≤ β*d := by
    have := le_max_right (1:ℝ) (c/β)
    exact (div_le_iff₀ hβ).mp this |>.trans_eq (mul_comm _ _)
  have hex : 0 ≤ Real.exp η-1 := sub_nonneg.mpr (Real.one_le_exp_iff.mpr hη)
  have hh := mul_le_mul_of_nonneg_left (exp_dilation d η hd) hβ.le
  calc
    _ ≤ Real.exp η * (β*(1+d*(Real.exp η-1))) := by
      apply mul_le_mul_of_nonneg_left _ (Real.exp_pos _).le
      nlinarith [mul_le_mul_of_nonneg_right hcb hex]
    _ ≤ Real.exp η * (β*Real.exp (d*η)) := mul_le_mul_of_nonneg_left hh (Real.exp_pos _).le
    _ = _ := by rw [mul_left_comm, ← Real.exp_add]; congr 2; dsimp [d]; ring

/-- A-ring Step 2 for an operator in the chosen adapted norm. `hM` is only
the deterministic transverse contraction; the random Jacobian bound is
derived. `hG` and `hA` are the primitive multiplier estimates from Step 1.
No stochastic uniform derivative estimate is assumed or concluded. -/
theorem jacobian_bound (A M : V →L[ℝ] V) (w : V →L[ℝ] ℝ) (y z : V)
    (β η b Y : ℝ) (hβ : 0 < β) (hη : 0 ≤ η) (hb : 0 ≤ b) (_hY : 0 ≤ Y)
    (hw : ∀ v, w (M v) = w v) (hz : w z = 0)
    (hM : ‖M z‖ ≤ β*‖z‖) (hy : ‖normalize A w y‖ ≤ Y)
    (hG : |(w (A y))⁻¹| ≤ Real.exp η)
    (hA : ‖A-M‖ ≤ b*(Real.exp η-1)) :
    ‖jacobian A w y z‖ ≤
      β*Real.exp ((1+max 1 (((1+‖w‖*Y)*b)/β))*η)*‖z‖ := by
  have he : 0 ≤ Real.exp η-1 := sub_nonneg.mpr (Real.one_le_exp_iff.mpr hη)
  have herr : ‖(A-M) z‖ ≤ b*(Real.exp η-1)*‖z‖ :=
    (A-M).le_opNorm z |>.trans (mul_le_mul_of_nonneg_right hA (norm_nonneg _))
  have hwz : w (A z) = w ((A-M) z) := by simp [hw, hz]
  have hid : A z - w (A z) • normalize A w y =
      M z + ((A-M) z - w ((A-M) z) • normalize A w y) := by
    rw [hwz]; simp only [sub_apply]; module
  have hinner : ‖A z - w (A z) • normalize A w y‖ ≤
      (β+(1+‖w‖*Y)*b*(Real.exp η-1))*‖z‖ := by
    rw [hid]
    calc
      _ ≤ ‖M z‖ + (‖(A-M) z‖ + ‖w ((A-M) z) • normalize A w y‖) :=
        (norm_add_le _ _).trans (add_le_add_right (norm_sub_le _ _) _)
      _ ≤ β*‖z‖ + (b*(Real.exp η-1)*‖z‖ +
          (‖w‖*(b*(Real.exp η-1)*‖z‖))*Y) := by
        rw [norm_smul]
        gcongr
        exact (w.le_opNorm _).trans (mul_le_mul_of_nonneg_left herr (norm_nonneg _))
      _ = _ := by ring
  calc
    _ = |(w (A y))⁻¹| * ‖A z - w (A z) • normalize A w y‖ := by
      simp [jacobian, norm_smul, Real.norm_eq_abs]
    _ ≤ Real.exp η * ((β+((1+‖w‖*Y)*b)*(Real.exp η-1))*‖z‖) := by
      exact mul_le_mul hG hinner (norm_nonneg _) (Real.exp_pos _).le
    _ ≤ _ := by
      rw [← mul_assoc]
      exact mul_le_mul_of_nonneg_right (exponential_absorption β ((1+‖w‖*Y)*b) η hβ hη)
        (norm_nonneg _)

end Dobson.Owed2_Projective

#print axioms Dobson.Owed2_Projective.normalize_hasFDerivAt
#print axioms Dobson.Owed2_Projective.jacobian_bound
