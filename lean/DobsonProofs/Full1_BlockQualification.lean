import DobsonProofs.ExactTest
import DobsonProofs.HalfPlane

/-! Qualification of the unreduced response polynomials in Table 1.
Stability means every complex root has negative real part. The internal
matrices and responses are defined explicitly; their linearization from the
nonlinear block equations is not derived here. Coverage correction,
2026-11-22: `prop:dyn-k3` is core in the version-2 inventory. -/

namespace Dobson.Full1.BlockQualification

def Hurwitz (D : ℂ → ℂ) : Prop := ∀ z, D z = 0 → z.re < 0

def NonAmplifying (D N : ℂ → ℂ) : Prop :=
  Hurwitz D ∧ ∀ ω : ℝ, ‖N (Complex.I * ω) / D (Complex.I * ω)‖ ≤ 1

def Damping (D N : ℂ → ℂ) : Prop :=
  Hurwitz D ∧ ∀ ω : ℝ, ‖N (Complex.I * ω) / D (Complex.I * ω)‖ < 1

def quad (A B : ℝ) (z : ℂ) : ℂ := z ^ 2 + A * z + B

theorem quadratic_hurwitz (A B : ℝ) : Hurwitz (quad A B) ↔ 0 < A ∧ 0 < B := by
  constructor
  · intro h
    by_cases hd : 0 ≤ A ^ 2 - 4 * B
    · let t := Real.sqrt (A ^ 2 - 4 * B)
      have ht : t ^ 2 = A ^ 2 - 4 * B := Real.sq_sqrt hd
      have hp : quad A B (((-A + t) / 2 : ℝ) : ℂ) = 0 := by
        have hr : ((-A + t) / 2) ^ 2 + A * ((-A + t) / 2) + B = 0 := by nlinarith
        unfold quad
        exact_mod_cast hr
      have hm : quad A B (((-A - t) / 2 : ℝ) : ℂ) = 0 := by
        have hr : ((-A - t) / 2) ^ 2 + A * ((-A - t) / 2) + B = 0 := by nlinarith
        unfold quad
        exact_mod_cast hr
      have hpr := h _ hp
      have hmr := h _ hm
      simp only [Complex.ofReal_re] at hpr hmr
      have hprod := mul_pos_of_neg_of_neg hpr hmr
      constructor <;> nlinarith
    · have hd' : 0 ≤ 4 * B - A ^ 2 := by linarith
      let t := Real.sqrt (4 * B - A ^ 2)
      have ht : t ^ 2 = 4 * B - A ^ 2 := Real.sq_sqrt hd'
      have hz : quad A B (⟨-A / 2, t / 2⟩ : ℂ) = 0 := by
        apply Complex.ext <;>
          simp [quad, pow_two, Complex.mul_re, Complex.mul_im] <;> nlinarith
      have hr := h _ hz
      change -A / 2 < 0 at hr
      constructor <;> nlinarith [sq_nonneg A]
  · rintro ⟨hA, hB⟩ z hz
    have hr := congrArg Complex.re hz
    have hi := congrArg Complex.im hz
    simp only [quad, pow_two, Complex.add_re, Complex.add_im, Complex.mul_re,
      Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im, zero_mul, add_zero,
      Complex.zero_re, Complex.zero_im] at hr hi
    by_contra hn
    have hx : 0 ≤ z.re := le_of_not_gt hn
    have hy : z.im = 0 := by
      have hf : z.im * (2 * z.re + A) = 0 := by nlinarith [hi]
      exact (mul_eq_zero.mp hf).resolve_right (by linarith)
    rw [hy] at hr
    nlinarith [mul_nonneg hA.le hx, sq_nonneg z.re]

theorem quadratic_nonnegative (U V : ℝ) :
    (∀ x : ℝ, 0 ≤ x → 0 ≤ x ^ 2 + U * x + V) ↔
      0 ≤ V ∧ 0 ≤ U + 2 * Real.sqrt V := by
  constructor
  · intro h
    have hv : 0 ≤ V := by simpa using h 0 le_rfl
    refine ⟨hv, ?_⟩
    by_cases hv0 : V = 0
    · simp only [hv0, Real.sqrt_zero, mul_zero, add_zero]
      by_contra hu
      have hx : 0 ≤ -U / 2 := by linarith
      have hh := h (-U / 2) hx
      rw [hv0] at hh
      nlinarith [sq_pos_of_neg (lt_of_not_ge hu)]
    · have ht := Real.sq_sqrt hv
      have hp := Real.sqrt_pos.mpr (lt_of_le_of_ne hv (Ne.symm hv0))
      have hh := h (Real.sqrt V) hp.le
      by_contra hn
      nlinarith [mul_neg_of_pos_of_neg hp (lt_of_not_ge hn)]
  · rintro ⟨hv, hu⟩ x hx
    have ht := Real.sq_sqrt hv
    nlinarith [sq_nonneg (x - Real.sqrt V), mul_nonneg hx hu]

def denominator (d h ρ : ℝ) (z : ℂ) : ℂ := (z + ρ) * (z + d) + d * h
def numerator (q ρ : ℝ) (z : ℂ) : ℂ := q * (z + ρ)
def U (d q h ρ : ℝ) : ℝ := ρ ^ 2 + d ^ 2 - 2 * d * h - q ^ 2
def V (d q h ρ : ℝ) : ℝ := d ^ 2 * (ρ + h) ^ 2 - q ^ 2 * ρ ^ 2

theorem denominator_hurwitz (d h ρ : ℝ) :
    Hurwitz (denominator d h ρ) ↔ 0 < ρ + d ∧ 0 < d * (ρ + h) := by
  have he : denominator d h ρ = quad (ρ + d) (d * (ρ + h)) := by
    funext z; simp only [denominator, quad]; push_cast; ring
  rw [he, quadratic_hurwitz]

theorem denominator_ne (d h ρ : ℝ) (hs : Hurwitz (denominator d h ρ))
    (z : ℂ) (hz : 0 ≤ z.re) : denominator d h ρ z ≠ 0 := by
  intro he; exact (not_lt_of_ge hz) (hs z he)

theorem axis_gap (d q h ρ ω : ℝ) :
    ‖denominator d h ρ (Complex.I * ω)‖ ^ 2 -
      ‖numerator q ρ (Complex.I * ω)‖ ^ 2 =
      (ω ^ 2) ^ 2 + U d q h ρ * ω ^ 2 + V d q h ρ := by
  rw [← Complex.normSq_eq_norm_sq, ← Complex.normSq_eq_norm_sq]
  simp only [denominator, numerator, Complex.normSq_mul, Complex.normSq_ofReal]
  simpa only [pow_two] using
    Dobson.ExactTest.X1 d q h ρ ω (ω ^ 2) (U d q h ρ) (V d q h ρ) rfl rfl rfl

theorem norm_ratio_lt (a b : ℂ) (hb : b ≠ 0) :
    ‖a / b‖ < 1 ↔ 0 < ‖b‖ ^ 2 - ‖a‖ ^ 2 := by
  rw [norm_div, div_lt_one (norm_pos_iff.mpr hb)]
  constructor <;> intro h <;> nlinarith [norm_nonneg a, norm_pos_iff.mpr hb]

theorem norm_ratio_le (a b : ℂ) (hb : b ≠ 0) :
    ‖a / b‖ ≤ 1 ↔ 0 ≤ ‖b‖ ^ 2 - ‖a‖ ^ 2 := by
  rw [norm_div, div_le_one (norm_pos_iff.mpr hb)]
  constructor <;> intro h <;> nlinarith [norm_nonneg a, norm_pos_iff.mpr hb]

/-- Part (e), strict test, with internal stability included. -/
theorem VM_damping_iff (d q h ρ : ℝ) :
    Damping (denominator d h ρ) (numerator q ρ) ↔
      0 < ρ + d ∧ 0 < d * (ρ + h) ∧ 0 < V d q h ρ ∧
        0 < U d q h ρ + 2 * Real.sqrt (V d q h ρ) := by
  constructor
  · rintro ⟨hs, hg⟩
    obtain ⟨hA, hB⟩ := (denominator_hurwitz d h ρ).mp hs
    have hall : ∀ x : ℝ, 0 ≤ x → 0 < x ^ 2 + U d q h ρ * x + V d q h ρ := by
      intro x hx
      have h := (norm_ratio_lt _ _ (denominator_ne d h ρ hs _ (by simp))).mp
        (hg (Real.sqrt x))
      rwa [axis_gap, Real.sq_sqrt hx] at h
    obtain ⟨hV, hU⟩ := (Dobson.ExactTest.X2 _ _).mp hall
    exact ⟨hA, hB, hV, by linarith⟩
  · rintro ⟨hA, hB, hV, hU⟩
    have hs := (denominator_hurwitz d h ρ).mpr ⟨hA, hB⟩
    refine ⟨hs, fun ω => ?_⟩
    rw [norm_ratio_lt _ _ (denominator_ne d h ρ hs _ (by simp)), axis_gap]
    exact (Dobson.ExactTest.X2 _ _).mpr ⟨hV, by linarith⟩ _ (sq_nonneg ω)

/-- Part (e), non-strict test, including V=0 and U=0. -/
theorem VM_nonAmplifying_iff (d q h ρ : ℝ) :
    NonAmplifying (denominator d h ρ) (numerator q ρ) ↔
      0 < ρ + d ∧ 0 < d * (ρ + h) ∧ 0 ≤ V d q h ρ ∧
        0 ≤ U d q h ρ + 2 * Real.sqrt (V d q h ρ) := by
  constructor
  · rintro ⟨hs, hg⟩
    obtain ⟨hA, hB⟩ := (denominator_hurwitz d h ρ).mp hs
    have hall : ∀ x : ℝ, 0 ≤ x → 0 ≤ x ^ 2 + U d q h ρ * x + V d q h ρ := by
      intro x hx
      have h := (norm_ratio_le _ _ (denominator_ne d h ρ hs _ (by simp))).mp
        (hg (Real.sqrt x))
      rwa [axis_gap, Real.sq_sqrt hx] at h
    exact ⟨hA, hB, (quadratic_nonnegative _ _).mp hall⟩
  · rintro ⟨hA, hB, hV, hU⟩
    have hs := (denominator_hurwitz d h ρ).mpr ⟨hA, hB⟩
    refine ⟨hs, fun ω => ?_⟩
    rw [norm_ratio_le _ _ (denominator_ne d h ρ hs _ (by simp)), axis_gap]
    exact (quadratic_nonnegative _ _).mpr ⟨hV, hU⟩ _ (sq_nonneg ω)

def linearD (a : ℝ) (z : ℂ) : ℂ := z + a
def linearN (b : ℝ) (_z : ℂ) : ℂ := b

theorem linear_hurwitz (a : ℝ) (ha : 0 < a) : Hurwitz (linearD a) := by
  intro z hz
  have hr := congrArg Complex.re hz
  simp only [linearD, Complex.add_re, Complex.ofReal_re, Complex.zero_re] at hr
  linarith

/-- Part (a): the origin witnesses failure of strict damping. -/
theorem L_qualification (a : ℝ) (ha : 0 < a) :
    NonAmplifying (linearD a) (linearN a) ∧
      ‖linearN a 0 / linearD a 0‖ = 1 ∧
        ¬ Damping (linearD a) (linearN a) := by
  have he : ‖linearN a 0 / linearD a 0‖ = 1 := by
    simp [linearN, linearD, ha.ne']
  refine ⟨⟨linear_hurwitz a ha, fun ω => ?_⟩, he, ?_⟩
  · exact Dobson.HalfPlane.H1 a _ ha (by simp)
  · intro h
    have hg := h.2 0
    simp only [Complex.ofReal_zero, mul_zero, he, lt_self_iff_false] at hg

/-- Part (b), both converses use the zero-frequency response. -/
theorem R_qualification (a κ : ℝ) (ha : 0 < a) :
    (NonAmplifying (linearD a) (linearN ((1 - κ) * a)) ↔ 0 ≤ κ ∧ κ ≤ 2) ∧
    (Damping (linearD a) (linearN ((1 - κ) * a)) ↔ 0 < κ ∧ κ < 2) := by
  have hzero : ‖linearN ((1 - κ) * a) 0 / linearD a 0‖ = |1 - κ| := by
    simp only [linearD, linearN, zero_add, Complex.ofReal_mul,
      mul_div_cancel_right₀ _ (Complex.ofReal_ne_zero.mpr ha.ne'), Complex.norm_real,
      Real.norm_eq_abs]
  have hbound (hk0 : 0 ≤ κ) (hk2 : κ ≤ 2) (ω : ℝ) :
      ‖linearN ((1 - κ) * a) (Complex.I * ω) /
        linearD a (Complex.I * ω)‖ ≤ |1 - κ| := by
    simpa [linearN, linearD] using
      (Dobson.HalfPlane.H5 κ a (Complex.I * ω) hk0 hk2 ha (by simp)).1
  constructor
  · constructor
    · intro hg
      have hh := hg.2 0
      simp only [Complex.ofReal_zero, mul_zero, hzero] at hh
      rw [abs_le] at hh
      constructor <;> linarith
    · rintro ⟨hk0, hk2⟩
      refine ⟨linear_hurwitz a ha, fun ω => (hbound hk0 hk2 ω).trans ?_⟩
      rw [abs_le]; constructor <;> linarith
  · constructor
    · intro hg
      have hh := hg.2 0
      simp only [Complex.ofReal_zero, mul_zero, hzero] at hh
      rw [abs_lt] at hh
      constructor <;> linarith
    · rintro ⟨hk0, hk2⟩
      refine ⟨linear_hurwitz a ha, fun ω => (hbound hk0.le hk2.le ω).trans_lt ?_⟩
      rw [abs_lt]; constructor <;> linarith

theorem S_qualification (a p : ℝ) (ha : 0 < a) (hp : 0 < p) (hp1 : p < 1) :
    (∀ ω : ℝ, ‖linearN a (Complex.I * ω) / linearD (a / (1 - p))
      (Complex.I * ω)‖ ≤ 1 - p) ∧
    Damping (linearD (a / (1 - p))) (linearN a) := by
  have hd : 0 < a / (1 - p) := div_pos ha (sub_pos.mpr hp1)
  have hratio : a / (a / (1 - p)) = 1 - p := by field_simp
  have hb (ω : ℝ) : ‖linearN a (Complex.I * ω) /
      linearD (a / (1 - p)) (Complex.I * ω)‖ ≤ 1 - p := by
    have hh := Dobson.HalfPlane.norm_ofReal_div_le a (a / (1 - p))
      (Complex.I * ω + (a / (1 - p) : ℝ)) ha.le hd
      (by simp only [Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im,
        Complex.ofReal_re, Complex.ofReal_im]; linarith)
    simpa only [linearN, linearD, hratio] using hh
  exact ⟨hb, linear_hurwitz _ hd, fun ω => (hb ω).trans_lt (by linarith)⟩

/-- The unreduced internal determinant, including its zero root when rho=h=0. -/
theorem internal_determinant (d h ρ : ℝ) (z : ℂ) :
    Matrix.det (!![z + (ρ : ℂ), (h : ℂ); -(d : ℂ), z + (d : ℂ)]) =
      denominator d h ρ z := by
  simp [Matrix.det_fin_two, denominator]; ring

/-- Part (d) on the full closed right half-plane, away from z+rho=0. -/
theorem VM_halfPlane_bound (d q h ρ : ℝ) (hd : 0 < d) (hq : 0 ≤ q)
    (hh : 0 ≤ h) (hρ : 0 ≤ ρ) (z : ℂ) (hz : 0 ≤ z.re) (hne : z + ρ ≠ 0) :
    ‖numerator q ρ z / denominator d h ρ z‖ ≤ q / d := by
  have hr := Dobson.HalfPlane.re_ofReal_div_nonneg (d * h) (z + ρ)
    (mul_nonneg hd.le hh) (by simpa using add_nonneg hz hρ)
  have he : numerator q ρ z / denominator d h ρ z =
      (q : ℂ) / (z + d + (d : ℂ) * h / (z + ρ)) := by
    simp only [numerator, denominator]
    field_simp
  apply he ▸ Dobson.HalfPlane.norm_ofReal_div_le q d _ hq hd ?_
  push_cast at hr
  simp only [Complex.add_re, Complex.ofReal_re]
  linarith

theorem VM_origin (d q h : ℝ) : numerator q 0 0 / denominator d h 0 0 = 0 := by
  simp [numerator, denominator]

theorem VM_axis_bound (d q h ρ : ℝ) (hd : 0 < d) (hq : 0 ≤ q)
    (hh : 0 ≤ h) (hρ : 0 ≤ ρ) (ω : ℝ) :
    ‖numerator q ρ (Complex.I * ω) / denominator d h ρ (Complex.I * ω)‖ ≤ q / d := by
  by_cases hn : Complex.I * (ω : ℂ) + ρ = 0
  · simp only [numerator, hn, mul_zero, zero_div, norm_zero]
    positivity
  · exact VM_halfPlane_bound d q h ρ hd hq hh hρ _ (by simp) hn

theorem V_qualification (a α p ρ : ℝ) (ha : 0 < a) (hα : 0 ≤ α)
    (hp : 0 < p) (hp1 : p < 1) (hρ : 0 ≤ ρ) (hρh : 0 < ρ + α * p) :
    a / (a / (1 - p)) = 1 - p ∧
    (∀ ω : ℝ, ‖numerator a ρ (Complex.I * ω) /
      denominator (a / (1 - p)) (α * p) ρ (Complex.I * ω)‖ ≤ 1 - p) ∧
    Damping (denominator (a / (1 - p)) (α * p) ρ) (numerator a ρ) := by
  have hd : 0 < a / (1 - p) := div_pos ha (sub_pos.mpr hp1)
  have he : a / (a / (1 - p)) = 1 - p := by field_simp
  have hb (ω : ℝ) := VM_axis_bound (a / (1 - p)) a (α * p) ρ hd ha.le
    (mul_nonneg hα hp.le) hρ ω
  simp only [he] at hb
  refine ⟨he, hb, ?_, fun ω => (hb ω).trans_lt (by linarith)⟩
  exact (denominator_hurwitz _ _ _).mpr ⟨by linarith, mul_pos hd hρh⟩

theorem M_qualification (ξ ε h ρ : ℝ) (hξ : 0 < ξ) (hε : 1 ≤ ε)
    (hh : 0 ≤ h) (hρ : 0 ≤ ρ) :
    (∀ ω : ℝ, ‖numerator (ξ + ε * h) ρ (Complex.I * ω) /
      denominator (ξ + 2 * ε * h) h ρ (Complex.I * ω)‖ ≤
        (ξ + ε * h) / (ξ + 2 * ε * h)) ∧
    (0 < h → Damping (denominator (ξ + 2 * ε * h) h ρ) (numerator (ξ + ε * h) ρ)) ∧
    (h = 0 → 0 < ρ → NonAmplifying (denominator (ξ + 2 * ε * h) h ρ)
      (numerator (ξ + ε * h) ρ)) := by
  have he : 0 < ε := by linarith
  have hd : 0 < ξ + 2 * ε * h := by positivity
  have hq : 0 ≤ ξ + ε * h := by positivity
  have hb := VM_axis_bound (ξ + 2 * ε * h) (ξ + ε * h) h ρ hd hq hh hρ
  refine ⟨hb, ?_, ?_⟩
  · intro hhpos
    refine ⟨(denominator_hurwitz _ _ _).mpr ⟨by linarith, by positivity⟩, fun ω => ?_⟩
    apply (hb ω).trans_lt
    rw [div_lt_one hd]
    nlinarith [mul_pos he hhpos]
  · intro hh0 hr
    refine ⟨(denominator_hurwitz _ _ _).mpr ⟨by linarith, by positivity⟩, fun ω => ?_⟩
    simpa [hh0, hξ.ne'] using hb ω

theorem VM_degenerate (d : ℝ) : ¬ Hurwitz (denominator d 0 0) := by
  intro h
  have hh := h 0 (by simp [denominator])
  simp at hh

/-- The complete qualification statement, parts (a)--(e), quantified over the
parameters in Table 1. The V statements hold even without the additional
admissibility restriction alpha <= a; hence include every admissible V block. -/
structure ManuscriptK3 : Prop where
  part_a : ∀ a : ℝ, 0 < a →
    NonAmplifying (linearD a) (linearN a) ∧ ‖linearN a 0 / linearD a 0‖ = 1 ∧
      ¬ Damping (linearD a) (linearN a)
  part_b : ∀ a κ : ℝ, 0 < a →
    (NonAmplifying (linearD a) (linearN ((1 - κ) * a)) ↔ 0 ≤ κ ∧ κ ≤ 2) ∧
    (Damping (linearD a) (linearN ((1 - κ) * a)) ↔ 0 < κ ∧ κ < 2)
  part_c : ∀ a p : ℝ, 0 < a → 0 < p → p < 1 →
    (∀ ω : ℝ, ‖linearN a (Complex.I * ω) /
      linearD (a / (1 - p)) (Complex.I * ω)‖ ≤ 1 - p) ∧
    Damping (linearD (a / (1 - p))) (linearN a)
  part_d_bound : ∀ d q h ρ : ℝ, 0 < d → 0 ≤ q → 0 ≤ h → 0 ≤ ρ →
    ∀ z : ℂ, 0 ≤ z.re → z + ρ ≠ 0 →
      ‖numerator q ρ z / denominator d h ρ z‖ ≤ q / d
  part_d_origin : ∀ d q h : ℝ, numerator q 0 0 / denominator d h 0 0 = 0
  part_d_V : ∀ a α p ρ : ℝ, 0 < a → 0 ≤ α → 0 < p → p < 1 →
    0 ≤ ρ → 0 < ρ + α * p →
    a / (a / (1 - p)) = 1 - p ∧
    (∀ ω : ℝ, ‖numerator a ρ (Complex.I * ω) /
      denominator (a / (1 - p)) (α * p) ρ (Complex.I * ω)‖ ≤ 1 - p) ∧
    Damping (denominator (a / (1 - p)) (α * p) ρ) (numerator a ρ)
  part_d_M : ∀ ξ ε h ρ : ℝ, 0 < ξ → 1 ≤ ε → 0 ≤ h → 0 ≤ ρ →
    (∀ ω : ℝ, ‖numerator (ξ + ε * h) ρ (Complex.I * ω) /
      denominator (ξ + 2 * ε * h) h ρ (Complex.I * ω)‖ ≤
        (ξ + ε * h) / (ξ + 2 * ε * h)) ∧
    (0 < h → Damping (denominator (ξ + 2 * ε * h) h ρ) (numerator (ξ + ε * h) ρ)) ∧
    (h = 0 → 0 < ρ → NonAmplifying (denominator (ξ + 2 * ε * h) h ρ)
      (numerator (ξ + ε * h) ρ))
  part_d_degenerate : ∀ d : ℝ, ¬ Hurwitz (denominator d 0 0)
  part_e_strict : ∀ d q h ρ : ℝ,
    Damping (denominator d h ρ) (numerator q ρ) ↔
      0 < ρ + d ∧ 0 < d * (ρ + h) ∧ 0 < V d q h ρ ∧
        0 < U d q h ρ + 2 * Real.sqrt (V d q h ρ)
  part_e_weak : ∀ d q h ρ : ℝ,
    NonAmplifying (denominator d h ρ) (numerator q ρ) ↔
      0 < ρ + d ∧ 0 < d * (ρ + h) ∧ 0 ≤ V d q h ρ ∧
        0 ≤ U d q h ρ + 2 * Real.sqrt (V d q h ρ)

theorem prop_dyn_k3 : ManuscriptK3 :=
  ⟨L_qualification, R_qualification, S_qualification, VM_halfPlane_bound,
    VM_origin, V_qualification, M_qualification, VM_degenerate,
    VM_damping_iff, VM_nonAmplifying_iff⟩

#print axioms prop_dyn_k3

end Dobson.Full1.BlockQualification
