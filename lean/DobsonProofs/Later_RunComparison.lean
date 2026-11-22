import DobsonProofs.GapOneMiddle
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-! Pathwise constant-run comparison from gap1mid Proposition 3.
The sequences below satisfy the actual ratio and tangent recurrences. No
contraction or error estimate is supplied as a hypothesis. -/
namespace Dobson.LaterRunComparison
open Filter Set MeasureTheory
open scoped Topology

noncomputable def logStep (a x : ℝ) : ℝ := Real.log (1 + a * Real.exp (-x))

lemma logStep_derivative (a x : ℝ) (ha : 0 ≤ a) :
    HasDerivAt (logStep a) (-(a * Real.exp (-x) / (1 + a * Real.exp (-x)))) x := by
  have hp : 0 < 1 + a * Real.exp (-x) := by positivity
  convert (((hasDerivAt_id x).neg.exp.const_mul a).const_add 1).log hp.ne' using 1
  · rfl
  · simp only [Pi.neg_apply, id_eq]; ring

lemma logStep_contraction (A B l u a x z : ℝ)
    (hA : 0 ≤ A) (ha : A ≤ a) (haB : a ≤ B) (hl : 1 < l)
    (hlu : l ≤ u) (hu : u = 1 + B / l)
    (hx : x ∈ Icc (Real.log l) (Real.log u))
    (hz : z ∈ Icc (Real.log l) (Real.log u)) :
    |logStep a x - logStep a z| ≤ (1 - 1 / u) * |x-z| := by
  have hl0 : 0 < l := by linarith
  have hu0 : 0 < u := hl0.trans_le hlu
  have ha0 : 0 ≤ a := hA.trans ha
  have hb0 : 0 ≤ B := ha0.trans haB
  have hd : ∀ t ∈ Icc (Real.log l) (Real.log u),
      HasDerivWithinAt (logStep a)
        (-(a * Real.exp (-t) / (1 + a * Real.exp (-t))))
        (Icc (Real.log l) (Real.log u)) t :=
    fun t _ => (logStep_derivative a t ha0).hasDerivWithinAt
  have hb : ∀ t ∈ Icc (Real.log l) (Real.log u),
      ‖-(a * Real.exp (-t) / (1 + a * Real.exp (-t)))‖ ≤ 1 - 1/u := by
    intro t ht
    have he : Real.exp (-t) ≤ 1/l := by
      calc
        _ ≤ Real.exp (-Real.log l) := Real.exp_le_exp.mpr (by linarith [ht.1])
        _ = 1/l := by rw [Real.exp_neg, Real.exp_log hl0]; simp
    have hv : a * Real.exp (-t) ≤ B/l := by
      calc
        _ ≤ B * (1/l) := mul_le_mul haB he (Real.exp_pos _).le hb0
        _ = _ := by ring
    have hp : 0 < 1 + a * Real.exp (-t) := by positivity
    have hq : a * Real.exp (-t) / (1+a*Real.exp (-t)) ≤ 1-1/u := by
      apply (div_le_iff₀ hp).mpr
      have hh := (div_le_div_of_nonneg_left (show (0:ℝ) ≤ 1 by norm_num) hp
        (show 1+a*Real.exp (-t) ≤ u by rw [hu]; linarith))
      have hid : a*Real.exp (-t)/(1+a*Real.exp (-t)) = 1-1/(1+a*Real.exp (-t)) := by
        field_simp; ring
      rw [← div_le_iff₀ hp]
      rw [hid]
      linarith
    rw [norm_neg, Real.norm_eq_abs, abs_of_nonneg (show 0 ≤ a*Real.exp (-t)/(1+a*Real.exp (-t)) by positivity)]
    exact hq
  simpa [Real.norm_eq_abs] using
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le hd hb (convex_Icc _ _) hz hx

/-- The two recurrences are the manuscript's (2), evaluated along a constant
host run. The initial ratio and sensitivity may depend on the entire past. -/
theorem constant_run_bounds (A B l u a lam M : ℝ) (y e : ℕ → ℝ)
    (hA : 0 ≤ A) (ha : A ≤ a) (haB : a ≤ B) (hl : 1 < l)
    (hlu : l ≤ u) (hlo : l = 1+A/u) (hhi : u = 1+B/l)
    (hlam : lam ∈ Icc l u) (hfixed : lam = 1+a/lam)
    (hy : y 0 ∈ Icc l u) (he : e 0 ∈ Icc (0:ℝ) 1)
    (hyrec : ∀ n, y (n+1) = 1+a/y n)
    (herec : ∀ n, e (n+1) = (1-1/y (n+1))*(1-e n))
    (hM : |Real.log (y 0)-Real.log lam| ≤ M) :
    ∀ n, |Real.log (y n)-Real.log lam| ≤ M*(1-1/u)^n ∧
      |e n-(lam-1)/(2*lam-1)| ≤ (1-1/u)^n*(1+(n:ℝ)*M) := by
  have hl0 : 0 < l := by linarith
  have hu0 : 0 < u := hl0.trans_le hlu
  have hlam1 : 1 < lam := hl.trans_le hlam.1
  have hlam0 : 0 < lam := by linarith
  have hk0 : 0 ≤ 1-1/u := by
    have hh : 1/u ≤ 1 := (div_le_one hu0).mpr (by linarith)
    linarith
  have hymem : ∀ n, y n ∈ Icc l u := by
    intro n
    induction n with
    | zero => exact hy
    | succ n ih =>
      rw [hyrec]
      exact GapOneMiddle.ratio_interval A B l u a (y n) hl0 ih.1 ih.2 hA ha haB hlo hhi
  have hyp (n : ℕ) : 0 < y n := hl0.trans_le (hymem n).1
  have hy1 (n : ℕ) : 1 < y n := hl.trans_le (hymem n).1
  have halpha (n : ℕ) : 0 ≤ 1-1/y n ∧ 1-1/y n ≤ 1-1/u := by
    constructor
    · have := (div_le_one (hyp n)).mpr (hy1 n).le; linarith
    · have := one_div_le_one_div_of_le (hyp n) (hymem n).2; linarith
  have hemem : ∀ n, e n ∈ Icc (0:ℝ) 1 := by
    intro n
    induction n with
    | zero => exact he
    | succ n ih =>
      rw [herec]
      constructor
      · exact mul_nonneg (halpha _).1 (by linarith [ih.2])
      · have hα : 1-1/y (n+1) ≤ 1 := by linarith [one_div_pos.mpr (hyp (n+1))]
        nlinarith [mul_nonneg (halpha (n+1)).1 ih.1]
  let v := (lam-1)/(2*lam-1)
  have hv0 : 0 ≤ v := div_nonneg (by linarith) (by linarith)
  have hv1 : v ≤ 1 := (div_le_one (by linarith : 0 < 2*lam-1)).mpr (by linarith)
  have hvfix : v = (1-1/lam)*(1-v) := by
    have hvrel : v*(2*lam-1)=lam-1 := div_mul_cancel₀ _ (by linarith)
    apply mul_right_cancel₀ hlam0.ne'
    calc
      v*lam = (lam-1)*(1-v) := by nlinarith [hvrel]
      _ = ((1-1/lam)*(1-v))*lam := by field_simp
  have hlog (n : ℕ) : Real.log (y n) ∈ Icc (Real.log l) (Real.log u) :=
    ⟨Real.log_le_log hl0 (hymem n).1, Real.log_le_log (hyp n) (hymem n).2⟩
  have hloglam : Real.log lam ∈ Icc (Real.log l) (Real.log u) :=
    ⟨Real.log_le_log hl0 hlam.1, Real.log_le_log hlam0 hlam.2⟩
  have hlogrec (n : ℕ) : Real.log (y (n+1)) = logStep a (Real.log (y n)) := by
    rw [hyrec]; simp [logStep, Real.exp_neg, Real.exp_log (hyp n), div_eq_mul_inv]
  have hlogfix : Real.log lam = logStep a (Real.log lam) := by
    conv_lhs => rw [hfixed]
    simp [logStep, Real.exp_neg, Real.exp_log hlam0, div_eq_mul_inv]
  intro n
  induction n with
  | zero =>
    refine ⟨by simpa using hM, ?_⟩
    simp only [pow_zero, Nat.cast_zero, zero_mul, add_zero, one_mul]
    change |e 0-v| ≤ 1
    exact abs_le.mpr ⟨by linarith [he.1], by linarith [he.2]⟩
  | succ n ih =>
    have hr : |Real.log (y (n+1))-Real.log lam| ≤ M*(1-1/u)^(n+1) := by
      rw [hlogrec, hlogfix]
      apply (logStep_contraction A B l u a _ _ hA ha haB hl hlu hhi (hlog n) hloglam).trans
      calc
        _ ≤ (1-1/u)*(M*(1-1/u)^n) := mul_le_mul_of_nonneg_left ih.1 hk0
        _ = _ := by rw [pow_succ]; ring
    refine ⟨hr, ?_⟩
    have hα : |(1-1/y (n+1))-(1-1/lam)| ≤ |Real.log (y (n+1))-Real.log lam| := by
      have hh := GapOneMiddle.exp_neg_lipschitz (Real.log (y (n+1))) (Real.log lam)
        (Real.log_nonneg (hy1 _).le) (Real.log_nonneg hlam1.le)
      simpa [Real.exp_neg, Real.exp_log (hyp (n+1)), Real.exp_log hlam0,
        one_div, abs_sub_comm] using hh
    have hid : e (n+1)-v = ((1-1/y (n+1))-(1-1/lam))*(1-v) -
        (1-1/y (n+1))*(e n-v) := by rw [herec]; linear_combination -hvfix
    change |e (n+1)-v| ≤ _
    rw [hid]
    calc
      _ ≤ |((1-1/y (n+1))-(1-1/lam))*(1-v)| +
          |(1-1/y (n+1))*(e n-v)| := abs_sub _ _
      _ = |(1-1/y (n+1))-(1-1/lam)| *(1-v) +
          (1-1/y (n+1))*|e n-v| := by
        rw [abs_mul, abs_mul, abs_of_nonneg (by linarith : 0 ≤ 1-v),
          abs_of_nonneg (halpha _).1]
      _ ≤ M*(1-1/u)^(n+1) + (1-1/u)*((1-1/u)^n*(1+(n:ℝ)*M)) := by
        have hh := mul_le_mul_of_nonneg_right (hα.trans hr) (show 0 ≤ 1-v by linarith)
        have hR0 : 0 ≤ M*(1-1/u)^(n+1) := (abs_nonneg _).trans hr
        have hmul := mul_le_mul (halpha (n+1)).2 ih.2 (abs_nonneg _) hk0
        nlinarith
      _ = _ := by rw [pow_succ]; push_cast; ring


/-- Exact geometric-run expectations, indexed by N=n+1, including epsilon=1. -/
theorem geometric_run_sums (ε κ : ℝ) (hε0 : 0 < ε) (hε1 : ε ≤ 1)
    (hκ0 : 0 ≤ κ) (hκ1 : κ < 1) :
    HasSum (fun n : ℕ => ε*(1-ε)^n*κ^(n+1)) (ε*κ/(1-(1-ε)*κ)) ∧
    HasSum (fun n : ℕ => ε*(1-ε)^n*((n+1:ℕ):ℝ)*κ^(n+1))
      (ε*κ/(1-(1-ε)*κ)^2) := by
  let q := (1-ε)*κ
  have hq0 : 0 ≤ q := mul_nonneg (by linarith) hκ0
  have hq1 : q < 1 := by dsimp [q]; nlinarith
  have hnorm : ‖q‖ < 1 := by simpa [Real.norm_eq_abs, abs_of_nonneg hq0] using hq1
  have hg := hasSum_geometric_of_norm_lt_one hnorm
  have hn := hasSum_coe_mul_geometric_of_norm_lt_one hnorm
  constructor
  · convert hg.mul_left (ε*κ) using 1
    · ext n; dsimp [q]; rw [mul_pow, pow_succ]; ring
    · dsimp [q]; ring
  · convert (hn.add hg).mul_left (ε*κ) using 1
    · ext n; dsimp [q]; rw [mul_pow, pow_succ]; push_cast; ring
    · dsimp [q] at *
      field_simp [show 1-(1-ε)*κ ≠ 0 by linarith]
      <;> ring

/-- Averaging the two pathwise bounds uses only the geometric distribution's
moments. These are named textbook probability inputs (their scalar series are
proved above), not the desired project-specific error estimates. -/
theorem expected_run_errors {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (A B l u M ε : ℝ) (a lam : Ω → ℝ) (y e : Ω → ℕ → ℝ) (N : Ω → ℕ)
    (hA : 0 ≤ A) (hl : 1 < l) (hlu : l ≤ u)
    (hlo : l = 1+A/u) (hhi : u = 1+B/l) (hM0 : 0 ≤ M) (hε : 0 ≤ ε)
    (ha : ∀ ω, A ≤ a ω ∧ a ω ≤ B)
    (hlam : ∀ ω, lam ω ∈ Icc l u ∧ lam ω = 1+a ω/lam ω)
    (hy : ∀ ω, y ω 0 ∈ Icc l u) (he : ∀ ω, e ω 0 ∈ Icc (0:ℝ) 1)
    (hyrec : ∀ ω n, y ω (n+1) = 1+a ω/y ω n)
    (herec : ∀ ω n, e ω (n+1) = (1-1/y ω (n+1))*(1-e ω n))
    (hM : ∀ ω, |Real.log (y ω 0)-Real.log (lam ω)| ≤ M)
    (textbook_geometric_moments :
      MeasureTheory.Integrable (fun ω => (1-1/u)^(N ω)) μ ∧
      MeasureTheory.Integrable (fun ω => (N ω:ℝ)*(1-1/u)^(N ω)) μ ∧
      (∫ ω, (1-1/u)^(N ω) ∂μ) = ε*(1-1/u)/(1-(1-ε)*(1-1/u)) ∧
      (∫ ω, (N ω:ℝ)*(1-1/u)^(N ω) ∂μ) = ε*(1-1/u)/(1-(1-ε)*(1-1/u))^2) :
    |∫ ω, Real.log (y ω (N ω))-Real.log (lam ω) ∂μ| ≤
        ε*(M*(1-1/u)/(1-(1-1/u))) ∧
    |∫ ω, e ω (N ω)-(lam ω-1)/(2*lam ω-1) ∂μ| ≤
        ε*((1-1/u)/(1-(1-1/u))+M*(1-1/u)/(1-(1-1/u))^2) := by
  have hpath (ω : Ω) := constant_run_bounds A B l u (a ω) (lam ω) M (y ω) (e ω)
    hA (ha ω).1 (ha ω).2 hl hlu hlo hhi (hlam ω).1 (hlam ω).2
    (hy ω) (he ω) (hyrec ω) (herec ω) (hM ω) (N ω)
  let κ := 1-1/u
  have hu0 : 0 < u := by linarith
  have hk0 : 0 ≤ κ := by dsimp [κ]; have := (div_le_one hu0).mpr (show 1 ≤ u by linarith); linarith
  have hk1 : κ < 1 := by dsimp [κ]; linarith [one_div_pos.mpr hu0]
  obtain ⟨i0,i1,h0,h1⟩ := textbook_geometric_moments
  have hd := GapOneMiddle.geometric_denominator_bounds ε κ hε hk0 hk1
  constructor
  · have hbound := MeasureTheory.norm_integral_le_of_norm_le
      (f := fun ω => Real.log (y ω (N ω))-Real.log (lam ω)) (i0.const_mul M)
      (Filter.Eventually.of_forall (fun ω => by simpa only [Real.norm_eq_abs] using (hpath ω).1))
    rw [MeasureTheory.integral_const_mul, h0, Real.norm_eq_abs] at hbound
    have hh := mul_le_mul_of_nonneg_left hd.1 hM0
    convert hbound.trans hh using 1 <;> dsimp [κ] <;> ring
  · have hbound := MeasureTheory.norm_integral_le_of_norm_le
      (f := fun ω => e ω (N ω)-(lam ω-1)/(2*lam ω-1)) (i0.add (i1.const_mul M))
      (Filter.Eventually.of_forall (fun ω => by
        simp only [Real.norm_eq_abs, Pi.add_apply]
        convert (hpath ω).2 using 1 <;> ring))
    simp only [Pi.add_apply] at hbound
    rw [MeasureTheory.integral_add i0 (i1.const_mul M), MeasureTheory.integral_const_mul,
      h0, h1, Real.norm_eq_abs] at hbound
    have hh := add_le_add hd.1 (mul_le_mul_of_nonneg_left hd.2 hM0)
    convert hbound.trans hh using 1 <;> dsimp [κ] <;> ring

end Dobson.LaterRunComparison

#print axioms Dobson.LaterRunComparison.constant_run_bounds

#print axioms Dobson.LaterRunComparison.geometric_run_sums
#print axioms Dobson.LaterRunComparison.expected_run_errors
