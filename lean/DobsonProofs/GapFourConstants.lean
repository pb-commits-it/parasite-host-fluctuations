import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Tactic

/-!
# GAP 4: exact proof constants for memA P5 and P7

All decimal constants below are exact real rationals. Bounds on the exponential
come from Mathlib's Taylor estimate, not floating-point evaluation. The complex
one-step disk bounds are proved directly. Stationary pullback convergence,
holomorphy, Rouché's theorem, and the probability model are not asserted here.
-/
namespace Dobson.GapFourConstants

/-- P5's parameter disk radius. -/
noncomputable def p5R : ℝ := 1 / 256

/-- A rational upper bound sufficient for every P5 constant. -/
theorem exp_quarter_bound : Real.exp (1 / 4 : ℝ) < 129 / 100 := by
  have h := Real.exp_bound' (x := (1 / 4 : ℝ)) (by norm_num) (by norm_num)
    (n := 4) (by norm_num)
  norm_num [Finset.sum_range_succ, Nat.factorial] at h
  linarith

/-- Rouché's strict numerical separation, including the stronger `0.21` margin. -/
theorem p5_rouche :
    (19 / 256 : ℝ) < 0.075 ∧
    (0.075 : ℝ) < 3 / 2 - Real.exp (1 / 4 : ℝ) ∧
    (0.21 : ℝ) < 3 / 2 - Real.exp (1 / 4 : ℝ) := by
  have h := exp_quarter_bound
  constructor
  · norm_num
  constructor <;> linarith

/-- P5's invariant-ball factor is strictly less than four. -/
theorem p5_ball_factor :
    1 + 2 * Real.exp (1 / 4 : ℝ) / (1 - 4 * p5R) < 4 := by
  have h := exp_quarter_bound
  norm_num [p5R]
  linarith

/-- P5's contraction constant is nonnegative and strictly below one. -/
theorem p5_contraction :
    0 ≤ 2 * p5R * Real.exp (1 / 4 : ℝ) / (1 - 4 * p5R) ^ 2 ∧
    2 * p5R * Real.exp (1 / 4 : ℝ) / (1 - 4 * p5R) ^ 2 < 1 := by
  constructor
  · unfold p5R; positivity
  · have h := exp_quarter_bound
    norm_num [p5R]
    linarith

/-- The coefficient in P5's analytic error estimate is strictly below nineteen. -/
theorem p5_error_coefficient :
    (8 * Real.exp (1 / 4 : ℝ) + 8) / (1 - 4 * p5R) < 19 := by
  have h := exp_quarter_bound
  norm_num [p5R]
  linarith

/-- P5's radius powers and the CE1 analytic derivative margin are exact rationals. -/
theorem p5_radii_and_ce1 :
    0 < p5R ∧ 4 * p5R < 1 ∧
    p5R ^ 2 / 4 = (1 / 262144 : ℝ) ∧
    p5R ^ 3 / 2 = (1 / 33554432 : ℝ) ∧
    (1 / 100000000 : ℝ) < (1 / 16) / 262144 ∧
    (1 / 32 : ℝ) < 1 / 16 - 2 * (1 / 100000000) / p5R ^ 2 := by
  norm_num [p5R]

/-- P5's actual complex one-step map sends the closed disk strictly inside itself;
the displayed entering-state derivative also has modulus strictly below one. -/
theorem p5_disk (ε D h w : ℂ) (hε : ‖ε‖ ≤ p5R) (hD : ‖D‖ ≤ 1 / 4)
    (hh : ‖h‖ ≤ 2) (hw : ‖w‖ ≤ 4 * p5R) :
    ‖-ε + ε * Complex.exp D * h / (1 + w)‖ < 4 * p5R ∧
    ‖-(ε * Complex.exp D * h) / (1 + w) ^ 2‖ < 1 := by
  have hR0 : 0 < p5R := by norm_num [p5R]
  have hd : 1 - 4 * p5R ≤ ‖1 + w‖ := by
    have ht := norm_sub_norm_le (1 : ℂ) (-w)
    simp only [norm_one, norm_neg, sub_neg_eq_add] at ht
    linarith
  have hd0 : 0 < 1 - 4 * p5R := by norm_num [p5R]
  have he : ‖Complex.exp D‖ ≤ Real.exp (1 / 4 : ℝ) := by
    rw [Complex.norm_exp]
    exact Real.exp_le_exp.mpr ((Complex.re_le_norm D).trans hD)
  have hn : ‖ε * Complex.exp D * h‖ ≤ 2 * p5R * Real.exp (1 / 4 : ℝ) := by
    simp only [norm_mul]
    calc ‖ε‖ * ‖Complex.exp D‖ * ‖h‖ ≤ p5R * Real.exp (1 / 4 : ℝ) * 2 := by
          gcongr
      _ = _ := by ring
  have hi : ‖ε * Complex.exp D * h / (1 + w)‖ ≤
      2 * p5R * Real.exp (1 / 4 : ℝ) / (1 - 4 * p5R) := by
    rw [norm_div]
    exact div_le_div₀ (by positivity) hn hd0 hd
  have hc : ‖-(ε * Complex.exp D * h) / (1 + w) ^ 2‖ ≤
      2 * p5R * Real.exp (1 / 4 : ℝ) / (1 - 4 * p5R) ^ 2 := by
    rw [norm_div, norm_neg, norm_pow]
    exact div_le_div₀ (by positivity) hn (sq_pos_of_pos hd0)
      (pow_le_pow_left₀ hd0.le hd 2)
  constructor
  · have ht := norm_add_le (-ε) (ε * Complex.exp D * h / (1 + w))
    rw [norm_neg] at ht
    have hf := mul_lt_mul_of_pos_left p5_ball_factor (by norm_num [p5R] : 0 < p5R)
    have hsum : ‖-ε + ε * Complex.exp D * h / (1 + w)‖ ≤
        p5R + 2 * p5R * Real.exp (1 / 4 : ℝ) / (1 - 4 * p5R) := by
      linarith only [ht, hε, hi]
    norm_num [p5R] at hsum hf ⊢
    linarith only [hsum, hf]
  · exact hc.trans_lt p5_contraction.2

/-- P7's amplitude radius and invariant-state radius. -/
noncomputable def p7R (b : ℝ) : ℝ := (1 - b) ^ 2 / (8 * b)
noncomputable def p7ρ (b : ℝ) : ℝ := (1 - b) / 2

/-- P7's radii are positive, the state radius is below `1/2`, and its real
logarithm bound is strictly below `log 2`. -/
theorem p7_radii (b : ℝ) (hb0 : 0 < b) (hb1 : b < 1) :
    0 < p7R b ∧ 0 < p7ρ b ∧ p7ρ b < 1 / 2 ∧
    -Real.log (1 - p7ρ b) < Real.log 2 := by
  have hρ0 : 0 < p7ρ b := by unfold p7ρ; linarith
  have hρ1 : p7ρ b < 1 / 2 := by unfold p7ρ; linarith
  refine ⟨by unfold p7R; positivity, hρ0, hρ1, ?_⟩
  have hlog : Real.log (1 / 2 : ℝ) < Real.log (1 - p7ρ b) :=
    Real.log_lt_log (by norm_num) (by linarith)
  have he : Real.log (1 / 2 : ℝ) = -Real.log 2 := by
    rw [one_div, Real.log_inv]
  linarith

/-- Strict disk invariance and the exact P7 contraction factor for every `0 < b < 1`. -/
theorem p7_scalar_bounds (b : ℝ) (hb0 : 0 < b) (hb1 : b < 1) :
    b * (p7R b + p7ρ b) / (1 - p7ρ b) < p7ρ b ∧
    b * (1 + p7R b) / (1 - p7ρ b) ^ 2 =
      (1 + 6 * b + b ^ 2) / (2 * (1 + b) ^ 2) ∧
    (1 + 6 * b + b ^ 2) / (2 * (1 + b) ^ 2) < 1 := by
  have hb : b ≠ 0 := hb0.ne'
  have hbp : 0 < 1 + b := by linarith
  have hρeq : 1 - p7ρ b = (1 + b) / 2 := by unfold p7ρ; ring
  have hi : p7ρ b - b * (p7R b + p7ρ b) / (1 - p7ρ b) =
      (1 - b) ^ 2 / (4 * (1 + b)) := by
    rw [hρeq]
    unfold p7R p7ρ
    field_simp [hb, hbp.ne']
    ring
  have hc : 1 - (1 + 6 * b + b ^ 2) / (2 * (1 + b) ^ 2) =
      (1 - b) ^ 2 / (2 * (1 + b) ^ 2) := by
    field_simp
    ring
  refine ⟨?_, ?_, ?_⟩
  · have hp : 0 < (1 - b) ^ 2 / (4 * (1 + b)) := by positivity
    linarith
  · rw [hρeq]
    unfold p7R
    field_simp [hb, hbp.ne']
    ring
  · have hp : 0 < (1 - b) ^ 2 / (2 * (1 + b) ^ 2) := by positivity
    linarith

/-- P7's complex recurrence and entering-state derivative satisfy the strict
bounds on the closed amplitude and state disks, uniformly in all host histories. -/
theorem p7_disk (b : ℝ) (hb0 : 0 < b) (hb1 : b < 1) (δ w : ℂ)
    (hδ : ‖δ‖ ≤ p7R b) (hw : ‖w‖ ≤ p7ρ b) :
    ‖(b : ℂ) * (δ - w) / (1 + w)‖ < p7ρ b ∧
    ‖-(b : ℂ) * (1 + δ) / (1 + w) ^ 2‖ < 1 := by
  obtain ⟨hR0, hρ0, hρ1, _⟩ := p7_radii b hb0 hb1
  obtain ⟨himage, hfactor, hcontract⟩ := p7_scalar_bounds b hb0 hb1
  have hd0 : 0 < 1 - p7ρ b := by linarith
  have hd : 1 - p7ρ b ≤ ‖1 + w‖ := by
    have ht := norm_sub_norm_le (1 : ℂ) (-w)
    simp only [norm_one, norm_neg, sub_neg_eq_add] at ht
    linarith
  have hb : ‖(b : ℂ)‖ = b := by simp [abs_of_pos hb0]
  have hn : ‖δ - w‖ ≤ p7R b + p7ρ b := (norm_sub_le δ w).trans (add_le_add hδ hw)
  have hn' : ‖1 + δ‖ ≤ 1 + p7R b := by
    have ht := norm_add_le (1 : ℂ) δ
    simp only [norm_one] at ht
    linarith
  constructor
  · apply lt_of_le_of_lt _ himage
    rw [norm_div, norm_mul, hb]
    exact div_le_div₀ (by positivity) (mul_le_mul_of_nonneg_left hn hb0.le) hd0 hd
  · apply lt_of_le_of_lt _ (hfactor.symm ▸ hcontract)
    rw [norm_div, norm_mul, norm_neg, hb, norm_pow]
    exact div_le_div₀ (by positivity) (mul_le_mul_of_nonneg_left hn' hb0.le)
      (sq_pos_of_pos hd0) (pow_le_pow_left₀ hd0.le hd 2)

/-- The scalar tail inequality used in P5's exceptional derivative estimate. -/
theorem p5_tail_bound (x : ℝ) (_hx0 : 0 ≤ x) (hx1 : x ≤ 1 / 2) :
    1 / (1 - x) ^ 2 - 1 - 2 * x ≤ 8 * x ^ 2 := by
  have hd : 0 < 1 - x := by linarith
  apply (mul_le_mul_iff_right₀ (sq_pos_of_pos hd)).mp
  have he : (1 - x) ^ 2 * (1 / (1 - x) ^ 2 - 1 - 2 * x) =
      3 * x ^ 2 - 2 * x ^ 3 := by
    field_simp
    ring
  rw [he]
  have hp : 0 ≤ x ^ 2 * ((1 - 2 * x) * (5 - 4 * x)) := by
    apply mul_nonneg (sq_nonneg x)
    exact mul_nonneg (by linarith) (by linarith)
  nlinarith only [hp]

/-- The scalar normalization bound used for P7's finite-chain noise radius. -/
theorem normalized_deviation_bound (R E : ℝ) (hR : 0 < R) (_hE0 : 0 ≤ E)
    (hE : E ≤ R / (2 + R)) : E < 1 ∧ 2 * E / (1 - E) ≤ R := by
  have hd : 0 < 2 + R := by linarith
  have hR1 : R / (2 + R) < 1 := (div_lt_one hd).mpr (by linarith)
  have hE1 : E < 1 := hE.trans_lt hR1
  refine ⟨hE1, ?_⟩
  rw [div_le_iff₀ (by linarith : 0 < 1 - E)]
  have hm := (le_div_iff₀ hd).mp hE
  nlinarith only [hm]

/-- P7's finite-chain sigma radius: the exponential deviation and its normalized
bound are controlled throughout the closed complex parameter disk. -/
theorem finite_chain_radius (R X : ℝ) (σ : ℂ) (hR : 0 < R) (hX : 0 < X)
    (hσ : ‖σ‖ ≤ Real.log (1 + R / (2 + R)) / X) :
    0 < Real.log (1 + R / (2 + R)) / X ∧
    Real.exp (X * ‖σ‖) - 1 ≤ R / (2 + R) ∧
    Real.exp (X * ‖σ‖) - 1 < 1 ∧
    2 * (Real.exp (X * ‖σ‖) - 1) / (1 - (Real.exp (X * ‖σ‖) - 1)) ≤ R := by
  have hr : 0 < R / (2 + R) := div_pos hR (by linarith)
  have hlog : 0 < Real.log (1 + R / (2 + R)) := Real.log_pos (by linarith)
  have he : Real.exp (X * ‖σ‖) - 1 ≤ R / (2 + R) := by
    have hm : X * ‖σ‖ ≤ Real.log (1 + R / (2 + R)) := by
      have hh := (le_div_iff₀ hX).mp hσ
      linarith
    have hh := Real.exp_le_exp.mpr hm
    rw [Real.exp_log (by linarith)] at hh
    linarith
  have he0 : 0 ≤ Real.exp (X * ‖σ‖) - 1 := by
    have hh := Real.add_one_le_exp (X * ‖σ‖)
    have hp := mul_nonneg hX.le (norm_nonneg σ)
    linarith
  obtain ⟨he1, hnorm⟩ := normalized_deviation_bound R (Real.exp (X * ‖σ‖) - 1) hR he0 he
  exact ⟨div_pos hlog hX, he, he1, hnorm⟩

end Dobson.GapFourConstants

#print axioms Dobson.GapFourConstants.exp_quarter_bound

#print axioms Dobson.GapFourConstants.p5_rouche

#print axioms Dobson.GapFourConstants.p5_ball_factor

#print axioms Dobson.GapFourConstants.p5_contraction

#print axioms Dobson.GapFourConstants.p5_error_coefficient

#print axioms Dobson.GapFourConstants.p5_radii_and_ce1

#print axioms Dobson.GapFourConstants.p5_disk

#print axioms Dobson.GapFourConstants.p7_radii

#print axioms Dobson.GapFourConstants.p7_scalar_bounds

#print axioms Dobson.GapFourConstants.p7_disk

#print axioms Dobson.GapFourConstants.p5_tail_bound

#print axioms Dobson.GapFourConstants.normalized_deviation_bound

#print axioms Dobson.GapFourConstants.finite_chain_radius
