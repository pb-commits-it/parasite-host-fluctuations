import DobsonProofs.NoiseAnalyticity
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Analytic.Basic
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Data.Nat.Choose.Central
import Mathlib.Tactic

/-!
# GAP 3: factorial integral bounds and a conditional non-analyticity criterion

Finite/scalar cores of `gap3holo2/gap3holo2.md`, sections 3–4, targets L4, L5,
L8 and the root-certificate part of L9. The hypotheses connecting these cores
to the ecological model are explicit; see the dated STATUS hypothesis ledger.
No invariant law, Tricomi connection formula or continuation theorem is asserted.
-/

/-! ## Hypothesis audit reconciliation — 2026-11-22

The independent hypothesis audit (2026-11-16) concerned earlier abstract
coefficient, series, growth-ratio, moment and continuation interfaces. Current
signatures distinguish private conditional helpers from actual integral results.
In particular the six former negative-shape limits/moment premises are discharged
by the actual Kummer integral argument below; the Tricomi continuation identity
remains supplied. The remaining model-to-series and growth-ratio identifications
must be supplied where stated; scalar remainder algebra alone does not prove them. -/

namespace Dobson.NoiseAsymptotics

open Filter Set MeasureTheory
open scoped Topology BigOperators

/-- The normalized half-integer moment `(1/2)_n`. -/
noncomputable def halfMoment : ℕ → ℝ
  | 0 => 1
  | n + 1 => ((n : ℝ) + 1 / 2) * halfMoment n

/-- The combinatorial input to the factorial majorant, for every order. -/
theorem central_binomial_bound (n : ℕ) :
    (Nat.choose (2 * n) n : ℝ) ≤ (4 : ℝ) ^ n := by
  exact_mod_cast Nat.centralBinom_le_four_pow n

/-- Exact factorial formula, including order zero. -/
theorem halfMoment_factorial (n : ℕ) :
    halfMoment n = ((2 * n).factorial : ℝ) / (4 ^ n * (n.factorial : ℝ)) := by
  induction n with
  | zero => norm_num [halfMoment]
  | succ n ih =>
    rw [halfMoment, ih, show 2 * (n + 1) = (2 * n + 1) + 1 by omega]
    rw [Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ]
    push_cast
    rw [pow_succ]
    have hn : (n.factorial : ℝ) ≠ 0 := by positivity
    have hn1 : (n : ℝ) + 1 ≠ 0 := by positivity
    field_simp
    ring

/-- Both signs and the all-order `n!` upper bound are proved, not assumed. -/
theorem halfMoment_bounds (n : ℕ) :
    0 < halfMoment n ∧ halfMoment n ≤ (n.factorial : ℝ) := by
  induction n with
  | zero => norm_num [halfMoment]
  | succ n ih =>
    rw [halfMoment, Nat.factorial_succ, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
    constructor
    · exact mul_pos (by positivity) ih.1
    · calc
        ((n : ℝ) + 1 / 2) * halfMoment n ≤ ((n : ℝ) + 1) * halfMoment n := by
          nlinarith [ih.1]
        _ ≤ ((n : ℝ) + 1) * (n.factorial : ℝ) := by gcongr; exact ih.2

/-- Identification with Mathlib's actual gamma function. -/
theorem halfMoment_gamma (n : ℕ) :
    Real.Gamma ((n : ℝ) + 1 / 2) / Real.sqrt Real.pi = halfMoment n := by
  induction n with
  | zero =>
    rw [Nat.cast_zero, zero_add, Real.Gamma_one_half_eq, halfMoment]
    exact div_self (Real.sqrt_pos.2 Real.pi_pos).ne'
  | succ n ih =>
    rw [Nat.cast_add, Nat.cast_one, add_right_comm, Real.Gamma_add_one (by positivity)]
    rw [mul_div_assoc, ih, halfMoment]

/-- A geometric coefficient bound becomes a factorial coefficient bound after
Laplace integration. This does not bound the implicit threshold coefficients. -/
theorem factorial_coefficient_bound (B : ℕ → ℝ) (M R : ℝ)
    (hM : 0 ≤ M) (hR : 0 < R) (hB : ∀ j, |B j| ≤ M / R ^ j) (j : ℕ) :
    |B j * halfMoment j| ≤ M / R ^ j * (j.factorial : ℝ) := by
  rw [abs_mul, abs_of_pos (halfMoment_bounds j).1]
  exact mul_le_mul (hB j) (halfMoment_bounds j).2 (halfMoment_bounds j).1.le
    (div_nonneg hM (pow_pos hR j).le)

/-- The truncated amplitude polynomial. -/
noncomputable def amplitudeSum (B : ℕ → ℝ) (N : ℕ) (u : ℝ) : ℝ :=
  ∑ j ∈ Finset.range N, B j * u ^ j

private lemma half_geometric_sum (N : ℕ) :
    ∑ j ∈ Finset.range N, (1 / 2 : ℝ) ^ j = 2 * (1 - (1 / 2 : ℝ) ^ N) := by
  induction N with
  | zero => simp
  | succ N ih => rw [Finset.sum_range_succ, ih, pow_succ]; ring

/-- The convergent series hypothesis yields the actual geometric tail on `[0,R/2]`. -/
theorem small_amplitude_remainder (B : ℕ → ℝ) (A M R u : ℝ) (N : ℕ)
    (hM : 0 ≤ M) (hR : 0 < R) (hu : 0 ≤ u) (huR : u ≤ R / 2)
    (hB : ∀ j, |B j| ≤ M / R ^ j)
    (hseries : HasSum (fun j => B j * u ^ j) A) :
    |A - amplitudeSum B N u| ≤ 2 * M * (u / R) ^ N := by
  have hr0 : 0 ≤ u / R := div_nonneg hu hR.le
  have hr : u / R ≤ 1 / 2 := (div_le_iff₀ hR).2 (by linarith)
  have ht : ∀ j, ‖B j * u ^ j‖ ≤ M * (u / R) ^ j := by
    intro j
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (pow_nonneg hu j), div_pow]
    calc
      |B j| * u ^ j ≤ (M / R ^ j) * u ^ j :=
        mul_le_mul_of_nonneg_right (hB j) (pow_nonneg hu j)
      _ = M * (u ^ j / R ^ j) := by ring
  have h := norm_sub_le_of_geometric_bound_of_hasSum (by linarith : u / R < 1)
    ht hseries N
  rw [Real.norm_eq_abs, abs_sub_comm] at h
  apply h.trans
  apply (div_le_iff₀ (by linarith : 0 < 1 - u / R)).2
  have hp : 0 ≤ M * (u / R) ^ N := mul_nonneg hM (pow_nonneg hr0 N)
  nlinarith

/-- For `u ≥ b`, every lower power is controlled by the order-`N` power. -/
theorem large_amplitude_polynomial (B : ℕ → ℝ) (M b u : ℝ) (N : ℕ)
    (hM : 0 ≤ M) (hb : 0 < b) (hu : b ≤ u)
    (hB : ∀ j, |B j| ≤ M / (2 * b) ^ j) :
    |amplitudeSum B N u| ≤ 2 * M * (u / b) ^ N := by
  have hx : 1 ≤ u / b := (le_div_iff₀ hb).2 (by simpa using hu)
  have hu0 : 0 ≤ u := hb.le.trans hu
  have hterm : ∀ j ∈ Finset.range N,
      |B j * u ^ j| ≤ M * (u / b) ^ N * (1 / 2 : ℝ) ^ j := by
    intro j hj
    have hp : (u / b) ^ j ≤ (u / b) ^ N :=
      pow_le_pow_right₀ hx (Nat.le_of_lt (Finset.mem_range.mp hj))
    rw [abs_mul, abs_of_nonneg (pow_nonneg hu0 j)]
    calc
      |B j| * u ^ j ≤ M / (2 * b) ^ j * u ^ j :=
        mul_le_mul_of_nonneg_right (hB j) (pow_nonneg hu0 j)
      _ = M * (u / b) ^ j * (1 / 2 : ℝ) ^ j := by
        rw [mul_pow, div_pow, div_pow]; ring
      _ ≤ M * (u / b) ^ N * (1 / 2 : ℝ) ^ j := by gcongr
  calc
    |amplitudeSum B N u| ≤ ∑ j ∈ Finset.range N, |B j * u ^ j| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j ∈ Finset.range N, M * (u / b) ^ N * (1 / 2 : ℝ) ^ j :=
      Finset.sum_le_sum hterm
    _ = M * (u / b) ^ N * (2 * (1 - (1 / 2 : ℝ) ^ N)) := by
      rw [← Finset.mul_sum, half_geometric_sum]
    _ ≤ 2 * M * (u / b) ^ N := by
      have hpos : 0 ≤ M * (u / b) ^ N := by positivity
      nlinarith [mul_nonneg hpos (pow_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2) N)]

/-- The polynomial growth term is absorbed when `N ≥ 2`. -/
theorem quadratic_growth_bound (b u : ℝ) (N : ℕ) (hb : 0 < b)
    (hu : b ≤ u) (hN : 2 ≤ N) :
    1 + u ^ 2 ≤ (1 + b ^ 2) * (u / b) ^ N := by
  have hx : 1 ≤ u / b := (le_div_iff₀ hb).2 (by simpa using hu)
  have hp : (u / b) ^ 2 ≤ (u / b) ^ N := pow_le_pow_right₀ hx hN
  have hp1 : 1 ≤ (u / b) ^ N := one_le_pow₀ hx
  have hid : b ^ 2 * (u / b) ^ 2 = u ^ 2 := by field_simp
  nlinarith [mul_le_mul_of_nonneg_left hp (sq_nonneg b)]

/-- L8's complete pointwise remainder, with the specified constant `4M+C(1+b²)`. -/
theorem amplitude_remainder (B : ℕ → ℝ) (A : ℝ → ℝ) (M C b : ℝ) (N : ℕ)
    (hM : 0 ≤ M) (hC : 0 ≤ C) (hb : 0 < b) (hN : 2 ≤ N)
    (hB : ∀ j, |B j| ≤ M / (2 * b) ^ j)
    (hseries : ∀ u ∈ Icc 0 b, HasSum (fun j => B j * u ^ j) (A u))
    (hgrowth : ∀ u, b ≤ u → |A u| ≤ C * (1 + u ^ 2))
    (u : ℝ) (hu : 0 ≤ u) :
    |A u - amplitudeSum B N u| ≤ (4 * M + C * (1 + b ^ 2)) / b ^ N * u ^ N := by
  have he : 2 * M ≤ 4 * M + C * (1 + b ^ 2) := by
    nlinarith [mul_nonneg hC (sq_nonneg b)]
  have hid : (4 * M + C * (1 + b ^ 2)) / b ^ N * u ^ N =
      (4 * M + C * (1 + b ^ 2)) * (u / b) ^ N := by rw [div_pow]; ring
  rw [hid]
  by_cases hub : u ≤ b
  · have h := small_amplitude_remainder B (A u) M (2 * b) u N hM (by positivity)
      hu (by linarith) hB (hseries u ⟨hu, hub⟩)
    apply h.trans
    apply mul_le_mul he
    · apply pow_le_pow_left₀ (by positivity)
      exact div_le_div_of_nonneg_left hu hb (by linarith)
    · positivity
    · positivity
  · have hbu : b ≤ u := le_of_lt (lt_of_not_ge hub)
    calc
      |A u - amplitudeSum B N u| ≤ |A u| + |amplitudeSum B N u| := abs_sub _ _
      _ ≤ C * (1 + u ^ 2) + 2 * M * (u / b) ^ N :=
        add_le_add (hgrowth u hbu) (large_amplitude_polynomial B M b u N hM hb hbu hB)
      _ ≤ C * ((1 + b ^ 2) * (u / b) ^ N) + 2 * M * (u / b) ^ N := by
        gcongr
        exact quadratic_growth_bound b u N hb hbu hN
      _ ≤ (4 * M + C * (1 + b ^ 2)) * (u / b) ^ N := by
        nlinarith [mul_nonneg hM (pow_nonneg (div_nonneg hu hb.le) N)]

/-- Finite Laplace expansion, with coefficients `B_j (1/2)_j`. -/
noncomputable def laplaceSum (B : ℕ → ℝ) (N : ℕ) (v : ℝ) : ℝ :=
  ∑ j ∈ Finset.range N, B j * halfMoment j * v ^ j

/-- Integration of the scalar bound against a nonnegative measure. The support,
integrability and half-moment identities are explicit analytic inputs. -/
theorem factorial_integral_remainder (μ : Measure ℝ) (B : ℕ → ℝ) (A : ℝ → ℝ)
    (M C b v : ℝ) (N : ℕ) (hM : 0 ≤ M) (hC : 0 ≤ C) (hb : 0 < b)
    (hv : 0 ≤ v) (hN : 2 ≤ N)
    (hB : ∀ j, |B j| ≤ M / (2 * b) ^ j)
    (hseries : ∀ u ∈ Icc 0 b, HasSum (fun j => B j * u ^ j) (A u))
    (hgrowth : ∀ u, b ≤ u → |A u| ≤ C * (1 + u ^ 2))
    (hsupport : ∀ᵐ u ∂μ, 0 ≤ u) (hA : Integrable A μ)
    (hpowers : ∀ j ≤ N, Integrable (fun u : ℝ => u ^ j) μ)
    (hmoments : ∀ j ≤ N, (∫ u : ℝ, u ^ j ∂μ) = halfMoment j * v ^ j) :
    |(∫ u, A u ∂μ) - laplaceSum B N v| ≤
      (4 * M + C * (1 + b ^ 2)) / b ^ N * (N.factorial : ℝ) * v ^ N := by
  let E := (4 * M + C * (1 + b ^ 2)) / b ^ N
  have hE : 0 ≤ E := by dsimp [E]; positivity
  have hi : ∀ j ∈ Finset.range N, Integrable (fun u : ℝ => B j * u ^ j) μ := by
    intro j hj
    exact (hpowers j (Nat.le_of_lt (Finset.mem_range.mp hj))).const_mul _
  have hpoly : Integrable (amplitudeSum B N) μ := integrable_finsetSum _ hi
  have hpolyval : (∫ u, amplitudeSum B N u ∂μ) = laplaceSum B N v := by
    unfold amplitudeSum laplaceSum
    rw [integral_finsetSum _ hi]
    apply Finset.sum_congr rfl
    intro j hj
    rw [integral_const_mul, hmoments j (Nat.le_of_lt (Finset.mem_range.mp hj))]
    ring
  have hbound : |∫ u, A u - amplitudeSum B N u ∂μ| ≤ E * (halfMoment N * v ^ N) := by
    have h := norm_integral_le_of_norm_le ((hpowers N le_rfl).const_mul E)
      (show ∀ᵐ u ∂μ, ‖A u - amplitudeSum B N u‖ ≤ E * u ^ N from by
        filter_upwards [hsupport] with u hu
        simpa only [Real.norm_eq_abs] using
          amplitude_remainder B A M C b N hM hC hb hN hB hseries hgrowth u hu)
    simpa only [Real.norm_eq_abs, integral_const_mul, hmoments N le_rfl] using h
  rw [integral_sub hA hpoly, hpolyval] at hbound
  apply hbound.trans
  calc
    E * (halfMoment N * v ^ N) = E * halfMoment N * v ^ N := by ring
    _ ≤ E * (N.factorial : ℝ) * v ^ N := by
      gcongr
      exact (halfMoment_bounds N).2

/-- The positive root appearing in the strict moment bound (3.4). -/
noncomputable def momentRoot (q X : ℝ) : ℝ :=
  (1 - X + Real.sqrt ((X - 1) ^ 2 + 4 * X * q)) / (2 * X)

/-- L5: strict variance and the integration-by-parts identity imply (3.4).
They are scalar hypotheses here, not asserted properties of a constructed law. -/
theorem quadratic_moment_bound (q X m1 m2 : ℝ) (hq : 0 < q) (hX : 0 < X)
    (_hm1 : 0 < m1) (hvariance : m1 ^ 2 < m2)
    (hibp : X * m2 + (X - 1) * m1 = q) : m1 < momentRoot q X := by
  have hdisc : 0 ≤ (X - 1) ^ 2 + 4 * X * q := by positivity
  have hsqrt := Real.sq_sqrt hdisc
  have hsqrt0 := Real.sqrt_nonneg ((X - 1) ^ 2 + 4 * X * q)
  have hpoly : X * m1 ^ 2 + (X - 1) * m1 < q := by
    nlinarith [mul_lt_mul_of_pos_left hvariance hX]
  unfold momentRoot
  apply (lt_div_iff₀ (by positivity : 0 < 2 * X)).2
  by_contra h
  have hside : Real.sqrt ((X - 1) ^ 2 + 4 * X * q) ≤ 2 * X * m1 + X - 1 := by
    linarith
  have hsquare := mul_self_le_mul_self hsqrt0 hside
  nlinarith [mul_pos hX (sub_pos.mpr hpoly)]

/-- Exact limiting positive root, with uniqueness among positive solutions. -/
theorem limiting_positive_root (s t : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (ht : 0 < t) :
    s ^ 2 / (1 - s) * (t ^ 2 + t) = 1 ↔ t = (1 - s) / s := by
  have hs : s ≠ 0 := hs0.ne'
  have ha : 1 - s ≠ 0 := by linarith
  have hid : s ^ 2 / (1 - s) * (((1 - s) / s) ^ 2 + (1 - s) / s) = 1 := by
    field_simp
    ring
  constructor
  · intro h
    have hpos : 0 < s ^ 2 / (1 - s) := by positivity
    have ht0 : 0 < (1 - s) / s := by positivity
    have hfactor : (t - (1 - s) / s) * (t + (1 - s) / s + 1) = 0 := by
      have : t ^ 2 + t = ((1 - s) / s) ^ 2 + (1 - s) / s :=
        (mul_left_cancel₀ hpos.ne') (h.trans hid.symm)
      nlinarith
    rcases mul_eq_zero.mp hfactor with he | he
    · linarith
    · linarith
  · rintro rfl
    exact hid

/-- L9: a derivative lower bound converts a residual into distance to the root. -/
theorem root_residual_bound (r : ℝ → ℝ) (η lam D Y : ℝ) (hlam : 0 < lam)
    (hcont : ContinuousOn r (Icc (-η) η))
    (hdiff : DifferentiableOn ℝ r (Ioo (-η) η))
    (hslope : ∀ x ∈ Ioo (-η) η, lam ≤ deriv r x)
    (hD : D ∈ Icc (-η) η) (hY : Y ∈ Icc (-η) η) (hroot : r D = 0) :
    |D - Y| ≤ |r Y| / lam := by
  have hmono := (convex_Icc (-η) η).mul_sub_le_image_sub_of_le_deriv hcont
    (by simpa only [interior_Icc] using hdiff)
    (by simpa only [interior_Icc] using hslope)
  apply (le_div_iff₀ hlam).2
  rcases le_total D Y with h | h
  · have hm := hmono D hD Y hY h
    rw [hroot] at hm
    rw [abs_of_nonpos (sub_nonpos.mpr h)]
    nlinarith [le_abs_self (r Y)]
  · have hm := hmono Y hY D hD h
    rw [hroot] at hm
    rw [abs_of_nonneg (sub_nonneg.mpr h)]
    nlinarith [neg_le_abs (r Y)]

/-- Exact numerator/denominator assembly in (4.5); `P` may be `laplaceSum B N v`.
The growth ratio and positive denominator lower bound remain explicit inputs. -/
theorem threshold_certificate (r : ℝ → ℝ) (η lam D Y Jg J1 L P E : ℝ)
    (hlam : 0 < lam) (hL : 0 < L)
    (hcont : ContinuousOn r (Icc (-η) η))
    (hdiff : DifferentiableOn ℝ r (Ioo (-η) η))
    (hslope : ∀ x ∈ Ioo (-η) η, lam ≤ deriv r x)
    (hD : D ∈ Icc (-η) η) (hY : Y ∈ Icc (-η) η) (hroot : r D = 0)
    (hratio : r Y = Jg / J1) (hden : L ≤ J1) (herr : |Jg - P| ≤ E) :
    |D - Y| ≤ (|P| + E) / (lam * L) := by
  have hJ1 : 0 < J1 := lt_of_lt_of_le hL hden
  have hnum : |Jg| ≤ |P| + E := by
    have htri := abs_add_le (Jg - P) P
    rw [sub_add_cancel] at htri
    linarith
  have hPE : 0 ≤ |P| + E := (abs_nonneg Jg).trans hnum
  calc
    |D - Y| ≤ |r Y| / lam := root_residual_bound r η lam D Y hlam hcont hdiff hslope hD hY hroot
    _ = |Jg| / J1 / lam := by rw [hratio, abs_div, abs_of_pos hJ1]
    _ ≤ (|P| + E) / L / lam := by gcongr
    _ = (|P| + E) / (lam * L) := by ring

/-- The negative half-integer phase has zero real part, for every integer `m`. -/
theorem negative_shape_phase (m : ℤ) :
    (Complex.exp (-((Real.pi * ((m : ℝ) + 1 / 2) : ℝ) : ℂ) * Complex.I)).re = 0 := by
  rw [Complex.exp_re]
  simp only [Complex.mul_re, Complex.mul_im, Complex.neg_re, Complex.neg_im,
    Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im,
    mul_zero, sub_zero, mul_one, add_zero, neg_zero, Real.exp_zero, one_mul]
  rw [Real.cos_neg, show Real.pi * ((m : ℝ) + 1 / 2) =
    (m : ℝ) * Real.pi + Real.pi / 2 by ring, Real.cos_add_pi_div_two,
    Real.sin_int_mul_pi, neg_zero]

/-- Real-parameter restriction of the two terms in the connection formula. -/
noncomputable def connectionWave (A B : ℝ → ℝ) (τ : ℂ) (x : ℝ) : ℂ :=
  Complex.exp (((Real.pi * (x - 1) : ℝ) : ℂ) * Complex.I) * (A x : ℂ) + τ * (B x : ℂ)

/-- L4 differentiation, before taking real parts. This uses real derivatives of
`A,B`; it does not assume a zero-free Tricomi function. -/
theorem connectionWave_derivative (A B : ℝ → ℝ) (A' B' : ℝ) (τ : ℂ)
    (hA : HasDerivAt A A' 1) (hB : HasDerivAt B B' 1) :
    HasDerivAt (connectionWave A B τ)
      ((Real.pi : ℂ) * Complex.I * (A 1 : ℂ) + (A' : ℂ) + τ * (B' : ℂ)) 1 := by
  have hphase : HasDerivAt (fun x : ℝ =>
      Complex.exp (((Real.pi * (x - 1) : ℝ) : ℂ) * Complex.I))
      ((Real.pi : ℂ) * Complex.I) 1 := by
    have hx := (((hasDerivAt_id (1 : ℝ)).sub_const 1).const_mul Real.pi).ofReal_comp
    convert (hx.mul_const Complex.I).cexp using 1 <;> simp
  unfold connectionWave
  convert (hphase.mul hA.ofReal_comp).add (hB.ofReal_comp.const_mul τ) using 1
  simp

/-- L4 real-part extraction from the undivided equation. -/
theorem connection_real_part (A B : ℝ → ℝ) (A' B' α : ℝ) (τ : ℂ)
    (hA : HasDerivAt A A' 1) (hB : HasDerivAt B B' 1) (hτ : τ.re = 0) :
    ((α : ℂ) * connectionWave A B τ 1 + deriv (connectionWave A B τ) 1).re =
      α * A 1 + A' := by
  rw [(connectionWave_derivative A B A' B' τ hA hB).deriv]
  simp [connectionWave, Complex.mul_re, Complex.mul_im, hτ]

/-- Division is only by the explicitly nonzero real amplitude `A(1)`. -/
theorem undivided_threshold_extraction (A B : ℝ → ℝ) (A' B' α : ℝ) (τ : ℂ)
    (hA : HasDerivAt A A' 1) (hB : HasDerivAt B B' 1) (hτ : τ.re = 0)
    (hA0 : A 1 ≠ 0)
    (hzero : (α : ℂ) * connectionWave A B τ 1 + deriv (connectionWave A B τ) 1 = 0) :
    α + A' / A 1 = 0 := by
  have h := connection_real_part A B A' B' α τ hA hB hτ
  rw [hzero, Complex.zero_re] at h
  rw [show α + A' / A 1 = (α * A 1 + A') / A 1 by field_simp]
  rw [← h, zero_div]

/-- Finite contradiction core of section 3, in normalized moment coordinates.
`k = X/q`, `e = 1/q`, `m1 = E T`, `m2 = E T²`, and `ell = E log T`. -/
theorem negative_shape_incompatible (s k e m1 m2 ell : ℝ)
    (hs : 0 < s) (hk : 0 < k) (hke : 0 ≤ k - e) (hm1 : 0 < m1)
    (hcritical : 1 < k * (1 / s) ^ 2 + (k - e) * (1 / s))
    (hvariance : m1 ^ 2 < m2) (hibp : k * m2 + (k - e) * m1 = 1)
    (hjensen : ell ≤ Real.log m1) (hzero : Real.log s + ell = 0) : False := by
  have hlower : 1 / s ≤ m1 := by
    apply (Real.log_le_log_iff (by positivity) hm1).mp
    rw [Real.log_div (by norm_num) hs.ne', Real.log_one]
    linarith
  have hquad : k * (1 / s) ^ 2 ≤ k * m1 ^ 2 := by gcongr
  have hlin : (k - e) * (1 / s) ≤ (k - e) * m1 :=
    mul_le_mul_of_nonneg_left hlower hke
  nlinarith [mul_lt_mul_of_pos_left hvariance hk]

/-! ## Dated correction — 2026-11-16, leanowed
The old abstract sequence criterion assumed six limits/moment facts (#125–130
in audit part 2). They are now discharged: half-integer coordinates are defined
from D, the limit follows under the contradicted analyticity assumption, and
`L3_kummer_moments` proves the needed integral bound. Only #131, the Tricomi
continuation identity, remains a paper-specific premise. Its paper proof is
`gap3holo2.md` §3, independently checked in the continuation review, Step 3.
This dated note supersedes the earlier class-(c) wording for this theorem. -/

/-- Positive Kummer shape after continuation to p=-(n+1/2): q=1-p. -/
noncomputable def negativeShapeQ (n : ℕ) : ℝ := (n : ℝ) + 3/2
/-- Actual positive lower-bank argument X=-x at v=-1/(n+1/2). -/
noncomputable def negativeShapeX (D : ℝ → ℝ) (s : ℝ) (n : ℕ) : ℝ :=
  ((n : ℝ)+1/2)*s^2/((1-s)*Real.exp (D (-((n:ℝ)+1/2)⁻¹)))

/-- The negative-shape scaling is derived from analyticity and D(0)=0. -/
theorem negative_shape_scaling (D : ℝ → ℝ) (s : ℝ)
    (hs0 : 0 < s) (hs1 : s < 1) (hD0 : D 0 = 0) (hD : AnalyticAt ℝ D 0) :
    Tendsto negativeShapeQ atTop atTop ∧
    Tendsto (fun n => negativeShapeX D s n / negativeShapeQ n)
      atTop (𝓝 (s^2/(1-s))) := by
  have hn : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop := tendsto_natCast_atTop_atTop
  have hq : Tendsto negativeShapeQ atTop atTop := tendsto_atTop_add_const_right atTop (3/2) hn
  have hp : Tendsto (fun n : ℕ => (n:ℝ)+1/2) atTop atTop := tendsto_atTop_add_const_right atTop (1/2) hn
  have he : Tendsto (fun n : ℕ => -((n:ℝ)+1/2)⁻¹) atTop (𝓝 0) := by
    simpa using (tendsto_inv_atTop_zero.comp hp).neg
  have hDlim : Tendsto (fun n : ℕ => D (-((n:ℝ)+1/2)⁻¹)) atTop (𝓝 0) := by
    simpa [hD0, Function.comp_def] using hD.continuousAt.tendsto.comp he
  have hexp : Tendsto (fun n : ℕ => Real.exp (D (-((n:ℝ)+1/2)⁻¹))) atTop (𝓝 1) := by
    simpa [Function.comp_def] using Real.continuous_exp.continuousAt.tendsto.comp hDlim
  have hnorm : Tendsto (fun n : ℕ => 1-(negativeShapeQ n)⁻¹) atTop (𝓝 1) := by
    simpa using tendsto_const_nhds.sub (tendsto_inv_atTop_zero.comp hq)
  refine ⟨hq, ?_⟩
  have hl := (hnorm.mul_const (s^2)).div (hexp.const_mul (1-s)) (by linarith : (1-s)*1 ≠ 0)
  simp only [one_mul, mul_one] at hl
  convert hl using 1
  ext n
  dsimp [negativeShapeX, negativeShapeQ]
  have hnq : (n:ℝ)+3/2 ≠ 0 := by positivity
  field_simp
  ring

/-- Iid-gamma nonanalyticity with actual Kummer moments. The only analytical
handoff is the undivided Tricomi continuation identity. The density shape is
n+3/2, whereas X and the negative variance use n+1/2. The noiseless
normalization D(0)=0 is explicit; it is not an asymptotic hypothesis. -/
theorem not_analytic_of_negative_shape (D : ℝ → ℝ) (s : ℝ) (hs0 : 0 < s) (hs1 : s < 1)
    (hD0 : D 0 = 0)
    (hcontinuation : AnalyticAt ℝ D 0 → ∀ᶠ n in atTop,
      Real.log s + NoiseAnalyticity.kummerExpect (negativeShapeQ n)
        (negativeShapeX D s n) Real.log = 0) : ¬ AnalyticAt ℝ D 0 := by
  intro hD
  obtain ⟨hq, hxq⟩ := negative_shape_scaling D s hs0 hs1 hD0 hD
  obtain ⟨hlim, hneg⟩ := NoiseAnalyticity.L3_threshold_limit s negativeShapeQ
    (negativeShapeX D s) hs0 hs1 hq hxq
  have hevent := hlim.eventually (eventually_lt_nhds hneg)
  have hfalse : ∀ᶠ _n : ℕ in atTop, False := by
    filter_upwards [hevent, hcontinuation hD] with n hn hzero
    have hqp : 0 < negativeShapeQ n := by dsimp [negativeShapeQ]; positivity
    have hxp : 0 < negativeShapeX D s n := by
      dsimp [negativeShapeX]
      apply div_pos (by positivity) (mul_pos (by linarith) (Real.exp_pos _))
    have hj := (NoiseAnalyticity.L3_kummer_moments (negativeShapeQ n)
      (negativeShapeX D s n) hqp hxp).2.2.2.2.2.2.2.2
    linarith
  exact hfalse.exists.elim fun _ h => h

/-- The actual normalized Laplace kernel on the positive half-line. Its density
is `exp(-u/v)/(sqrt(pi*v)*sqrt(u))`; moments and integrability are not hidden. -/
noncomputable def laplaceMeasure (v : ℝ) : Measure ℝ :=
  (volume.withDensity (fun u : ℝ => ENNReal.ofReal
    (Real.exp (-u / v) / (Real.sqrt (Real.pi * v) * Real.sqrt u)))).restrict (Ioi 0)

/-- Equation (4.2) for the explicit Laplace measure. Identifying the original
saddle integral with this amplitude integral remains a separate analytic step. -/
theorem laplace_remainder (B : ℕ → ℝ) (A : ℝ → ℝ) (M C b v : ℝ) (N : ℕ)
    (hM : 0 ≤ M) (hC : 0 ≤ C) (hb : 0 < b) (hv : 0 < v) (hN : 2 ≤ N)
    (hB : ∀ j, |B j| ≤ M / (2 * b) ^ j)
    (hseries : ∀ u ∈ Icc 0 b, HasSum (fun j => B j * u ^ j) (A u))
    (hgrowth : ∀ u, b ≤ u → |A u| ≤ C * (1 + u ^ 2))
    (hA : Integrable A (laplaceMeasure v))
    (hpowers : ∀ j ≤ N, Integrable (fun u : ℝ => u ^ j) (laplaceMeasure v))
    (hmoments : ∀ j ≤ N, (∫ u : ℝ, u ^ j ∂laplaceMeasure v) = halfMoment j * v ^ j) :
    |(∫ u, A u ∂laplaceMeasure v) - laplaceSum B N v| ≤
      (4 * M + C * (1 + b ^ 2)) / b ^ N * (N.factorial : ℝ) * v ^ N := by
  apply factorial_integral_remainder (laplaceMeasure v) B A M C b v N
    hM hC hb hv.le hN hB hseries hgrowth ?_ hA hpowers hmoments
  filter_upwards [ae_restrict_mem (μ := volume.withDensity _) measurableSet_Ioi] with u hu
  exact le_of_lt hu

/-- Equation (4.3)'s explicit lower constant from a local integrand envelope.
The envelope is a hypothesis; interval integration and cancellation of `sqrt v`
are proved here. `f` is extended by zero off its original integration domain. -/
theorem local_integral_lower_bound (f : ℝ → ℝ) (t0 d M2 v : ℝ)
    (hd : 0 < d) (htd : 0 < t0 + d) (hv : 0 < v)
    (hint : Integrable f) (hnonneg : ∀ t, 0 ≤ f t)
    (hlocal : ∀ t ∈ Icc (t0 - d * Real.sqrt v) (t0 + d * Real.sqrt v),
      Real.exp (-M2 * d ^ 2 / 2) / (t0 + d) ≤ f t) :
    0 < 2 * d * Real.exp (-M2 * d ^ 2 / 2) / (Real.sqrt Real.pi * (t0 + d)) ∧
    2 * d * Real.exp (-M2 * d ^ 2 / 2) / (Real.sqrt Real.pi * (t0 + d)) ≤
      (∫ t, f t) / Real.sqrt (Real.pi * v) := by
  have hvroot : 0 < Real.sqrt v := Real.sqrt_pos.2 hv
  have hpiroot : 0 < Real.sqrt Real.pi := Real.sqrt_pos.2 Real.pi_pos
  have hwidth : 0 ≤ (t0 + d * Real.sqrt v) - (t0 - d * Real.sqrt v) := by
    nlinarith [mul_pos hd hvroot]
  have hlow : 2 * d * Real.sqrt v * (Real.exp (-M2 * d ^ 2 / 2) / (t0 + d)) ≤
      ∫ t, f t := by
    calc
      _ = ∫ t in Icc (t0 - d * Real.sqrt v) (t0 + d * Real.sqrt v),
          Real.exp (-M2 * d ^ 2 / 2) / (t0 + d) := by
        rw [setIntegral_const, Real.volume_real_Icc, max_eq_left hwidth, smul_eq_mul]
        ring
      _ ≤ ∫ t in Icc (t0 - d * Real.sqrt v) (t0 + d * Real.sqrt v), f t := by
        apply setIntegral_mono_on (integrableOn_const isCompact_Icc.measure_ne_top)
          hint.integrableOn measurableSet_Icc hlocal
      _ ≤ ∫ t, f t := setIntegral_le_integral hint (Eventually.of_forall hnonneg)
  constructor
  · positivity
  · rw [Real.sqrt_mul Real.pi_pos.le]
    apply (le_div_iff₀ (mul_pos hpiroot hvroot)).2
    convert hlow using 1
    field_simp

/-- The complete explicit residual certificate (4.5), using the proved integral
remainder as its error term. Only the growth ratio and denominator comparison
connect this scalar calculation to a threshold of the original stochastic model. -/
theorem laplace_threshold_certificate (r A : ℝ → ℝ) (B : ℕ → ℝ)
    (η lam D Y J1 L M C b v : ℝ) (N : ℕ)
    (hlam : 0 < lam) (hL : 0 < L) (hM : 0 ≤ M) (hC : 0 ≤ C)
    (hb : 0 < b) (hv : 0 < v) (hN : 2 ≤ N)
    (hcont : ContinuousOn r (Icc (-η) η))
    (hdiff : DifferentiableOn ℝ r (Ioo (-η) η))
    (hslope : ∀ x ∈ Ioo (-η) η, lam ≤ deriv r x)
    (hD : D ∈ Icc (-η) η) (hY : Y ∈ Icc (-η) η) (hroot : r D = 0)
    (hratio : r Y = (∫ u, A u ∂laplaceMeasure v) / J1) (hden : L ≤ J1)
    (hB : ∀ j, |B j| ≤ M / (2 * b) ^ j)
    (hseries : ∀ u ∈ Icc 0 b, HasSum (fun j => B j * u ^ j) (A u))
    (hgrowth : ∀ u, b ≤ u → |A u| ≤ C * (1 + u ^ 2))
    (hA : Integrable A (laplaceMeasure v))
    (hpowers : ∀ j ≤ N, Integrable (fun u : ℝ => u ^ j) (laplaceMeasure v))
    (hmoments : ∀ j ≤ N, (∫ u : ℝ, u ^ j ∂laplaceMeasure v) = halfMoment j * v ^ j) :
    |D - Y| ≤ (|laplaceSum B N v| +
      (4 * M + C * (1 + b ^ 2)) / b ^ N * (N.factorial : ℝ) * v ^ N) / (lam * L) := by
  exact threshold_certificate r η lam D Y (∫ u, A u ∂laplaceMeasure v) J1 L
    (laplaceSum B N v)
    ((4 * M + C * (1 + b ^ 2)) / b ^ N * (N.factorial : ℝ) * v ^ N)
    hlam hL hcont hdiff hslope hD hY hroot hratio hden
    (laplace_remainder B A M C b v N hM hC hb hv hN hB hseries hgrowth hA hpowers hmoments)

end Dobson.NoiseAsymptotics

#print axioms Dobson.NoiseAsymptotics.central_binomial_bound
#print axioms Dobson.NoiseAsymptotics.halfMoment_factorial
#print axioms Dobson.NoiseAsymptotics.halfMoment_bounds
#print axioms Dobson.NoiseAsymptotics.halfMoment_gamma
#print axioms Dobson.NoiseAsymptotics.factorial_coefficient_bound
#print axioms Dobson.NoiseAsymptotics.small_amplitude_remainder
#print axioms Dobson.NoiseAsymptotics.large_amplitude_polynomial
#print axioms Dobson.NoiseAsymptotics.quadratic_growth_bound
#print axioms Dobson.NoiseAsymptotics.amplitude_remainder
#print axioms Dobson.NoiseAsymptotics.factorial_integral_remainder
#print axioms Dobson.NoiseAsymptotics.quadratic_moment_bound
#print axioms Dobson.NoiseAsymptotics.limiting_positive_root
#print axioms Dobson.NoiseAsymptotics.root_residual_bound
#print axioms Dobson.NoiseAsymptotics.threshold_certificate
#print axioms Dobson.NoiseAsymptotics.negative_shape_phase
#print axioms Dobson.NoiseAsymptotics.connectionWave_derivative
#print axioms Dobson.NoiseAsymptotics.connection_real_part
#print axioms Dobson.NoiseAsymptotics.undivided_threshold_extraction
#print axioms Dobson.NoiseAsymptotics.negative_shape_incompatible
#print axioms Dobson.NoiseAsymptotics.not_analytic_of_negative_shape
#print axioms Dobson.NoiseAsymptotics.laplace_remainder
#print axioms Dobson.NoiseAsymptotics.local_integral_lower_bound
#print axioms Dobson.NoiseAsymptotics.laplace_threshold_certificate

#print axioms Dobson.NoiseAsymptotics.negative_shape_scaling
