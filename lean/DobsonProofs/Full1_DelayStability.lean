import DobsonProofs.Full1_BlockQualification
import DobsonProofs.OneHostOneDelay
import DobsonProofs.Island
import DobsonProofs.Coverage2Dynamics

/-! Root-based Routh equivalence and stability calculations for the
displayed three-state host/delay matrix. The Jacobian is entered as a
definition, not derived by differentiating the nonlinear model.
Coverage correction, 2026-11-22: `prop:dyn-k4` has core coverage. -/

namespace Dobson.Full1.DelayStability

open Dobson.Full1.BlockQualification
open scoped Topology
open Filter

def cubic (b₂ b₁ b₀ : ℝ) (z : ℂ) : ℂ := z^3 + b₂*z^2 + b₁*z + b₀

theorem real_cubic_root (b₂ b₁ b₀ : ℝ) :
    ∃ r : ℝ, r^3 + b₂*r^2 + b₁*r + b₀ = 0 := by
  let p : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^3 +
    Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₀
  have hd : p.natDegree = 3 := Polynomial.natDegree_cubic (by norm_num)
  obtain ⟨r, hr⟩ := IsRealClosed.exists_isRoot_of_odd_natDegree (f := p)
    (by rw [hd]; decide)
  exact ⟨r, by simpa [Polynomial.IsRoot, p] using hr⟩

theorem cubic_factor (b₂ b₁ b₀ r : ℝ)
    (hr : r^3 + b₂*r^2 + b₁*r + b₀ = 0) (z : ℂ) :
    cubic b₂ b₁ b₀ z = (z - r) * quad (b₂ + r) (b₁ + b₂*r + r^2) z := by
  have hc : (r : ℂ)^3 + (b₂ : ℂ)*r^2 + (b₁ : ℂ)*r + b₀ = 0 := by exact_mod_cast hr
  simp only [cubic, quad]
  push_cast
  linear_combination hc

/-- Full cubic Routh--Hurwitz equivalence, without a Routh hypothesis. -/
theorem cubic_hurwitz_iff (b₂ b₁ b₀ : ℝ) :
    Hurwitz (cubic b₂ b₁ b₀) ↔
      0 < b₂ ∧ 0 < b₁ ∧ 0 < b₀ ∧ b₀ < b₂*b₁ := by
  constructor
  · intro hs
    obtain ⟨r, hr⟩ := real_cubic_root b₂ b₁ b₀
    have hrC : cubic b₂ b₁ b₀ (r : ℂ) = 0 := by
      unfold cubic; exact_mod_cast hr
    have hrneg : r < 0 := hs r hrC
    have hq : Hurwitz (quad (b₂+r) (b₁+b₂*r+r^2)) := by
      intro z hz
      apply hs z
      rw [cubic_factor _ _ _ _ hr, hz, mul_zero]
    obtain ⟨hA, hB⟩ := (quadratic_hurwitz _ _).mp hq
    have h2 : 0 < b₂ := by linarith
    have h1 : 0 < b₁ := by nlinarith [mul_neg_of_neg_of_pos hrneg hA]
    have h0 : 0 < b₀ := by nlinarith [mul_neg_of_neg_of_pos hrneg hB]
    have hm : 0 < (b₂+r)*(b₁+r^2) := mul_pos hA (by nlinarith [sq_nonneg r])
    exact ⟨h2, h1, h0, by nlinarith [hr]⟩
  · rintro ⟨h2, h1, h0, hm⟩ z hz
    exact Dobson.OneHostOneDelay.D3 1 b₂ b₁ b₀ (by norm_num) h2 h1 h0
      (by simpa using hm) z (by simpa [cubic] using hz)

theorem quadratic_unstable_of_neg (A B : ℝ) (hA : A < 0) :
    ∃ z : ℂ, quad A B z = 0 ∧ 0 < z.re := by
  by_cases hd : 0 ≤ A^2 - 4*B
  · let t := Real.sqrt (A^2 - 4*B)
    have ht : t^2 = A^2-4*B := Real.sq_sqrt hd
    refine ⟨(((-A+t)/2 : ℝ) : ℂ), ?_, ?_⟩
    · unfold quad
      have he : ((-A+t)/2)^2 + A*((-A+t)/2) + B = 0 := by nlinarith
      exact_mod_cast he
    · simp only [Complex.ofReal_re]
      dsimp [t]; linarith [Real.sqrt_nonneg (A^2-4*B)]
  · have hd' : 0 ≤ 4*B-A^2 := by linarith
    let t := Real.sqrt (4*B-A^2)
    have ht : t^2 = 4*B-A^2 := Real.sq_sqrt hd'
    refine ⟨⟨-A/2,t/2⟩, ?_, by change 0 < -A/2; linarith⟩
    apply Complex.ext <;> simp [quad, pow_two, Complex.mul_re, Complex.mul_im] <;>
      nlinarith

/-- A negative Routh margin gives a root with positive real part, not merely
failure of strict stability. -/
theorem cubic_unstable_of_margin (b₂ b₁ b₀ : ℝ)
    (h2 : 0 < b₂) (h1 : 0 < b₁) (h0 : 0 < b₀) (hm : b₂*b₁ < b₀) :
    ∃ z : ℂ, cubic b₂ b₁ b₀ z = 0 ∧ 0 < z.re := by
  obtain ⟨r, hr⟩ := real_cubic_root b₂ b₁ b₀
  have hrneg : r < 0 := by
    by_contra hn
    have hp : 0 < r^3 + b₂*r^2 + b₁*r + b₀ := by
      have : 0 ≤ r := le_of_not_gt hn
      positivity
    linarith
  have hA : b₂+r < 0 := by
    have hpos : 0 < b₁+r^2 := by positivity
    have he : (b₂+r)*(b₁+r^2) = b₂*b₁-b₀ := by nlinarith [hr]
    by_contra hn
    have hh := mul_nonneg (le_of_not_gt hn) hpos.le
    rw [he] at hh
    linarith
  obtain ⟨z, hz, hre⟩ := quadratic_unstable_of_neg (b₂+r) (b₁+b₂*r+r^2) hA
  refine ⟨z, ?_, hre⟩
  rw [cubic_factor _ _ _ _ hr, hz, mul_zero]

def A (d ρ : ℝ) : ℝ := ρ+d
def B (d h ρ : ℝ) : ℝ := d*(ρ+h)
def J (d q ρ : ℝ) : ℝ := ρ+d-q
def C (d q h ρ : ℝ) : ℝ := ρ*(d-q)+d*h
def margin (d q h ρ a : ℝ) : ℝ :=
  J d q ρ*a^2 + (A d ρ*J d q ρ+q*ρ)*a + A d ρ*B d h ρ

def characteristic (d q h ρ a : ℝ) (z : ℂ) : ℂ :=
  Matrix.det (!![z+(ρ : ℂ), (h : ℂ), 0;
    -(d : ℂ), z+(d : ℂ), -(q : ℂ); 0, -(a : ℂ), z+(a : ℂ)])

def jacobian (d q h ρ a : ℝ) : Matrix (Fin 3) (Fin 3) ℂ :=
  !![-(ρ : ℂ), -(h : ℂ), 0; (d : ℂ), -(d : ℂ), (q : ℂ); 0, (a : ℂ), -(a : ℂ)]

theorem characteristic_is_det (d q h ρ a : ℝ) (z : ℂ) :
    characteristic d q h ρ a z = (Matrix.scalar (Fin 3) z - jacobian d q h ρ a).det := by
  unfold characteristic
  congr 1
  ext i j
  fin_cases i <;> fin_cases j <;> simp [jacobian, Matrix.scalar, Matrix.sub_apply]

/-- The spectral bridge uses Mathlib's finite-dimensional spectrum theorem. -/
theorem eigenvalue_iff (d q h ρ a : ℝ) (z : ℂ) :
    z ∈ spectrum ℂ (jacobian d q h ρ a) ↔ characteristic d q h ρ a z = 0 := by
  rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot, Matrix.eval_charpoly,
    ← characteristic_is_det]

theorem matrix_hurwitz_iff (d q h ρ a : ℝ) :
    (∀ z ∈ spectrum ℂ (jacobian d q h ρ a), z.re < 0) ↔
      Hurwitz (characteristic d q h ρ a) := by
  simp only [eigenvalue_iff, Hurwitz]

theorem characteristic_cubic (d q h ρ a : ℝ) :
    characteristic d q h ρ a = cubic (a+A d ρ) (B d h ρ+a*J d q ρ) (a*C d q h ρ) := by
  funext z
  simp [characteristic, Matrix.det_fin_three, cubic, A, B, C, J]
  ring

theorem margin_identity (d q h ρ a : ℝ) :
    (a+A d ρ)*(B d h ρ+a*J d q ρ)-a*C d q h ρ = margin d q h ρ a := by
  simp only [margin, A, B, C, J]; ring

/-- The common admissibility relation d-q >= h includes both V and M. -/
theorem admissible_positive (d q h ρ : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) :
    0 < A d ρ ∧ 0 < B d h ρ ∧ 0 < C d q h ρ ∧ 0 < J d q ρ := by
  have hd : 0 < d := by linarith
  have hsum : 0 < ρ+h := by linarith
  have hb : 0 < B d h ρ := mul_pos hd hsum
  have hc : C d q h ρ = B d h ρ-q*ρ := by unfold B C; ring
  exact ⟨by unfold A; linarith, hb, by rw [hc]; nlinarith [mul_neg_of_pos_of_neg hq hr0],
    by unfold J; linarith⟩

theorem host_delay_stability_iff (d q h ρ a : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) (ha : 0 < a) :
    Hurwitz (characteristic d q h ρ a) ↔ 0 < margin d q h ρ a := by
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  rw [characteristic_cubic, cubic_hurwitz_iff]
  have h2 : 0 < a+A d ρ := by positivity
  have h1 : 0 < B d h ρ+a*J d q ρ := by positivity
  have h0 : 0 < a*C d q h ρ := by positivity
  simp only [h2, h1, h0, true_and]
  rw [← margin_identity]
  constructor <;> intro hh <;> linarith

theorem host_delay_unstable (d q h ρ a : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) (ha : 0 < a) (hm : margin d q h ρ a < 0) :
    ∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re := by
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  rw [characteristic_cubic]
  apply cubic_unstable_of_margin <;> try positivity
  have he := margin_identity d q h ρ a
  linarith

/-- Boundary factorization and frequency, including the equality case. -/
theorem boundary_frequency (d q h ρ a : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) (ha : 0 < a) (hm : margin d q h ρ a = 0) :
    let ω₂ := B d h ρ+a*J d q ρ
    0 < ω₂ ∧ ω₂ = a*C d q h ρ/(a+A d ρ) ∧
      ∀ z : ℂ, characteristic d q h ρ a z =
        (z+(a+A d ρ : ℝ))*(z^2+(ω₂ : ℂ)) := by
  dsimp
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  have h2 : 0 < a+A d ρ := by positivity
  have he := margin_identity d q h ρ a
  have heq : (a+A d ρ)*(B d h ρ+a*J d q ρ) = a*C d q h ρ := by linarith
  refine ⟨by positivity, (eq_div_iff h2.ne').mpr (by nlinarith [heq]), ?_⟩
  intro z
  rw [characteristic_cubic]
  have heC : ((a+A d ρ : ℝ) : ℂ)*((B d h ρ+a*J d q ρ : ℝ) : ℂ) =
      ((a*C d q h ρ : ℝ) : ℂ) := by exact_mod_cast heq
  unfold cubic
  linear_combination -heC

/-- At a zero margin the only roots are a negative real root and an imaginary
pair; a neutral boundary is not called unstable. -/
theorem boundary_no_positive (b₂ b₁ : ℝ) (h2 : 0 < b₂) (h1 : 0 < b₁)
    (z : ℂ) (hz : (z+b₂)*(z^2+(b₁ : ℂ)) = 0) : z.re ≤ 0 := by
  rcases mul_eq_zero.mp hz with hz | hz
  · have he := congrArg Complex.re hz
    simp only [Complex.add_re, Complex.ofReal_re, Complex.zero_re] at he
    linarith
  · have hr := congrArg Complex.re hz
    have hi := congrArg Complex.im hz
    simp only [pow_two, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im,
      Complex.ofReal_re, Complex.ofReal_im, Complex.zero_re, Complex.zero_im, add_zero] at hr hi
    by_contra hn
    have hp : 0 < z.re := lt_of_not_ge hn
    have hy : z.im = 0 := by nlinarith [hi]
    rw [hy] at hr
    nlinarith [sq_nonneg z.re]

theorem host_delay_unstable_iff (d q h ρ a : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) (ha : 0 < a) :
    (∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔ margin d q h ρ a < 0 := by
  constructor
  · rintro ⟨z,hz,hre⟩
    by_contra hn
    have hm : 0 ≤ margin d q h ρ a := le_of_not_gt hn
    rcases hm.eq_or_lt with he | he
    · obtain ⟨hω,_,hf⟩ := boundary_frequency d q h ρ a hq hadm hr hr0 ha he.symm
      obtain ⟨hA,_,_,_⟩ := admissible_positive d q h ρ hq hadm hr hr0
      have hnpos := boundary_no_positive (a+A d ρ) (B d h ρ+a*J d q ρ)
        (by positivity) hω z (by rw [← hf]; exact hz)
      linarith
    · have hs := (host_delay_stability_iff d q h ρ a hq hadm hr hr0 ha).mpr he
      have hh := hs z hz
      linarith
  · exact host_delay_unstable d q h ρ a hq hadm hr hr0 ha

theorem quadratic_endpoints (j k c : ℝ) (hj : 0 < j) (hk : 0 < k) :
    ∃ ε M : ℝ, 0 < ε ∧ 0 < M ∧ ∀ a : ℝ, 0 < a →
      (a < ε ∨ M < a) → 0 < j*a^2+c*a+k := by
  refine ⟨k/(|c|+1), (|c|+1)/j, by positivity, by positivity, ?_⟩
  intro a ha hab
  have hc := neg_abs_le c
  have hca : (-|c|) * a ≤ c*a := mul_le_mul_of_nonneg_right hc ha.le
  rcases hab with hs | hl
  · have hh := (lt_div_iff₀ (by positivity : 0 < |c|+1)).mp hs
    nlinarith [mul_nonneg hj.le (sq_nonneg a)]
  · have hh := (div_lt_iff₀ hj).mp hl
    have hmul := mul_pos ha (sub_pos.mpr hh)
    nlinarith

theorem small_large_stable (d q h ρ : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) :
    ∃ ε M : ℝ, 0 < ε ∧ 0 < M ∧ ∀ a : ℝ, 0 < a →
      (a < ε ∨ M < a) → Hurwitz (characteristic d q h ρ a) := by
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  obtain ⟨ε,M,hε,hM,hf⟩ := quadratic_endpoints (J d q ρ) (A d ρ*B d h ρ)
    (A d ρ*J d q ρ+q*ρ) hJ (mul_pos hA hB)
  exact ⟨ε,M,hε,hM, fun a ha he =>
    (host_delay_stability_iff d q h ρ a hq hadm hr hr0 ha).mpr (hf a ha he)⟩

def islandCondition (d q h ρ : ℝ) : Prop :=
  2*Real.sqrt (J d q ρ*(A d ρ*B d h ρ)) < q*|ρ|-A d ρ*J d q ρ

noncomputable def lowerRate (d q h ρ : ℝ) : ℝ :=
  (q*|ρ|-A d ρ*J d q ρ - Real.sqrt ((q*|ρ|-A d ρ*J d q ρ)^2 -
    4*J d q ρ*(A d ρ*B d h ρ)))/(2*J d q ρ)

noncomputable def upperRate (d q h ρ : ℝ) : ℝ :=
  (q*|ρ|-A d ρ*J d q ρ + Real.sqrt ((q*|ρ|-A d ρ*J d q ρ)^2 -
    4*J d q ρ*(A d ρ*B d h ρ)))/(2*J d q ρ)

theorem instability_exists_iff (d q h ρ : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) :
    (∃ a : ℝ, 0 < a ∧ ∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔
      islandCondition d q h ρ := by
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  have he := Dobson.Island.I1 (J d q ρ) (A d ρ*B d h ρ)
    (A d ρ*J d q ρ+q*ρ) hJ (mul_pos hA hB)
  have ha : ∀ a : ℝ, 0 < a → ((∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔
      margin d q h ρ a < 0) := fun a hpos =>
        host_delay_unstable_iff d q h ρ a hq hadm hr hr0 hpos
  have hh : (∃ a : ℝ, 0 < a ∧ ∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔
      ∃ a : ℝ, 0 < a ∧ margin d q h ρ a < 0 := by
    constructor <;> rintro ⟨a,hpos,hz⟩
    · exact ⟨a,hpos,(ha a hpos).mp hz⟩
    · exact ⟨a,hpos,(ha a hpos).mpr hz⟩
  rw [hh]
  change (∃ a : ℝ, 0 < a ∧ J d q ρ*a^2+(A d ρ*J d q ρ+q*ρ)*a+A d ρ*B d h ρ < 0) ↔ _
  rw [he]
  unfold islandCondition
  rw [abs_of_neg hr0]
  ring_nf

theorem instability_interval (d q h ρ : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) (hc : islandCondition d q h ρ) :
    (0 < lowerRate d q h ρ ∧ lowerRate d q h ρ < upperRate d q h ρ) ∧
    ∀ a : ℝ, 0 < a → ((∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔
      lowerRate d q h ρ < a ∧ a < upperRate d q h ρ) := by
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  have habs : q*|ρ|-A d ρ*J d q ρ = -(A d ρ*J d q ρ+q*ρ) := by
    rw [abs_of_neg hr0]; ring
  have hc' : 2*Real.sqrt (J d q ρ*(A d ρ*B d h ρ)) < -(A d ρ*J d q ρ+q*ρ) := by
    simpa only [islandCondition, habs] using hc
  obtain ⟨hends,hint⟩ := Dobson.Island.I2 (J d q ρ) (A d ρ*B d h ρ)
    (A d ρ*J d q ρ+q*ρ) ((A d ρ*J d q ρ+q*ρ)^2-4*J d q ρ*(A d ρ*B d h ρ))
    (lowerRate d q h ρ) (upperRate d q h ρ) hJ (mul_pos hA hB) hc' rfl
    (by simp only [lowerRate, habs, neg_sq]) (by simp only [upperRate, habs, neg_sq])
  refine ⟨hends, fun a ha => ?_⟩
  rw [host_delay_unstable_iff d q h ρ a hq hadm hr hr0 ha]
  exact hint a ha

theorem tangent_equality (j k c : ℝ) (hj : 0 < j) (hk : 0 < k)
    (hc : -c = 2*Real.sqrt (j*k)) :
    let a₀ := Real.sqrt (j*k)/j
    0 < a₀ ∧ ∀ a : ℝ, (j*a^2+c*a+k = 0 ↔ a = a₀) ∧
      (a ≠ a₀ → 0 < j*a^2+c*a+k) := by
  dsimp
  have ht := Real.sq_sqrt (mul_pos hj hk).le
  have hpos : 0 < Real.sqrt (j*k) := Real.sqrt_pos.mpr (mul_pos hj hk)
  refine ⟨div_pos hpos hj, fun a => ?_⟩
  have he : j*a^2+c*a+k = j*(a-Real.sqrt (j*k)/j)^2 := by
    field_simp
    linear_combination -j*a*hc - ht
  rw [he]
  constructor
  · simp [hj.ne', sub_eq_zero]
  · intro hn
    exact mul_pos hj (sq_pos_of_ne_zero (sub_ne_zero.mpr hn))

theorem single_neutral_rate (d q h ρ : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0)
    (hc : q*|ρ|-A d ρ*J d q ρ = 2*Real.sqrt (J d q ρ*(A d ρ*B d h ρ))) :
    ∃ a₀ : ℝ, 0 < a₀ ∧ margin d q h ρ a₀ = 0 ∧
      ∀ a : ℝ, 0 < a → (Hurwitz (characteristic d q h ρ a) ↔ a ≠ a₀) := by
  obtain ⟨hA,hB,hC,hJ⟩ := admissible_positive d q h ρ hq hadm hr hr0
  have hc' : -(A d ρ*J d q ρ+q*ρ) = 2*Real.sqrt (J d q ρ*(A d ρ*B d h ρ)) := by
    rw [abs_of_neg hr0] at hc
    linarith
  obtain ⟨hpos,hall⟩ := tangent_equality (J d q ρ) (A d ρ*B d h ρ)
    (A d ρ*J d q ρ+q*ρ) hJ (mul_pos hA hB) hc'
  let a₀ := Real.sqrt (J d q ρ*(A d ρ*B d h ρ))/J d q ρ
  have he : margin d q h ρ a₀ = 0 := (hall a₀).1.mpr rfl
  refine ⟨a₀,hpos,he,fun a ha => ?_⟩
  rw [host_delay_stability_iff d q h ρ a hq hadm hr hr0 ha]
  constructor
  · intro hp hn
    subst a
    rw [he] at hp
    exact lt_irrefl 0 hp
  · exact (hall a).2

theorem V_admissible (a α p : ℝ) (ha : 0 < a) (hα : 0 ≤ α)
    (hαa : α ≤ a) (hp : 0 < p) (hp1 : p < 1) : α*p ≤ a/(1-p)-a := by
  have hden : 0 < 1-p := by linarith
  have he : a/(1-p)-a = a*p/(1-p) := by field_simp; ring
  rw [he, le_div_iff₀ hden]
  have hαp : 0 ≤ α*p := mul_nonneg hα hp.le
  nlinarith [mul_nonneg (sub_nonneg.mpr hαa) hp.le, mul_nonneg hαp hp.le]

theorem M_admissible (ξ ε h : ℝ) (hε : 1 ≤ ε) (hh : 0 ≤ h) :
    h ≤ (ξ+2*ε*h)-(ξ+ε*h) := by
  nlinarith [mul_nonneg (sub_nonneg.mpr hε) hh]

theorem boundary_imaginary_root (d q h ρ a : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) (ha : 0 < a) (hm : margin d q h ρ a = 0) :
    ∃ ω : ℝ, 0 < ω ∧ characteristic d q h ρ a (Complex.I*ω) = 0 ∧
      (Complex.I*(ω : ℂ)).re = 0 ∧ ω^2 = B d h ρ+a*J d q ρ := by
  obtain ⟨hw,he,hf⟩ := boundary_frequency d q h ρ a hq hadm hr hr0 ha hm
  let w := Real.sqrt (B d h ρ+a*J d q ρ)
  have hw2 : w^2 = B d h ρ+a*J d q ρ := Real.sq_sqrt hw.le
  refine ⟨w,Real.sqrt_pos.mpr hw,?_,by simp,hw2⟩
  rw [hf]
  have heC : (w : ℂ)^2 = ((B d h ρ+a*J d q ρ : ℝ) : ℂ) := by exact_mod_cast hw2
  rw [mul_pow, Complex.I_sq, heC]
  ring

/-- Stability and island conclusions for the defined three-state matrix,
under the common V/M admissibility relation proved from the table parameters.
The characteristic polynomial is its determinant; deriving the matrix from
the nonlinear model is outside the `prop:dyn-k4` core. -/
def SelfRegulatedConclusions (d q h ρ : ℝ) : Prop :=
    (∀ a : ℝ, characteristic d q h ρ a =
      cubic (a+A d ρ) (B d h ρ+a*J d q ρ) (a*C d q h ρ)) ∧
    (0 < A d ρ ∧ 0 < B d h ρ ∧ 0 < C d q h ρ ∧ 0 < J d q ρ) ∧
    (∀ a : ℝ, 0 < a →
      ((a+A d ρ)*(B d h ρ+a*J d q ρ)-a*C d q h ρ = margin d q h ρ a) ∧
      (Hurwitz (characteristic d q h ρ a) ↔ 0 < margin d q h ρ a)) ∧
    (∃ ε M : ℝ, 0 < ε ∧ 0 < M ∧ ∀ a : ℝ, 0 < a →
      (a < ε ∨ M < a) → Hurwitz (characteristic d q h ρ a)) ∧
    ((∃ a : ℝ, 0 < a ∧ ∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔
      islandCondition d q h ρ) ∧
    (islandCondition d q h ρ →
      (0 < lowerRate d q h ρ ∧ lowerRate d q h ρ < upperRate d q h ρ) ∧
      ∀ a : ℝ, 0 < a → ((∃ z : ℂ, characteristic d q h ρ a z = 0 ∧ 0 < z.re) ↔
        lowerRate d q h ρ < a ∧ a < upperRate d q h ρ)) ∧
    (q*|ρ|-A d ρ*J d q ρ = 2*Real.sqrt (J d q ρ*(A d ρ*B d h ρ)) →
      ∃ a₀ : ℝ, 0 < a₀ ∧ margin d q h ρ a₀ = 0 ∧
        ∀ a : ℝ, 0 < a → (Hurwitz (characteristic d q h ρ a) ↔ a ≠ a₀)) ∧
    (∀ a : ℝ, 0 < a → margin d q h ρ a = 0 →
      let ω₂ := B d h ρ+a*J d q ρ
      (0 < ω₂ ∧ ω₂ = a*C d q h ρ/(a+A d ρ) ∧
        ∀ z : ℂ, characteristic d q h ρ a z = (z+(a+A d ρ : ℝ))*(z^2+(ω₂ : ℂ))) ∧
      (∃ ω : ℝ, 0 < ω ∧ characteristic d q h ρ a (Complex.I*ω) = 0 ∧
        (Complex.I*(ω : ℂ)).re = 0 ∧ ω^2 = ω₂))

theorem depressed_host (d q h ρ : ℝ) (hq : 0 < q) (hadm : h ≤ d-q)
    (hr : -h < ρ) (hr0 : ρ < 0) : SelfRegulatedConclusions d q h ρ := by
  refine ⟨characteristic_cubic d q h ρ, admissible_positive d q h ρ hq hadm hr hr0,
    ?_, small_large_stable d q h ρ hq hadm hr hr0,
    instability_exists_iff d q h ρ hq hadm hr hr0,
    instability_interval d q h ρ hq hadm hr hr0,
    single_neutral_rate d q h ρ hq hadm hr hr0, ?_⟩
  · intro a ha
    exact ⟨margin_identity d q h ρ a, host_delay_stability_iff d q h ρ a hq hadm hr hr0 ha⟩
  · intro a ha hm
    exact ⟨boundary_frequency d q h ρ a hq hadm hr hr0 ha hm,
      boundary_imaginary_root d q h ρ a hq hadm hr hr0 ha hm⟩

/-- The cubic formula is unconditional in rho; the stability-island conclusions
apply on the depressed self-regulated host domain. -/
theorem prop_dyn_k4 :
    (∀ d q h ρ a : ℝ, characteristic d q h ρ a =
      cubic (a+A d ρ) (B d h ρ+a*J d q ρ) (a*C d q h ρ)) ∧
    (∀ d q h ρ : ℝ, 0 < q → h ≤ d-q → -h < ρ → ρ < 0 →
      SelfRegulatedConclusions d q h ρ) :=
  ⟨characteristic_cubic, depressed_host⟩

/-- Additional progress on dyn-k5: the entire one-delay stability criterion,
including necessity. The multi-delay and singular asymptotic claims are separate. -/
theorem unregulated_single_delay_iff (a d q h : ℝ) (ha : 0 < a) (hq : 0 < q) (hh : 0 < h) :
    Hurwitz (characteristic d q h (-h) a) ↔
      0 < d-h-q ∧ q*h < (a+(d-h))*(d-h-q) := by
  rw [characteristic_cubic, cubic_hurwitz_iff]
  have heB : B d h (-h) = 0 := by unfold B; ring
  have heC : C d q h (-h) = q*h := by unfold C; ring
  have heA : A d (-h) = d-h := by unfold A; ring
  have heJ : J d q (-h) = d-h-q := by unfold J; ring
  rw [heB,heC,heA,heJ,zero_add]
  have h0 : 0 < a*(q*h) := by positivity
  simp only [h0, true_and]
  have he := Dobson.Coverage2Dynamics.routh_reduction a (d-h) (d-h-q) q h ha hq hh
  simpa only [mul_assoc] using he

#print axioms prop_dyn_k4
#print axioms unregulated_single_delay_iff

end Dobson.Full1.DelayStability
