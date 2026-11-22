import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Tactic

/-! Scalar coefficient comparisons for `lem:mech-contact-consequences`.
The negative-memory product estimate is proved for finite rings with at
least two stages. The positive-memory derivative is for the displayed
substituted scalar expression. Coverage correction, 2026-11-22: this is core;
the `d = 0` endpoint/domain gaps and the derivative bridge to the actual
changed stay coefficient remain outside full manuscript coverage. -/

namespace Dobson.Full2Contact
open Finset

noncomputable def coeff (A R q d : ℝ) : ℝ :=
  d^2*A-d*(R-(1-d)*q)/(1-R)

lemma coefficient_split (A R q d : ℝ) (hR : R ≠ 1) :
    coeff A R q d = d^2*(A-R/(1-R))+d*(1-d)*(q-R)/(1-R) := by
  have hd : 1-R ≠ 0 := sub_ne_zero.mpr hR.symm
  unfold coeff
  field_simp
  <;> ring

lemma coefficient_comparison (A R q d K Kc : ℝ)
    (hK : K = A-R/(1-R))
    (hKc : Kc = d^2*K+d*(1-d)*(q-R)/(1-R))
    (hR : R < 1) (_hd : 0 ≤ d) (hd1 : d < 1) :
    K < Kc ↔ (1+d)*A < (R+d*q)/(1-R) := by
  have hdR : 1-R ≠ 0 := by linarith
  have he : Kc-K = (1-d)*((R+d*q)/(1-R)-(1+d)*A) := by
    rw [hKc, hK]
    field_simp
    <;> ring
  rw [← sub_pos, he, mul_pos_iff_of_pos_left (by linarith : 0 < 1-d), sub_pos]

noncomputable def stay (φ τ : ℝ) : ℝ := φ / (τ - (τ - 1) * φ)

theorem negative_stay (φ τ : ℝ) (hφ : -1 < φ) (hφ' : φ < 0) (hτ : 1 ≤ τ) :
    -1 < stay φ τ ∧ stay φ τ < 0 := by
  have hd : 0 < τ - (τ - 1) * φ := by nlinarith
  constructor
  · unfold stay
    apply (lt_div_iff₀ hd).2
    nlinarith
  · exact div_neg_of_neg_of_pos hφ' hd

theorem negative_memory {ι : Type*} [Fintype ι] [DecidableEq ι]
    (j : ι) (hother : ∃ k : ι, k ≠ j) (τ : ι → ℝ)
    (hτ : ∀ k, 1 ≤ τ k) (φ : ℝ) (hφ : -1 < φ) (hφ' : φ < 0) :
    let q := fun k => stay φ (τ k)
    let R := ∏ k, q k
    R < 1 ∧ (q j - R) / (1 - R) < 0 := by
  dsimp
  let q := fun k => stay φ (τ k)
  have hq (k : ι) : -1 < q k ∧ q k < 0 := negative_stay φ (τ k) hφ hφ' (hτ k)
  have habs (k : ι) : |q k| < 1 := by
    rw [abs_of_neg (hq k).2]
    linarith [(hq k).1]
  have hne : (univ.erase j).Nonempty := by
    obtain ⟨k, hk⟩ := hother
    exact ⟨k, mem_erase.mpr ⟨hk, mem_univ k⟩⟩
  have hp : (∏ k ∈ univ.erase j, |q k|) < 1 := by
    simpa using Finset.prod_lt_prod_of_nonempty₀
      (s := univ.erase j) (f := fun k => |q k|) (g := fun _ => (1 : ℝ))
      (fun k _ => abs_pos.mpr (ne_of_lt (hq k).2)) (fun k _ => habs k) hne
  have hpabs : |∏ k ∈ univ.erase j, q k| < 1 := by
    simpa only [Finset.abs_prod] using hp
  have he : (∏ k, q k) = q j * ∏ k ∈ univ.erase j, q k :=
    (mul_prod_erase univ q (mem_univ j)).symm
  have hprod : |∏ k, q k| < 1 := by
    rw [he, abs_mul]
    calc
      |q j| * |∏ k ∈ univ.erase j, q k| < |q j| * 1 :=
        mul_lt_mul_of_pos_left hpabs (abs_pos.mpr (ne_of_lt (hq j).2))
      _ < 1 := by simpa using habs j
  have hR : (∏ k, q k) < 1 := (le_abs_self _).trans_lt hprod
  refine ⟨hR, div_neg_of_neg_of_pos ?_ (sub_pos.mpr hR)⟩
  rw [he]
  have hr : (∏ k ∈ univ.erase j, q k) < 1 := (le_abs_self _).trans_lt hpabs
  nlinarith [(hq j).2]

theorem contact_signs (K L d : ℝ) (hL : L < 0) (hd : 0 < d) :
    (0 < K → (0 < d^2*K+d*(1-d)*L ↔ -L/(K-L) < d)) ∧
    (K ≤ 0 → d < 1 → d^2*K+d*(1-d)*L < 0) ∧
    (d = 1 → d^2*K+d*(1-d)*L = K) := by
  have he : d^2*K+d*(1-d)*L = d*(L+d*(K-L)) := by ring
  refine ⟨?_, ?_, ?_⟩
  · intro hK
    rw [he, mul_pos_iff_of_pos_left hd, div_lt_iff₀ (by linarith : 0 < K-L)]
    constructor <;> intro h <;> nlinarith
  · intro hK hd1
    have h1 : d^2*K ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (sq_nonneg d) hK
    have h2 : d*(1-d)*L < 0 := mul_neg_of_pos_of_neg (mul_pos hd (by linarith)) hL
    linarith
  · rintro rfl
    ring

/-- Negative-memory and comparison conclusions for the displayed stay
factors, under the stated positive-contact and multi-stage hypotheses. -/
theorem contact_consequences {ι : Type*} [Fintype ι] [DecidableEq ι]
    (j : ι) (hother : ∃ k : ι, k ≠ j) (τ : ι → ℝ)
    (hτ : ∀ k, 1 ≤ τ k) (φ d : ℝ) (hφ : -1 < φ) (hφ' : φ < 0)
    (hd : 0 < d) (hd1 : d ≤ 1) :
    let q := fun k => stay φ (τ k)
    let R := ∏ k, q k
    let A := (1+2*(φ/(1-φ)))/(2*∑ k, τ k)
    let K := A-R/(1-R)
    let L := (q j-R)/(1-R)
    let Kc := coeff A R (q j) d
    L < 0 ∧
    (0 < K → (0 < Kc ↔ -L/(K-L) < d)) ∧
    (K ≤ 0 → d < 1 → Kc < 0) ∧
    (d = 1 → Kc = K) ∧
    (d < 1 → (K < Kc ↔ (1+d)*A < (R+d*q j)/(1-R))) := by
  dsimp only
  obtain ⟨hR, hL⟩ := negative_memory j hother τ hτ φ hφ hφ'
  have he := coefficient_split
    ((1+2*(φ/(1-φ)))/(2*∑ k, τ k)) (∏ k, stay φ (τ k)) (stay φ (τ j)) d
    (ne_of_lt hR)
  change coeff _ _ _ _ = _ at he
  rw [he]
  obtain ⟨hpos, hneg, hone⟩ := contact_signs
    ((1+2*(φ/(1-φ)))/(2*∑ k, τ k)-(∏ k, stay φ (τ k))/(1-∏ k, stay φ (τ k)))
    ((stay φ (τ j)-(∏ k, stay φ (τ k)))/(1-∏ k, stay φ (τ k))) d hL hd
  simp only [mul_div_assoc] at hpos hneg hone ⊢
  refine ⟨hL, hpos, hneg, hone, fun hdlt => ?_⟩
  simpa only [mul_div_assoc] using
    coefficient_comparison _ _ _ _ _ _ rfl rfl hR hd.le hdlt

/-- Differentiation before any long-stay limit, at every nonsingular stay. -/
theorem contact_stay_derivative (θ S U q d b : ℝ)
    (hS : b+S ≠ 0) (hθ : b+θ ≠ 0) (hU : b+θ-U ≠ 0) :
    HasDerivAt (fun x : ℝ =>
      coeff ((1+2*θ)/(2*(x+S))) (U/(x+θ)) q d)
      (-d^2*(θ+1/2)/(b+S)^2+d*U*(1-(1-d)*q)/(b+θ-U)^2) b := by
  have hret : 1-U/(b+θ) ≠ 0 := by
    rw [sub_div' hθ]
    simpa only [one_mul] using div_ne_zero hU hθ
  have hr := (hasDerivAt_const b U).div ((hasDerivAt_id b).add_const θ) hθ
  have ht := (hasDerivAt_const b (1+2*θ)).div
    (((hasDerivAt_id b).add_const S).const_mul 2) (mul_ne_zero two_ne_zero hS)
  have hc := (ht.const_mul (d^2)).sub
    (((hr.sub_const ((1-d)*q)).const_mul d).div
      ((hasDerivAt_const b 1).sub hr) hret)
  convert hc using 1
  · rfl
  · dsimp
    field_simp
    <;> ring


lemma stay_abs_lt_one (φ τ : ℝ) (hφ0 : -1 < φ) (hφ1 : φ < 1) (hτ : 1 ≤ τ) :
    |stay φ τ| < 1 := by
  have hd1 : 1 ≤ τ-(τ-1)*φ := by
    nlinarith [mul_nonneg (show 0 ≤ τ-1 by linarith) (show 0 ≤ 1-φ by linarith)]
  have hd : 0 < τ-(τ-1)*φ := by linarith
  rw [abs_lt]
  constructor
  · unfold stay
    rw [lt_div_iff₀ hd]
    linarith
  · unfold stay
    rw [div_lt_one hd]
    linarith

/-- The manuscript comparison is valid for both signs of memory. -/
theorem contact_comparison_all_memory {ι : Type*} [Fintype ι] [Nonempty ι]
    (j : ι) (τ : ι → ℝ) (hτ : ∀ k, 1 ≤ τ k) (φ d : ℝ)
    (hφ0 : -1 < φ) (hφ1 : φ < 1) (hd0 : 0 < d) (hd1 : d < 1) :
    let q := fun k => stay φ (τ k)
    let R := ∏ k, q k
    let A := (1+2*(φ/(1-φ)))/(2*∑ k, τ k)
    A-R/(1-R) < coeff A R (q j) d ↔ (1+d)*A < (R+d*q j)/(1-R) := by
  dsimp only
  have hprod : |∏ k, stay φ (τ k)| < 1 := by
    rw [Finset.abs_prod]
    by_cases hzero : ∃ k, stay φ (τ k) = 0
    · obtain ⟨k, hk⟩ := hzero
      rw [Finset.prod_eq_zero (mem_univ k) (by simp [hk])]
      norm_num
    · push_neg at hzero
      simpa using Finset.prod_lt_prod_of_nonempty₀
        (s := univ) (f := fun k => |stay φ (τ k)|) (g := fun _ => (1:ℝ))
        (fun k _ => abs_pos.mpr (hzero k))
        (fun k _ => stay_abs_lt_one φ (τ k) hφ0 hφ1 (hτ k)) univ_nonempty
  have hR := (le_abs_self _).trans_lt hprod
  apply coefficient_comparison _ _ _ _ _ _ rfl
    (coefficient_split _ _ _ _ hR.ne) hR hd0.le hd1

/-- Positive-memory derivative, with `S` and `U` constructed from the actual
other stages. All denominator conditions are derived, not assumed. -/
theorem contact_positive_stay_derivative {ι : Type*} [Fintype ι] [DecidableEq ι]
    (park j : ι) (_hdifferent : park ≠ j) (τ : ι → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (φ d b : ℝ) (hφ0 : 0 < φ) (hφ1 : φ < 1) (hb : 1 ≤ b) :
    let θ := φ/(1-φ)
    let S := ∑ k ∈ univ.erase park, τ k
    let U := θ*∏ k ∈ univ.erase park, stay φ (τ k)
    let q := stay φ (τ j)
    HasDerivAt (fun x : ℝ => coeff ((1+2*θ)/(2*(x+S))) (U/(x+θ)) q d)
      (-d^2*(θ+1/2)/(b+S)^2+d*U*(1-(1-d)*q)/(b+θ-U)^2) b := by
  dsimp only
  have hθ : 0 < φ/(1-φ) := div_pos hφ0 (by linarith)
  have hS : 0 ≤ ∑ k ∈ univ.erase park, τ k := sum_nonneg (fun k _ => (by linarith [hτ k]))
  have hprod : (∏ k ∈ univ.erase park, stay φ (τ k)) ≤ 1 := by
    apply Finset.prod_le_one₀
    · intro k _
      unfold stay
      apply div_nonneg hφ0.le
      nlinarith [mul_nonneg (show 0 ≤ τ k-1 by linarith [hτ k]) (show 0 ≤ 1-φ by linarith)]
    · intro k _
      exact ((le_abs_self _).trans_lt (stay_abs_lt_one φ (τ k) (by linarith) hφ1 (hτ k))).le
  have hU : φ/(1-φ)*(∏ k ∈ univ.erase park, stay φ (τ k)) ≤ φ/(1-φ) := by
    simpa using mul_le_mul_of_nonneg_left hprod hθ.le
  exact contact_stay_derivative _ _ _ _ _ _ (by linarith) (by linarith) (by linarith)


/-- Combined scalar contact conclusions on `0 < d ≤ 1` for a ring with
another stage. The comparison covers both signs of memory; the derivative
is for the displayed substituted coefficient. The `d = 0` endpoint and
actual-stay derivative bridge are outside this statement. -/
theorem manuscript_contact_consequences {ι : Type*} [Fintype ι] [DecidableEq ι]
    (j : ι) (hother : ∃ k : ι, k ≠ j) (τ : ι → ℝ) (hτ : ∀ k, 1 ≤ τ k)
    (φ d : ℝ) (hφ0 : -1 < φ) (hφ1 : φ < 1) (hd0 : 0 < d) (hd1 : d ≤ 1) :
    let θ := φ/(1-φ)
    let q := fun k => stay φ (τ k)
    let R := ∏ k, q k
    let A := (1+2*θ)/(2*∑ k, τ k)
    let K := A-R/(1-R)
    let L := (q j-R)/(1-R)
    let Kc := coeff A R (q j) d
    (φ < 0 → L < 0 ∧
      (0 < K → (0 < Kc ↔ -L/(K-L) < d)) ∧
      (K ≤ 0 → d < 1 → Kc < 0) ∧ (d = 1 → Kc = K)) ∧
    (d < 1 → (K < Kc ↔ (1+d)*A < (R+d*q j)/(1-R))) ∧
    (0 < φ → ∀ park : ι, park ≠ j → ∀ b : ℝ, 1 ≤ b →
      let S := ∑ k ∈ univ.erase park, τ k
      let U := θ*∏ k ∈ univ.erase park, q k
      HasDerivAt (fun x : ℝ => coeff ((1+2*θ)/(2*(x+S))) (U/(x+θ)) (q j) d)
        (-d^2*(θ+1/2)/(b+S)^2+d*U*(1-(1-d)*q j)/(b+θ-U)^2) b) := by
  letI : Nonempty ι := ⟨j⟩
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · intro hneg
    obtain ⟨hL, hp, hn, he, _⟩ := contact_consequences j hother τ hτ φ d hφ0 hneg hd0 hd1
    exact ⟨hL, hp, hn, he⟩
  · intro hdlt
    exact contact_comparison_all_memory j τ hτ φ d hφ0 hφ1 hd0 hdlt
  · intro hpos park hdiff b hb
    exact contact_positive_stay_derivative park j hdiff τ hτ φ d b hpos hφ1 hb

end Dobson.Full2Contact

#print axioms Dobson.Full2Contact.contact_consequences
#print axioms Dobson.Full2Contact.contact_stay_derivative

#print axioms Dobson.Full2Contact.contact_comparison_all_memory
#print axioms Dobson.Full2Contact.contact_positive_stay_derivative

#print axioms Dobson.Full2Contact.manuscript_contact_consequences
