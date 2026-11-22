import Mathlib

/-!
# When an intermediate stage length destabilises (items I1 to I3)

All variables are real. `Real.sqrt` is the real square root `√`.

In I2 the derived quantities `Δ = c² − 4JK`, `a₋ = (−c − √Δ)/(2J)` and `a₊ = (−c + √Δ)/(2J)`
are variables (`Δ`, `am`, `ap`) carrying those equations as hypotheses.
-/

namespace Dobson.Island

/-- **I1.** For real `J > 0`, `K > 0` and real `c`: there is `a > 0` with
`J·a² + c·a + K < 0` if and only if `2·√(J·K) < −c`. -/
theorem I1 (J K c : ℝ) (hJ : 0 < J) (hK : 0 < K) :
    (∃ a : ℝ, 0 < a ∧ J * a ^ 2 + c * a + K < 0) ↔ 2 * Real.sqrt (J * K) < -c := by
  have hJK : 0 ≤ J * K := (mul_pos hJ hK).le
  have hr0 : 0 ≤ Real.sqrt (J * K) := Real.sqrt_nonneg _
  have hr2 : Real.sqrt (J * K) ^ 2 = J * K := Real.sq_sqrt hJK
  constructor
  · rintro ⟨a, ha, h⟩
    have hsJ : Real.sqrt J ^ 2 = J := Real.sq_sqrt hJ.le
    have hsK : Real.sqrt K ^ 2 = K := Real.sq_sqrt hK.le
    have hmul : Real.sqrt (J * K) = Real.sqrt J * Real.sqrt K := Real.sqrt_mul hJ.le K
    rw [hmul]
    by_contra hc
    rw [not_lt] at hc
    have h1 : 0 ≤ (2 * (Real.sqrt J * Real.sqrt K) + c) * a :=
      mul_nonneg (by linarith) ha.le
    have h2 : (Real.sqrt J * a - Real.sqrt K) ^ 2
        = Real.sqrt J ^ 2 * a ^ 2 - 2 * (Real.sqrt J * Real.sqrt K) * a + Real.sqrt K ^ 2 := by
      ring
    rw [hsJ, hsK] at h2
    have h3 := sq_nonneg (Real.sqrt J * a - Real.sqrt K)
    rw [h2] at h3
    nlinarith
  · intro h
    have h4 : 4 * (J * K) < c ^ 2 := by
      nlinarith [mul_pos (by linarith : (0 : ℝ) < -c - 2 * Real.sqrt (J * K))
        (by linarith : (0 : ℝ) < -c + 2 * Real.sqrt (J * K))]
    have hJ0 : J ≠ 0 := hJ.ne'
    refine ⟨-c / (2 * J), div_pos (by linarith) (by positivity), ?_⟩
    have e : J * (-c / (2 * J)) ^ 2 + c * (-c / (2 * J)) + K
        = (4 * (J * K) - c ^ 2) / (4 * J) := by
      field_simp
      ring
    rw [e]
    exact div_neg_of_neg_of_pos (by linarith) (by positivity)

/-- **I2.** Let `J > 0`, `K > 0` and `2·√(J·K) < −c`, and put `Δ = c² − 4JK`,
`a₋ = (−c − √Δ)/(2J)`, `a₊ = (−c + √Δ)/(2J)`. Then `0 < a₋ < a₊`, and for every `a > 0`:
`J·a² + c·a + K < 0` if and only if `a₋ < a` and `a < a₊`. -/
theorem I2 (J K c Δ am ap : ℝ) (hJ : 0 < J) (hK : 0 < K)
    (hc : 2 * Real.sqrt (J * K) < -c)
    (hΔ : Δ = c ^ 2 - 4 * J * K)
    (ham : am = (-c - Real.sqrt Δ) / (2 * J))
    (hap : ap = (-c + Real.sqrt Δ) / (2 * J)) :
    (0 < am ∧ am < ap) ∧
      ∀ a : ℝ, 0 < a → (J * a ^ 2 + c * a + K < 0 ↔ (am < a ∧ a < ap)) := by
  have hJK : 0 < J * K := mul_pos hJ hK
  have hr0 : 0 ≤ Real.sqrt (J * K) := Real.sqrt_nonneg _
  have hr2 : Real.sqrt (J * K) ^ 2 = J * K := Real.sq_sqrt hJK.le
  have hcneg : 0 < -c := by linarith
  have hΔpos : 0 < Δ := by
    rw [hΔ]
    nlinarith [mul_pos (by linarith : (0 : ℝ) < -c - 2 * Real.sqrt (J * K))
      (by linarith : (0 : ℝ) < -c + 2 * Real.sqrt (J * K))]
  have hs0 : 0 < Real.sqrt Δ := Real.sqrt_pos.mpr hΔpos
  have hs2 : Real.sqrt Δ ^ 2 = c ^ 2 - 4 * J * K := by rw [Real.sq_sqrt hΔpos.le, hΔ]
  have hslt : Real.sqrt Δ < -c := by
    by_contra hcon
    rw [not_lt] at hcon
    nlinarith [mul_self_le_mul_self hcneg.le hcon]
  have h2J : 0 < 2 * J := by positivity
  have hJ0 : J ≠ 0 := hJ.ne'
  have ham_pos : 0 < am := by
    rw [ham]
    exact div_pos (by linarith) h2J
  have hlt : am < ap := by
    rw [ham, hap]
    exact (div_lt_div_iff_of_pos_right h2J).mpr (by linarith)
  refine ⟨⟨ham_pos, hlt⟩, ?_⟩
  intro a _ha
  -- `0 < a` is not needed: the equivalence holds for every real `a`. It is cleared here so
  -- that no later step can use it.
  clear _ha
  have hfac : J * a ^ 2 + c * a + K = J * ((a - am) * (a - ap)) := by
    have e : J * ((a - am) * (a - ap))
        = J * a ^ 2 + c * a + (c ^ 2 - Real.sqrt Δ ^ 2) / (4 * J) := by
      rw [ham, hap]
      field_simp
      ring
    rw [e, hs2]
    field_simp
    ring
  rw [hfac]
  constructor
  · intro h
    have h2 : (a - am) * (a - ap) < 0 := by
      by_contra hcon
      rw [not_lt] at hcon
      nlinarith [mul_nonneg hJ.le hcon]
    rcases mul_neg_iff.mp h2 with ⟨h3, h4⟩ | ⟨h3, h4⟩
    · exact ⟨by linarith, by linarith⟩
    · exfalso
      linarith
  · rintro ⟨h1, h2⟩
    exact mul_neg_of_pos_of_neg hJ (mul_neg_of_pos_of_neg (by linarith) (by linarith))

/-- **I3.** (The quadratic of the earlier file in the other variable.) For real `J`, `c`, `K`
and real `τ ≠ 0`: `τ²·(J·(1/τ)² + c·(1/τ) + K) = K·τ² + c·τ + J`. -/
theorem I3 (J c K τ : ℝ) (hτ : τ ≠ 0) :
    τ ^ 2 * (J * (1 / τ) ^ 2 + c * (1 / τ) + K) = K * τ ^ 2 + c * τ + J := by
  field_simp
  ring

end Dobson.Island

#print axioms Dobson.Island.I1
#print axioms Dobson.Island.I2
#print axioms Dobson.Island.I3
