import DobsonProofs.Coverage6ExactMemory

/-!
The complete rational sign table in `prop:exact-mem-coefficient`.
This is a coefficient theorem; it does not assert a nonlinear Gaussian
growth expansion or a differentiated fourth-order remainder.
-/

namespace Dobson.Full2MemorySigns
open Dobson.Coverage6ExactMemory

def numerator (s φ : ℝ) : ℝ := 1+2*φ+2*(1-s)*φ-(1-s)^2*φ^2

theorem numerator_strictMono (φ : ℝ) (hφ : φ < 0) :
    StrictMonoOn (fun s => numerator s φ) (Set.Icc 0 1) := by
  intro s hs t ht hst
  have h1 : 0 < -2*φ*(t-s) := mul_pos (by linarith) (sub_pos.mpr hst)
  have h2 : 0 ≤ φ^2*(t-s)*(2-s-t) := by
    apply mul_nonneg
    · exact mul_nonneg (sq_nonneg _) (sub_nonneg.mpr hst.le)
    · linarith [hs.2, ht.2]
  dsimp [numerator]
  nlinarith

theorem derivative_sign (s φ : ℝ) (hs : 0 < s) (hs1 : s < 1) (hφ : -1 < φ) :
    (deriv (fun t => arCoeff t φ) s < 0 ↔ 0 < numerator s φ) ∧
    (deriv (fun t => arCoeff t φ) s = 0 ↔ numerator s φ = 0) ∧
    (0 < deriv (fun t => arCoeff t φ) s ↔ numerator s φ < 0) := by
  have hd : 0 < 1+(1-s)*φ := by nlinarith
  rw [(ar_coefficient_derivative s φ (by linarith) hd.ne').deriv]
  change (-numerator s φ / _ < 0 ↔ _) ∧ (-numerator s φ / _ = 0 ↔ _) ∧
    (0 < -numerator s φ / _ ↔ _)
  have hs2 : 0 < 2-s := by linarith
  have hden : 0 < 2*(2-s)^2*(1+(1-s)*φ)^2 := by positivity
  rw [div_lt_iff₀ hden, div_eq_zero_iff, lt_div_iff₀ hden]
  simp [hden.ne']

theorem upper_memory_sign (s φ : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (hφ : 2-Real.sqrt 5 ≤ φ) (hφ1 : φ < 1) :
    deriv (fun t => arCoeff t φ) s < 0 := by
  have hsqrt := Real.sq_sqrt (show (0:ℝ) ≤ 5 by norm_num)
  have hsqrt0 := Real.sqrt_nonneg (5:ℝ)
  have hφ0 : -1 < φ := by nlinarith
  apply (derivative_sign s φ hs hs1 hφ0).1.mpr
  by_cases hp : 0 ≤ φ
  · have hq : 0 ≤ (1-s)*φ := mul_nonneg (by linarith) hp
    have hq1 : (1-s)*φ < 1 := by nlinarith
    dsimp [numerator]
    nlinarith [mul_nonneg hq (sub_nonneg.mpr hq1.le)]
  · have hneg : φ < 0 := lt_of_not_ge hp
    have hbase : 0 ≤ numerator 0 φ := by
      dsimp [numerator]
      nlinarith [sq_nonneg (φ-2+Real.sqrt 5)]
    exact hbase.trans_lt (numerator_strictMono φ hneg ⟨le_rfl, by norm_num⟩
      ⟨hs.le, hs1.le⟩ hs)

theorem lower_memory_sign (s φ : ℝ) (hs : 0 < s) (hs1 : s < 1)
    (hφ : -1 < φ) (hφ1 : φ ≤ -1/2) :
    0 < deriv (fun t => arCoeff t φ) s := by
  apply (derivative_sign s φ hs hs1 hφ).2.2.mpr
  have hbase : numerator 1 φ ≤ 0 := by dsimp [numerator]; linarith
  exact (numerator_strictMono φ (by linarith) ⟨hs.le, hs1.le⟩
    ⟨by norm_num, le_rfl⟩ hs1).trans_le hbase

noncomputable def turningStay (φ : ℝ) : ℝ :=
  1-(1-Real.sqrt (2*(1+φ)))/φ

theorem turning_stay (φ : ℝ) (hlo : -1/2 < φ) (hhi : φ < 2-Real.sqrt 5) :
    0 < turningStay φ ∧ turningStay φ < 1 ∧ numerator (turningStay φ) φ = 0 := by
  have h5 := Real.sq_sqrt (show (0:ℝ) ≤ 5 by norm_num)
  have h50 := Real.sqrt_nonneg (5:ℝ)
  have hneg : φ < 0 := by nlinarith
  have hr0 := Real.sqrt_nonneg (2*(1+φ))
  have hr := Real.sq_sqrt (show 0 ≤ 2*(1+φ) by linarith)
  have hr1 : 1 < Real.sqrt (2*(1+φ)) := by nlinarith
  have hn : 1+4*φ-φ^2 < 0 := by
    have hp : 0 < (2-φ-Real.sqrt 5)*(2-φ+Real.sqrt 5) :=
      mul_pos (by linarith) (by linarith)
    nlinarith
  have hrupper : Real.sqrt (2*(1+φ)) < 1-φ := by nlinarith
  have hq0 : 0 < (1-Real.sqrt (2*(1+φ)))/φ := div_pos_of_neg_of_neg (by linarith) hneg
  have hq1 : (1-Real.sqrt (2*(1+φ)))/φ < 1 := by
    rw [div_lt_one_of_neg hneg]
    linarith
  refine ⟨by dsimp [turningStay]; linarith, by dsimp [turningStay]; linarith, ?_⟩
  have he : 1-(1-turningStay φ)*φ = Real.sqrt (2*(1+φ)) := by
    unfold turningStay
    field_simp [hneg.ne]
    <;> ring
  calc
    numerator (turningStay φ) φ = 2*(1+φ)-(1-(1-turningStay φ)*φ)^2 := by
      unfold numerator
      ring
    _ = 0 := by rw [he, hr]; ring

/-- All three signs at the actual stated turning point, on the full
intermediate-memory interval. No sign or root is supplied as a premise. -/
theorem intermediate_memory_signs (φ : ℝ) (hlo : -1/2 < φ)
    (hhi : φ < 2-Real.sqrt 5) :
    0 < turningStay φ ∧ turningStay φ < 1 ∧
    ∀ s : ℝ, 0 < s → s < 1 →
      (0 < deriv (fun t => arCoeff t φ) s ↔ s < turningStay φ) ∧
      (deriv (fun t => arCoeff t φ) s = 0 ↔ s = turningStay φ) ∧
      (deriv (fun t => arCoeff t φ) s < 0 ↔ turningStay φ < s) := by
  obtain ⟨ht0, ht1, htzero⟩ := turning_stay φ hlo hhi
  refine ⟨ht0, ht1, fun s hs hs1 => ?_⟩
  have h5 := Real.sq_sqrt (show (0:ℝ) ≤ 5 by norm_num)
  have h50 := Real.sqrt_nonneg (5:ℝ)
  have hneg : φ < 0 := by nlinarith
  have hm := numerator_strictMono φ hneg
  have hd := derivative_sign s φ hs hs1 (by linarith)
  rw [hd.2.2, hd.2.1, hd.1, ← htzero]
  exact ⟨hm.lt_iff_lt ⟨hs.le, hs1.le⟩ ⟨ht0.le, ht1.le⟩,
    hm.injOn.eq_iff ⟨hs.le, hs1.le⟩ ⟨ht0.le, ht1.le⟩,
    hm.lt_iff_lt ⟨ht0.le, ht1.le⟩ ⟨hs.le, hs1.le⟩⟩

end Dobson.Full2MemorySigns

#print axioms Dobson.Full2MemorySigns.upper_memory_sign
#print axioms Dobson.Full2MemorySigns.lower_memory_sign
#print axioms Dobson.Full2MemorySigns.intermediate_memory_signs
