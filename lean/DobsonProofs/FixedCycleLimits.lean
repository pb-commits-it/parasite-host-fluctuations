import DobsonProofs.FixedCycle
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Topology.Order.Lattice

namespace Dobson.FixedCycle
open scoped BigOperators
open Filter
variable {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι]

/-- Finitely many positive cohorts with the same exponent have that sum exponent. -/
theorem finite_sum_log_limit (a : ι → ℕ → ℝ) (d : ℕ → ℝ) (r : ℝ)
    (ha : ∀ i k, 0 < a i k) (hd : ∀ k, 0 < d k)
    (hinv : Tendsto (fun k => (d k)⁻¹) atTop (nhds 0))
    (hl : ∀ i, Tendsto (fun k => Real.log (a i k) / d k) atTop (nhds r)) :
    Tendsto (fun k => Real.log (∑ i, a i k) / d k) atTop (nhds r) := by
  let i₀ : ι := Classical.arbitrary ι
  have hne : (Finset.univ : Finset ι).Nonempty := Finset.univ_nonempty
  let m : ℕ → ℝ := fun k => Finset.univ.sup' hne (fun i => Real.log (a i k) / d k)
  have hm : Tendsto m atTop (nhds r) := by
    simpa [m, Finset.sup'_const] using
      (Tendsto.finset_sup'_nhds_apply hne (fun i _ => hl i))
  have hcard : (0 : ℝ) < Fintype.card ι := Nat.cast_pos.mpr Fintype.card_pos
  have hconst : Tendsto (fun k => Real.log (Fintype.card ι : ℝ) / d k)
      atTop (nhds 0) := by
    simpa [div_eq_mul_inv] using hinv.const_mul (Real.log (Fintype.card ι : ℝ))
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' (hl i₀) (by simpa using hconst.add hm)
  · exact Eventually.of_forall (fun k => div_le_div_of_nonneg_right
      (Real.log_le_log (ha i₀ k)
        (Finset.single_le_sum (fun i _ => (ha i k).le) (Finset.mem_univ i₀))) (hd k).le)
  · apply Eventually.of_forall
    intro k
    have hi (i : ι) : a i k ≤ Real.exp (m k * d k) := by
      have hm' : Real.log (a i k) / d k ≤ m k :=
        Finset.le_sup' (fun j => Real.log (a j k) / d k) (Finset.mem_univ i)
      have hh := (div_le_iff₀ (hd k)).mp hm'
      simpa [Real.exp_log (ha i k)] using Real.exp_le_exp.mpr hh
    have hsum : (∑ i, a i k) ≤ (Fintype.card ι : ℝ) * Real.exp (m k * d k) := by
      simpa using Finset.sum_le_card_nsmul Finset.univ (fun i => a i k)
        (Real.exp (m k * d k)) (fun i _ => hi i)
    have hsumpos : 0 < ∑ i, a i k := Finset.sum_pos (fun i _ => ha i k) hne
    have hb := div_le_div_of_nonneg_right (Real.log_le_log hsumpos hsum) (hd k).le
    rw [Real.log_mul (ne_of_gt hcard) (ne_of_gt (Real.exp_pos _)), Real.log_exp] at hb
    convert hb using 1 <;> field_simp [(hd k).ne'] <;> ring

theorem population_pos (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (hg : ∀ i, 0 < g i) (hh : ∀ t, 0 < h t)
    (hx : ∀ i, 0 < x₀ i) (t : ℕ) (i : ι) :
    0 < population next g birth h x₀ t i := by
  induction t generalizing i with
  | zero => exact hx i
  | succ t ih =>
    simp only [population, weight]
    apply mul_pos (mul_pos (hg _) _) (ih _)
    split_ifs
    · exact hh t
    · norm_num

/-- A Birkhoff average along the complete-cycle shift determines each cohort's exponent. -/
theorem cohort_log_limit (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth i : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (hg : ∀ j, 0 < g j) (hh : ∀ t, 0 < h t)
    (hx : 0 < x₀ i) (r : ℕ) (hr : r < Fintype.card ι)
    (hri : (next^r) i = birth) (meanLog : ℝ)
    (hBirkhoff_cycle : Tendsto (fun q : ℕ =>
      (∑ j ∈ Finset.range (q+1), Real.log (h (j*Fintype.card ι+r))) / (q+1 : ℕ))
      atTop (nhds meanLog)) :
    Tendsto (fun q : ℕ => Real.log (population next g birth h x₀
      ((q+1)*Fintype.card ι) i) / ((q+1 : ℕ) * (Fintype.card ι : ℝ)))
      atTop (nhds ((Real.log (∏ j, g j) + meanLog) / Fintype.card ι)) := by
  have hN : (Fintype.card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hinv : Tendsto (fun q : ℕ => ((q+1 : ℕ) : ℝ)⁻¹) atTop (nhds 0) := by
    simpa [one_div, Function.comp_def] using
      (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ)).comp (tendsto_add_atTop_nat 1)
  have hinit : Tendsto (fun q : ℕ => Real.log (x₀ i) / ((q+1 : ℕ) : ℝ))
      atTop (nhds 0) := by
    simpa [div_eq_mul_inv] using hinv.const_mul (Real.log (x₀ i))
  have hconst0 : Tendsto (fun _ : ℕ => Real.log (∏ j, g j)) atTop
      (nhds (Real.log (∏ j, g j))) := tendsto_const_nhds
  have hhlim := ((hconst0.add hBirkhoff_cycle).add hinit).div_const
    (Fintype.card ι : ℝ)
  apply Tendsto.congr' _ (by simpa using hhlim)
  apply Eventually.of_forall
  intro q
  dsimp only
  rw [cohort_product next hc g birth i h x₀ (q+1) r hr hri,
    Real.log_mul (ne_of_gt (mul_pos
      (pow_pos (Finset.prod_pos (fun j _ => hg j)) _)
      (Finset.prod_pos (fun j _ => hh _)))) (ne_of_gt hx),
    Real.log_mul (ne_of_gt (pow_pos (Finset.prod_pos (fun j _ => hg j)) _))
      (ne_of_gt (Finset.prod_pos (fun j _ => hh _))),
    Real.log_pow, Real.log_prod (fun j _ => ne_of_gt (hh _))]
  have hq : ((q+1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.succ_ne_zero q)
  push_cast at hq ⊢
  field_simp
  <;> ring

/-- Circuit-boundary growth of the whole daily population, with every residue averaged. -/
theorem circuit_log_limit (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (hg : ∀ j, 0 < g j) (hh : ∀ t, 0 < h t)
    (hx : ∀ j, 0 < x₀ j) (meanLog : ℝ)
    (hBirkhoff_cycle : ∀ r < Fintype.card ι, Tendsto (fun q : ℕ =>
      (∑ j ∈ Finset.range (q+1), Real.log (h (j*Fintype.card ι+r))) / (q+1 : ℕ))
      atTop (nhds meanLog)) :
    Tendsto (fun q : ℕ => Real.log (∑ i, population next g birth h x₀
      ((q+1)*Fintype.card ι) i) / ((q+1 : ℕ) * (Fintype.card ι : ℝ)))
      atTop (nhds ((Real.log (∏ j, g j) + meanLog) / Fintype.card ι)) := by
  have hN : (0 : ℝ) < Fintype.card ι := Nat.cast_pos.mpr Fintype.card_pos
  apply finite_sum_log_limit
  · intro i q
    exact population_pos next g birth h x₀ hg hh hx _ i
  · intro q
    exact mul_pos (Nat.cast_pos.mpr (Nat.succ_pos q)) hN
  · have hinv : Tendsto (fun q : ℕ => ((q+1 : ℕ) : ℝ)⁻¹) atTop (nhds 0) := by
      simpa [one_div, Function.comp_def] using
        (tendsto_one_div_atTop_nhds_zero_nat (𝕜 := ℝ)).comp (tendsto_add_atTop_nat 1)
    simpa [mul_inv_rev, mul_comm] using hinv.mul_const (Fintype.card ι : ℝ)⁻¹
  · intro i
    obtain ⟨r, hr, hri⟩ := hc.exists_pow_eq (Finset.mem_univ i) (Finset.mem_univ birth)
    exact cohort_log_limit next hc g birth i h x₀ hg hh (hx i) r (by simpa using hr)
      hri meanLog (hBirkhoff_cycle r (by simpa using hr))

/-- Positivity and order propagate through the actual daily update. -/
theorem population_nonneg (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (hg : ∀ i, 0 < g i) (hh : ∀ t, 0 < h t)
    (hx : ∀ i, 0 ≤ x₀ i) (t : ℕ) (i : ι) :
    0 ≤ population next g birth h x₀ t i := by
  induction t generalizing i with
  | zero => exact hx i
  | succ t ih =>
    simp only [population, weight]
    apply mul_nonneg (mul_nonneg (hg _).le _) (ih _)
    split_ifs
    · exact (hh t).le
    · norm_num

theorem population_mono (next : Equiv.Perm ι) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x y : ι → ℝ) (hg : ∀ i, 0 < g i) (hh : ∀ t, 0 < h t)
    (hxy : ∀ i, x i ≤ y i) (t : ℕ) (i : ι) :
    population next g birth h x t i ≤ population next g birth h y t i := by
  induction t generalizing i with
  | zero => exact hxy i
  | succ t ih =>
    simp only [population, weight]
    apply mul_le_mul_of_nonneg_left (ih _)
    apply mul_nonneg (hg _).le
    split_ifs
    · exact (hh t).le
    · norm_num

/-- Zero initial cohorts are allowed: sandwich between one occupied cohort
and an everywhere positive initial population. -/
theorem circuit_log_limit_nonneg (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (hg : ∀ j, 0 < g j) (hh : ∀ t, 0 < h t)
    (hx : ∀ j, 0 ≤ x₀ j) (hxne : ∃ j, 0 < x₀ j) (meanLog : ℝ)
    (hBirkhoff_cycle : ∀ r < Fintype.card ι, Tendsto (fun q : ℕ =>
      (∑ j ∈ Finset.range (q+1), Real.log (h (j*Fintype.card ι+r))) / (q+1 : ℕ))
      atTop (nhds meanLog)) :
    Tendsto (fun q : ℕ => Real.log (∑ i, population next g birth h x₀
      ((q+1)*Fintype.card ι) i) / ((q+1 : ℕ) * (Fintype.card ι : ℝ)))
      atTop (nhds ((Real.log (∏ j, g j) + meanLog) / Fintype.card ι)) := by
  obtain ⟨i, hi⟩ := hxne
  obtain ⟨r, hr, hri⟩ := hc.exists_pow_eq (Finset.mem_univ i) (Finset.mem_univ birth)
  have hr' : r < Fintype.card ι := by simpa using hr
  have hl := cohort_log_limit next hc g birth i h x₀ hg hh hi r hr' hri
    meanLog (hBirkhoff_cycle r hr')
  have hu := circuit_log_limit next hc g birth h (fun j => x₀ j+1) hg hh
    (fun j => by linarith [hx j]) meanLog hBirkhoff_cycle
  have hpos (q : ℕ) : 0 < population next g birth h x₀ ((q+1)*Fintype.card ι) i := by
    rw [cohort_product next hc g birth i h x₀ (q+1) r hr' hri]
    exact mul_pos (mul_pos (pow_pos (Finset.prod_pos (fun j _ => hg j)) _)
      (Finset.prod_pos (fun j _ => hh _))) hi
  have hsum (q : ℕ) : population next g birth h x₀ ((q+1)*Fintype.card ι) i ≤
      ∑ j, population next g birth h x₀ ((q+1)*Fintype.card ι) j :=
    Finset.single_le_sum (fun j _ => population_nonneg next g birth h x₀ hg hh hx _ j)
      (Finset.mem_univ i)
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hl hu
  · apply Eventually.of_forall
    intro q
    exact div_le_div_of_nonneg_right (Real.log_le_log (hpos q) (hsum q)) (by positivity)
  · apply Eventually.of_forall
    intro q
    apply div_le_div_of_nonneg_right _ (by positivity)
    apply Real.log_le_log (lt_of_lt_of_le (hpos q) (hsum q))
    exact Finset.sum_le_sum (fun j _ => population_mono next g birth h x₀
      (fun j => x₀ j+1) hg hh (fun j => by linarith) _ j)

/-- Applying a reproductive divisor to an arbitrary designated edge. -/
noncomputable def dividedFactors (g : ι → ℝ) (birth : ι) (R : ℝ) (i : ι) : ℝ :=
  g i * if i = birth then R⁻¹ else 1

theorem dividedFactors_prod (g : ι → ℝ) (birth : ι) (R : ℝ) :
    (∏ i, dividedFactors g birth R i) = (∏ i, g i) / R := by
  unfold dividedFactors
  rw [Finset.prod_mul_distrib]
  simp [div_eq_mul_inv]

theorem dividedFactors_pos (g : ι → ℝ) (birth : ι) (R : ℝ)
    (hg : ∀ i, 0 < g i) (hR : 0 < R) (i : ι) : 0 < dividedFactors g birth R i := by
  dsimp [dividedFactors]
  apply mul_pos (hg i)
  split_ifs
  · exact inv_pos.mpr hR
  · norm_num

/-- An actual zero-growth reproductive divisor has the manuscript log-loss.
The zero-growth premise concerns the daily population, not a supplied growth formula.
The only statistical input is the named complete-cycle Birkhoff average. -/
theorem threshold_log_loss (next : Equiv.Perm ι)
    (hc : next.IsCycleOn (Finset.univ : Finset ι)) (g : ι → ℝ) (birth noise : ι)
    (h : ℕ → ℝ) (x₀ : ι → ℝ) (hg : ∀ j, 0 < g j) (hh : ∀ t, 0 < h t)
    (hx : ∀ j, 0 ≤ x₀ j) (hxne : ∃ j, 0 < x₀ j) (Rstar meanLog : ℝ) (hR : 0 < Rstar)
    (hBirkhoff_cycle : ∀ r < Fintype.card ι, Tendsto (fun q : ℕ =>
      (∑ j ∈ Finset.range (q+1), Real.log (h (j*Fintype.card ι+r))) / (q+1 : ℕ))
      atTop (nhds meanLog))
    (hzero_growth : Tendsto (fun t : ℕ => Real.log
      (∑ i, population next (dividedFactors g birth Rstar) noise h x₀ t i) / t)
      atTop (nhds 0)) :
    Real.log (∏ i, g i) - Real.log Rstar = -meanLog := by
  have hsub : Tendsto (fun q : ℕ => (q+1)*Fintype.card ι) atTop atTop := by
    apply tendsto_atTop_mono (f := fun q : ℕ => q+1) _ (tendsto_add_atTop_nat 1)
    intro q
    exact Nat.le_mul_of_pos_right (q+1) Fintype.card_pos
  have hz : Tendsto (fun q : ℕ => Real.log
      (∑ i, population next (dividedFactors g birth Rstar) noise h x₀
        ((q+1)*Fintype.card ι) i) / ((q+1 : ℕ) * (Fintype.card ι : ℝ)))
      atTop (nhds 0) := by
    simpa [Function.comp_def, Nat.cast_mul] using hzero_growth.comp hsub
  have hformula := circuit_log_limit_nonneg next hc (dividedFactors g birth Rstar) noise h x₀
    (dividedFactors_pos g birth Rstar hg hR) hh hx hxne meanLog hBirkhoff_cycle
  have he := tendsto_nhds_unique hformula hz
  rw [dividedFactors_prod, Real.log_div
    (ne_of_gt (Finset.prod_pos (fun i _ => hg i))) (ne_of_gt hR)] at he
  have hN : (Fintype.card ι : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have := (div_eq_zero_iff).mp he
  rcases this with hnum | hden
  · linarith
  · exact (hN hden).elim

end Dobson.FixedCycle
#print axioms Dobson.FixedCycle.finite_sum_log_limit
#print axioms Dobson.FixedCycle.cohort_log_limit
#print axioms Dobson.FixedCycle.threshold_log_loss
