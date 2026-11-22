import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic

/-! Further deterministic conclusions of `prop:dyn-k8`.
The polynomial used here is exactly the S/L characteristic polynomial after
equilibrium balance, not an arbitrary supplied root or stability conclusion.
The nonlinear identical-S equilibrium and its full Jacobian identification
remain outside this file; no full-row claim is made.
-/

namespace Dobson.Full2ReturnRoot
open Finset Set Filter
open scoped Topology

noncomputable def returnProduct {ι : Type*} [Fintype ι] (d : ι → ℝ) (γ : ℝ) : ℝ :=
  ∏ k, (1-γ/d k)

theorem returnProduct_strictAnti {ι : Type*} [Fintype ι] [Nonempty ι]
    (d : ι → ℝ) (hd : ∀ k, 0 < d k) (m : ℝ) (hm : ∀ k, m ≤ d k) :
    StrictAntiOn (returnProduct d) (Ico 0 m) := by
  intro x hx y hy hxy
  unfold returnProduct
  apply Finset.prod_lt_prod_of_nonempty₀
    (fun k _ => ?_) (fun k _ => ?_) Finset.univ_nonempty
  · have : y/d k < 1 := (div_lt_one (hd k)).mpr (hy.2.trans_le (hm k))
    linarith
  · exact sub_lt_sub_left ((div_lt_div_iff_of_pos_right (hd k)).mpr hxy) 1

/-- Existence and uniqueness are derived by the intermediate value theorem,
on the entire interval below the slowest diagonal rate. -/
theorem unique_return_root {ι : Type*} [Fintype ι] [Nonempty ι]
    (d : ι → ℝ) (hd : ∀ k, 0 < d k) (R : ℝ) (hR : 1 < R) :
    ∃! γ : ℝ, 0 < γ ∧ (∀ k, γ < d k) ∧ returnProduct d γ = 1/R := by
  classical
  let m := univ.inf' Finset.univ_nonempty d
  have hm : ∀ k, m ≤ d k := fun k => inf'_le _ (mem_univ k)
  obtain ⟨j, _, hj⟩ := exists_mem_eq_inf' Finset.univ_nonempty d
  change m = d j at hj
  have hm0 : 0 < m := by rw [hj]; exact hd j
  have hf0 : returnProduct d 0 = 1 := by simp [returnProduct]
  have hfm : returnProduct d m = 0 := by
    unfold returnProduct
    apply Finset.prod_eq_zero (mem_univ j)
    rw [hj, div_self (hd j).ne']
    ring
  have hc : Continuous (returnProduct d) := by unfold returnProduct; fun_prop
  have ht0 : 0 < 1/R := one_div_pos.mpr (by linarith)
  have ht1 : 1/R < 1 := (div_lt_one (by linarith : 0 < R)).mpr hR
  obtain ⟨γ, hγ, he⟩ := intermediate_value_Icc' hm0.le hc.continuousOn
    (show 1/R ∈ Icc (returnProduct d m) (returnProduct d 0) by
      rw [hfm, hf0]; exact ⟨ht0.le, ht1.le⟩)
  have hγ0 : 0 < γ := by
    apply lt_of_le_of_ne hγ.1
    intro hz
    have : 1 = 1/R := by simpa [hz.symm, hf0] using he
    linarith
  have hγm : γ < m := by
    apply lt_of_le_of_ne hγ.2
    intro hz
    have : 0 = 1/R := by simpa [hz, hfm] using he
    linarith
  refine ⟨γ, ⟨hγ0, fun k => hγm.trans_le (hm k), he⟩, ?_⟩
  intro x hx
  have hxm : x < m := by rw [hj]; exact hx.2.1 j
  exact (returnProduct_strictAnti d hd m hm).injOn ⟨hx.1.le, hxm⟩
    ⟨hγ0.le, hγm⟩ (hx.2.2.trans he.symm)

/-- Every complex root lies on or to the left of the real return root.
In particular the whole characteristic polynomial is Hurwitz. -/
theorem spectral_bound {ι : Type*} [Fintype ι] [Nonempty ι]
    (d : ι → ℝ) (hd : ∀ k, 0 < d k) (R γ : ℝ) (hR : 1 < R)
    (hγ : 0 < γ) (hγd : ∀ k, γ < d k) (hroot : returnProduct d γ = 1/R)
    (z : ℂ) (hz : (∏ k, (z+(d k : ℂ))) = ((∏ k, d k)/R : ℝ)) :
    z.re ≤ -γ ∧ z.re < 0 := by
  have hR0 : 0 < R := by linarith
  have hprod : (∏ k, (d k-γ)) = (∏ k, d k)/R := by
    have he : ∀ k, 1-γ/d k = (d k-γ)/d k := by
      intro k; field_simp [(hd k).ne']
    simp only [returnProduct, he, prod_div_distrib] at hroot
    have hp : (∏ k, d k) ≠ 0 := (prod_pos (fun k _ => hd k)).ne'
    apply (div_eq_iff hp).mp at hroot
    rw [hroot]
    ring
  have hle : z.re ≤ -γ := by
    by_contra hn
    have hgt : -γ < z.re := lt_of_not_ge hn
    have hlt : (∏ k, (d k-γ)) < ∏ k, ‖z+(d k : ℂ)‖ := by
      apply prod_lt_prod_of_nonempty₀ (fun k _ => sub_pos.mpr (hγd k))
        (fun k _ => ?_) univ_nonempty
      have hnorm := Complex.re_le_norm (z+(d k : ℂ))
      simp only [Complex.add_re, Complex.ofReal_re] at hnorm
      linarith
    rw [← norm_prod, hz, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (div_pos (prod_pos (fun k _ => hd k)) hR0), hprod] at hlt
    exact (lt_irrefl _ hlt)
  exact ⟨hle, hle.trans_lt (by linarith)⟩

/-- Equilibrium balance is obtained by cancellation around the actual cycle.
`r k` is `c N / a` in S and `c/a` in L, and `p k=0` in L. -/
theorem equilibrium_balance {ι : Type*} [Fintype ι] [DecidableEq ι]
    (next : Equiv.Perm ι) (u r p : ι → ℝ) (hu : ∀ k, 0 < u k)
    (heq : ∀ k, u (next k) = r k*(1-p k)*u k) :
    (∏ k, r k)*(∏ k, (1-p k)) = 1 := by
  have hp := congrArg (fun v : ι → ℝ => ∏ k, v k) (funext heq)
  simp only [prod_mul_distrib, Equiv.prod_comp] at hp
  have hune : (∏ k, u k) ≠ 0 := (prod_pos (fun k _ => hu k)).ne'
  apply mul_right_cancel₀ hune
  simpa using hp.symm

/-- The previously absent increasing comparison at fixed total filter time. -/
theorem fixed_filter_strictMono (L : ℝ) (hL : 0 < L) :
    StrictMonoOn (fun x : ℝ => x*(1-Real.exp (-L/x))) (Ioi 0) := by
  apply strictMonoOn_of_deriv_pos (convex_Ioi 0)
  · intro x hx
    fun_prop (disch := exact ne_of_gt hx)
  · intro x hx
    rw [interior_Ioi] at hx
    have hx0 : x ≠ 0 := ne_of_gt hx
    have hh := ((hasDerivAt_const x (-L)).div (hasDerivAt_id x) hx0).exp
    have hd := (hasDerivAt_id x).mul ((hasDerivAt_const x 1).sub hh)
    have hd' : HasDerivAt (fun x : ℝ => x*(1-Real.exp (-L/x)))
        (1-Real.exp (-L/x)*(1+L/x)) x := by
      convert hd using 1
      · rfl
      · dsimp
        field_simp
        <;> ring
    rw [hd'.deriv]
    have ht : 0 < L/x := div_pos hL hx
    have he := Real.add_one_lt_exp ht.ne'
    have hm := mul_lt_mul_of_pos_left he (Real.exp_pos (-L/x))
    have hc : Real.exp (-L/x)*Real.exp (L/x) = 1 := by
      rw [← Real.exp_add]
      convert Real.exp_zero using 1
      congr 1
      ring
    rw [hc] at hm
    nlinarith


/-- Both actual limits, for the two duration conventions in the manuscript. -/
theorem duration_limits (L T : ℝ) (_hL : 0 < L) (_hT : 0 < T) :
    Tendsto (fun x : ℝ => x*(Real.exp (L/x)-1)/T) atTop (nhds (L/T)) ∧
    Tendsto (fun x : ℝ => x*(1-Real.exp (-L/x))/T) atTop (nhds (L/T)) := by
  have hp : HasDerivAt (fun t : ℝ => Real.exp (L*t)) L 0 := by
    convert ((hasDerivAt_id (0:ℝ)).const_mul L).exp using 1 <;> simp
  have hn : HasDerivAt (fun t : ℝ => 1-Real.exp (-L*t)) L 0 := by
    convert (hasDerivAt_const (0:ℝ) 1).sub
      (((hasDerivAt_id (0:ℝ)).const_mul (-L)).exp) using 1
    · rfl
    · simp
  constructor
  · simpa [smul_eq_mul, div_eq_mul_inv] using
      (hp.tendsto_slope_zero_right.comp tendsto_inv_atTop_nhdsGT_zero).div_const T
  · simpa [smul_eq_mul, div_eq_mul_inv] using
      (hn.tendsto_slope_zero_right.comp tendsto_inv_atTop_nhdsGT_zero).div_const T

end Dobson.Full2ReturnRoot

#print axioms Dobson.Full2ReturnRoot.unique_return_root
#print axioms Dobson.Full2ReturnRoot.spectral_bound
#print axioms Dobson.Full2ReturnRoot.equilibrium_balance
#print axioms Dobson.Full2ReturnRoot.fixed_filter_strictMono

#print axioms Dobson.Full2ReturnRoot.duration_limits
