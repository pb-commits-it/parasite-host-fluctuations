import DobsonProofs.Coverage2Exact
import Mathlib.NumberTheory.Harmonic.GammaDeriv
import Mathlib.Analysis.Calculus.SmoothSeries
import Mathlib.Analysis.SpecialFunctions.Gamma.BohrMollerup

namespace Dobson.GammaDigammaSeries
open Filter Set
open scoped Topology
open Coverage2Exact

private theorem logGamma_deriv (x : ℝ) (hx : 0 < x) :
    HasDerivAt (fun y => Real.log (Real.Gamma y)) (gammaLogDerivative x) x := by
  have hg := (Real.hasDerivAt_Gamma_of_pos hx)
  convert hg.log (Real.Gamma_pos_of_pos hx).ne' using 1
  rw [gammaLogDerivative, hg.deriv]

private theorem logGamma_recurrence (x : ℝ) (hx : 0 < x) :
    Real.log (Real.Gamma (x + 1)) = Real.log (Real.Gamma x) + Real.log x := by
  rw [Real.Gamma_add_one hx.ne', Real.log_mul hx.ne' (Real.Gamma_pos_of_pos hx).ne']
  ring

theorem digamma_recurrence (x : ℝ) (hx : 0 < x) :
    gammaLogDerivative (x + 1) = gammaLogDerivative x + 1 / x := by
  have hleft := (logGamma_deriv (x+1) (by linarith)).comp x ((hasDerivAt_id x).add_const 1)
  have hright := (logGamma_deriv x hx).add (Real.hasDerivAt_log hx.ne')
  have he : (fun y => Real.log (Real.Gamma (y+1))) =ᶠ[𝓝 x]
      (fun y => Real.log (Real.Gamma y) + Real.log y) := by
    filter_upwards [eventually_gt_nhds hx] with y hy
    exact logGamma_recurrence y hy
  have hd := hleft.congr_of_eventuallyEq he.symm
  simpa using hd.unique hright

theorem digamma_nat_recurrence (x : ℝ) (hx : 0 < x) (n : ℕ) :
    gammaLogDerivative (x+n) = gammaLogDerivative x +
      ∑ k ∈ Finset.range n, 1 / (x+k) := by
  induction n with
  | zero => simp
  | succ n hn =>
    rw [Nat.cast_succ, ← add_assoc, digamma_recurrence (x+n) (by positivity), hn, Finset.sum_range_succ]
    ring

theorem digamma_log_upper (x : ℝ) (hx : 0 < x) : gammaLogDerivative x ≤ Real.log x := by
  have h := Real.convexOn_log_Gamma.deriv_le_slope hx (by linarith : 0 < x+1)
    (by linarith : x < x+1) (logGamma_deriv x hx).differentiableAt
  dsimp only [Function.comp_def] at h
  rw [(logGamma_deriv x hx).deriv] at h
  simpa only [slope_def_field, logGamma_recurrence x hx, add_sub_cancel_left,
    add_sub_cancel_right, div_one] using h

theorem digamma_log_lower (x : ℝ) (hx : 1 < x) : Real.log (x-1) ≤ gammaLogDerivative x := by
  have hx0 : 0 < x := by linarith
  have hx1 : 0 < x-1 := by linarith
  have h := Real.convexOn_log_Gamma.slope_le_deriv hx1 hx0
    (by linarith : x-1 < x) (logGamma_deriv x hx0).differentiableAt
  dsimp only [Function.comp_def] at h
  rw [(logGamma_deriv x hx0).deriv] at h
  have he : Real.log (Real.Gamma x) = Real.log (Real.Gamma (x-1)) + Real.log (x-1) := by
    convert logGamma_recurrence (x-1) hx1 using 1 <;> ring_nf
  have hden : x-(x-1) = 1 := by ring
  simpa only [slope_def_field, he, hden, add_sub_cancel_left, div_one] using h


private theorem log_nat_add_sub_log_tendsto (c : ℝ) :
    Tendsto (fun n : ℕ => Real.log (c+n) - Real.log n) atTop (𝓝 0) := by
  have hn : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop := tendsto_natCast_atTop_atTop
  have hr : Tendsto (fun n : ℕ => 1 + c / n) atTop (𝓝 1) := by
    simpa using tendsto_const_nhds.add (tendsto_const_nhds.div_atTop hn :
      Tendsto (fun n : ℕ => c / n) atTop (𝓝 0))
  have hl := (Real.continuousAt_log (by norm_num : (1 : ℝ) ≠ 0)).tendsto.comp hr
  rw [Real.log_one] at hl
  apply hl.congr'
  filter_upwards [hn.eventually (eventually_gt_atTop 0),
    hn.eventually (eventually_gt_atTop (-c))] with n hn0 hnc
  have hcn : 0 < c + n := by linarith
  dsimp only [Function.comp_def]
  rw [← Real.log_div hcn.ne' hn0.ne']
  congr 1
  field_simp
  ring

/-- The logarithmic derivative of Gamma has its standard logarithmic growth,
proved using log-convexity and the functional equation. -/
theorem digamma_nat_log_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun n : ℕ => gammaLogDerivative (p+n) - Real.log n) atTop (𝓝 0) := by
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le'
    (log_nat_add_sub_log_tendsto (p-1)) (log_nat_add_sub_log_tendsto p)
  · filter_upwards [eventually_gt_atTop (1 : ℕ)] with n hn
    have hn' : (1 : ℝ) < n := by exact_mod_cast hn
    have hh := digamma_log_lower (p+n) (by linarith)
    have he : p-1+(n:ℝ) = p+n-1 := by ring
    rw [he]
    linarith
  · filter_upwards with n
    have hh := digamma_log_upper (p+n) (by positivity)
    linarith

/-- Gauss' convergent partial sums for the actual logarithmic derivative. -/
theorem digamma_series_tendsto (p : ℝ) (hp : 0 < p) :
    Tendsto (fun n : ℕ => ∑ k ∈ Finset.range n,
      (1 / ((k : ℝ)+1) - 1 / (p+k))) atTop
      (𝓝 (gammaLogDerivative p + Real.eulerMascheroniConstant)) := by
  have hh := Real.tendsto_harmonic_sub_log.add (digamma_nat_log_tendsto p hp).neg
  have hlim : Real.eulerMascheroniConstant + - (0 : ℝ) = Real.eulerMascheroniConstant := by ring
  rw [hlim] at hh
  have hsum := hh.const_add (gammaLogDerivative p)
  change Tendsto (fun n : ℕ => gammaLogDerivative p +
    ((harmonic n : ℝ) - Real.log n + -(gammaLogDerivative (p+n) - Real.log n)))
    atTop (𝓝 (gammaLogDerivative p + Real.eulerMascheroniConstant)) at hsum
  apply hsum.congr'
  filter_upwards with n
  rw [digamma_nat_recurrence p hp n]
  have hH : (harmonic n : ℝ) = ∑ k ∈ Finset.range n, 1 / ((k : ℝ)+1) := by
    simp only [harmonic, Rat.cast_sum, Rat.cast_inv, Rat.cast_add, Rat.cast_one, Rat.cast_natCast, Nat.cast_add,
      Nat.cast_one, one_div]
  rw [hH, Finset.sum_sub_distrib]
  ring


private theorem summable_digamma_series (p : ℝ) (hp : 0 < p) :
    Summable (fun n : ℕ => 1 / ((n : ℝ)+1) - 1 / (p+n)) := by
  let a := min p 1
  have ha : 0 < a := lt_min hp zero_lt_one
  have hs : Summable (fun n : ℕ => 1 / (a+n)^2) := by
    have hh := (Real.summable_one_div_nat_add_rpow a (s := 2)).mpr (by norm_num)
    simpa only [Real.rpow_two, sq_abs, add_comm] using hh
  apply Summable.of_norm_bounded (hs.mul_left |p-1|)
  intro n
  have han : 0 < a+n := by positivity
  have hpn : 0 < p+n := by positivity
  have hn1 : 0 < (n:ℝ)+1 := by positivity
  have he : 1 / ((n:ℝ)+1) - 1 / (p+n) = (p-1) / (((n:ℝ)+1)*(p+n)) := by
    field_simp
    <;> ring
  rw [he, Real.norm_eq_abs, abs_div, abs_of_pos (mul_pos hn1 hpn)]
  have ha1 : a ≤ 1 := min_le_right _ _
  have hap : a ≤ p := min_le_left _ _
  have hden : (a+n)^2 ≤ ((n:ℝ)+1)*(p+n) := by
    nlinarith [mul_nonneg (show 0 ≤ (n:ℝ)+1-(a+n) by linarith)
      (show 0 ≤ p+n-(a+n) by linarith)]
  simpa only [mul_one_div] using div_le_div_of_nonneg_left (abs_nonneg (p-1))
    (sq_pos_of_pos han) hden

theorem digamma_series (p : ℝ) (hp : 0 < p) :
    (∑' n : ℕ, (1 / ((n:ℝ)+1) - 1 / (p+n))) =
      gammaLogDerivative p + Real.eulerMascheroniConstant := by
  exact tendsto_nhds_unique (summable_digamma_series p hp).hasSum.tendsto_sum_nat
    (digamma_series_tendsto p hp)

/-- The manuscript's positive trigamma series is the derivative of the actual
logarithmic derivative of Gamma, proved by differentiating Gauss' series. -/
theorem gammaLogDerivative_hasDerivAt (p : ℝ) (hp : 0 < p) :
    HasDerivAt gammaLogDerivative (gammaTrigamma p) p := by
  let g := fun n : ℕ => fun x : ℝ => 1 / ((n:ℝ)+1) - 1 / (x+n)
  let g' := fun n : ℕ => fun x : ℝ => 1 / (x+n)^2
  let u := fun n : ℕ => 1 / (p/2+n)^2
  have hu : Summable u := by
    have hh := (Real.summable_one_div_nat_add_rpow (p/2) (s := 2)).mpr (by norm_num)
    simpa only [u, Real.rpow_two, sq_abs, add_comm] using hh
  have hg : ∀ n x, x ∈ Ioi (p/2) → HasDerivAt (g n) (g' n x) x := by
    intro n x hx
    have hx0 : 0 < x := lt_trans (by linarith : 0 < p/2) hx
    have hn : x+(n:ℝ) ≠ 0 := ne_of_gt (by positivity)
    have hh := (((hasDerivAt_id x).add_const (n:ℝ)).inv hn).const_sub (1 / ((n:ℝ)+1))
    convert hh using 1
    · funext y
      simp [g, one_div]
    · dsimp [g']
      ring
  have hb : ∀ n x, x ∈ Ioi (p/2) → ‖g' n x‖ ≤ u n := by
    intro n x hx
    have hl : 0 < p/2+(n:ℝ) := by positivity
    have hr : 0 < x+(n:ℝ) := by have : p/2 < x := hx; linarith
    dsimp [g',u]
    rw [abs_of_pos (by positivity : 0 < 1 / (x+(n:ℝ))^2)]
    apply div_le_div_of_nonneg_left zero_le_one (sq_pos_of_pos hl)
    have hx' : p/2 < x := hx
    nlinarith
  have hmem : p ∈ Ioi (p/2) := by change p/2 < p; linarith
  have hh := hasDerivAt_tsum_of_isPreconnected hu isOpen_Ioi (convex_Ioi (p/2)).isPreconnected
    hg hb hmem (summable_digamma_series p hp) hmem
  have ht := hh.sub_const Real.eulerMascheroniConstant
  have he : (fun x => (∑' n : ℕ, g n x) - Real.eulerMascheroniConstant) =ᶠ[𝓝 p]
      gammaLogDerivative := by
    filter_upwards [eventually_gt_nhds hp] with x hx
    dsimp only [g]
    rw [digamma_series x hx]
    ring
  exact ht.congr_of_eventuallyEq he.symm

theorem gammaTrigamma_eq_deriv (p : ℝ) (hp : 0 < p) :
    gammaTrigamma p = deriv gammaLogDerivative p :=
  (gammaLogDerivative_hasDerivAt p hp).deriv.symm

theorem gammaTrigamma_pos (p : ℝ) (hp : 0 < p) : 0 < gammaTrigamma p := by
  have hs : Summable (fun n : ℕ => 1 / (p+n)^2) := by
    have hh := (Real.summable_one_div_nat_add_rpow p (s := 2)).mpr (by norm_num)
    simpa only [Real.rpow_two, sq_abs, add_comm] using hh
  exact hs.tsum_pos (fun n => by positivity) 0 (by simp only [Nat.cast_zero, add_zero]; positivity)

end Dobson.GammaDigammaSeries
#print axioms Dobson.GammaDigammaSeries.digamma_series_tendsto

#print axioms Dobson.GammaDigammaSeries.gammaTrigamma_eq_deriv
