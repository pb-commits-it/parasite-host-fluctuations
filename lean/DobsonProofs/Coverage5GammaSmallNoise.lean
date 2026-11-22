import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
# Batch 5, item 35 (`prop:exact-R7`): exact Laplace derivatives and coefficient

The smooth potential is written as `log(s+exp z)-z+(s/c)exp z`, algebraically equal
to the manuscript's `log(1+s exp(-z))+(s/c)exp z`. The derivatives, their values at
the deterministic threshold, the normalized Laplace coefficient and its `1/4`
endpoint are checked. The probabilistic integral identification, uniform Laplace
error, implicit-root error, gamma-gap expansion and iterated model limits remain
analytic/probability inputs that this file does not discharge.
-/
namespace Dobson.Coverage5GammaSmallNoise

noncomputable def observable (s z : ℝ) : ℝ := Real.log (s + Real.exp z)
noncomputable def observableFirst (s z : ℝ) : ℝ := Real.exp z / (s + Real.exp z)
noncomputable def observableSecond (s z : ℝ) : ℝ := s * Real.exp z / (s + Real.exp z) ^ 2
noncomputable def potential (s c z : ℝ) : ℝ := observable s z - z + (s / c) * Real.exp z
noncomputable def potentialFirst (s c z : ℝ) : ℝ := observableFirst s z - 1 + (s / c) * Real.exp z
noncomputable def potentialSecond (s c z : ℝ) : ℝ := observableSecond s z + (s / c) * Real.exp z
noncomputable def potentialThird (s c z : ℝ) : ℝ :=
  s * Real.exp z * (s - Real.exp z) / (s + Real.exp z) ^ 3 + (s / c) * Real.exp z

/-- The differentiated potential is exactly the manuscript's original function. -/
theorem potential_eq_manuscript (s c z : ℝ) (hs : 0 < s) :
    potential s c z = Real.log (1 + s * Real.exp (-z)) + (s / c) * Real.exp z := by
  have he : Real.exp z ≠ 0 := (Real.exp_pos z).ne'
  have hsum : s + Real.exp z ≠ 0 := by positivity
  have hid : 1 + s * Real.exp (-z) = (s + Real.exp z) / Real.exp z := by
    rw [Real.exp_neg]
    field_simp <;> ring
  rw [hid, Real.log_div hsum he, Real.log_exp]
  rfl

/-- The first two observable derivatives are calculated from the exponential and
logarithm, rather than given as Laplace-theory inputs. -/
theorem observable_derivatives (s z : ℝ) (hs : 0 < s) :
    HasDerivAt (observable s) (observableFirst s z) z ∧
      HasDerivAt (observableFirst s) (observableSecond s z) z := by
  have hden : s + Real.exp z ≠ 0 := by positivity
  have he := Real.hasDerivAt_exp z
  constructor
  · change HasDerivAt (fun y : ℝ => Real.log (s + Real.exp y))
      (Real.exp z / (s + Real.exp z)) z
    simpa only [Pi.add_apply, zero_add] using (((hasDerivAt_const z s).add he).log hden)
  · convert (he.div ((hasDerivAt_const z s).add he) hden) using 1
    · rfl
    · dsimp [observableSecond]
      field_simp <;> ring

/-- Three derivatives of the Laplace potential, including the cubic term needed
for its first correction. No derivative of the potential is assumed. -/
theorem potential_derivatives (s c z : ℝ) (hs : 0 < s) :
    HasDerivAt (potential s c) (potentialFirst s c z) z ∧
      HasDerivAt (potentialFirst s c) (potentialSecond s c z) z ∧
      HasDerivAt (potentialSecond s c) (potentialThird s c z) z := by
  have he := Real.hasDerivAt_exp z
  have hobs := observable_derivatives s z hs
  have hden : s + Real.exp z ≠ 0 := by positivity
  have hobs2 : HasDerivAt (observableSecond s)
      (s * Real.exp z * (s - Real.exp z) / (s + Real.exp z) ^ 3) z := by
    convert ((he.const_mul s).div (((hasDerivAt_const z s).add he).pow 2)
      (pow_ne_zero 2 hden)) using 1
    · rfl
    · dsimp
      field_simp <;> ring
  refine ⟨?_, ?_, ?_⟩
  · convert ((hobs.1.sub (hasDerivAt_id z)).add (he.const_mul (s / c))) using 1 <;> rfl
  · convert ((hobs.2.sub_const 1).add (he.const_mul (s / c))) using 1 <;> rfl
  · convert (hobs2.add (he.const_mul (s / c))) using 1 <;> rfl

/-- Exact saddle values at `c=1-s`, `z=log(1-s)`, including stationarity. -/
theorem deterministic_saddle_values (s : ℝ) (hs0 : 0 < s) (hs1 : s < 1) :
    observableFirst s (Real.log (1 - s)) = 1 - s ∧
      observableSecond s (Real.log (1 - s)) = s * (1 - s) ∧
      potentialFirst s (1 - s) (Real.log (1 - s)) = 0 ∧
      potentialSecond s (1 - s) (Real.log (1 - s)) = s * (2 - s) ∧
      potentialThird s (1 - s) (Real.log (1 - s)) = s ^ 2 * (3 - 2 * s) := by
  have ht : 0 < 1 - s := by linarith
  simp only [observableFirst, observableSecond, potentialFirst, potentialSecond,
    potentialThird, Real.exp_log ht]
  have hsum : s + (1 - s) = 1 := by ring
  simp only [hsum, one_pow, div_one]
  have hcancel : s / (1 - s) * (1 - s) = s := div_mul_cancel₀ s ht.ne'
  rw [hcancel]
  refine ⟨by trivial, by trivial, ?_, ?_, ?_⟩ <;> ring

/-- The explicit first Laplace correction is negative, and division by the
positive deterministic log-transfer derivative gives `(1-s)/(2(2-s))`. -/
theorem laplace_coefficient (s : ℝ) (hs0 : 0 < s) (hs1 : s < 1) :
    ((s * (1 - s)) / (s * (2 - s)) -
      (1 - s) * (s ^ 2 * (3 - 2 * s)) / (s * (2 - s)) ^ 2) / 2 =
      -(1 - s) ^ 2 / (2 * (2 - s) ^ 2) ∧
    -(-(1 - s) ^ 2 / (2 * (2 - s) ^ 2)) / ((1 - s) / (2 - s)) =
      (1 - s) / (2 * (2 - s)) ∧
    0 < (1 - s) / (2 * (2 - s)) := by
  have ht : 0 < 1 - s := by linarith
  have hd : 0 < 2 - s := by linarith
  refine ⟨?_, ?_, div_pos ht (mul_pos (by norm_num) hd)⟩
  · field_simp <;> ring
  · field_simp <;> ring

/-- The exact coefficient's endpoint is `1/4`; the opposite-order coefficient
is `1/2` if the independent gamma-gap expansion supplies that value. This is a
limit of the proved scalar coefficient, not a claim about the stochastic D. -/
theorem laplace_coefficient_endpoint :
    Filter.Tendsto (fun s : ℝ => (1 - s) / (2 * (2 - s))) (nhds 0) (nhds (1 / 4)) ∧
      (1 / 4 : ℝ) ≠ 1 / 2 := by
  constructor
  · have h : ContinuousAt (fun s : ℝ => (1 - s) / (2 * (2 - s))) 0 := by fun_prop (disch := norm_num)
    convert h.tendsto using 1 <;> norm_num
  · norm_num

/-- The deterministic logarithmic-transfer derivative used in solving the first
Laplace correction is also calculated from the explicit Perron root. -/
theorem deterministic_log_transfer_derivative (s : ℝ) (hs : s < 1) :
    HasDerivAt (fun y : ℝ => Real.log
      ((s + Real.sqrt (s ^ 2 + 4 * (1 - s) * Real.exp y)) / 2))
      ((1 - s) / (2 - s)) 0 := by
  have hd : 0 < 2 - s := by linarith
  have hrad : s ^ 2 + 4 * (1 - s) = (2 - s) ^ 2 := by ring
  have hsqrt : Real.sqrt (s ^ 2 + 4 * (1 - s)) = 2 - s := by
    rw [hrad, Real.sqrt_sq_eq_abs, abs_of_pos hd]
  have hr : s ^ 2 + 4 * (1 - s) * Real.exp 0 ≠ 0 := by
    simp only [Real.exp_zero, mul_one, hrad]
    exact pow_ne_zero 2 hd.ne'
  have hbase := (hasDerivAt_const (0 : ℝ) (s ^ 2)).add
    ((Real.hasDerivAt_exp 0).const_mul (4 * (1 - s)))
  have hroot := ((hbase.sqrt hr).const_add s).div_const 2
  have hn : (s + Real.sqrt (s ^ 2 + 4 * (1 - s) * Real.exp 0)) / 2 ≠ 0 := by
    simp only [Real.exp_zero, mul_one, hsqrt]
    ring_nf <;> norm_num
  convert hroot.log hn using 1
  dsimp
  simp only [Real.exp_zero, mul_one, hsqrt, zero_add]
  field_simp <;> ring

end Dobson.Coverage5GammaSmallNoise
#print axioms Dobson.Coverage5GammaSmallNoise.potential_eq_manuscript
#print axioms Dobson.Coverage5GammaSmallNoise.observable_derivatives
#print axioms Dobson.Coverage5GammaSmallNoise.potential_derivatives
#print axioms Dobson.Coverage5GammaSmallNoise.deterministic_saddle_values
#print axioms Dobson.Coverage5GammaSmallNoise.laplace_coefficient
#print axioms Dobson.Coverage5GammaSmallNoise.laplace_coefficient_endpoint
#print axioms Dobson.Coverage5GammaSmallNoise.deterministic_log_transfer_derivative
