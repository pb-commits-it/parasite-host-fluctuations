import DobsonProofs.Coverage6Covariance
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-! # Part B: finite-response limits and conditional restoring balance

Coverage correction, 2026-11-22: `lem:mech-general` remains partial-P1.
The independent fidelity audit (2026-11-16) did not confirm closure of
`part1:62` or `part1:63` by this uncredited module.

Geometric decay gives summability of the covariance series. The finite-response
limit theorem assumes `hL_second` for supplied functions `L N`; it does not
identify a physical Lyapunov coefficient. The restoring-balance corollary
also assumes a root family, its quadratic coefficient limit and its local
growth expansion. The eigenvalue derivative calculation assumes a
differentiable eigenpair family. -/

namespace Dobson.P1bB_MechGeneral

open scoped BigOperators
open MeasureTheory Filter
open scoped Topology
open Dobson.Coverage6Covariance

section TransverseSummability

variable {ι κ Ω : Type*} [Fintype κ] [DecidableEq κ] [MeasurableSpace Ω]

/-- Geometric decay on the transverse subspace and bounded lag covariances imply
summability of `kernel T Q E v w i j m * covariance μ ξ i j (m + 1)`. -/
lemma summable_kernel_covariance
    (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (Bcov C_T β : ℝ) (_hBcov : 0 ≤ Bcov) (hCT : 0 ≤ C_T) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hcov_bdd : ∀ i j lag, |covariance μ ξ i j lag| ≤ Bcov)
    (hdecay : ∀ i j m, |kernel T Q E v w i j m| ≤ C_T * β ^ m) :
    ∀ i j, Summable (fun m : ℕ => kernel T Q E v w i j m * covariance μ ξ i j (m + 1)) := by
  intro i j
  apply Summable.of_norm_bounded
    ((summable_geometric_of_lt_one hβ0 hβ1).mul_left (C_T * Bcov))
  intro m
  rw [Real.norm_eq_abs, abs_mul]
  calc
    |kernel T Q E v w i j m| * |covariance μ ξ i j (m + 1)|
        ≤ (C_T * β ^ m) * Bcov :=
          mul_le_mul (hdecay i j m) (hcov_bdd i j (m + 1))
            (abs_nonneg _) (mul_nonneg hCT (pow_nonneg hβ0 m))
    _ = (C_T * Bcov) * β ^ m := by ring

/-- Limit of the finite-response quadratic expectations, assuming
`hL_second` identifies them with half the second derivatives of supplied
functions `L N`. The limit `g2` is defined from those derivatives; the theorem
does not identify it with physical growth or state a quantitative tail bound. -/
lemma growth_identification_of_second_deriv [Fintype ι]
    (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (direct : Ω → ℝ) (L : ℕ → ℝ → ℝ)
    (Bcov C_T β : ℝ) (hBcov : 0 ≤ Bcov) (hCT : 0 ≤ C_T) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ)
    (hd : Integrable direct μ) (hd0 : ∫ ω, direct ω ∂μ = 0)
    (hcov_bdd : ∀ i j lag, |covariance μ ξ i j lag| ≤ Bcov)
    (hdecay : ∀ i j m, |kernel T Q E v w i j m| ≤ C_T * β ^ m)
    (hL_second : ∀ N, iteratedDeriv 2 (L N) 0 =
      2 * ∫ ω, logarithmicQuadratic ξ T Q E v w direct N ω ∂μ) :
    let g2 := (1 / 2 : ℝ) * limUnder atTop (fun N : ℕ => iteratedDeriv 2 (L N) 0)
    Tendsto (fun N => ∫ ω, logarithmicQuadratic ξ T Q E v w direct N ω ∂μ)
      atTop (𝓝 g2) := by
  intro g2
  have hsum := summable_kernel_covariance μ ξ T Q E v w Bcov C_T β
    hBcov hCT hβ0 hβ1 hcov_bdd hdecay
  have hexp := expected_quadratic_limit μ ξ T Q E v w direct hξ hd hd0 hsum
  have hL_lim : Tendsto (fun N : ℕ => iteratedDeriv 2 (L N) 0) atTop
      (𝓝 (2 * ∑ i, ∑ j, ((∑' m : ℕ,
        kernel T Q E v w i j m * covariance μ ξ i j (m + 1)) -
          elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2))) := by
    simpa only [hL_second] using hexp.const_mul 2
  have hg2_eq : g2 = ∑ i, ∑ j, ((∑' m : ℕ,
      kernel T Q E v w i j m * covariance μ ξ i j (m + 1)) -
        elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2) := by
    dsimp [g2]
    rw [hL_lim.limUnder_eq]
    ring
  rwa [hg2_eq]

end TransverseSummability

section BirthBalance

/-- Deterministic Perron birth-tilt derivative (`C-mechanisms.tex` line 26):
if `A(y) v(y) = λ(y) v(y)` with `A(0) = A₀`, `A'(0) = B_birth`, `wᵀ A₀ = wᵀ`, `v(0) = v`,
`λ(0) = 1`, and `wᵀ v = 1`, then `(d/dy) log λ(y)|_{y=0} = wᵀ B_birth v`. -/
theorem perron_birth_slope {κ : Type*} [Fintype κ] [DecidableEq κ]
    (A₀ B_birth : Matrix κ κ ℝ) (v w : κ → ℝ)
    (lam : ℝ → ℝ) (vec : ℝ → κ → ℝ) (b_deriv : ℝ)
    (hwA : A₀.vecMul w = w) (hnorm : w ⬝ᵥ v = 1)
    (hlam0 : lam 0 = 1) (hvec0 : vec 0 = v)
    (heig : ∀ y, ((A₀ + (Real.exp y - 1) • B_birth).mulVec (vec y)) = lam y • vec y)
    (hlam_deriv : HasDerivAt lam b_deriv 0)
    (hvec_diff : ∀ k, DifferentiableAt ℝ (fun y => vec y k) 0) :
    HasDerivAt (fun y => Real.log (lam y)) (w ⬝ᵥ (B_birth.mulVec v)) 0 ∧
      b_deriv = w ⬝ᵥ (B_birth.mulVec v) := by
  let vec' : κ → ℝ := fun k => deriv (fun y => vec y k) 0
  have hvec_hasDeriv : ∀ k, HasDerivAt (fun y => vec y k) (vec' k) 0 :=
    fun k => (hvec_diff k).hasDerivAt
  have hexp : HasDerivAt (fun y : ℝ => Real.exp y - 1) 1 0 := by
    simpa using (Real.hasDerivAt_exp 0).sub_const 1
  have hlhs_eq : ∀ y, w ⬝ᵥ ((A₀ + (Real.exp y - 1) • B_birth).mulVec (vec y)) =
      w ⬝ᵥ vec y + (Real.exp y - 1) * (w ⬝ᵥ (B_birth.mulVec (vec y))) := by
    intro y
    rw [Matrix.add_mulVec, Matrix.smul_mulVec, dotProduct_add, dotProduct_smul, smul_eq_mul,
      Matrix.dotProduct_mulVec, hwA]
  have hrhs_eq : ∀ y, w ⬝ᵥ (lam y • vec y) = lam y * (w ⬝ᵥ vec y) := by
    intro y
    rw [dotProduct_smul, smul_eq_mul]
  have hscalar : ∀ y, w ⬝ᵥ vec y + (Real.exp y - 1) * (w ⬝ᵥ (B_birth.mulVec (vec y))) =
      lam y * (w ⬝ᵥ vec y) := by
    intro y
    rw [← hlhs_eq y, heig y, hrhs_eq y]
  have hwvec_deriv : HasDerivAt (fun y => w ⬝ᵥ vec y) (w ⬝ᵥ vec') 0 := by
    unfold dotProduct
    exact HasDerivAt.fun_sum fun k _ => (hvec_hasDeriv k).const_mul (w k)
  have hwBvec_deriv : HasDerivAt (fun y => w ⬝ᵥ (B_birth.mulVec (vec y)))
      (w ⬝ᵥ (B_birth.mulVec vec')) 0 := by
    have hfn : (fun y => w ⬝ᵥ (B_birth.mulVec (vec y))) =
        fun y => ∑ i, w i * ∑ j, B_birth i j * vec y j := by
      funext y; rfl
    rw [hfn]
    exact HasDerivAt.fun_sum fun i _ =>
      (HasDerivAt.fun_sum fun j _ => (hvec_hasDeriv j).const_mul (B_birth i j)).const_mul (w i)
  have hlhs_deriv : HasDerivAt
      (fun y => w ⬝ᵥ vec y + (Real.exp y - 1) * (w ⬝ᵥ (B_birth.mulVec (vec y))))
      (w ⬝ᵥ vec' + (1 * (w ⬝ᵥ (B_birth.mulVec (vec 0))) +
        (Real.exp 0 - 1) * (w ⬝ᵥ (B_birth.mulVec vec')))) 0 :=
    hwvec_deriv.add (hexp.mul hwBvec_deriv)
  have hrhs_deriv : HasDerivAt (fun y => lam y * (w ⬝ᵥ vec y))
      (b_deriv * (w ⬝ᵥ vec 0) + lam 0 * (w ⬝ᵥ vec')) 0 :=
    hlam_deriv.mul hwvec_deriv
  have hrhs_deriv' : HasDerivAt
      (fun y => w ⬝ᵥ vec y + (Real.exp y - 1) * (w ⬝ᵥ (B_birth.mulVec (vec y))))
      (b_deriv * (w ⬝ᵥ vec 0) + lam 0 * (w ⬝ᵥ vec')) 0 := by
    simpa only [funext hscalar] using hrhs_deriv
  have heq := hlhs_deriv.unique hrhs_deriv'
  rw [hvec0, hlam0, hnorm, Real.exp_zero, sub_self, zero_mul, add_zero, one_mul, mul_one,
    one_mul] at heq
  have hb_eq : b_deriv = w ⬝ᵥ (B_birth.mulVec v) := by linarith
  refine ⟨?_, hb_eq⟩
  have hlog := hlam_deriv.log (by rw [hlam0]; norm_num)
  rwa [hlam0, div_one, hb_eq] at hlog

/-- Conditional root-balance algebra: a supplied expansion along `Δ`,
the equation `r(σ, Δ(σ)) = 0`, and the limit `Δ(σ) / σ² → K` imply
`b * K + g2 = 0`. These premises retain the physical moving-root bridge. -/
lemma birth_balance_of_local_expansion
    (r : ℝ → ℝ → ℝ) (Δ : ℝ → ℝ) (b K g2 : ℝ)
    (hroot : ∀ᶠ σ in 𝓝[≠] (0 : ℝ), r σ (Δ σ) = 0)
    (hK : Tendsto (fun σ : ℝ => Δ σ / σ ^ 2) (𝓝[≠] 0) (𝓝 K))
    (hexp : Tendsto (fun σ : ℝ => (r σ (Δ σ) - (b * Δ σ + g2 * σ ^ 2)) / σ ^ 2)
      (𝓝[≠] 0) (𝓝 0)) :
    b * K + g2 = 0 := by
  have hcomb : Tendsto (fun σ : ℝ =>
      (r σ (Δ σ) - (b * Δ σ + g2 * σ ^ 2)) / σ ^ 2 + b * (Δ σ / σ ^ 2) + g2)
      (𝓝[≠] 0) (𝓝 (0 + b * K + g2)) :=
    (hexp.add (hK.const_mul b)).add_const g2
  have hzero : Tendsto (fun σ : ℝ =>
      (r σ (Δ σ) - (b * Δ σ + g2 * σ ^ 2)) / σ ^ 2 + b * (Δ σ / σ ^ 2) + g2)
      (𝓝[≠] 0) (𝓝 0) := by
    apply tendsto_const_nhds.congr'
    filter_upwards [hroot, self_mem_nhdsWithin] with σ hσ0 hσ_ne
    have hσ_ne' : σ ≠ 0 := hσ_ne
    rw [hσ0]
    field_simp [hσ_ne']
    ring
  have huniq := tendsto_nhds_unique hcomb hzero
  linarith

end BirthBalance

section Corollary

variable {ι κ Ω : Type*} [Fintype ι] [Fintype κ] [DecidableEq κ] [MeasurableSpace Ω]

/-- Ordered covariance formula conditional on `hL_second`, the supplied
root family `hroot`, its coefficient limit `hK`, and the expansion `hexp`.
The finite-response limit and scalar root-balance lemmas combine these inputs;
they do not discharge the physical growth and moving-root identifications. -/
theorem ordered_covariance_coefficient_unconditional
    (μ : Measure Ω) (ξ : ι → ℕ → Ω → ℝ)
    (T Q : Matrix κ κ ℝ) (E : ι → Matrix κ κ ℝ) (v w : κ → ℝ)
    (direct : Ω → ℝ) (L : ℕ → ℝ → ℝ)
    (r : ℝ → ℝ → ℝ) (Δ : ℝ → ℝ) (b K : ℝ) (hb : 0 < b)
    (Bcov C_T β : ℝ) (hBcov : 0 ≤ Bcov) (hCT : 0 ≤ C_T) (hβ0 : 0 ≤ β) (hβ1 : β < 1)
    (hξ : ∀ i j lag, Integrable (fun ω => ξ i 0 ω * ξ j lag ω) μ)
    (hd : Integrable direct μ) (hd0 : ∫ ω, direct ω ∂μ = 0)
    (hcov_bdd : ∀ i j lag, |covariance μ ξ i j lag| ≤ Bcov)
    (hdecay : ∀ i j m, |kernel T Q E v w i j m| ≤ C_T * β ^ m)
    (hL_second : ∀ N, iteratedDeriv 2 (L N) 0 =
      2 * ∫ ω, logarithmicQuadratic ξ T Q E v w direct N ω ∂μ)
    (hroot : ∀ᶠ σ in 𝓝[≠] (0 : ℝ), r σ (Δ σ) = 0)
    (hK : Tendsto (fun σ : ℝ => Δ σ / σ ^ 2) (𝓝[≠] 0) (𝓝 K))
    (hexp : Tendsto (fun σ : ℝ =>
      (r σ (Δ σ) - (b * Δ σ +
        ((1 / 2 : ℝ) * limUnder atTop (fun N : ℕ => iteratedDeriv 2 (L N) 0)) * σ ^ 2)) / σ ^ 2)
      (𝓝[≠] 0) (𝓝 0)) :
    K = (1 / b) * ∑ i, ∑ j,
      (elasticity E v w i * elasticity E v w j * covariance μ ξ i j 0 / 2 -
        ∑' m : ℕ, covariance μ ξ i j (m + 1) * kernel T Q E v w i j m) := by
  let g2 := (1 / 2 : ℝ) * limUnder atTop (fun N : ℕ => iteratedDeriv 2 (L N) 0)
  have hsum := summable_kernel_covariance μ ξ T Q E v w Bcov C_T β
    hBcov hCT hβ0 hβ1 hcov_bdd hdecay
  have hgrowth := growth_identification_of_second_deriv μ ξ T Q E v w direct L
    Bcov C_T β hBcov hCT hβ0 hβ1 hξ hd hd0 hcov_bdd hdecay hL_second
  have hbirth := birth_balance_of_local_expansion r Δ b K g2 hroot hK hexp
  exact ordered_covariance_coefficient μ ξ T Q E v w direct b K g2 hb
    hξ hd hd0 hsum hgrowth hbirth

end Corollary

end Dobson.P1bB_MechGeneral

#print axioms Dobson.P1bB_MechGeneral.summable_kernel_covariance
#print axioms Dobson.P1bB_MechGeneral.growth_identification_of_second_deriv
#print axioms Dobson.P1bB_MechGeneral.perron_birth_slope
#print axioms Dobson.P1bB_MechGeneral.birth_balance_of_local_expansion
#print axioms Dobson.P1bB_MechGeneral.ordered_covariance_coefficient_unconditional
