import DobsonProofs.Full2_ReturnRoot
import DobsonProofs.Full3_CyclicDet
import Mathlib.LinearAlgebra.Matrix.Charpoly.Eigs
import Mathlib.Analysis.Calculus.FDeriv.Pi
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Logic.Equiv.Fin.Rotate

/-!
The nonlinear S/L model, its actual derivative, and the full constant-host
return-rate statement (manuscript Proposition dyn-k8). `S i` selects a
constant-number host; other stages are linear. No equilibrium balance,
Jacobian formula, or spectral conclusion is assumed.
-/

namespace Dobson.Full3.ConstantHosts
open Finset Set Filter
open scoped Topology

attribute [-simp] finRotate_apply finRotate_symm_apply

variable {n : ℕ}

noncomputable def prev (n : ℕ) : Equiv.Perm (Fin n) := (finRotate n).symm

noncomputable def input (c x : Fin n → ℝ) (i : Fin n) : ℝ :=
  c (prev n i) * x (prev n i)

noncomputable def field (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (i : Fin n) : ℝ :=
  input c x i * (if S i then N i - x i else 1) - a i * x i

noncomputable def rate (S : Fin n → Prop) [DecidablePred S]
    (a c x : Fin n → ℝ) (i : Fin n) : ℝ :=
  a i + if S i then input c x i else 0

noncomputable def susceptibility (S : Fin n → Prop) [DecidablePred S]
    (N x : Fin n → ℝ) (i : Fin n) : ℝ := if S i then N i-x i else 1

noncomputable def differential (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) : (Fin n → ℝ) →L[ℝ] (Fin n → ℝ) :=
  ContinuousLinearMap.pi fun i =>
    (c (prev n i) * susceptibility S N x i) • ContinuousLinearMap.proj (prev n i) -
    rate S a c x i • ContinuousLinearMap.proj i

noncomputable def jacobian (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  LinearMap.toMatrix' (differential S a c N x).toLinearMap

theorem actual_derivative (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) :
    HasFDerivAt (field S a c N) (differential S a c N x) x := by
  apply hasFDerivAt_pi.mpr
  intro i
  have hi := (hasFDerivAt_apply (𝕜 := ℝ) (prev n i) x).const_mul (c (prev n i))
  have hx := hasFDerivAt_apply (𝕜 := ℝ) i x
  by_cases hS : S i
  · convert (hi.mul ((hasFDerivAt_const (N i) x).sub hx)).sub
      (hx.const_mul (a i)) using 1
    · funext y; simp [field, input, hS]
    · ext v
      simp [differential, susceptibility, rate, input, hS]
      ring
  · convert hi.sub (hx.const_mul (a i)) using 1
    · funext y; simp [field, input, hS]
    · ext v
      simp [differential, susceptibility, rate, input, hS]

theorem jacobian_entry (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (i j : Fin n) :
    jacobian S a c N x i j =
      (if i = finRotate n j then c j * susceptibility S N x i else 0) -
      (if i = j then rate S a c x i else 0) := by
  have he : prev n i = j ↔ i = finRotate n j := by
    exact (finRotate n).symm_apply_eq
  simp only [jacobian, LinearMap.toMatrix'_apply, differential,
    ContinuousLinearMap.coe_coe, ContinuousLinearMap.pi_apply,
    ContinuousLinearMap.sub_apply, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul, Pi.single_apply]
  by_cases h : prev n i = j
  · have hj := he.mp h
    subst i
    simp [prev, eq_comm, mul_ite]
  · simp [h, he.not.mp h, Ne.symm h, eq_comm, mul_ite]

/-- The equilibrium assumptions are the literal nonlinear equations. -/
structure PositiveEquilibrium (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) : Prop where
  ha : ∀ i, 0 < a i
  hc : ∀ i, 0 < c i
  hx : ∀ i, 0 < x i
  equilibrium : ∀ i, field S a c N x i = 0

theorem PositiveEquilibrium.host {S : Fin n → Prop} [DecidablePred S]
    {a c N x : Fin n → ℝ} (h : PositiveEquilibrium S a c N x)
    (i : Fin n) (hS : S i) : x i < N i := by
  have hu : 0 < input c x i := mul_pos (h.hc _) (h.hx _)
  have he := h.equilibrium i
  simp only [field, if_pos hS] at he
  have hax := mul_pos (h.ha i) (h.hx i)
  by_contra hn
  have hmul := mul_nonpos_of_nonneg_of_nonpos hu.le
    (sub_nonpos.mpr (le_of_not_gt hn))
  linarith

noncomputable def prevalence (S : Fin n → Prop) [DecidablePred S]
    (N x : Fin n → ℝ) (i : Fin n) : ℝ := if S i then x i/N i else 0

noncomputable def reproduction (S : Fin n → Prop) [DecidablePred S]
    (a c N : Fin n → ℝ) : ℝ := ∏ i, c i*(if S i then N i else 1)/a i

theorem equilibrium_local (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x) (i : Fin n) :
    0 < rate S a c x i ∧
    rate S a c x i = a i / (1-prevalence S N x i) ∧
    input c x i * susceptibility S N x i = a i*x i := by
  have hu : 0 < input c x i := mul_pos (h.hc _) (h.hx _)
  have he := h.equilibrium i
  change input c x i * susceptibility S N x i - a i*x i = 0 at he
  refine ⟨?_, ?_, by linarith⟩
  · dsimp [rate]; split_ifs <;> linarith [h.ha i]
  · by_cases hS : S i
    · have hN : 0 < N i := (h.hx i).trans (h.host i hS)
      simp only [susceptibility, if_pos hS] at he
      simp only [rate, prevalence, if_pos hS]
      field_simp [hN.ne', (sub_pos.mpr (h.host i hS)).ne']
      nlinarith
    · simp [rate, prevalence, hS]

theorem equilibrium_products (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x) :
    reproduction S a c N * (∏ i, (1-prevalence S N x i)) = 1 ∧
    (∏ i, c i * susceptibility S N x (finRotate n i)) = ∏ i, a i := by
  have hpx : (∏ i, x i) ≠ 0 := (prod_pos (fun i _ => h.hx i)).ne'
  have he : ∀ i, c i * susceptibility S N x (finRotate n i) * x i =
      a (finRotate n i) * x (finRotate n i) := by
    intro i
    have hi := (equilibrium_local S a c N x h (finRotate n i)).2.2
    simpa only [input, prev, Equiv.symm_apply_apply, mul_assoc, mul_comm, mul_left_comm] using hi
  have hp := congrArg (fun f : Fin n → ℝ => ∏ i, f i) (funext he)
  simp only [prod_mul_distrib, Equiv.prod_comp] at hp
  have hprod := mul_right_cancel₀ hpx hp
  refine ⟨?_, by simpa only [prod_mul_distrib, Equiv.prod_comp] using hprod⟩
  have hs : ∀ i, (if S i then N i else 1)*(1-prevalence S N x i) =
      susceptibility S N x i := by
    intro i
    by_cases hS : S i
    · have hN : N i ≠ 0 := ne_of_gt ((h.hx i).trans (h.host i hS))
      simp [prevalence, susceptibility, hS]; field_simp
    · simp [prevalence, susceptibility, hS]
  have hp' : (∏ i, c i)*(∏ i, susceptibility S N x i) = ∏ i, a i := by
    simpa [prod_mul_distrib, Equiv.prod_comp] using hprod
  dsimp [reproduction]
  rw [prod_div_distrib, prod_mul_distrib]
  have ha : (∏ i, a i) ≠ 0 := (prod_pos (fun i _ => h.ha i)).ne'
  rw [div_mul_eq_mul_div, mul_assoc (∏ i, c i), ← prod_mul_distrib]
  simp_rw [hs]
  rw [hp', div_self ha]

theorem rotate_val (j : Fin n) : (finRotate n j).val = (j.val+1)%n := by
  cases n with
  | zero => exact j.elim0
  | succ m =>
    rw [coe_finRotate]
    split_ifs with h
    · simp [h]
    · rw [Nat.mod_eq_of_lt]
      exact Nat.succ_lt_succ (Fin.val_lt_last h)

theorem characteristic_raw (hn : 1 ≤ n) (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (z : ℂ) :
    ((jacobian S a c N x).map Complex.ofRealHom).charpoly.eval z =
      (∏ i, (z + (rate S a c x i : ℂ))) -
        ∏ i, (c i * susceptibility S N x (finRotate n i) : ℝ) := by
  rw [Matrix.eval_charpoly]
  have he : Matrix.scalar (Fin n) z - (jacobian S a c N x).map Complex.ofRealHom =
      Dobson.Full3.CyclicDet.diagSubCyclic n (fun i => z+(rate S a c x i : ℂ))
        (fun i => (c i * susceptibility S N x (finRotate n i) : ℝ)) := by
    ext i j
    have hr : i.val = (j.val+1)%n ↔ i = finRotate n j := by
      rw [← rotate_val, Fin.ext_iff]
    simp only [Matrix.sub_apply, Matrix.scalar, Matrix.diagonal_apply,
      Matrix.map_apply, jacobian_entry, map_sub, map_zero,
      Dobson.Full3.CyclicDet.diagSubCyclic, Matrix.of_apply, hr]
    by_cases hrj : i = finRotate n j
    · simp only [hrj, ite_true]
      rw [← hrj]
      by_cases hij : i = j <;> simp [hij, Matrix.diagonal_apply] <;> ring
    · simp only [hrj, ite_false, map_zero]
      by_cases hij : i = j <;> simp [hij, Matrix.diagonal_apply] <;> ring
  rw [he, Dobson.Full3.CyclicDet.det_diagSubCyclic n hn]
  push_cast
  rfl

theorem characteristic (hn : 1 ≤ n) (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x) (z : ℂ) :
    ((jacobian S a c N x).map Complex.ofRealHom).charpoly.eval z =
      (∏ i, (z + (rate S a c x i : ℂ))) - (∏ i, (a i : ℂ)) := by
  rw [characteristic_raw hn]
  congr 1
  exact_mod_cast (equilibrium_products S a c N x h).2

theorem reproduction_gt_one (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x)
    (hS : ∃ i, S i) : 1 < reproduction S a c N := by
  have hp : ∀ i, 0 < 1-prevalence S N x i ∧ 1-prevalence S N x i ≤ 1 := by
    intro i
    by_cases hi : S i
    · have hN : 0 < N i := (h.hx i).trans (h.host i hi)
      have h0 : 0 < x i/N i := div_pos (h.hx i) hN
      have h1 : x i/N i < 1 := (div_lt_one hN).mpr (h.host i hi)
      simp only [prevalence, if_pos hi]
      constructor <;> linarith
    · simp [prevalence, hi]
  have hprodpos : 0 < ∏ i, (1-prevalence S N x i) := prod_pos fun i _ => (hp i).1
  have hprodlt : (∏ i, (1-prevalence S N x i)) < 1 := by
    obtain ⟨j,hj⟩ := hS
    have ho : (∏ _i : Fin n, (1 : ℝ)) = 1 := prod_const_one
    apply lt_of_lt_of_eq ?_ ho
    apply prod_lt_prod₀ (fun i _ => (hp i).1) (fun i _ => (hp i).2)
    refine ⟨j, mem_univ _, ?_⟩
    have h0 := div_pos (h.hx j) ((h.hx j).trans (h.host j hj))
    simp only [prevalence, if_pos hj]
    linarith
  have he := (equilibrium_products S a c N x h).1
  nlinarith

/-- The root is an eigenvalue of the actual derivative, dominates its entire
complex spectrum, and is the unique root below every diagonal rate. -/
theorem return_rate (hn : 1 ≤ n) (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x)
    (hS : ∃ i, S i) :
    reproduction S a c N * (∏ i, (1-prevalence S N x i)) = 1 ∧
    (∏ i, a i/rate S a c x i) = 1/reproduction S a c N ∧
    ∃ γ : ℝ, 0 < γ ∧ (∀ i, γ < rate S a c x i) ∧
      Full2ReturnRoot.returnProduct (rate S a c x) γ = 1/reproduction S a c N ∧
      (∀ γ' : ℝ, 0 < γ' → (∀ i, γ' < rate S a c x i) →
        Full2ReturnRoot.returnProduct (rate S a c x) γ' = 1/reproduction S a c N → γ'=γ) ∧
      (-γ : ℂ) ∈ spectrum ℂ ((jacobian S a c N x).map Complex.ofRealHom) ∧
      (∀ z ∈ spectrum ℂ ((jacobian S a c N x).map Complex.ofRealHom),
        z.re ≤ -γ ∧ z.re < 0) ∧
      (∀ γ' : ℝ, 0 < γ' → γ' ≠ γ →
        Full2ReturnRoot.returnProduct (rate S a c x) γ' = 1/reproduction S a c N →
        γ < γ') := by
  letI : NeZero n := ⟨by omega⟩
  let d := rate S a c x
  let R := reproduction S a c N
  have hd : ∀ i, 0 < d i := fun i => (equilibrium_local S a c N x h i).1
  have hR : 1 < R := reproduction_gt_one S a c N x h hS
  have hbalance := (equilibrium_products S a c N x h).1
  have hp : (∏ i, a i/d i) = 1/R := by
    have he : ∀ i, a i/d i = 1-prevalence S N x i := by
      intro i
      dsimp [d]
      rw [(equilibrium_local S a c N x h i).2.1, div_div_eq_mul_div]
      exact mul_div_cancel_left₀ _ (h.ha i).ne'
    simp_rw [he]
    exact (eq_div_iff (show R ≠ 0 by linarith)).mpr (by nlinarith [hbalance])
  have hpa : (∏ i, a i) = (∏ i, d i)/R := by
    rw [prod_div_distrib] at hp
    have := (div_eq_iff (prod_pos (fun i _ => hd i)).ne').mp hp
    rw [this]; ring
  have root_iff (y : ℝ) :
      (-y : ℂ) ∈ spectrum ℂ ((jacobian S a c N x).map Complex.ofRealHom) ↔
        Full2ReturnRoot.returnProduct d y = 1/R := by
    rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
      characteristic hn S a c N x h, sub_eq_zero]
    have he : ∀ i, 1-y/d i = (d i-y)/d i := by
      intro i; field_simp [(hd i).ne']
    simp only [Full2ReturnRoot.returnProduct, he, prod_div_distrib,
      div_eq_iff (prod_pos (fun i _ => hd i)).ne']
    have hc : (∏ i, (- (y : ℂ) + (d i : ℂ))) = ((∏ i, (d i-y) : ℝ) : ℂ) := by
      push_cast
      apply prod_congr rfl
      intro i _
      ring
    change (∏ i, (-(y : ℂ)+(d i : ℂ))) = ∏ i, (a i : ℂ) ↔ _
    rw [hc, ← Complex.ofReal_prod, Complex.ofReal_inj, hpa]
    simp only [sub_eq_add_neg, add_comm, div_eq_mul_inv, one_mul, mul_comm]
  obtain ⟨γ,hγ,hunique⟩ := Full2ReturnRoot.unique_return_root d hd R hR
  have hspec : ∀ z ∈ spectrum ℂ ((jacobian S a c N x).map Complex.ofRealHom),
      z.re ≤ -γ ∧ z.re < 0 := by
    intro z hz
    rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot,
      characteristic hn S a c N x h, sub_eq_zero] at hz
    apply Full2ReturnRoot.spectral_bound d hd R γ hR hγ.1 hγ.2.1 hγ.2.2 z
    convert hz using 1
    exact_mod_cast hpa.symm
  refine ⟨hbalance, hp, γ, hγ.1, hγ.2.1, hγ.2.2,
    fun y hy hyd he => hunique y ⟨hy,hyd,he⟩, (root_iff γ).mpr hγ.2.2, hspec, ?_⟩
  intro y _ hy he
  have hb := (hspec (-y) ((root_iff y).mpr he)).1
  simp only [Complex.neg_re, Complex.ofReal_re] at hb
  exact lt_of_le_of_ne (by linarith) (Ne.symm hy)

/-- Identical S blocks cannot have a nonuniform positive equilibrium. The
proof uses the smallest and largest population, not a symmetry assumption. -/
theorem identical_equilibrium (hn : 1 ≤ n) (a c N : ℝ) (x : Fin n → ℝ)
    (h : PositiveEquilibrium (fun _ => True) (fun _ => a) (fun _ => c)
      (fun _ => N) x) :
    (∀ i, x i = N-a/c) ∧
    (∀ i, rate (fun _ => True) (fun _ => a) (fun _ => c) x i = c*N) ∧
    reproduction (n := n) (fun _ => True) (fun _ => a) (fun _ => c)
      (fun _ => N) = (c*N/a)^n := by
  letI : NeZero n := ⟨by omega⟩
  obtain ⟨lo,_,hlo⟩ := univ.exists_min_image x univ_nonempty
  obtain ⟨hi,_,hhi⟩ := univ.exists_max_image x univ_nonempty
  have hc : 0 < c := h.hc lo
  have he (i : Fin n) : c*x (prev n i)*(N-x i) = a*x i := by
    have := h.equilibrium i
    simpa [field, input] using (sub_eq_zero.mp this)
  have hl : c*(N-x lo) ≤ a := by
    apply (mul_le_mul_iff_right₀ (h.hx lo)).mp
    calc x lo*(c*(N-x lo)) = c*x lo*(N-x lo) := by ring
      _ ≤ c*x (prev n lo)*(N-x lo) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (hlo _ (mem_univ _)) hc.le)
            (sub_nonneg.mpr (h.host lo trivial).le)
      _ = x lo*a := by rw [he]; ring
  have hh : a ≤ c*(N-x hi) := by
    apply (mul_le_mul_iff_right₀ (h.hx hi)).mp
    calc x hi*a = c*x (prev n hi)*(N-x hi) := by rw [he]; ring
      _ ≤ c*x hi*(N-x hi) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left (hhi _ (mem_univ _)) hc.le)
            (sub_nonneg.mpr (h.host hi trivial).le)
      _ = x hi*(c*(N-x hi)) := by ring
  have hx (i : Fin n) : x i = N-a/c := by
    have hil := hlo i (mem_univ i)
    have hih := hhi i (mem_univ i)
    have heq : c*(N-x i) = a := by nlinarith
    apply (eq_sub_iff_add_eq).mpr
    apply (mul_left_cancel₀ hc.ne')
    field_simp
    nlinarith
  refine ⟨hx, ?_, ?_⟩
  · intro i
    simp [rate, input, hx, hc.ne', mul_sub, mul_div_cancel₀]
  · simp [reproduction, div_pow]

noncomputable def lifetimeReturn (R T x : ℝ) : ℝ := x*(Real.exp (Real.log R/x)-1)/T
noncomputable def filterReturn (R T x : ℝ) : ℝ := x*(1-Real.exp (-Real.log R/x))/T

theorem duration_comparisons (R T : ℝ) (hR : 1 < R) (hT : 0 < T) :
    StrictAntiOn (lifetimeReturn R T) (Ioi 0) ∧
    StrictMonoOn (filterReturn R T) (Ioi 0) ∧
    Tendsto (lifetimeReturn R T) atTop (nhds (Real.log R/T)) ∧
    Tendsto (filterReturn R T) atTop (nhds (Real.log R/T)) := by
  have hL : 0 < Real.log R := Real.log_pos hR
  refine ⟨?_, ?_, (Full2ReturnRoot.duration_limits (Real.log R) T hL hT).1,
    (Full2ReturnRoot.duration_limits (Real.log R) T hL hT).2⟩
  · intro x hx y hy hxy
    apply (div_lt_div_iff_of_pos_right hT).mpr
    have hLx : 0 < Real.log R/x := div_pos hL hx
    have hLy : 0 < Real.log R/y := div_pos hL hy
    have hlt : Real.log R/y < Real.log R/x := div_lt_div_of_pos_left hL hx hxy
    have hh := strictConvexOn_exp.secant_strict_mono (a := 0)
      (x := Real.log R/y) (y := Real.log R/x)
      (mem_univ _) (mem_univ _) (mem_univ _) hLy.ne' hLx.ne' hlt
    simp only [Real.exp_zero, sub_zero, div_div_eq_mul_div] at hh
    have hh' : y*(Real.exp (Real.log R/y)-1)/Real.log R <
        x*(Real.exp (Real.log R/x)-1)/Real.log R := by
      convert hh using 1 <;> ring
    exact (div_lt_div_iff_of_pos_right hL).mp hh' 
  · intro x hx y hy hxy
    exact (div_lt_div_iff_of_pos_right hT).mpr
      (Full2ReturnRoot.fixed_filter_strictMono _ hL hx hy hxy)

/-- The rare-population derivative is evaluated at zero in the same nonlinear
model. Its dominant eigenvalue is `c*N-a`. -/
theorem identical_invasion (hn : 1 ≤ n) (a c N : ℝ) (hcN : 0 < c*N) :
    ((c*N-a : ℝ) : ℂ) ∈ spectrum ℂ
      ((jacobian (fun _ : Fin n => True) (fun _ => a) (fun _ => c)
        (fun _ => N) 0).map Complex.ofRealHom) ∧
    ∀ z ∈ spectrum ℂ
      ((jacobian (fun _ : Fin n => True) (fun _ => a) (fun _ => c)
        (fun _ => N) 0).map Complex.ofRealHom), z.re ≤ c*N-a := by
  letI : NeZero n := ⟨by omega⟩
  have hc (z : ℂ) :
      z ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
        (fun _ => c) (fun _ => N) 0).map Complex.ofRealHom) ↔
        (z+(a : ℂ))^n = ((c*N : ℝ) : ℂ)^n := by
    rw [Matrix.mem_spectrum_iff_isRoot_charpoly, Polynomial.IsRoot, characteristic_raw hn]
    simp [rate, input, susceptibility, sub_eq_zero, Complex.ofReal_pow]
  refine ⟨(hc _).mpr (by push_cast; ring), ?_⟩
  intro z hz
  by_contra hnle
  have hzgt : c*N < z.re+a := by linarith [lt_of_not_ge hnle]
  have hnrm : c*N < ‖z+(a : ℂ)‖ := by
    have hb := Complex.re_le_norm (z+(a : ℂ))
    simp only [Complex.add_re, Complex.ofReal_re] at hb
    linarith
  have hp : (c*N)^n < ‖z+(a : ℂ)‖^n :=
    pow_lt_pow_left₀ hnrm hcN.le (by omega)
  have he := congrArg norm ((hc z).mp hz)
  simp only [norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hcN] at he
  exact (ne_of_gt hp) he

/-- Complete identical-S specialization, including the actual invasion
identification and both fixed-total-duration formulas. -/
theorem identical_return (hn : 1 ≤ n) (a c N : ℝ) (x : Fin n → ℝ)
    (h : PositiveEquilibrium (fun _ => True) (fun _ => a) (fun _ => c)
      (fun _ => N) x) :
    let R := reproduction (n := n) (fun _ => True) (fun _ => a) (fun _ => c) (fun _ => N)
    let γ := c*N-a
    1 < R ∧ 0 < γ ∧
    (∀ i, x i = N-a/c) ∧
    (∀ i, rate (fun _ => True) (fun _ => a) (fun _ => c) x i = c*N) ∧
    R = (c*N/a)^n ∧
    γ = a*(R^(1/(n:ℝ))-1) ∧
    γ = lifetimeReturn R ((n:ℝ)/a) n ∧
    γ = filterReturn R ((n:ℝ)/(c*N)) n ∧
    (-γ : ℂ) ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) x).map Complex.ofRealHom) ∧
    (∀ z ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) x).map Complex.ofRealHom), z.re ≤ -γ ∧ z.re < 0) ∧
    (γ : ℂ) ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) 0).map Complex.ofRealHom) ∧
    (∀ z ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) 0).map Complex.ofRealHom), z.re ≤ γ) := by
  letI : NeZero n := ⟨by omega⟩
  intro R γ
  have hn0 : (0:ℝ) < n := by exact_mod_cast (lt_of_lt_of_le Nat.zero_lt_one hn)
  have ha : 0 < a := h.ha 0
  have hc : 0 < c := h.hc 0
  have hN : 0 < N := (h.hx 0).trans (h.host 0 trivial)
  have hcN : 0 < c*N := mul_pos hc hN
  obtain ⟨hx,hd,hRpow⟩ := identical_equilibrium hn a c N x h
  have hγ : 0 < γ := by
    have := h.hx 0
    rw [hx] at this
    dsimp [γ]
    have hh := (div_lt_iff₀ hc).mp (show a/c < N by linarith)
    nlinarith
  have hR : 1 < R := reproduction_gt_one _ _ _ _ _ h ⟨0,trivial⟩
  have hexp : Real.exp (Real.log R/(n:ℝ)) = c*N/a := by
    have hl : Real.log R = (n:ℝ)*Real.log (c*N/a) := by
      change Real.log (reproduction _ _ _ _) = _
      rw [hRpow, Real.log_pow]
    rw [hl, mul_div_cancel_left₀ _ hn0.ne', Real.exp_log (div_pos hcN ha)]
  have hrpow : R^(1/(n:ℝ)) = c*N/a := by
    rw [Real.rpow_def_of_pos (by linarith : 0 < R)]
    convert hexp using 1
    congr 1
    ring
  have hneg : Real.exp (-Real.log R/(n:ℝ)) = a/(c*N) := by
    rw [neg_div, Real.exp_neg, hexp, inv_div]
  obtain ⟨_,hstat,γ₀,h0,hd0,hroot,hunique,hspec,hbound,hfaster⟩ :=
    return_rate hn (fun _ => True) (fun _ => a) (fun _ => c) (fun _ => N) x h ⟨0,trivial⟩
  have hgroot : Full2ReturnRoot.returnProduct
      (rate (fun _ => True) (fun _ => a) (fun _ => c) x) γ = 1/R := by
    convert hstat using 1
    unfold Full2ReturnRoot.returnProduct
    apply prod_congr rfl
    intro i _
    rw [hd]
    dsimp [γ]
    field_simp
    <;> ring
  have heq : γ = γ₀ := hunique γ hγ (by intro i; rw [hd]; dsimp [γ]; linarith) hgroot
  rw [← heq] at hspec hbound
  obtain ⟨hinv,hdom⟩ := identical_invasion hn a c N hcN
  refine ⟨hR,hγ,hx,hd,hRpow,?_,?_,?_,hspec,hbound,hinv,hdom⟩
  · rw [hrpow]
    dsimp [γ]
    field_simp
    <;> ring
  · dsimp [lifetimeReturn]
    rw [hexp]
    dsimp [γ]
    field_simp
    <;> ring
  · dsimp [filterReturn]
    rw [hneg]
    dsimp [γ]
    field_simp
    <;> ring

/-- Relative population perturbations obey the manuscript's local scalar
response equation. Input pressure has the same relative perturbation as the
previous stage because its output coefficient is constant. -/
theorem normalized_derivative (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x)
    (y : Fin n → ℝ) (i : Fin n) :
    differential S a c N x (fun j => x j*y j) i / x i =
      a i*y (prev n i) - rate S a c x i*y i := by
  have he := (equilibrium_local S a c N x h i).2.2
  dsimp [differential]
  simp only [ContinuousLinearMap.pi_apply, sub_apply, smul_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul]
  apply (div_eq_iff (h.hx i).ne').mpr
  dsimp [input] at he
  linear_combination y (prev n i) * he

/-- Solving that normalized local equation identifies the physical transfer
function, with the unreduced denominator retained. -/
theorem local_response (a d : ℝ) (z u v : ℂ) (hz : z+(d:ℂ) ≠ 0) :
    (z*v = (a:ℂ)*u-(d:ℂ)*v) ↔ v = ((a:ℂ)/(z+(d:ℂ)))*u := by
  field_simp
  constructor <;> intro h <;> linear_combination h


/-- Complete manuscript Proposition dyn-k8. The general ring, identical-host
specialization, and both duration comparisons are quantified independently,
exactly as the manuscript's three clauses require. -/
structure ManuscriptK8 : Prop where
  characteristic {n : ℕ} (hn : 1 ≤ n) (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x) (z : ℂ) :
    ((jacobian S a c N x).map Complex.ofRealHom).charpoly.eval z =
      (∏ i, (z + (rate S a c x i : ℂ))) - (∏ i, (a i : ℂ))
  reproduction_gt_one {n : ℕ} (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x)
    (hS : ∃ i, S i) : 1 < reproduction S a c N
  actual_derivative {n : ℕ} (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) :
    HasFDerivAt (field S a c N) (differential S a c N x) x
  equilibrium_local {n : ℕ} (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x) (i : Fin n) :
    0 < rate S a c x i ∧
    rate S a c x i = a i / (1-prevalence S N x i) ∧
    input c x i * susceptibility S N x i = a i*x i
  normalized_derivative {n : ℕ} (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x)
    (y : Fin n → ℝ) (i : Fin n) :
    differential S a c N x (fun j => x j*y j) i / x i =
      a i*y (prev n i) - rate S a c x i*y i
  local_response (a d : ℝ) (z u v : ℂ) (hz : z+(d:ℂ) ≠ 0) :
    (z*v = (a:ℂ)*u-(d:ℂ)*v) ↔ v = ((a:ℂ)/(z+(d:ℂ)))*u
  return_rate {n : ℕ} (hn : 1 ≤ n) (S : Fin n → Prop) [DecidablePred S]
    (a c N x : Fin n → ℝ) (h : PositiveEquilibrium S a c N x)
    (hS : ∃ i, S i) :
    reproduction S a c N * (∏ i, (1-prevalence S N x i)) = 1 ∧
    (∏ i, a i/rate S a c x i) = 1/reproduction S a c N ∧
    ∃ γ : ℝ, 0 < γ ∧ (∀ i, γ < rate S a c x i) ∧
      Full2ReturnRoot.returnProduct (rate S a c x) γ = 1/reproduction S a c N ∧
      (∀ γ' : ℝ, 0 < γ' → (∀ i, γ' < rate S a c x i) →
        Full2ReturnRoot.returnProduct (rate S a c x) γ' = 1/reproduction S a c N → γ'=γ) ∧
      (-γ : ℂ) ∈ spectrum ℂ ((jacobian S a c N x).map Complex.ofRealHom) ∧
      (∀ z ∈ spectrum ℂ ((jacobian S a c N x).map Complex.ofRealHom),
        z.re ≤ -γ ∧ z.re < 0) ∧
      (∀ γ' : ℝ, 0 < γ' → γ' ≠ γ →
        Full2ReturnRoot.returnProduct (rate S a c x) γ' = 1/reproduction S a c N →
        γ < γ')
  identical_return {n : ℕ} (hn : 1 ≤ n) (a c N : ℝ) (x : Fin n → ℝ)
    (h : PositiveEquilibrium (fun _ => True) (fun _ => a) (fun _ => c)
      (fun _ => N) x) :
    let R := reproduction (n := n) (fun _ => True) (fun _ => a) (fun _ => c) (fun _ => N)
    let γ := c*N-a
    1 < R ∧ 0 < γ ∧
    (∀ i, x i = N-a/c) ∧
    (∀ i, rate (fun _ => True) (fun _ => a) (fun _ => c) x i = c*N) ∧
    R = (c*N/a)^n ∧
    γ = a*(R^(1/(n:ℝ))-1) ∧
    γ = lifetimeReturn R ((n:ℝ)/a) n ∧
    γ = filterReturn R ((n:ℝ)/(c*N)) n ∧
    (-γ : ℂ) ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) x).map Complex.ofRealHom) ∧
    (∀ z ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) x).map Complex.ofRealHom), z.re ≤ -γ ∧ z.re < 0) ∧
    (γ : ℂ) ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) 0).map Complex.ofRealHom) ∧
    (∀ z ∈ spectrum ℂ ((jacobian (fun _ : Fin n => True) (fun _ => a)
      (fun _ => c) (fun _ => N) 0).map Complex.ofRealHom), z.re ≤ γ)
  duration_comparisons (R T : ℝ) (hR : 1 < R) (hT : 0 < T) :
    StrictAntiOn (lifetimeReturn R T) (Ioi 0) ∧
    StrictMonoOn (filterReturn R T) (Ioi 0) ∧
    Tendsto (lifetimeReturn R T) atTop (nhds (Real.log R/T)) ∧
    Tendsto (filterReturn R T) atTop (nhds (Real.log R/T))

theorem prop_dyn_k8 : ManuscriptK8 where
  characteristic := @characteristic
  reproduction_gt_one := @reproduction_gt_one
  actual_derivative := @actual_derivative
  equilibrium_local := @equilibrium_local
  normalized_derivative := @normalized_derivative
  local_response := @local_response
  return_rate := @return_rate
  identical_return := @identical_return
  duration_comparisons := @duration_comparisons

end Dobson.Full3.ConstantHosts

#print axioms Dobson.Full3.ConstantHosts.prop_dyn_k8
