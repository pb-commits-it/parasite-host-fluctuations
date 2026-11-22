import Mathlib

/-!
# The algebra behind a lower bound on growth (items J1 to J3)

All variables are real. `Real.log` is the natural logarithm, `Real.exp` the exponential.

Encoding of a finite family: an arbitrary index type `ι`, a finite set of indices
(`s` in J1, `t` in J2 and J3), and functions `ι → ℝ`; hypotheses are assumed for indices in
the finite set, and sums and products run over it.

Names. In J1 the statement's `λ` is `lam` and its `s_k` is `σ k` (`s` is the index set); the
derived quantities `q_k`, `τ_k`, `𝒯` are the definitions `q`, `tau`, `total` below. In J3 the
derived quantities `ℓ` and `φ(D)` are the definitions `ell` and `phi` below.
-/

namespace Dobson.RefChain

/-- `q_k = s_k / λ`. -/
noncomputable def q {ι : Type*} (lam : ℝ) (σ : ι → ℝ) (k : ι) : ℝ :=
  σ k / lam

/-- `τ_k = 1 / (1 − q_k)`. -/
noncomputable def tau {ι : Type*} (lam : ℝ) (σ : ι → ℝ) (k : ι) : ℝ :=
  1 / (1 - q lam σ k)

/-- `𝒯 = ∑_{k ∈ s} τ_k`. -/
noncomputable def total {ι : Type*} (s : Finset ι) (lam : ℝ) (σ : ι → ℝ) : ℝ :=
  ∑ k ∈ s, tau lam σ k

/-- **J1.** Let `s` be a finite nonempty index set, `λ > 0`, and for `k` in `s` let
`0 ≤ s_k < λ` and `g_k > 0`, with `∏ (λ − s_k) = ∏ g_k`. Put `q_k = s_k/λ`,
`τ_k = 1/(1 − q_k)` and `𝒯 = ∑ τ_k`. Then
`∑_k (τ_k/𝒯)·( q_k·log λ + (1 − q_k)·log(g_k/(1 − q_k)) ) = log λ`. -/
theorem J1 {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (lam : ℝ) (hlam : 0 < lam)
    (σ g : ι → ℝ) (_hσ0 : ∀ k ∈ s, 0 ≤ σ k) (hσ : ∀ k ∈ s, σ k < lam)
    (hg : ∀ k ∈ s, 0 < g k)
    (hprod : ∏ k ∈ s, (lam - σ k) = ∏ k ∈ s, g k) :
    ∑ k ∈ s, (tau lam σ k / total s lam σ)
        * (q lam σ k * Real.log lam
            + (1 - q lam σ k) * Real.log (g k / (1 - q lam σ k)))
      = Real.log lam := by
  have hd : ∀ k ∈ s, 0 < 1 - q lam σ k := by
    intro k hk
    unfold q
    rw [sub_pos, div_lt_one hlam]
    exact hσ k hk
  have hτpos : ∀ k ∈ s, 0 < tau lam σ k := by
    intro k hk
    unfold tau
    exact one_div_pos.mpr (hd k hk)
  have hT : 0 < total s lam σ := Finset.sum_pos hτpos hs
  have hlog : ∑ k ∈ s, Real.log (lam - σ k) = ∑ k ∈ s, Real.log (g k) := by
    rw [← Real.log_prod (fun k hk => (sub_pos.mpr (hσ k hk)).ne'),
      ← Real.log_prod (fun k hk => (hg k hk).ne'), hprod]
  have hterm : ∀ k ∈ s, (tau lam σ k / total s lam σ)
        * (q lam σ k * Real.log lam
            + (1 - q lam σ k) * Real.log (g k / (1 - q lam σ k)))
      = (tau lam σ k * Real.log lam + Real.log (g k) - Real.log (lam - σ k))
          / total s lam σ := by
    intro k hk
    have hdk := hd k hk
    have h1 : 1 - q lam σ k = (lam - σ k) / lam := by
      unfold q
      field_simp
    have h2 : Real.log (g k / (1 - q lam σ k))
        = Real.log (g k) - (Real.log (lam - σ k) - Real.log lam) := by
      rw [Real.log_div (hg k hk).ne' hdk.ne', h1,
        Real.log_div (sub_pos.mpr (hσ k hk)).ne' hlam.ne']
    have h3 : tau lam σ k * (1 - q lam σ k) = 1 := by
      unfold tau
      exact one_div_mul_cancel hdk.ne'
    have h4 : tau lam σ k * q lam σ k = tau lam σ k - 1 := by
      linear_combination -h3
    rw [h2, div_mul_eq_mul_div]
    congr 1
    linear_combination (Real.log lam) * h4
      + (Real.log (g k) - (Real.log (lam - σ k) - Real.log lam)) * h3
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_div, Finset.sum_sub_distrib,
    Finset.sum_add_distrib, ← Finset.sum_mul, hlog]
  have hT' : total s lam σ ≠ 0 := hT.ne'
  unfold total at hT' ⊢
  field_simp
  ring

/-- Jensen's inequality for the logarithm and a finite sum, without assuming the index set
nonempty (that follows from `∑ Q i = 1`). Used by J2 and J3. -/
theorem jensen_log {ι : Type*} (t : Finset ι) (Q a : ι → ℝ)
    (hQ : ∀ i ∈ t, 0 < Q i) (hsum : ∑ i ∈ t, Q i = 1) (ha : ∀ i ∈ t, 0 < a i) :
    ∑ i ∈ t, Q i * Real.log (a i) ≤ Real.log (∑ i ∈ t, Q i * a i) := by
  have h := strictConcaveOn_log_Ioi.concaveOn.le_map_sum (t := t) (w := Q) (p := a)
    (fun i hi => (hQ i hi).le) hsum (fun i hi => Set.mem_Ioi.mpr (ha i hi))
  simpa only [smul_eq_mul] using h

/-- **J2.** (Jensen for a finite sum.) For a finite nonempty family of weights `Q i > 0` with
`∑ Q i = 1` and reals `a i > 0`: `∑ Q i · log (a i) ≤ log (∑ Q i · a i)`. -/
theorem J2 {ι : Type*} (t : Finset ι) (_ht : t.Nonempty) (Q a : ι → ℝ)
    (hQ : ∀ i ∈ t, 0 < Q i) (hsum : ∑ i ∈ t, Q i = 1) (ha : ∀ i ∈ t, 0 < a i) :
    ∑ i ∈ t, Q i * Real.log (a i) ≤ Real.log (∑ i ∈ t, Q i * a i) :=
  jensen_log t Q a hQ hsum ha

/-- `ℓ = −∑_{i ∈ t} w_i · log h_i`. -/
noncomputable def ell {ι : Type*} (t : Finset ι) (w h : ι → ℝ) : ℝ :=
  -∑ i ∈ t, w i * Real.log (h i)

/-- `φ(D) = ∑_{i ∈ t} w_i · log(s + (1 − s)·exp(D)·h_i)`. -/
noncomputable def phi {ι : Type*} (t : Finset ι) (w h : ι → ℝ) (s D : ℝ) : ℝ :=
  ∑ i ∈ t, w i * Real.log (s + (1 - s) * Real.exp D * h i)

/-- **J3 (a).** For finitely many weights `w i > 0` with `∑ w i = 1` and `h i > 0` with
`∑ w i · h i = 1`: `ℓ = −∑ w i · log (h i) ≥ 0`. -/
theorem J3a {ι : Type*} (t : Finset ι) (w h : ι → ℝ)
    (hw : ∀ i ∈ t, 0 < w i) (hw1 : ∑ i ∈ t, w i = 1) (hh : ∀ i ∈ t, 0 < h i)
    (hwh : ∑ i ∈ t, w i * h i = 1) :
    0 ≤ ell t w h := by
  have h1 := jensen_log t w h hw hw1 hh
  rw [hwh, Real.log_one] at h1
  unfold ell
  linarith

/-- **J3 (b).** For finitely many weights `w i > 0` with `∑ w i = 1`, `h i > 0` with
`∑ w i · h i = 1`, and `0 ≤ s < 1`: `φ(0) ≤ 0`, where
`φ(D) = ∑ w i · log(s + (1 − s)·exp(D)·h i)`. -/
theorem J3b {ι : Type*} (t : Finset ι) (w h : ι → ℝ) (s : ℝ)
    (hw : ∀ i ∈ t, 0 < w i) (hw1 : ∑ i ∈ t, w i = 1) (hh : ∀ i ∈ t, 0 < h i)
    (hwh : ∑ i ∈ t, w i * h i = 1) (hs0 : 0 ≤ s) (hs1 : s < 1) :
    phi t w h s 0 ≤ 0 := by
  have hpos : ∀ i ∈ t, 0 < s + (1 - s) * Real.exp 0 * h i := by
    intro i hi
    have : 0 < (1 - s) * Real.exp 0 * h i :=
      mul_pos (mul_pos (sub_pos.mpr hs1) (Real.exp_pos 0)) (hh i hi)
    linarith
  have h1 := jensen_log t w (fun i => s + (1 - s) * Real.exp 0 * h i) hw hw1 hpos
  have e : ∑ i ∈ t, w i * (s + (1 - s) * Real.exp 0 * h i) = 1 := by
    have hrew : ∀ i ∈ t, w i * (s + (1 - s) * Real.exp 0 * h i)
        = s * w i + (1 - s) * (w i * h i) := by
      intro i _
      rw [Real.exp_zero]
      ring
    rw [Finset.sum_congr rfl hrew, Finset.sum_add_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum, hw1, hwh]
    ring
  rw [e, Real.log_one] at h1
  unfold phi
  exact h1

/-- **J3 (c).** For finitely many weights `w i > 0` with `∑ w i = 1`, `h i > 0` with
`∑ w i · h i = 1`, and `0 ≤ s < 1`: `φ(ℓ) ≥ 0`, where `ℓ = −∑ w i · log (h i)` and
`φ(D) = ∑ w i · log(s + (1 − s)·exp(D)·h i)`.

The hypothesis `∑ w i · h i = 1` is not needed for this part. -/
theorem J3c {ι : Type*} (t : Finset ι) (w h : ι → ℝ) (s : ℝ)
    (hw : ∀ i ∈ t, 0 < w i) (hw1 : ∑ i ∈ t, w i = 1) (hh : ∀ i ∈ t, 0 < h i)
    (_hwh : ∑ i ∈ t, w i * h i = 1) (hs0 : 0 ≤ s) (hs1 : s < 1) :
    0 ≤ phi t w h s (ell t w h) := by
  -- concavity of `log`: `log (s·1 + (1 − s)·y) ≥ (1 − s)·log y` for `y > 0`
  have hlow : ∀ i ∈ t, w i * ((1 - s) * (ell t w h + Real.log (h i)))
      ≤ w i * Real.log (s + (1 - s) * Real.exp (ell t w h) * h i) := by
    intro i hi
    have hy : 0 < Real.exp (ell t w h) * h i := mul_pos (Real.exp_pos _) (hh i hi)
    have hc := strictConcaveOn_log_Ioi.concaveOn.2 (Set.mem_Ioi.mpr one_pos)
      (Set.mem_Ioi.mpr hy) hs0 (sub_nonneg.mpr hs1.le) (by ring : s + (1 - s) = 1)
    simp only [smul_eq_mul, Real.log_one, mul_zero, zero_add, mul_one] at hc
    rw [Real.log_mul (Real.exp_pos _).ne' (hh i hi).ne', Real.log_exp, ← mul_assoc] at hc
    exact mul_le_mul_of_nonneg_left hc (hw i hi).le
  have hsum : ∑ i ∈ t, w i * ((1 - s) * (ell t w h + Real.log (h i))) = 0 := by
    have hrew : ∀ i ∈ t, w i * ((1 - s) * (ell t w h + Real.log (h i)))
        = (1 - s) * ell t w h * w i + (1 - s) * (w i * Real.log (h i)) := by
      intro i _
      ring
    rw [Finset.sum_congr rfl hrew, Finset.sum_add_distrib, ← Finset.mul_sum,
      ← Finset.mul_sum, hw1]
    unfold ell
    ring
  calc (0 : ℝ) = ∑ i ∈ t, w i * ((1 - s) * (ell t w h + Real.log (h i))) := hsum.symm
    _ ≤ phi t w h s (ell t w h) := by
      unfold phi
      exact Finset.sum_le_sum hlow

/-- **J3.** (One stage, a finite environment: the loss lies between zero and the bound.)
Let `w i > 0` be finitely many weights with `∑ w i = 1`, `h i > 0` with `∑ w i · h i = 1`,
and `0 ≤ s < 1`. Put `ℓ = −∑ w i · log (h i)` and
`φ(D) = ∑ w i · log(s + (1 − s)·exp(D)·h i)`. Then (a) `ℓ ≥ 0`; (b) `φ(0) ≤ 0`;
(c) `φ(ℓ) ≥ 0`. -/
theorem J3 {ι : Type*} (t : Finset ι) (w h : ι → ℝ) (s : ℝ)
    (hw : ∀ i ∈ t, 0 < w i) (hw1 : ∑ i ∈ t, w i = 1) (hh : ∀ i ∈ t, 0 < h i)
    (hwh : ∑ i ∈ t, w i * h i = 1) (hs0 : 0 ≤ s) (hs1 : s < 1) :
    0 ≤ ell t w h ∧ phi t w h s 0 ≤ 0 ∧ 0 ≤ phi t w h s (ell t w h) :=
  ⟨J3a t w h hw hw1 hh hwh, J3b t w h s hw hw1 hh hwh hs0 hs1,
    J3c t w h s hw hw1 hh hwh hs0 hs1⟩

end Dobson.RefChain

#print axioms Dobson.RefChain.J1
#print axioms Dobson.RefChain.J2
#print axioms Dobson.RefChain.J3a
#print axioms Dobson.RefChain.J3b
#print axioms Dobson.RefChain.J3c
#print axioms Dobson.RefChain.J3
