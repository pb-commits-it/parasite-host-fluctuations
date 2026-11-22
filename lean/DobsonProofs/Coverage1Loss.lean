import DobsonProofs.GrowthShape
import DobsonProofs.RingGrowth
import Mathlib

/-!
# Coverage batch 1 — Item 6: Existence and invariance of the ring threshold loss (`prop:model-loss`)

Contains the deterministic similarity and conditional threshold-root helpers for
`prop:model-loss` (`sections/02-model.tex`, lines 69–76, and `appendices/B-bounds.tex`,
lines 52–54):

1. **Deterministic positive diagonal conjugacy:** For any `n ≥ 1`, staying diagonal `s : Fin n → ℝ`,
   positive baseline transfers `g, g' : Fin n → ℝ` with equal circuit product
   `∏_k g_k = ∏_k g'_k`, and arbitrary step multipliers `h : Fin n → ℝ`, there exist strictly
   positive stage scales `d : Fin n → ℝ` (`d_k > 0`) constructed solely from `g` and `g'` such that
   `diag(d) * ringStepMat(s, g, h) = ringStepMat(s, g', h) * diag(d)`.
   Consequently, if any positive-transfer functional `r : (Fin n → ℝ) → ℝ` is invariant under
   constant positive diagonal conjugacy of `ringStepMat`, then `r g` depends on `g` solely through
   `y = ∑_k log g_k` (`r g = r (uniformTransfer n y)`).
2. **Birth-divisor log-product shift:** Designating `B ⊆ Fin n` of size `m = B.card ≥ 1` as birth
   transfers and dividing them by `R > 0` preserves positivity of transfers and shifts
   `∑_k log g_k` to `∑_k log g_k − m log R`.
3. **Unique threshold root and `(g, m)`-invariance of `D`:** Under explicit probability hypotheses
   on the stochastic growth exponent `r`:
   - `(H1_conj)` positive diagonal conjugacy invariance (`r g = r g'` whenever positive `g, g'`
     have conjugate step matrices),
   - `(H2_conv)` convexity and continuity of `y ↦ r (uniformTransfer n y)`,
   - `(H3_neg)` eventual negativity as `y → −∞` (`∀ᶠ y in atBot, r (uniformTransfer n y) < 0`),
   - `(H4_pos)` divergence above zero as `y → +∞`
     (`Tendsto (fun y => r (uniformTransfer n y)) atTop atTop`),
   there exists a **unique** real root `y₀` of `r (uniformTransfer n y₀) = 0`, for every positive
   `g` and nonempty birth set `B` there exists a **unique** divisor `R₀* > 0` with
   `r (tiltBirths B R₀* g) = 0`, and the threshold loss `D = log R₀ − B.card · log R₀*` equals
   `y₀ − ∑_k log (1 − s_k)`, invariant in both `(g_1, ..., g_n)` and `(B, m)`.
-/

/-! ## Coverage correction — 2026-11-22

`prop:model-loss` has full coverage in the version-2 inventory. The later
ring-product results prove threshold existence, uniqueness and invariance
under transfer and birth choices for the actual ring products. The named
Furstenberg–Kesten expected-growth limit is a textbook input.
The earlier `h_neg` and `h_pos` audit entries describe the abstract helper,
not outstanding premises of the credited physical-ring theorem. -/

namespace Dobson.Coverage1Loss

open scoped BigOperators

/-- One-step `n × n` ring projection matrix `A(t)` (`eq:model-ring`):
diagonal `s_i` plus cyclic transfer `(g_j * h_j)` in entry `((j + 1) mod n, j)` (added in the
single entry when `n = 1`). -/
def ringStepMat (n : ℕ) (s g h : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of fun i j =>
    (if i = j then s i else 0) +
      (if (i : ℕ) = ((j : ℕ) + 1) % n then g j * h j else 0)

/-- Canonical positive transfer vector `g_k = exp(y / n)` with log-product `y` (`n ≥ 1`). -/
noncomputable def uniformTransfer (n : ℕ) (y : ℝ) : Fin n → ℝ :=
  fun _ => Real.exp (y / (n : ℝ))

/-- Dividing designated birth transfers `j ∈ B` by `R`. -/
noncomputable def tiltBirths {n : ℕ} (B : Finset (Fin n)) (R : ℝ) (g : Fin n → ℝ) : Fin n → ℝ :=
  fun j => if j ∈ B then g j / R else g j

/-- Deterministic circuit reproduction number `R₀(s, g) = ∏_k g_k / (1 − s_k)`. -/
noncomputable def circuitR0 (n : ℕ) (s g : Fin n → ℝ) : ℝ :=
  ∏ k : Fin n, (g k / (1 - s k))

/-- **Part 1a (exact positive diagonal conjugacy of ring step matrices with equal transfer product).**
If `n ≥ 1`, `g_k > 0`, `g'_k > 0`, and `∏_k g_k = ∏_k g'_k`, there exists `d : Fin n → ℝ` with
`d_k > 0` for all `k` (constructed solely from `g` and `g'`) such that for every `s, h : Fin n → ℝ`,
`diag(d) * ringStepMat n s g h = ringStepMat n s g' h * diag(d)`. -/
theorem exists_pos_diag_conj_of_prod_eq (n : ℕ) (_hn : 1 ≤ n)
    (g g' : Fin n → ℝ) (hg : ∀ k, 0 < g k) (hg' : ∀ k, 0 < g' k)
    (hprod : ∏ k : Fin n, g k = ∏ k : Fin n, g' k) :
    ∃ d : Fin n → ℝ, (∀ k, 0 < d k) ∧
      ∀ s h : Fin n → ℝ,
        Matrix.diagonal d * ringStepMat n s g h =
          ringStepMat n s g' h * Matrix.diagonal d := by
  let ratio : ℕ → ℝ := fun r =>
    if hr : r < n then g' ⟨r, hr⟩ / g ⟨r, hr⟩ else 1
  let d : Fin n → ℝ := fun i => ∏ r ∈ Finset.range (i : ℕ), ratio r
  have hratio_pos : ∀ r, 0 < ratio r := by
    intro r
    dsimp [ratio]
    split_ifs with hr
    · exact div_pos (hg' ⟨r, hr⟩) (hg ⟨r, hr⟩)
    · exact one_pos
  have hd_pos : ∀ k : Fin n, 0 < d k := fun k =>
    Finset.prod_pos fun r _ => hratio_pos r
  have hfull_prod : ∏ r ∈ Finset.range n, ratio r = 1 := by
    have hbij : ∏ r ∈ Finset.range n, ratio r = ∏ k : Fin n, (g' k / g k) := by
      rw [← Fin.prod_univ_eq_prod_range (fun r => ratio r) n]
      refine Finset.prod_congr rfl fun k _ => ?_
      simp [ratio, k.isLt]
    rw [hbij, Finset.prod_div_distrib, ← hprod, div_self]
    exact Finset.prod_ne_zero_iff.mpr fun k _ => (hg k).ne'
  have hstep : ∀ j : Fin n, ∀ i : Fin n, (i : ℕ) = ((j : ℕ) + 1) % n →
      d i * g j = d j * g' j := by
    intro j i hij
    have hjlt : (j : ℕ) < n := j.isLt
    by_cases hlast : (j : ℕ) + 1 = n
    · have hi0 : (i : ℕ) = 0 := by rw [hij, hlast, Nat.mod_self]
      have hdi : d i = 1 := by simp [d, hi0]
      have hprod_split : (∏ r ∈ Finset.range ((j : ℕ) + 1), ratio r) =
          (∏ r ∈ Finset.range (j : ℕ), ratio r) * ratio (j : ℕ) :=
        Finset.prod_range_succ ratio (j : ℕ)
      have h1 : (∏ r ∈ Finset.range (j : ℕ), ratio r) * (g' j / g j) = 1 := by
        rw [← hlast] at hfull_prod
        rw [hprod_split] at hfull_prod
        simpa [ratio, hjlt] using hfull_prod
      have hgj : g j ≠ 0 := (hg j).ne'
      have h2 : d j * g' j = g j := by
        dsimp [d]
        calc (∏ r ∈ Finset.range (j : ℕ), ratio r) * g' j
            = ((∏ r ∈ Finset.range (j : ℕ), ratio r) * (g' j / g j)) * g j := by
              field_simp [hgj]
          _ = 1 * g j := by rw [h1]
          _ = g j := one_mul _
      rw [hdi, one_mul, h2]
    · have hj1lt : (j : ℕ) + 1 < n := by omega
      have hi_eq : (i : ℕ) = (j : ℕ) + 1 := by rw [hij, Nat.mod_eq_of_lt hj1lt]
      have hdi : d i = d j * (g' j / g j) := by
        dsimp [d]
        rw [hi_eq, Finset.prod_range_succ]
        simp [ratio, hjlt]
      have hgj : g j ≠ 0 := (hg j).ne'
      rw [hdi]
      field_simp [hgj]
  refine ⟨d, hd_pos, fun s h => ?_⟩
  ext i j
  rw [Matrix.diagonal_mul, Matrix.mul_diagonal]
  simp only [ringStepMat, Matrix.of_apply]
  have hdiag : d i * (if i = j then s i else 0) = (if i = j then s i else 0) * d j := by
    split_ifs with hij
    · subst hij; ring
    · ring
  have htrans : d i * (if (i : ℕ) = ((j : ℕ) + 1) % n then g j * h j else 0) =
      (if (i : ℕ) = ((j : ℕ) + 1) % n then g' j * h j else 0) * d j := by
    split_ifs with hcyc
    · have hrel := hstep j i hcyc
      linear_combination (h j) * hrel
    · ring
  linear_combination hdiag + htrans

/-- **Part 1b (reduction to `uniformTransfer n (∑ log g_k)`).**
For `n ≥ 1` and `g_k > 0`, `uniformTransfer n (∑_k log g_k)` has strictly positive entries and
the exact same circuit product `∏_k g_k`. -/
theorem prod_uniformTransfer_eq (n : ℕ) (hn : 1 ≤ n) (y : ℝ) :
    (∀ k : Fin n, 0 < uniformTransfer n y k) ∧
      ∏ k : Fin n, uniformTransfer n y k = Real.exp y ∧
      ∑ k : Fin n, Real.log (uniformTransfer n y k) = y := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  refine ⟨fun _ => Real.exp_pos _, ?_, ?_⟩
  · unfold uniformTransfer
    rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin, ← Real.exp_nat_mul]
    congr 1
    field_simp [hn_pos.ne']
  · unfold uniformTransfer
    simp only [Real.log_exp, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp [hn_pos.ne']

/-- **Part 2 (birth-divisor shift on log transfer product).**
Dividing the `m = B.card` birth transfers `B` by `R > 0` keeps every transfer positive and shifts
`∑_k log g_k` to `∑_k log g_k − B.card · log R`. -/
theorem sum_log_tiltBirths {n : ℕ} (B : Finset (Fin n)) (R : ℝ) (hR : 0 < R)
    (g : Fin n → ℝ) (hg : ∀ k, 0 < g k) :
    (∀ k, 0 < tiltBirths B R g k) ∧
      ∑ k : Fin n, Real.log (tiltBirths B R g k) =
        (∑ k : Fin n, Real.log (g k)) - (B.card : ℝ) * Real.log R := by
  refine ⟨fun k => ?_, ?_⟩
  · unfold tiltBirths
    split_ifs
    · exact div_pos (hg k) hR
    · exact hg k
  · have hterm : ∀ k : Fin n, Real.log (tiltBirths B R g k) =
        Real.log (g k) - if k ∈ B then Real.log R else 0 := by
      intro k
      unfold tiltBirths
      split_ifs with hk
      · rw [Real.log_div (hg k).ne' hR.ne']
      · ring
    simp_rw [hterm]
    rw [Finset.sum_sub_distrib]
    congr 1
    simp

/-- **Part 3a (unique root of a convex continuous function eventually `< 0` near `−∞` and
tending to `+∞` near `+∞`).**
Formalises the root-existence and convexity-uniqueness argument of `prop:model-loss`. -/
theorem existsUnique_zero_of_convex_eventually_neg_tendsto_atTop (f : ℝ → ℝ)
    (hconv : ConvexOn ℝ Set.univ f) (hcont : Continuous f)
    (hneg : ∀ᶠ y in Filter.atBot, f y < 0)
    (hpos : Filter.Tendsto f Filter.atTop Filter.atTop) :
    ∃! y₀ : ℝ, f y₀ = 0 := by
  obtain ⟨Y_bot, hY_bot⟩ := Filter.eventually_atBot.mp hneg
  obtain ⟨Y_top, hY_top⟩ := Filter.eventually_atTop.mp (hpos.eventually_gt_atTop 0)
  let a := min Y_bot 0
  let b := max a Y_top
  have hab : a ≤ b := le_max_left _ _
  have hfa : f a < 0 := hY_bot a (min_le_left _ _)
  have hfb : 0 < f b := hY_top b (le_max_right _ _)
  obtain ⟨y₀, _, hy₀⟩ := intermediate_value_Icc hab hcont.continuousOn ⟨hfa.le, hfb.le⟩
  refine ⟨y₀, hy₀, ?_⟩
  suffices huniq : ∀ z₁ z₂ : ℝ, f z₁ = 0 → f z₂ = 0 → z₁ < z₂ → False by
    intro y hy
    rcases lt_trichotomy y y₀ with hlt | heq | hgt
    · exact False.elim (huniq y y₀ hy hy₀ hlt)
    · exact heq
    · exact False.elim (huniq y₀ y hy₀ hy hgt)
  intro z₁ z₂ hz₁ hz₂ hlt
  let u := min Y_bot (z₁ - 1)
  have hu_lt : u < z₁ := lt_of_le_of_lt (min_le_right _ _) (by linarith)
  have hfu : f u < 0 := hY_bot u (min_le_left _ _)
  have hlen : 0 < z₂ - u := by linarith
  let w₁ : ℝ := (z₂ - z₁) / (z₂ - u)
  let w₂ : ℝ := (z₁ - u) / (z₂ - u)
  have hw₁_pos : 0 < w₁ := div_pos (by linarith) hlen
  have hw₂_nn : 0 ≤ w₂ := div_nonneg (by linarith) hlen.le
  have hw_sum : w₁ + w₂ = 1 := by
    dsimp [w₁, w₂]; field_simp [hlen.ne']; ring
  have hz₁_comb : w₁ * u + w₂ * z₂ = z₁ := by
    dsimp [w₁, w₂]; field_simp [hlen.ne']; ring
  have hle := hconv.2 (Set.mem_univ u) (Set.mem_univ z₂) hw₁_pos.le hw₂_nn hw_sum
  simp only [smul_eq_mul, hz₁_comb, hz₁, hz₂, mul_zero, add_zero] at hle
  have : w₁ * f u < 0 := mul_neg_of_pos_of_neg hw₁_pos hfu
  linarith

/-- **Part 3b (`prop:model-loss`: unique `R₀* > 0` and `(g, m)`-invariance of `D`).**
Fix `n ≥ 1` and `s : Fin n → ℝ` with `0 ≤ s_k < 1`. Suppose the stochastic growth exponent
`r : (Fin n → ℝ) → ℝ` satisfies the four explicit probability hypotheses:
- `(h_conj)` `r g = r g'` whenever positive `g, g'` have positive-diagonally conjugate step
  matrices for `s` and all `h`;
- `(h_conv)` `y ↦ r (uniformTransfer n y)` is convex and continuous on `ℝ`;
- `(h_neg)` `∀ᶠ y in atBot, r (uniformTransfer n y) < 0`;
- `(h_pos)` `Tendsto (fun y => r (uniformTransfer n y)) atTop atTop`.

Then there exists a unique `y₀ : ℝ` with `r (uniformTransfer n y₀) = 0`, and putting
`D_star = y₀ − ∑_k log (1 − s_k)` (which depends only on `s` and `r`, not on `g`, `B`, or `m`):
for every positive transfer vector `g` (`g_k > 0`) and every nonempty birth-transfer subset
`B ⊆ Fin n` (`m = B.card ≥ 1`), there exists a **unique** `R₀* > 0` with
`r (tiltBirths B R₀* g) = 0`, and it satisfies
`Real.log (circuitR0 n s g) - (B.card : ℝ) * Real.log R₀* = D_star`. -/
private theorem existsUnique_R0Star_and_D_invariant_of_tails (n : ℕ) (hn : 1 ≤ n)
    (s : Fin n → ℝ) (_hs0 : ∀ k, 0 ≤ s k) (hs1 : ∀ k, s k < 1)
    (r : (Fin n → ℝ) → ℝ)
    (h_conj : ∀ g g' : Fin n → ℝ, (∀ k, 0 < g k) → (∀ k, 0 < g' k) →
      (∃ d : Fin n → ℝ, (∀ k, 0 < d k) ∧ ∀ h : Fin n → ℝ,
        Matrix.diagonal d * ringStepMat n s g h =
          ringStepMat n s g' h * Matrix.diagonal d) → r g = r g')
    (h_conv : ConvexOn ℝ Set.univ (fun y => r (uniformTransfer n y)))
    (h_cont : Continuous (fun y => r (uniformTransfer n y)))
    (h_neg : ∀ᶠ y in Filter.atBot, r (uniformTransfer n y) < 0)
    (h_pos : Filter.Tendsto (fun y => r (uniformTransfer n y)) Filter.atTop Filter.atTop) :
    ∃! y₀ : ℝ, r (uniformTransfer n y₀) = 0 ∧
      ∀ (g : Fin n → ℝ), (∀ k, 0 < g k) →
      ∀ (B : Finset (Fin n)), B.Nonempty →
        ∃! Rstar : ℝ, 0 < Rstar ∧ r (tiltBirths B Rstar g) = 0 ∧
          Real.log (circuitR0 n s g) - (B.card : ℝ) * Real.log Rstar =
            y₀ - ∑ k : Fin n, Real.log (1 - s k) := by
  -- First reduce `r g'` for any positive `g'` to `r (uniformTransfer n (∑ log g'_k))`.
  have hr_eq : ∀ g' : Fin n → ℝ, (∀ k, 0 < g' k) →
      r g' = r (uniformTransfer n (∑ k : Fin n, Real.log (g' k))) := by
    intro g' hg'
    let y := ∑ k : Fin n, Real.log (g' k)
    obtain ⟨hu_pos, hu_prod, _⟩ := prod_uniformTransfer_eq n hn y
    have hprod_g' : ∏ k : Fin n, g' k = Real.exp y := by
      dsimp [y]
      rw [Real.exp_sum]
      exact Finset.prod_congr rfl fun k _ => (Real.exp_log (hg' k)).symm
    obtain ⟨d, hd_pos, hd_conj⟩ := exists_pos_diag_conj_of_prod_eq n hn g' (uniformTransfer n y)
      hg' hu_pos (hprod_g'.trans hu_prod.symm)
    exact h_conj g' (uniformTransfer n y) hg' hu_pos ⟨d, hd_pos, fun h => hd_conj s h⟩
  -- Unique zero `y₀` of `y ↦ r (uniformTransfer n y)`
  obtain ⟨y₀, hy₀_zero, hy₀_uniq⟩ :=
    existsUnique_zero_of_convex_eventually_neg_tendsto_atTop
      (fun y => r (uniformTransfer n y)) h_conv h_cont h_neg h_pos
  -- Log of `circuitR0 n s g`
  have hlogR0 : ∀ g : Fin n → ℝ, (∀ k, 0 < g k) →
      Real.log (circuitR0 n s g) =
        (∑ k : Fin n, Real.log (g k)) - ∑ k : Fin n, Real.log (1 - s k) := by
    intro g hg
    unfold circuitR0
    have hterm_pos : ∀ k : Fin n, g k / (1 - s k) ≠ 0 := fun k =>
      (div_pos (hg k) (sub_pos.mpr (hs1 k))).ne'
    rw [Real.log_prod (fun k _ => hterm_pos k)]
    have hsplit : ∀ k : Fin n, Real.log (g k / (1 - s k)) =
        Real.log (g k) - Real.log (1 - s k) := fun k =>
      Real.log_div (hg k).ne' (sub_pos.mpr (hs1 k)).ne'
    simp_rw [hsplit]
    rw [Finset.sum_sub_distrib]
  -- Given `y₀`, construct and prove uniqueness of `Rstar` for each `(g, B)`.
  have hRstar_all : ∀ (g : Fin n → ℝ), (∀ k, 0 < g k) →
      ∀ (B : Finset (Fin n)), B.Nonempty →
        ∃! Rstar : ℝ, 0 < Rstar ∧ r (tiltBirths B Rstar g) = 0 ∧
          Real.log (circuitR0 n s g) - (B.card : ℝ) * Real.log Rstar =
            y₀ - ∑ k : Fin n, Real.log (1 - s k) := by
    intro g hg B hBne
    have hm_pos : (0 : ℝ) < (B.card : ℝ) := by exact_mod_cast Finset.card_pos.mpr hBne
    let S := ∑ k : Fin n, Real.log (g k)
    let Rstar := Real.exp ((S - y₀) / (B.card : ℝ))
    have hRstar_pos : 0 < Rstar := Real.exp_pos _
    have hlog_Rstar : (B.card : ℝ) * Real.log Rstar = S - y₀ := by
      dsimp [Rstar]
      rw [Real.log_exp]
      field_simp [hm_pos.ne']
    obtain ⟨htilt_pos, htilt_sum⟩ := sum_log_tiltBirths B Rstar hRstar_pos g hg
    have htilt_y₀ : ∑ k : Fin n, Real.log (tiltBirths B Rstar g k) = y₀ := by
      dsimp [S] at hlog_Rstar
      linarith [htilt_sum, hlog_Rstar]
    have hr_tilt : r (tiltBirths B Rstar g) = 0 := by
      rw [hr_eq (tiltBirths B Rstar g) htilt_pos, htilt_y₀, hy₀_zero]
    have hD_eq : Real.log (circuitR0 n s g) - (B.card : ℝ) * Real.log Rstar =
        y₀ - ∑ k : Fin n, Real.log (1 - s k) := by
      rw [hlogR0 g hg]
      dsimp [S] at hlog_Rstar
      linarith
    refine ⟨Rstar, ⟨hRstar_pos, hr_tilt, hD_eq⟩, ?_⟩
    intro R' ⟨hR'_pos, hr_R', _⟩
    obtain ⟨htilt'_pos, htilt'_sum⟩ := sum_log_tiltBirths B R' hR'_pos g hg
    rw [hr_eq (tiltBirths B R' g) htilt'_pos] at hr_R'
    have hy_eq : ∑ k : Fin n, Real.log (tiltBirths B R' g k) = y₀ := hy₀_uniq _ hr_R'
    have hlog_eq : Real.log R' = (S - y₀) / (B.card : ℝ) := by
      have h1 : Real.log R' * (B.card : ℝ) = S - y₀ := by
        dsimp [S]; linarith [htilt'_sum, hy_eq]
      exact (eq_div_iff hm_pos.ne').mpr h1
    calc R' = Real.exp (Real.log R') := (Real.exp_log hR'_pos).symm
      _ = Rstar := by rw [hlog_eq]
  refine ⟨y₀, ⟨hy₀_zero, hRstar_all⟩, ?_⟩
  intro y' ⟨hy'_zero, _⟩
  exact hy₀_uniq y' hy'_zero


/-- Actual physical finite growth with positive reference initial masses.
The transpose identity below identifies the exact manuscript ring matrix. -/
noncomputable def ringNormalizedGrowth {Ω : Type*} [MeasurableSpace Ω]
    (n : ℕ) (hn : 1 ≤ n) (s : Fin n → ℝ) (host : Ω → Fin n → ℝ)
    (μ : MeasureTheory.Measure Ω) (T : Ω → Ω) (g : Fin n → ℝ) : ℕ → ℝ :=
  Coverage4ThermalFixed.physicalRingGrowth (RingGrowth.times s) (finRotate n)
    (fun ω i => g i * host ω i) (fun _ => s) 1 ⟨0, by omega⟩ μ T

lemma physical_step_transpose (n : ℕ) (hn : 1 ≤ n) (s g h : Fin n → ℝ) :
    (Coverage4ThermalFixed.physicalRingStep (finRotate n)
      (fun (_ : Unit) i => g i * h i) (fun _ => s) 1 ⟨0, by omega⟩ ()).transpose =
        ringStepMat n s g h := by
  letI : NeZero n := ⟨by omega⟩
  ext i j
  have hrot : ((finRotate n j : Fin n) : ℕ) = (j.val + 1) % n := by
    rw [finRotate_apply, Fin.val_add, Fin.val_one']
    simp [Nat.add_mod]
  have heq : finRotate n j = i ↔ (i : ℕ) = ((j : ℕ) + 1) % n := by
    rw [Fin.ext_iff, hrot]
    exact eq_comm
  simp only [Matrix.transpose_apply, Coverage4ThermalFixed.physicalRingStep,
    ringStepMat, Matrix.of_apply, ite_self, div_one, heq]
  by_cases hij : i = j
  · subst j
    rfl
  · simp [hij, Ne.symm hij]

/-- 2026-11-16 leanp1: both tail conclusions follow from the actual ring matrix
product, the stationary path lower bound, and the finite-mean row-sum upper bound. The named FK input is only the
textbook expected growth limit. The root and loss conclusions are unchanged. -/
theorem existsUnique_R0Star_and_D_invariant {Ω : Type*} [MeasurableSpace Ω] (n : ℕ) (hn : 1 ≤ n)
    (s : Fin n → ℝ) (hs0 : ∀ k, 0 ≤ s k) (hs1 : ∀ k, s k < 1)
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (T : Ω ≃ᵐ Ω) (hT : MeasureTheory.MeasurePreserving T μ μ)
    (host : Ω → Fin n → ℝ)
    (hhost_pos : ∀ᵐ ω ∂μ, ∀ i, 0 < host ω i)
    (hhost_mean : ∀ i, MeasureTheory.Integrable (fun ω => host ω i) μ)
    (hhost_log : ∀ i, MeasureTheory.Integrable (fun ω => Real.log (host ω i)) μ)
    (r : (Fin n → ℝ) → ℝ)
    (hFurstenbergKesten : ∀ g : Fin n → ℝ, (∀ i, 0 < g i) →
      Filter.Tendsto (ringNormalizedGrowth n hn s host μ T g)
        Filter.atTop (nhds (r g))) :
    ∃! y₀ : ℝ, r (uniformTransfer n y₀) = 0 ∧
      ∀ (g : Fin n → ℝ), (∀ k, 0 < g k) →
      ∀ (B : Finset (Fin n)), B.Nonempty →
        ∃! Rstar : ℝ, 0 < Rstar ∧ r (tiltBirths B Rstar g) = 0 ∧
          Real.log (circuitR0 n s g) - (B.card : ℝ) * Real.log Rstar =
            y₀ - ∑ k : Fin n, Real.log (1 - s k) := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  have h_conv : ConvexOn ℝ Set.univ (fun y => r (uniformTransfer n y)) := by
    apply GrowthShape.physical_growth_convex s host μ T hT hs0 hs1 hhost_pos hhost_log
      (finRotate n) ⟨0, by omega⟩ (fun y => r (uniformTransfer n y))
    intro y
    simpa only [ringNormalizedGrowth, uniformTransfer, Fintype.card_fin] using
      hFurstenbergKesten (uniformTransfer n y) (fun _ => Real.exp_pos _)
  have h_cont : Continuous (fun y => r (uniformTransfer n y)) :=
    GrowthShape.continuous_of_convex _ h_conv
  have h_conj : ∀ g g' : Fin n → ℝ, (∀ k, 0 < g k) → (∀ k, 0 < g' k) →
      (∃ d : Fin n → ℝ, (∀ k, 0 < d k) ∧ ∀ h : Fin n → ℝ,
        Matrix.diagonal d * ringStepMat n s g h =
          ringStepMat n s g' h * Matrix.diagonal d) → r g = r g' := by
    intro g g' hg hg' hc
    obtain ⟨d, hd, hD⟩ := hc
    apply GrowthShape.physical_growth_eq_of_conj s g g' host μ T hT hs0 hs1 hg hg'
      hhost_pos hhost_log (finRotate n) ⟨0, by omega⟩ d hd
    · apply Filter.Eventually.of_forall
      intro ω
      have ht := congrArg Matrix.transpose (hD (host ω))
      simp only [Matrix.transpose_mul, Matrix.diagonal_transpose] at ht
      have hA : ringStepMat n s g (host ω) =
          (Coverage4ThermalFixed.physicalRingStep (finRotate n)
            (fun ω i => g i*host ω i) (fun _ => s) 1 ⟨0, by omega⟩ ω).transpose :=
        (physical_step_transpose n hn s g (host ω)).symm
      have hB : ringStepMat n s g' (host ω) =
          (Coverage4ThermalFixed.physicalRingStep (finRotate n)
            (fun ω i => g' i*host ω i) (fun _ => s) 1 ⟨0, by omega⟩ ω).transpose :=
        (physical_step_transpose n hn s g' (host ω)).symm
      rw [hA, hB, Matrix.transpose_transpose, Matrix.transpose_transpose] at ht
      exact ht
    · exact hFurstenbergKesten g hg
    · exact hFurstenbergKesten g' hg'
  have h_pos : Filter.Tendsto (fun y => r (uniformTransfer n y)) Filter.atTop Filter.atTop := by
    apply RingGrowth.growth_tendsto_atTop s host μ T hT hs0 hs1 hhost_pos hhost_log
      (finRotate n) ⟨0, by omega⟩ (fun y => r (uniformTransfer n y))
    intro y
    simpa only [ringNormalizedGrowth, uniformTransfer, Fintype.card_fin] using
      hFurstenbergKesten (uniformTransfer n y) (fun _ => Real.exp_pos _)
  have h_neg : ∀ᶠ y in Filter.atBot, r (uniformTransfer n y) < 0 := by
    apply RingGrowth.growth_eventually_neg s host μ T hT hs0 hs1 hhost_pos hhost_mean hhost_log
      (finRotate n) ⟨0, by omega⟩ (fun y => r (uniformTransfer n y))
    intro y
    simpa only [ringNormalizedGrowth, uniformTransfer, Fintype.card_fin] using
      hFurstenbergKesten (uniformTransfer n y) (fun _ => Real.exp_pos _)
  exact existsUnique_R0Star_and_D_invariant_of_tails n hn s hs0 hs1 r
    h_conj h_conv h_cont h_neg h_pos

end Dobson.Coverage1Loss

#print axioms Dobson.Coverage1Loss.exists_pos_diag_conj_of_prod_eq
#print axioms Dobson.Coverage1Loss.prod_uniformTransfer_eq
#print axioms Dobson.Coverage1Loss.sum_log_tiltBirths
#print axioms Dobson.Coverage1Loss.existsUnique_zero_of_convex_eventually_neg_tendsto_atTop
#print axioms Dobson.Coverage1Loss.existsUnique_R0Star_and_D_invariant
