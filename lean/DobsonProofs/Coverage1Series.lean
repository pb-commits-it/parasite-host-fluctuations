import Mathlib
import DobsonProofs.Band
import DobsonProofs.ReturnRate
import DobsonProofs.JensenGap

/-!
# Coverage batch 1 — Item 2: Hosts in series (`prop:ring-series`)

Formalises the five scalar comparisons corresponding to parts (i)–(v) of `prop:ring-series` (`sections/03-ring.tex`, lines 51–62,
and `appendices/A-ring.tex`, lines 47–77):

Identification with the Gaussian-host growth coefficient is outside this core.

Let `n ≥ 1`, lineage stays `τ_k > 0` (`k : Fin n`) with `𝒯 = ∑_k τ_k` and mean stay
`τ̄ = 𝒯 / n`, log-host variance `v = σ² > 0`, memory `0 ≤ φ < 1` (`θ = φ / (1 − φ)`),
`P = ∏_k (1 + τ_k / θ)` (`φ > 0`), and `n`-host series loss
`Γ_series = n v / (2 𝒯)` at `φ = 0` and `n v [(1 + 2 θ)/(2 𝒯) − 1/(P − 1)]` for `0 < φ < 1`.

- **(i)** `v / (2 τ̄) ≤ Γ_series ≤ v (1 + 2 θ) / (2 τ̄)`, with left equality iff `φ = 0 ∨ n = 1`;
  plus the manuscript's non-ordering counterexample at `φ = 9/10`: stays `(10, 10)` (`13/35 ≈ 0.371`)
  exceed stays `(1, 1, 28)` (`12709/59420 ≈ 0.214`) at the same `τ̄ = 10`.
- **(ii)** At equal stays `τ > 0` (`a = 1 + τ / θ` for `φ > 0`), `Γ_n = v [(1 + 2 θ)/(2 τ) − n/(a^n − 1)]`
  (`v / (2 τ)` at `φ = 0`), strictly increasing in `n ≥ 1` (`0 < φ < 1`) and strictly decreasing
  in `τ > 0` (`0 ≤ φ < 1`).
- **(iii)** Appending stage `τ_{n+1} = 𝒯 / n` on its own host changes `Γ_series` by `0` at `φ = 0`
  and by `v [P (x − 1) + 1] / [(P − 1) (P (1 + x / n) − 1)] > 0` (`x = 𝒯 / θ`) when `0 < φ < 1`.
- **(iv)** Multiplying every stay `τ_k` by `c > 1` strictly lowers `Γ_series` for all `0 ≤ φ < 1`
  (proved via polynomial log-concavity / Newton's inequality on `∏_k (1 + c y_k)`).
- **(v)** At equal stays `τ > 1` and `0 < φ < 1` (`a = 1 + τ / θ > 1`), the per-step coefficient
  `f_n = (1 + 2 θ) / (2 n τ²) − 1 / (τ (a^n − 1))` has strictly decreasing scaled increments
  `n (n + 1) (f_{n+1} − f_n)` on `n ≥ 1` (changing sign from `+` to `−` at most once), and
  `(∃ n ≥ 2, f_n > f_1) ↔ f_2 > f_1 ↔ θ > τ / (2 (τ − 1))`.
-/

namespace Dobson.Coverage1Series

open scoped BigOperators

/-- Memory scale `θ(φ) = φ / (1 − φ)`. -/
noncomputable def theta (φ : ℝ) : ℝ :=
  φ / (1 - φ)

/-- Total generation time `𝒯 = ∑_k τ_k`. -/
noncomputable def totalStay {n : ℕ} (τ : Fin n → ℝ) : ℝ :=
  ∑ k : Fin n, τ k

/-- Mean stage stay `τ̄ = 𝒯 / n`. -/
noncomputable def meanStay {n : ℕ} (τ : Fin n → ℝ) : ℝ :=
  totalStay τ / (n : ℝ)

/-- Circuit product `P(θ, τ) = ∏_k (1 + τ_k / θ)` (`θ > 0`). -/
noncomputable def prodOneAdd {n : ℕ} (θ : ℝ) (τ : Fin n → ℝ) : ℝ :=
  ∏ k : Fin n, (1 + τ k / θ)

/-- Series loss `Γ` across `n` independent equal-`(v, φ)` `AR(1)` hosts (`0 ≤ φ < 1`). -/
noncomputable def seriesGamma (n : ℕ) (v φ : ℝ) (τ : Fin n → ℝ) : ℝ :=
  if φ = 0 then
    (n : ℝ) * v / (2 * totalStay τ)
  else
    (n : ℝ) * v * ((1 + 2 * theta φ) / (2 * totalStay τ) - 1 / (prodOneAdd (theta φ) τ - 1))

/-! ## Part (i): Fixed mean stay `τ̄`, strictness for `n ≥ 2`, and `(10, 10)` vs `(1, 1, 28)` -/

/-- Strict refinement of `Band.one_add_sum_le_prod_one_add` for `n ≥ 2` and `x_k > 0`:
`1 + ∑_k x_k < ∏_k (1 + x_k)`. -/
theorem one_add_sum_lt_prod_one_add_of_two_le (n : ℕ) (hn : 2 ≤ n)
    (x : Fin n → ℝ) (hx : ∀ k, 0 < x k) :
    1 + ∑ k : Fin n, x k < ∏ k : Fin n, (1 + x k) := by
  obtain ⟨m, rfl⟩ : ∃ m : ℕ, n = m + 2 := ⟨n - 2, by omega⟩
  rw [Fin.sum_univ_castSucc, Fin.prod_univ_castSucc]
  rw [Fin.sum_univ_castSucc, Fin.prod_univ_castSucc]
  let S := ∑ i : Fin m, x i.castSucc.castSucc
  let Pr := ∏ i : Fin m, (1 + x i.castSucc.castSucc)
  have hS_nn : 0 ≤ S := Finset.sum_nonneg fun i _ => (hx _).le
  have hB1 : 1 + S ≤ Pr :=
    Dobson.Band.one_add_sum_le_prod_one_add Finset.univ
      (fun i : Fin m => x i.castSucc.castSucc) (fun _ _ => (hx _).le)
  have hx1 : 0 < x (Fin.last m).castSucc := hx _
  have hx2 : 0 < x (Fin.last (m + 1)) := hx _
  have hstep1 : 1 + S + x (Fin.last m).castSucc ≤ Pr * (1 + x (Fin.last m).castSucc) := by
    nlinarith [mul_le_mul_of_nonneg_right hB1 (by linarith : 0 ≤ 1 + x (Fin.last m).castSucc),
      mul_nonneg hS_nn hx1.le]
  have hmul2 : (1 + S + x (Fin.last m).castSucc) * (1 + x (Fin.last (m + 1))) ≤
      (Pr * (1 + x (Fin.last m).castSucc)) * (1 + x (Fin.last (m + 1))) :=
    mul_le_mul_of_nonneg_right hstep1 (by linarith)
  have hpos_cross : 0 < (S + x (Fin.last m).castSucc) * x (Fin.last (m + 1)) :=
    mul_pos (by linarith) hx2
  change 1 + (S + x (Fin.last m).castSucc + x (Fin.last (m + 1))) <
    Pr * (1 + x (Fin.last m).castSucc) * (1 + x (Fin.last (m + 1)))
  linarith

/-- **Part (i): The series band at fixed `τ̄` and exact left-equality characterization.** -/
theorem series_part_i (n : ℕ) (hn : 1 ≤ n) (v φ : ℝ) (hv : 0 < v)
    (hφ0 : 0 ≤ φ) (hφ1 : φ < 1) (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k) :
    v / (2 * meanStay τ) ≤ seriesGamma n v φ τ ∧
      seriesGamma n v φ τ ≤ v * (1 + 2 * theta φ) / (2 * meanStay τ) ∧
      (seriesGamma n v φ τ = v / (2 * meanStay τ) ↔ φ = 0 ∨ n = 1) := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hT : 0 < totalStay τ := Finset.sum_pos (fun k _ => hτ k) hne
  have hlow_rewrite : v / (2 * meanStay τ) = (n : ℝ) * v / (2 * totalStay τ) := by
    unfold meanStay
    field_simp [hn_pos.ne', hT.ne']
  have hhigh_rewrite : v * (1 + 2 * theta φ) / (2 * meanStay τ) =
      (n : ℝ) * v * (1 + 2 * theta φ) / (2 * totalStay τ) := by
    unfold meanStay
    field_simp [hn_pos.ne', hT.ne']
  by_cases hφz : φ = 0
  · subst hφz
    have hθ0 : theta 0 = 0 := by simp [theta]
    refine ⟨?_, ?_, ?_⟩
    · rw [hlow_rewrite]; simp [seriesGamma]
    · rw [hhigh_rewrite, hθ0]; simp [seriesGamma]
    · rw [hlow_rewrite]; simp [seriesGamma]
  · have hφ_pos : 0 < φ := lt_of_le_of_ne hφ0 (Ne.symm hφz)
    have hθ : 0 < theta φ := div_pos hφ_pos (sub_pos.mpr hφ1)
    have hP_gt1 : 1 < prodOneAdd (theta φ) τ := by
      have hB1 := Dobson.Band.one_add_sum_le_prod_one_add Finset.univ
        (fun k => τ k / theta φ) (fun k _ => (div_pos (hτ k) hθ).le)
      have hsum : ∑ k : Fin n, τ k / theta φ = totalStay τ / theta φ := by
        unfold totalStay; rw [Finset.sum_div]
      unfold prodOneAdd
      linarith [div_pos hT hθ]
    have hratio_eq : Dobson.Band.R (theta φ) Finset.univ τ /
        (1 - Dobson.Band.R (theta φ) Finset.univ τ) =
        1 / (prodOneAdd (theta φ) τ - 1) := by
      change (1 / prodOneAdd (theta φ) τ) / (1 - 1 / prodOneAdd (theta φ) τ) =
        1 / (prodOneAdd (theta φ) τ - 1)
      have hP0 : prodOneAdd (theta φ) τ ≠ 0 := (by linarith : 0 < prodOneAdd (theta φ) τ).ne'
      have hP1 : prodOneAdd (theta φ) τ - 1 ≠ 0 := (sub_pos.mpr hP_gt1).ne'
      field_simp [hP0, hP1]
    have hGamma_eq : seriesGamma n v φ τ =
        (n : ℝ) * Dobson.Band.Gamma v (theta φ) Finset.univ τ := by
      unfold seriesGamma Dobson.Band.Gamma Dobson.Band.total totalStay
      rw [ite_eq_right hφz, hratio_eq]
      ring
    have hBand := Dobson.Band.band (theta φ) hθ Finset.univ hne τ (fun k _ => hτ k) v hv.le
    refine ⟨?_, ?_, ?_⟩
    · rw [hlow_rewrite, hGamma_eq]
      have hmul := mul_le_mul_of_nonneg_left hBand.1 hn_pos.le
      calc (n : ℝ) * v / (2 * totalStay τ)
          = (n : ℝ) * (v / (2 * Dobson.Band.total Finset.univ τ)) := by
            unfold totalStay Dobson.Band.total; ring
        _ ≤ (n : ℝ) * Dobson.Band.Gamma v (theta φ) Finset.univ τ := hmul
    · rw [hhigh_rewrite, hGamma_eq]
      have hmul := mul_le_mul_of_nonneg_left hBand.2 hn_pos.le
      calc (n : ℝ) * Dobson.Band.Gamma v (theta φ) Finset.univ τ
          ≤ (n : ℝ) * (v * (1 + 2 * theta φ) / (2 * Dobson.Band.total Finset.univ τ)) := hmul
        _ = (n : ℝ) * v * (1 + 2 * theta φ) / (2 * totalStay τ) := by
            unfold totalStay Dobson.Band.total; ring
    · rw [hlow_rewrite]
      unfold seriesGamma
      rw [ite_eq_right hφz]
      constructor
      · intro heq
        right
        by_contra hn_ne1
        have hn2 : 2 ≤ n := by omega
        have hstrict := one_add_sum_lt_prod_one_add_of_two_le n hn2
          (fun k => τ k / theta φ) (fun k => div_pos (hτ k) hθ)
        have hsum : ∑ k : Fin n, τ k / theta φ = totalStay τ / theta φ := by
          unfold totalStay; rw [Finset.sum_div]
        rw [hsum] at hstrict
        have hP_strict : totalStay τ / theta φ < prodOneAdd (theta φ) τ - 1 := by
          unfold prodOneAdd; linarith
        have hinv_lt : 1 / (prodOneAdd (theta φ) τ - 1) < theta φ / totalStay τ := by
          rw [div_lt_div_iff₀ (sub_pos.mpr hP_gt1) hT]
          have h1 : theta φ * (totalStay τ / theta φ) = totalStay τ := by
            field_simp [hθ.ne']
          have h2 : theta φ * (totalStay τ / theta φ) < theta φ * (prodOneAdd (theta φ) τ - 1) :=
            mul_lt_mul_of_pos_left hP_strict hθ
          linarith
        have hbracket : 1 / (2 * totalStay τ) <
            (1 + 2 * theta φ) / (2 * totalStay τ) - 1 / (prodOneAdd (theta φ) τ - 1) := by
          have hid : (1 + 2 * theta φ) / (2 * totalStay τ) - theta φ / totalStay τ =
              1 / (2 * totalStay τ) := by
            field_simp [hT.ne']
            ring
          linarith
        have hm := mul_lt_mul_of_pos_left hbracket (mul_pos hn_pos hv)
        have hmul_div : (n : ℝ) * v * (1 / (2 * totalStay τ)) =
            (n : ℝ) * v / (2 * totalStay τ) := mul_one_div _ _
        linarith
      · rintro (h0 | rfl)
        · exact False.elim (hφz h0)
        · have hT1 : totalStay τ = τ 0 := by simp [totalStay]
          have hP1 : prodOneAdd (theta φ) τ - 1 = τ 0 / theta φ := by
            simp [prodOneAdd]
          rw [hT1, hP1]
          have hτ0 : τ 0 ≠ 0 := (hτ 0).ne'
          simp only [Nat.cast_one, one_mul]
          field_simp [hτ0, hθ.ne']
          ring

/-- **Part (i) concrete non-ordering example:** At `φ = 9/10`, `(10, 10)` and `(1, 1, 28)` both
have `τ̄ = 10`, with exact coefficients `13/35 ≈ 0.371` and `12709/59420 ≈ 0.214`. -/
theorem series_part_i_nonmonotone_example :
    meanStay ![10, 10] = 10 ∧
      meanStay ![1, 1, 28] = 10 ∧
      seriesGamma 2 1 (9 / 10) ![10, 10] = 13 / 35 ∧
      seriesGamma 3 1 (9 / 10) ![1, 1, 28] = 12709 / 59420 ∧
      seriesGamma 3 1 (9 / 10) ![1, 1, 28] < seriesGamma 2 1 (9 / 10) ![10, 10] := by
  norm_num [meanStay, totalStay, seriesGamma, prodOneAdd, theta,
    Fin.sum_univ_two, Fin.prod_univ_two, Fin.sum_univ_three, Fin.prod_univ_three]

/-! ## Part (iv): Scaling stays `τ_k ↦ c τ_k` (`c > 1`) via polynomial log-concavity -/

/-- Multiply a degree-`m` positive-root polynomial's coefficient sequence by `(1 + r X)` (`r > 0`),
preserving positivity on `0 ≤ k ≤ m + 1`, vanishing on `k > m + 1`, and log-concavity. -/
private theorem logConcave_step (m : ℕ) (a : ℕ → ℝ) (r : ℝ) (hr : 0 < r)
    (ha_pos : ∀ k, k ≤ m → 0 < a k)
    (ha_zero : ∀ k, m < k → a k = 0)
    (ha_lc : ∀ i j, i ≤ j → a (i + 1) * a j ≥ a i * a (j + 1)) :
    let b : ℕ → ℝ := fun k => if k = 0 then a 0 else a k + r * a (k - 1)
    (∀ k, k ≤ m + 1 → 0 < b k) ∧
      (∀ k, m + 1 < k → b k = 0) ∧
      (∀ i j, i ≤ j → b (i + 1) * b j ≥ b i * b (j + 1)) := by
  intro b
  have ha_nn : ∀ k, 0 ≤ a k := fun k => by
    by_cases hk : k ≤ m
    · exact (ha_pos k hk).le
    · rw [ha_zero k (by omega)]
  have hb_pos : ∀ k, k ≤ m + 1 → 0 < b k := by
    intro k hk
    dsimp [b]
    rcases k with _ | k
    · simp [ha_pos 0 (Nat.zero_le _)]
    · simp only [Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte, Nat.add_sub_cancel]
      have hpos : 0 < r * a k := mul_pos hr (ha_pos k (by omega))
      linarith [ha_nn (k + 1)]
  have hb_zero : ∀ k, m + 1 < k → b k = 0 := by
    intro k hk
    dsimp [b]
    have hk0 : k ≠ 0 := by omega
    rw [ite_eq_right hk0, ha_zero k (by omega), ha_zero (k - 1) (by omega)]
    ring
  refine ⟨hb_pos, hb_zero, fun i j hij => ?_⟩
  rcases i with _ | i
  · rcases j with _ | j
    · dsimp [b]
      linarith
    · simp only [b, Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte, Nat.add_sub_cancel,
        Nat.sub_self, zero_add]
      have h1 : a 0 * a (j + 1 + 1) ≤ a 1 * a (j + 1) := ha_lc 0 (j + 1) (Nat.zero_le _)
      have h2 : 0 ≤ r * (a 1 * a j) := mul_nonneg hr.le (mul_nonneg (ha_nn 1) (ha_nn j))
      have h3 : 0 ≤ r ^ 2 * (a 0 * a j) := mul_nonneg (sq_nonneg r) (mul_nonneg (ha_nn 0) (ha_nn j))
      have hid : (a 1 + r * a 0) * (a (j + 1) + r * a j) -
          a 0 * (a (j + 1 + 1) + r * a (j + 1)) =
          (a 1 * a (j + 1) - a 0 * a (j + 1 + 1)) +
            r * (a 1 * a j) + r ^ 2 * (a 0 * a j) := by ring
      linarith
  · obtain ⟨j, rfl⟩ : ∃ j' : ℕ, j = j' + 1 := ⟨j - 1, by omega⟩
    have hij' : i ≤ j := by omega
    simp only [b, Nat.add_eq_zero_iff, one_ne_zero, and_false, ↓reduceIte, Nat.add_sub_cancel]
    have h1 : a (i + 2) * a (j + 1) ≥ a (i + 1) * a (j + 2) := ha_lc (i + 1) (j + 1) (by omega)
    have h3 : a (i + 1) * a j ≥ a i * a (j + 1) := ha_lc i j hij'
    have h2 : a (i + 2) * a j ≥ a i * a (j + 2) := by
      by_cases hi1 : i + 1 ≤ m
      · have hai1 : 0 < a (i + 1) := ha_pos (i + 1) hi1
        have hmul1 : (a (i + 2) * a j) * a (i + 1) ≥ a (i + 2) * (a i * a (j + 1)) := by
          have := mul_le_mul_of_nonneg_left h3 (ha_nn (i + 2))
          linarith
        have hmul2 : a (i + 2) * (a i * a (j + 1)) ≥ (a i * a (j + 2)) * a (i + 1) := by
          have := mul_le_mul_of_nonneg_left h1 (ha_nn i)
          linarith
        exact le_of_mul_le_mul_right (hmul2.trans hmul1) hai1
      · have hj2 : m < j + 2 := by omega
        rw [ha_zero (j + 2) hj2, mul_zero]
        exact mul_nonneg (ha_nn (i + 2)) (ha_nn j)
    nlinarith [mul_le_mul_of_nonneg_left h2 hr.le,
      mul_le_mul_of_nonneg_left h3 (sq_nonneg r)]

/-- Elementary symmetric polynomial coefficients `e_k` of `∏_{i : Fin n} (1 + y i * X)` (`y i > 0`)
are positive on `0 ≤ k ≤ n`, zero on `k > n`, satisfy `e_0 = 1`, `e_1 = ∑ y i`,
`∑_{k=0}^n e_k c^k = ∏_i (1 + c * y i)`, and are log-concave (`e_{i+1} e_j ≥ e_i e_{j+1}` for `i ≤ j`). -/
private theorem symPoly_logConcave :
    ∀ (n : ℕ) (y : Fin n → ℝ), (∀ i, 0 < y i) →
      ∃ e : ℕ → ℝ,
        e 0 = 1 ∧
        e 1 = ∑ i : Fin n, y i ∧
        (∀ c : ℝ, ∑ k ∈ Finset.range (n + 1), e k * c ^ k = ∏ i : Fin n, (1 + c * y i)) ∧
        (∀ k, k ≤ n → 0 < e k) ∧
        (∀ k, n < k → e k = 0) ∧
        (∀ i j, i ≤ j → e (i + 1) * e j ≥ e i * e (j + 1)) := by
  intro n
  induction n with
  | zero =>
    intro y _
    refine ⟨fun k => if k = 0 then 1 else 0, rfl, by simp, ?_, ?_, ?_, ?_⟩
    · intro c; simp
    · intro k hk
      have hk0 : k = 0 := Nat.eq_zero_of_le_zero hk
      rw [hk0]
      norm_num
    · intro k hk
      have hk0 : k ≠ 0 := Nat.ne_of_gt hk
      exact ite_eq_right hk0
    · intro i j hij
      rcases i with _ | i <;> rcases j with _ | j <;> simp at hij ⊢
  | succ m ih =>
    intro y hy
    let y' : Fin m → ℝ := fun i => y i.castSucc
    let r : ℝ := y (Fin.last m)
    obtain ⟨a, ha0, ha1, ha_eval, ha_pos, ha_zero, ha_lc⟩ := ih y' (fun i => hy i.castSucc)
    have hr : 0 < r := hy (Fin.last m)
    let b : ℕ → ℝ := fun k => if k = 0 then a 0 else a k + r * a (k - 1)
    obtain ⟨hb_pos, hb_zero, hb_lc⟩ := logConcave_step m a r hr ha_pos ha_zero ha_lc
    refine ⟨b, ha0, ?_, ?_, hb_pos, hb_zero, hb_lc⟩
    · dsimp [b, r, y']
      rw [ha1, ha0, mul_one, Fin.sum_univ_castSucc]
    · intro c
      rw [Fin.prod_univ_castSucc, ← ha_eval c]
      rw [Finset.sum_range_succ']
      simp only [b, ↓reduceIte, pow_zero, mul_one, Nat.add_eq_zero_iff, one_ne_zero, and_false,
        Nat.add_sub_cancel]
      have hshift1 : ∑ k ∈ Finset.range (m + 1), (a (k + 1) + r * a k) * c ^ (k + 1) =
          (∑ k ∈ Finset.range (m + 1), a (k + 1) * c ^ (k + 1)) +
            c * r * ∑ k ∈ Finset.range (m + 1), a k * c ^ k := by
        rw [Finset.mul_sum]
        rw [← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun k _ => by rw [pow_succ]; ring
      have hshift2 : a 0 + ∑ k ∈ Finset.range (m + 1), a (k + 1) * c ^ (k + 1) =
          ∑ k ∈ Finset.range (m + 1), a k * c ^ k := by
        rw [Finset.sum_range_succ, ha_zero (m + 1) (by omega), zero_mul, add_zero]
        have h := Finset.sum_range_succ' (fun k => a k * c ^ k) m
        simp only [pow_zero, mul_one] at h
        linarith
      rw [hshift1]
      dsimp [r]
      linarith

/-- **Part (iv): Multiplying every stay `τ_k` by `c > 1` strictly lowers `seriesGamma`.** -/
theorem series_part_iv (n : ℕ) (hn : 1 ≤ n) (v φ : ℝ) (hv : 0 < v)
    (hφ0 : 0 ≤ φ) (hφ1 : φ < 1) (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k)
    (c : ℝ) (hc : 1 < c) :
    seriesGamma n v φ (fun k => c * τ k) < seriesGamma n v φ τ := by
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hT : 0 < totalStay τ := Finset.sum_pos (fun k _ => hτ k) hne
  have hc_pos : 0 < c := zero_lt_one.trans hc
  have hTc : totalStay (fun k => c * τ k) = c * totalStay τ := by
    unfold totalStay; rw [← Finset.mul_sum]
  by_cases hφz : φ = 0
  · unfold seriesGamma
    rw [ite_eq_left hφz, ite_eq_left hφz, hTc]
    have hnv : 0 < (n : ℝ) * v := mul_pos hn_pos hv
    exact div_lt_div_of_pos_left hnv (by positivity) (by nlinarith)
  · have hφ_pos : 0 < φ := lt_of_le_of_ne hφ0 (Ne.symm hφz)
    have hθ : 0 < theta φ := div_pos hφ_pos (sub_pos.mpr hφ1)
    let y : Fin n → ℝ := fun k => τ k / theta φ
    have hy : ∀ k, 0 < y k := fun k => div_pos (hτ k) hθ
    obtain ⟨e, he0, he1, he_eval, he_pos, he_zero, he_lc⟩ := symPoly_logConcave n y hy
    have he1_eq : e 1 = totalStay τ / theta φ := by
      rw [he1]; unfold totalStay; dsimp [y]; rw [Finset.sum_div]
    have he1_pos : 0 < e 1 := he_pos 1 hn
    let D : ℝ → ℝ := fun t => ∑ k ∈ Finset.range n, e (k + 1) * t ^ (k + 1)
    let N : ℝ → ℝ := fun t => ∑ k ∈ Finset.range n, e (k + 2) * t ^ (k + 1)
    have hD_prod : ∀ t : ℝ, D t = ∏ k : Fin n, (1 + t * y k) - 1 := by
      intro t
      have h0 := he_eval t
      rw [Finset.sum_range_succ'] at h0
      dsimp [D]
      rw [he0, pow_zero, mul_one] at h0
      linarith
    have hD_pos : ∀ t : ℝ, 0 < t → 0 < D t := by
      intro t ht
      dsimp [D]
      exact Finset.sum_pos
        (fun k hk => mul_pos (he_pos (k + 1) (by linarith [Finset.mem_range.mp hk]))
          (pow_pos ht (k + 1)))
        ⟨0, Finset.mem_range.mpr hn⟩
    have hDN_id : ∀ t : ℝ, e 1 * t + t * N t = D t := by
      intro t
      obtain ⟨m, rfl⟩ : ∃ m : ℕ, n = m + 1 := ⟨n - 1, by omega⟩
      dsimp [D, N]
      rw [Finset.sum_range_succ (fun k => e (k + 2) * t ^ (k + 1)) m]
      rw [he_zero (m + 2) (by omega), zero_mul, add_zero]
      rw [Finset.sum_range_succ' (fun k => e (k + 1) * t ^ (k + 1)) m]
      rw [Finset.mul_sum]
      have htail : ∑ x ∈ Finset.range m, t * (e (x + 2) * t ^ (x + 1)) =
          ∑ k ∈ Finset.range m, e (k + 1 + 1) * t ^ (k + 1 + 1) := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [pow_succ]
        ring
      rw [htail]
      ring
    have hND_le : N c * D 1 ≤ N 1 * D c := by
      have hdouble : 2 * (N 1 * D c - N c * D 1) =
          ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
            (e (i + 2) * e (j + 1) - e (j + 2) * e (i + 1)) *
              (c ^ (j + 1) - c ^ (i + 1)) := by
        have hN1D : N 1 * D c = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
            e (i + 2) * e (j + 1) * c ^ (j + 1) := by
          dsimp [N, D]
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun j _ => by simp [one_pow]; ring
        have hNcD1 : N c * D 1 = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
            e (i + 2) * e (j + 1) * c ^ (i + 1) := by
          dsimp [N, D]
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun j _ => by simp [one_pow]; ring
        have hN1D_swap : N 1 * D c = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
            e (j + 2) * e (i + 1) * c ^ (i + 1) := by
          rw [hN1D, Finset.sum_comm]
        have hNcD1_swap : N c * D 1 = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range n,
            e (j + 2) * e (i + 1) * c ^ (j + 1) := by
          rw [hNcD1, Finset.sum_comm]
        have hlhs : 2 * (N 1 * D c - N c * D 1) =
            (N 1 * D c - N c * D 1) + (N 1 * D c - N c * D 1) := by ring
        rw [hlhs]
        conv_lhs =>
          lhs
          rw [hN1D, hNcD1]
        conv_lhs =>
          rhs
          rw [hN1D_swap, hNcD1_swap]
        rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun j _ => by ring
      have hnn : 0 ≤ 2 * (N 1 * D c - N c * D 1) := by
        rw [hdouble]
        refine Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ => ?_
        rcases lt_trichotomy i j with hij | rfl | hji
        · have hlc : e (j + 2) * e (i + 1) ≤ e (i + 2) * e (j + 1) := by
            have h := he_lc (i + 1) (j + 1) (by omega)
            linarith
          have hpow : c ^ (i + 1) ≤ c ^ (j + 1) :=
            pow_le_pow_right₀ hc.le (by omega)
          exact mul_nonneg (sub_nonneg.mpr hlc) (sub_nonneg.mpr hpow)
        · simp
        · have hlc : e (i + 2) * e (j + 1) ≤ e (j + 2) * e (i + 1) := by
            have h := he_lc (j + 1) (i + 1) (by omega)
            linarith
          have hpow : c ^ (j + 1) ≤ c ^ (i + 1) :=
            pow_le_pow_right₀ hc.le (by omega)
          exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr hlc) (sub_nonpos.mpr hpow)
      linarith
    have hD1_pos : 0 < D 1 := hD_pos 1 one_pos
    have hDc_pos : 0 < D c := hD_pos c hc_pos
    have hfrac_le : N c / (e 1 * D c) ≤ N 1 / (e 1 * D 1) := by
      rw [div_le_div_iff₀ (mul_pos he1_pos hDc_pos) (mul_pos he1_pos hD1_pos)]
      nlinarith [he1_pos]
    have hP1 : prodOneAdd (theta φ) τ - 1 = D 1 := by
      rw [hD_prod 1]; unfold prodOneAdd; simp [y]
    have hPc : prodOneAdd (theta φ) (fun k => c * τ k) - 1 = D c := by
      rw [hD_prod c]; unfold prodOneAdd
      congr 1
      refine Finset.prod_congr rfl fun k _ => ?_
      dsimp [y]; ring
    have hTe1 : totalStay τ = theta φ * e 1 := by
      rw [he1_eq]; field_simp [hθ.ne']
    have hbr1 : (1 + 2 * theta φ) / (2 * totalStay τ) - 1 / D 1 =
        1 / (2 * totalStay τ) + N 1 / (e 1 * D 1) := by
      have hN1_eq : N 1 = D 1 - e 1 := by linarith [hDN_id 1]
      rw [hN1_eq, hTe1]
      field_simp [hθ.ne', he1_pos.ne', hD1_pos.ne']
      ring
    have hbrc : (1 + 2 * theta φ) / (2 * (c * totalStay τ)) - 1 / D c =
        1 / (2 * (c * totalStay τ)) + N c / (e 1 * D c) := by
      have hNc_eq : N c = (D c - e 1 * c) / c := by
        rw [eq_div_iff hc_pos.ne']
        linarith [hDN_id c]
      rw [hNc_eq, hTe1]
      field_simp [hc_pos.ne', hθ.ne', he1_pos.ne', hDc_pos.ne']
      ring
    have hterm1_lt : 1 / (2 * (c * totalStay τ)) < 1 / (2 * totalStay τ) :=
      div_lt_div_of_pos_left one_pos (by positivity) (by nlinarith)
    unfold seriesGamma
    rw [ite_eq_right hφz, ite_eq_right hφz, hTc, hPc, hP1, hbrc, hbr1]
    exact mul_lt_mul_of_pos_left (add_lt_add_of_lt_of_le hterm1_lt hfrac_le) (mul_pos hn_pos hv)

/-! ## Part (ii): Equal stays `τ`: closed form, strict increase in `n`, strict decrease in `τ` -/

/-- **Part (ii): Equal stays `τ > 0` (`n ≥ 1`, `0 ≤ φ < 1`).**
Gives the closed form `Γ_n = v / (2 τ)` at `φ = 0` and
`Γ_n = v [(1 + 2 θ)/(2 τ) − n / ((1 + τ / θ)^n − 1)]` for `0 < φ < 1`, proves strict increase
in `n` (`1 ≤ m < n`) when `0 < φ < 1`, and proves strict decrease in `τ` (`0 < τ₁ < τ₂`) for
all `0 ≤ φ < 1`. -/
theorem series_part_ii (v φ : ℝ) (hv : 0 < v) (hφ0 : 0 ≤ φ) (hφ1 : φ < 1) :
    (∀ (n : ℕ), 1 ≤ n → ∀ τ : ℝ, 0 < τ →
      (φ = 0 → seriesGamma n v φ (fun _ => τ) = v / (2 * τ)) ∧
      (0 < φ → seriesGamma n v φ (fun _ => τ) =
        v * ((1 + 2 * theta φ) / (2 * τ) - (n : ℝ) / ((1 + τ / theta φ) ^ n - 1)))) ∧
    (0 < φ → ∀ (m n : ℕ), 1 ≤ m → m < n → ∀ τ : ℝ, 0 < τ →
      seriesGamma m v φ (fun _ => τ) < seriesGamma n v φ (fun _ => τ)) ∧
    (∀ (n : ℕ), 1 ≤ n → ∀ τ₁ τ₂ : ℝ, 0 < τ₁ → τ₁ < τ₂ →
      seriesGamma n v φ (fun _ => τ₂) < seriesGamma n v φ (fun _ => τ₁)) := by
  have heval : ∀ (n : ℕ), 1 ≤ n → ∀ τ : ℝ, 0 < τ →
      (φ = 0 → seriesGamma n v φ (fun _ => τ) = v / (2 * τ)) ∧
      (0 < φ → seriesGamma n v φ (fun _ => τ) =
        v * ((1 + 2 * theta φ) / (2 * τ) - (n : ℝ) / ((1 + τ / theta φ) ^ n - 1))) := by
    intro n hn τ hτ
    have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
    have hT : totalStay (fun _ : Fin n => τ) = (n : ℝ) * τ := by
      unfold totalStay
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    refine ⟨fun h0 => ?_, fun hpos => ?_⟩
    · subst h0
      unfold seriesGamma
      rw [ite_eq_left rfl, hT]
      field_simp [hn_pos.ne', hτ.ne']
    · have hθ : 0 < theta φ := div_pos hpos (sub_pos.mpr hφ1)
      have ha1 : 1 < (1 + τ / theta φ) ^ n :=
        one_lt_pow₀ (by linarith [div_pos hτ hθ]) (by omega)
      have hP : prodOneAdd (theta φ) (fun _ : Fin n => τ) = (1 + τ / theta φ) ^ n := by
        unfold prodOneAdd
        rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
      unfold seriesGamma
      rw [ite_eq_right hpos.ne', hT, hP]
      field_simp [hn_pos.ne', hτ.ne', (sub_pos.mpr ha1).ne']
  refine ⟨heval, ?_, ?_⟩
  · intro hφ_pos m n hm hmn τ hτ
    have hn : 1 ≤ n := by omega
    rw [(heval m hm τ hτ).2 hφ_pos, (heval n hn τ hτ).2 hφ_pos]
    have hθ : 0 < theta φ := div_pos hφ_pos (sub_pos.mpr hφ1)
    let a : ℝ := 1 + τ / theta φ
    have ha : 1 < a := by dsimp [a]; linarith [div_pos hτ hθ]
    let L : ℝ := Real.log a
    have hL : 0 < L := Real.log_pos ha
    have hm_pos : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
    have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
    have hmn_real : (m : ℝ) < (n : ℝ) := by exact_mod_cast hmn
    have hinv_lt : 1 / (n : ℝ) < 1 / (m : ℝ) :=
      one_div_lt_one_div_of_lt hm_pos hmn_real
    have hN1 := Dobson.ReturnRate.N1 L hL (Set.mem_Ioi.mpr (one_div_pos.mpr hn_pos))
      (Set.mem_Ioi.mpr (one_div_pos.mpr hm_pos)) hinv_lt
    dsimp only at hN1
    have hexp_k : ∀ k : ℕ, 1 ≤ k → Real.exp (L / (1 / (k : ℝ))) = a ^ k := by
      intro k hk
      have hk_ne : (k : ℝ) ≠ 0 := (by exact_mod_cast hk : (0 : ℝ) < (k : ℝ)).ne'
      have hdiv : L / (1 / (k : ℝ)) = (k : ℝ) * L := by field_simp [hk_ne]
      rw [hdiv, Real.exp_nat_mul, Real.exp_log (zero_lt_one.trans ha)]
    rw [hexp_k m hm, hexp_k n hn] at hN1
    have ham1 : 0 < a ^ m - 1 := sub_pos.mpr (one_lt_pow₀ ha (Nat.ne_of_gt hm))
    have han1 : 0 < a ^ n - 1 := sub_pos.mpr (one_lt_pow₀ ha (Nat.ne_of_gt hn))
    have hpm : 0 < (1 / (m : ℝ)) * (a ^ m - 1) := mul_pos (one_div_pos.mpr hm_pos) ham1
    have hrec := one_div_lt_one_div_of_lt hpm hN1
    have hrm : 1 / ((1 / (m : ℝ)) * (a ^ m - 1)) = (m : ℝ) / (a ^ m - 1) := by
      field_simp [hm_pos.ne', ham1.ne']
    have hrn : 1 / ((1 / (n : ℝ)) * (a ^ n - 1)) = (n : ℝ) / (a ^ n - 1) := by
      field_simp [hn_pos.ne', han1.ne']
    rw [hrm, hrn] at hrec
    change v * ((1 + 2 * theta φ) / (2 * τ) - (m : ℝ) / (a ^ m - 1)) <
      v * ((1 + 2 * theta φ) / (2 * τ) - (n : ℝ) / (a ^ n - 1))
    exact mul_lt_mul_of_pos_left (by linarith) hv
  · intro n hn τ₁ τ₂ hτ₁ hlt
    let c : ℝ := τ₂ / τ₁
    have hc : 1 < c := (one_lt_div hτ₁).mpr hlt
    have hfun : (fun _ : Fin n => τ₂) = (fun k : Fin n => c * (fun _ : Fin n => τ₁) k) := by
      ext _; dsimp [c]; exact (div_mul_cancel₀ τ₂ hτ₁.ne').symm
    rw [hfun]
    exact series_part_iv n hn v φ hv hφ0 hφ1 (fun _ => τ₁) (fun _ => hτ₁) c hc

/-! ## Part (iii): Appending a stage of stay `𝒯 / n` on its own host -/

/-- Helper: for `n ≥ 1` and `y : Fin n → ℝ` with `y_k > 0`, `∏_k (1 + y_k) < exp (∑_k y_k)`. -/
private theorem prod_one_add_lt_exp_sum :
    ∀ (n : ℕ), 1 ≤ n → ∀ (y : Fin n → ℝ), (∀ k, 0 < y k) →
      ∏ k : Fin n, (1 + y k) < Real.exp (∑ k : Fin n, y k) := by
  intro n hn
  obtain ⟨m, rfl⟩ : ∃ m : ℕ, n = m + 1 := ⟨n - 1, by omega⟩
  clear hn
  induction m with
  | zero =>
    intro y hy
    rw [Fin.prod_univ_castSucc, Fin.sum_univ_castSucc]
    simp only [Finset.univ_eq_empty, Finset.prod_empty, one_mul, Finset.sum_empty, zero_add]
    linarith [Real.add_one_lt_exp (hy (Fin.last 0)).ne']
  | succ k ih =>
    intro y hy
    rw [Fin.prod_univ_castSucc, Fin.sum_univ_castSucc, Real.exp_add]
    have h1 := ih (fun i => y i.castSucc) (fun i => hy i.castSucc)
    have h2 : 1 + y (Fin.last (k + 1)) < Real.exp (y (Fin.last (k + 1))) := by
      linarith [Real.add_one_lt_exp (hy (Fin.last (k + 1))).ne']
    have hp1 : 0 < ∏ i : Fin (k + 1), (1 + y i.castSucc) :=
      Finset.prod_pos fun i _ => by linarith [hy i.castSucc]
    have hp2 : 0 < 1 + y (Fin.last (k + 1)) := by linarith [hy (Fin.last (k + 1))]
    exact mul_lt_mul'' h1 h2 hp1.le hp2.le

/-- **Part (iii): Appending stage `τ_{n+1} = 𝒯 / n` (`Fin.snoc τ (meanStay τ)`).**
Leaves `seriesGamma` unchanged at `φ = 0`, and for `0 < φ < 1` increases `seriesGamma` by
`v * (P * (x − 1) + 1) / ((P − 1) * (P * (1 + x / n) − 1)) > 0` (`x = 𝒯 / θ`). -/
theorem series_part_iii (n : ℕ) (hn : 1 ≤ n) (v φ : ℝ) (hv : 0 < v)
    (_hφ0 : 0 ≤ φ) (hφ1 : φ < 1) (τ : Fin n → ℝ) (hτ : ∀ k, 0 < τ k) :
    let τ_ext : Fin (n + 1) → ℝ := Fin.snoc τ (meanStay τ)
    (φ = 0 → seriesGamma (n + 1) v φ τ_ext - seriesGamma n v φ τ = 0) ∧
      (0 < φ →
        let θ := theta φ
        let P := prodOneAdd θ τ
        let x := totalStay τ / θ
        seriesGamma (n + 1) v φ τ_ext - seriesGamma n v φ τ =
          v * (P * (x - 1) + 1) / ((P - 1) * (P * (1 + x / (n : ℝ)) - 1)) ∧
        0 < seriesGamma (n + 1) v φ τ_ext - seriesGamma n v φ τ) := by
  intro τ_ext
  have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hne : (Finset.univ : Finset (Fin n)).Nonempty := ⟨⟨0, by omega⟩, Finset.mem_univ _⟩
  have hT : 0 < totalStay τ := Finset.sum_pos (fun k _ => hτ k) hne
  have hT_ext : totalStay τ_ext = totalStay τ * (((n + 1 : ℕ) : ℝ) / (n : ℝ)) := by
    simp only [τ_ext, totalStay, meanStay, Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last]
    push_cast
    field_simp [hn_pos.ne']
  refine ⟨fun h0 => ?_, fun hpos => ?_⟩
  · subst h0
    unfold seriesGamma
    rw [ite_eq_left rfl, ite_eq_left rfl, hT_ext]
    push_cast
    field_simp [hn_pos.ne', (by positivity : (n : ℝ) + 1 ≠ 0), hT.ne']
    ring
  · dsimp only
    let θ := theta φ
    let P := prodOneAdd θ τ
    let x := totalStay τ / θ
    have hθ : 0 < θ := div_pos hpos (sub_pos.mpr hφ1)
    have hx : 0 < x := div_pos hT hθ
    have hP_gt1 : 1 < P := by
      have hB1 := Dobson.Band.one_add_sum_le_prod_one_add Finset.univ
        (fun k => τ k / θ) (fun k _ => (div_pos (hτ k) hθ).le)
      have hsum : ∑ k : Fin n, τ k / θ = x := by
        dsimp [x]; unfold totalStay; rw [Finset.sum_div]
      dsimp [P, prodOneAdd]
      linarith
    have hP_ext : prodOneAdd θ τ_ext = P * (1 + x / (n : ℝ)) := by
      dsimp [τ_ext, P, x]
      unfold prodOneAdd meanStay
      rw [Fin.prod_univ_castSucc]
      simp only [Fin.snoc_castSucc, Fin.snoc_last]
      congr 1
      field_simp [hn_pos.ne', hθ.ne']
    have hPext_gt1 : 1 < P * (1 + x / (n : ℝ)) := by
      have h1 : 1 < 1 + x / (n : ℝ) := by linarith [div_pos hx hn_pos]
      nlinarith
    have hdiff_eq : seriesGamma (n + 1) v φ τ_ext - seriesGamma n v φ τ =
        v * (P * (x - 1) + 1) / ((P - 1) * (P * (1 + x / (n : ℝ)) - 1)) := by
      unfold seriesGamma
      rw [ite_eq_right hpos.ne', ite_eq_right hpos.ne', hT_ext]
      let Q : ℝ := P * (1 + x / (n : ℝ))
      have hQ_gt1 : 1 < Q := hPext_gt1
      have hnQ : (n : ℝ) * Q = P * ((n : ℝ) + x) := by
        dsimp [Q]; field_simp [hn_pos.ne']
      change (((n + 1 : ℕ) : ℝ) * v *
            ((1 + 2 * θ) / (2 * (totalStay τ * (((n + 1 : ℕ) : ℝ) / (n : ℝ)))) -
              1 / (prodOneAdd θ τ_ext - 1)) -
          (n : ℝ) * v * ((1 + 2 * θ) / (2 * totalStay τ) - 1 / (P - 1))) =
        v * (P * (x - 1) + 1) / ((P - 1) * (Q - 1))
      have h1 : (((n + 1 : ℕ) : ℝ) * v *
            ((1 + 2 * θ) / (2 * (totalStay τ * (((n + 1 : ℕ) : ℝ) / (n : ℝ)))))) =
          (n : ℝ) * v * ((1 + 2 * θ) / (2 * totalStay τ)) := by
        push_cast
        field_simp [hn_pos.ne', (by positivity : (n : ℝ) + 1 ≠ 0), hT.ne']
      have h2 : (n : ℝ) * v * (1 / (P - 1)) - ((n + 1 : ℕ) : ℝ) * v * (1 / (Q - 1)) =
          v * ((n : ℝ) * (Q - 1) - ((n + 1 : ℕ) : ℝ) * (P - 1)) / ((P - 1) * (Q - 1)) := by
        field_simp [(sub_pos.mpr hP_gt1).ne', (sub_pos.mpr hQ_gt1).ne']
      have h3 : (n : ℝ) * (Q - 1) - ((n + 1 : ℕ) : ℝ) * (P - 1) = P * (x - 1) + 1 := by
        push_cast
        linarith [hnQ]
      have hP_ext_Q : prodOneAdd θ τ_ext = Q := hP_ext
      rw [hP_ext_Q]
      have hsplit_sub : (((n + 1 : ℕ) : ℝ) * v *
            ((1 + 2 * θ) / (2 * (totalStay τ * (((n + 1 : ℕ) : ℝ) / (n : ℝ)))) -
              1 / (Q - 1)) -
          (n : ℝ) * v * ((1 + 2 * θ) / (2 * totalStay τ) - 1 / (P - 1))) =
          ((((n + 1 : ℕ) : ℝ) * v *
            ((1 + 2 * θ) / (2 * (totalStay τ * (((n + 1 : ℕ) : ℝ) / (n : ℝ)))))) -
            (n : ℝ) * v * ((1 + 2 * θ) / (2 * totalStay τ))) +
          ((n : ℝ) * v * (1 / (P - 1)) - ((n + 1 : ℕ) : ℝ) * v * (1 / (Q - 1))) := by ring
      rw [hsplit_sub, h1, sub_self, zero_add, h2, h3]
    refine ⟨hdiff_eq, ?_⟩
    rw [hdiff_eq]
    have hden_pos : 0 < (P - 1) * (P * (1 + x / (n : ℝ)) - 1) :=
      mul_pos (sub_pos.mpr hP_gt1) (sub_pos.mpr hPext_gt1)
    have hnum_pos : 0 < P * (x - 1) + 1 := by
      by_cases hx1 : 1 ≤ x
      · nlinarith
      · have hx_lt1 : x < 1 := not_le.mp hx1
        have hP_lt_exp : P < Real.exp x := by
          have hp := prod_one_add_lt_exp_sum n hn (fun k => τ k / θ)
            (fun k => div_pos (hτ k) hθ)
          have hsum : ∑ k : Fin n, τ k / θ = x := by
            dsimp [x]; unfold totalStay; rw [Finset.sum_div]
          rwa [hsum] at hp
        have h1mx : 0 < 1 - x := sub_pos.mpr hx_lt1
        have hexp_neg : 1 - x < Real.exp (-x) := by
          linarith [Real.add_one_lt_exp (neg_ne_zero.mpr hx.ne')]
        have hprod_lt1 : (1 - x) * Real.exp x < 1 := by
          calc (1 - x) * Real.exp x
              < Real.exp (-x) * Real.exp x := mul_lt_mul_of_pos_right hexp_neg (Real.exp_pos x)
            _ = Real.exp (-x + x) := (Real.exp_add (-x) x).symm
            _ = 1 := by simp
        have h1mP : (1 - x) * P < 1 :=
          lt_trans (mul_lt_mul_of_pos_left hP_lt_exp h1mx) hprod_lt1
        linarith
    exact div_pos (mul_pos hv hnum_pos) hden_pos

/-! ## Part (v): Per-step coefficient `f_n` at equal rates (`τ > 1`, `0 < φ < 1`) -/

/-- Per-step series cost coefficient at equal stays `τ`:
`f_n = (1 + 2 θ) / (2 n τ²) − 1 / (τ ((1 + τ / θ)^n − 1))`. -/
noncomputable def fStep (φ τ : ℝ) (n : ℕ) : ℝ :=
  (1 + 2 * theta φ) / (2 * (n : ℝ) * τ ^ 2) - 1 / (τ * ((1 + τ / theta φ) ^ n - 1))

/-- Algebraic helper for Part (v): for `a > 1` and `n ≥ 1`,
`(n + 2) * a * (a ^ n - 1) < n * (a ^ (n + 2) - 1)`. -/
private theorem poly_step_ineq (a : ℝ) (ha : 1 < a) (n : ℕ) (hn : 1 ≤ n) :
    ((n + 2 : ℕ) : ℝ) * a * (a ^ n - 1) < (n : ℝ) * (a ^ (n + 2) - 1) := by
  have ha1 : 0 < a - 1 := sub_pos.mpr ha
  have hgeom : ∀ m : ℕ, a ^ m - 1 = (a - 1) * ∑ k ∈ Finset.range m, a ^ k := fun m =>
    (geom_sum_mul a m).symm.trans (mul_comm _ _)
  have hsum_id : (n : ℝ) * (∑ k ∈ Finset.range (n + 2), a ^ k) -
      ((n + 2 : ℕ) : ℝ) * a * (∑ k ∈ Finset.range n, a ^ k) =
      ∑ k ∈ Finset.range n, (a ^ (k + 1) - 1) * (a ^ (n - k) - 1) := by
    have hsplit : ∑ k ∈ Finset.range (n + 2), a ^ k =
        1 + a * (∑ k ∈ Finset.range n, a ^ k) + a ^ (n + 1) := by
      rw [Finset.sum_range_succ (fun k => a ^ k) (n + 1)]
      rw [Finset.sum_range_succ' (fun k => a ^ k) n]
      rw [Finset.mul_sum]
      have hpow : ∑ k ∈ Finset.range n, a ^ (k + 1) = ∑ k ∈ Finset.range n, a * a ^ k :=
        Finset.sum_congr rfl fun k _ => by rw [pow_succ, mul_comm]
      rw [hpow, pow_zero]
      ring
    have hrev : ∑ k ∈ Finset.range n, a ^ (n - k) = a * ∑ k ∈ Finset.range n, a ^ k := by
      have hr := Finset.sum_range_reflect (fun j => a ^ (j + 1)) n
      have hlhs : ∑ j ∈ Finset.range n, a ^ (n - 1 - j + 1) =
          ∑ k ∈ Finset.range n, a ^ (n - k) := by
        refine Finset.sum_congr rfl fun j hj => ?_
        have hjlt : j < n := Finset.mem_range.mp hj
        congr 1; omega
      rw [← hlhs, hr, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j _ => by rw [pow_succ, mul_comm]
    have hexp : ∑ k ∈ Finset.range n, (a ^ (k + 1) - 1) * (a ^ (n - k) - 1) =
        ∑ k ∈ Finset.range n, (a ^ (n + 1) + 1 - a * a ^ k - a ^ (n - k)) := by
      refine Finset.sum_congr rfl fun k hk => ?_
      have hklt : k < n := Finset.mem_range.mp hk
      have hadd : (k + 1) + (n - k) = n + 1 := by omega
      have hmul_pow : a ^ (k + 1) * a ^ (n - k) = a ^ (n + 1) := by
        rw [← pow_add, hadd]
      linear_combination hmul_pow
    rw [hexp]
    simp only [sub_eq_add_neg, Finset.sum_add_distrib, Finset.sum_neg_distrib, Finset.sum_const,
      Finset.card_range, nsmul_eq_mul, ← Finset.mul_sum, hrev, hsplit]
    push_cast
    ring
  have hpos_sum : 0 < ∑ k ∈ Finset.range n, (a ^ (k + 1) - 1) * (a ^ (n - k) - 1) := by
    refine Finset.sum_pos (fun k hk => ?_) ⟨0, Finset.mem_range.mpr hn⟩
    have hklt : k < n := Finset.mem_range.mp hk
    exact mul_pos (sub_pos.mpr (one_lt_pow₀ ha (by omega)))
      (sub_pos.mpr (one_lt_pow₀ ha (by omega)))
  have hmul_pos := mul_pos ha1 hpos_sum
  rw [← hsum_id] at hmul_pos
  have hrew : (a - 1) *
      ((n : ℝ) * (∑ k ∈ Finset.range (n + 2), a ^ k) -
        ((n + 2 : ℕ) : ℝ) * a * (∑ k ∈ Finset.range n, a ^ k)) =
      (n : ℝ) * (a ^ (n + 2) - 1) - ((n + 2 : ℕ) : ℝ) * a * (a ^ n - 1) := by
    rw [hgeom (n + 2), hgeom n]
    ring
  linarith

/-- **Part (v): Single-crossing increments of `f_n` and exact criterion for `sup_{n ≥ 2} f_n > f_1`.**
For `τ > 1` and `0 < φ < 1`:
1. `n (n + 1) (f_{n+1} − f_n)` strictly decreases on `n ≥ 1`;
2. `f_{n+1} − f_n` changes sign from positive to nonpositive at most once: if `f_{m+1} ≤ f_m`
   for some `m ≥ 1`, then `f_{n+1} < f_n` for all `n > m`;
3. `(∃ n ≥ 2, f_n > f_1) ↔ f_2 > f_1 ↔ theta φ > τ / (2 * (τ − 1))`. -/
theorem series_part_v (φ τ : ℝ) (hφ0 : 0 < φ) (hφ1 : φ < 1) (hτ : 1 < τ) :
    let d : ℕ → ℝ := fun n => (n : ℝ) * ((n + 1 : ℕ) : ℝ) * (fStep φ τ (n + 1) - fStep φ τ n)
    (∀ m n : ℕ, 1 ≤ m → m < n → d n < d m) ∧
    (∀ m n : ℕ, 1 ≤ m → m < n → fStep φ τ (m + 1) ≤ fStep φ τ m →
      fStep φ τ (n + 1) < fStep φ τ n) ∧
    ((∃ n : ℕ, 2 ≤ n ∧ fStep φ τ n > fStep φ τ 1) ↔ fStep φ τ 2 > fStep φ τ 1) ∧
    (fStep φ τ 2 > fStep φ τ 1 ↔ theta φ > τ / (2 * (τ - 1))) := by
  intro d
  have hτ_pos : 0 < τ := zero_lt_one.trans hτ
  have hθ : 0 < theta φ := div_pos hφ0 (sub_pos.mpr hφ1)
  let a : ℝ := 1 + τ / theta φ
  have ha : 1 < a := by dsimp [a]; linarith [div_pos hτ_pos hθ]
  have ha_pos : 0 < a := zero_lt_one.trans ha
  have hd_eq : ∀ n : ℕ, 1 ≤ n →
      d n = -(1 + 2 * theta φ) / (2 * τ ^ 2) +
        ((a - 1) / τ) * (((n : ℝ) * ((n + 1 : ℕ) : ℝ) * a ^ n) /
          ((a ^ n - 1) * (a ^ (n + 1) - 1))) := by
    intro n hn
    have hn_pos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
    have han1 : 0 < a ^ n - 1 := sub_pos.mpr (one_lt_pow₀ ha (by omega))
    have han2 : 0 < a ^ (n + 1) - 1 := sub_pos.mpr (one_lt_pow₀ ha (by omega))
    have hf : ∀ k : ℕ, fStep φ τ k =
        (1 + 2 * theta φ) / (2 * (k : ℝ) * τ ^ 2) - 1 / (τ * (a ^ k - 1)) := fun _ => rfl
    dsimp [d]
    rw [hf (n + 1), hf n]
    have hapow : a ^ (n + 1) = a ^ n * a := pow_succ a n
    have han2' : a ^ n * a - 1 ≠ 0 := by rw [← hapow]; exact han2.ne'
    rw [hapow]
    push_cast
    field_simp [hn_pos.ne', (by positivity : (n : ℝ) + 1 ≠ 0), hτ_pos.ne', han1.ne', han2']
    ring
  have hd_step : ∀ n : ℕ, 1 ≤ n → d (n + 1) < d n := by
    intro n hn
    rw [hd_eq (n + 1) (by omega), hd_eq n hn]
    have han0 : 0 < a ^ n - 1 := sub_pos.mpr (one_lt_pow₀ ha (by omega))
    have han1 : 0 < a ^ (n + 1) - 1 := sub_pos.mpr (one_lt_pow₀ ha (by omega))
    have han2 : 0 < a ^ (n + 2) - 1 := sub_pos.mpr (one_lt_pow₀ ha (by omega))
    have hpoly := poly_step_ineq a ha n hn
    have hdiv : (((n + 2 : ℕ) : ℝ) * a) / (a ^ (n + 2) - 1) < (n : ℝ) / (a ^ n - 1) :=
      (div_lt_div_iff₀ han2 han0).mpr hpoly
    have hcommon_pos : 0 < (((n + 1 : ℕ) : ℝ) * a ^ n) / (a ^ (n + 1) - 1) :=
      div_pos (mul_pos (by positivity) (pow_pos ha_pos n)) han1
    have hmul := mul_lt_mul_of_pos_left hdiv hcommon_pos
    have hlhs : (((n + 1 : ℕ) : ℝ) * a ^ n / (a ^ (n + 1) - 1)) *
        ((((n + 2 : ℕ) : ℝ) * a) / (a ^ (n + 2) - 1)) =
        (((n + 1 : ℕ) : ℝ) * ((n + 1 + 1 : ℕ) : ℝ) * a ^ (n + 1)) /
          ((a ^ (n + 1) - 1) * (a ^ (n + 1 + 1) - 1)) := by
      rw [pow_succ]
      field_simp [han1.ne', han2.ne']
    have hrhs : (((n + 1 : ℕ) : ℝ) * a ^ n / (a ^ (n + 1) - 1)) *
        ((n : ℝ) / (a ^ n - 1)) =
        ((n : ℝ) * ((n + 1 : ℕ) : ℝ) * a ^ n) / ((a ^ n - 1) * (a ^ (n + 1) - 1)) := by
      field_simp [han0.ne', han1.ne']
    rw [hlhs, hrhs] at hmul
    have hcoeff_pos : 0 < (a - 1) / τ := div_pos (sub_pos.mpr ha) hτ_pos
    linarith [mul_lt_mul_of_pos_left hmul hcoeff_pos]
  have hd_anti : ∀ m n : ℕ, 1 ≤ m → m < n → d n < d m := by
    intro m n hm hmn
    induction n, hmn using Nat.le_induction with
    | base => exact hd_step m hm
    | succ k _ ih => exact lt_trans (hd_step k (by omega)) ih
  have hsign : ∀ m n : ℕ, 1 ≤ m → m < n → fStep φ τ (m + 1) ≤ fStep φ τ m →
      fStep φ τ (n + 1) < fStep φ τ n := by
    intro m n hm hmn hle
    have hdm_nonpos : d m ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by positivity) (sub_nonpos.mpr hle)
    have hdn_neg : d n < 0 := lt_of_lt_of_le (hd_anti m n hm hmn) hdm_nonpos
    have hn_fac_pos : 0 < (n : ℝ) * ((n + 1 : ℕ) : ℝ) := by
      have : (0 : ℝ) < (n : ℝ) := by exact_mod_cast (show 1 ≤ n by omega)
      positivity
    dsimp [d] at hdn_neg
    have hsub_neg : fStep φ τ (n + 1) - fStep φ τ n < 0 := by
      by_contra hnn
      have : 0 ≤ (n : ℝ) * ((n + 1 : ℕ) : ℝ) * (fStep φ τ (n + 1) - fStep φ τ n) :=
        mul_nonneg hn_fac_pos.le (not_lt.mp hnn)
      linarith
    linarith
  refine ⟨hd_anti, hsign, ?_, ?_⟩
  · constructor
    · rintro ⟨n, hn2, hgt⟩
      by_contra hnot
      have hf2_le : fStep φ τ 2 ≤ fStep φ τ 1 := not_lt.mp hnot
      have hle_all : ∀ k : ℕ, 1 ≤ k → fStep φ τ k ≤ fStep φ τ 1 := by
        intro k hk
        induction k, hk using Nat.le_induction with
        | base => exact le_rfl
        | succ r hr ihr =>
          rcases eq_or_lt_of_le hr with rfl | h1r
          · exact hf2_le
          · exact (hsign 1 r le_rfl h1r hf2_le).le.trans ihr
      linarith [hle_all n (by omega)]
    · intro h21
      exact ⟨2, le_rfl, h21⟩
  · have hf21 : fStep φ τ 2 - fStep φ τ 1 =
        (2 * theta φ * (τ - 1) - τ) / (4 * τ ^ 2 * (2 * theta φ + τ)) := by
      unfold fStep
      have ha1_eq : (1 + τ / theta φ) ^ 1 - 1 = τ / theta φ := by ring
      have ha2_eq : (1 + τ / theta φ) ^ 2 - 1 = τ * (2 * theta φ + τ) / theta φ ^ 2 := by
        field_simp [hθ.ne']
        ring
      have h2θτ_pos : 0 < 2 * theta φ + τ := by positivity
      rw [ha1_eq, ha2_eq]
      push_cast
      field_simp [hτ_pos.ne', hθ.ne', h2θτ_pos.ne']
      ring
    have hden_pos : 0 < 4 * τ ^ 2 * (2 * theta φ + τ) := by positivity
    have h2τ1_pos : 0 < 2 * (τ - 1) := by linarith
    constructor
    · intro hgt
      have hdiff_pos : 0 < fStep φ τ 2 - fStep φ τ 1 := sub_pos.mpr hgt
      rw [hf21, div_pos_iff_of_pos_right hden_pos] at hdiff_pos
      rw [gt_iff_lt, div_lt_iff₀ h2τ1_pos]
      linarith
    · intro hθgt
      rw [gt_iff_lt, div_lt_iff₀ h2τ1_pos] at hθgt
      have hdiff_pos : 0 < fStep φ τ 2 - fStep φ τ 1 := by
        rw [hf21]
        exact div_pos (by linarith) hden_pos
      linarith

end Dobson.Coverage1Series

#print axioms Dobson.Coverage1Series.one_add_sum_lt_prod_one_add_of_two_le
#print axioms Dobson.Coverage1Series.series_part_i
#print axioms Dobson.Coverage1Series.series_part_i_nonmonotone_example
#print axioms Dobson.Coverage1Series.series_part_iv
#print axioms Dobson.Coverage1Series.series_part_ii
#print axioms Dobson.Coverage1Series.series_part_iii
#print axioms Dobson.Coverage1Series.series_part_v
