import Mathlib

/-!
# The rejection region of a rank test on six sites (items U1 and U2)

`S π = ∑_{i : Fin 6} (i − π i)²` for a permutation `π` of `Fin 6` (`Equiv.Perm (Fin 6)`),
with `i` and `π i` read as integers. The counts in U1 are cardinalities of
`Finset.univ.filter` over all 720 permutations.
-/

namespace Dobson.Cutoff

/-- `S(π) = ∑_{i : Fin 6} ((i : ℤ) − (π i : ℤ))²`. -/
def S (π : Equiv.Perm (Fin 6)) : ℤ :=
  ∑ i : Fin 6, ((i : ℤ) - (π i : ℤ)) ^ 2

set_option maxRecDepth 100000 in
/-- **U1, cutoff 64.** Exactly 21 permutations of `Fin 6` have `S(π) ≥ 64`. -/
theorem U1_64 :
    (Finset.univ.filter (fun π : Equiv.Perm (Fin 6) => S π ≥ 64)).card = 21 := by
  decide +kernel

set_option maxRecDepth 100000 in
/-- **U1, cutoff 66.** Exactly 12 permutations of `Fin 6` have `S(π) ≥ 66`. -/
theorem U1_66 :
    (Finset.univ.filter (fun π : Equiv.Perm (Fin 6) => S π ≥ 66)).card = 12 := by
  decide +kernel

set_option maxRecDepth 100000 in
/-- **U1, cutoff 62.** Exactly 37 permutations of `Fin 6` have `S(π) ≥ 62`. -/
theorem U1_62 :
    (Finset.univ.filter (fun π : Equiv.Perm (Fin 6) => S π ≥ 62)).card = 37 := by
  decide +kernel

/-- **U1.** The number of permutations `π` of `Fin 6` with `S(π) ≥ 64` is 21, the number with
`S(π) ≥ 66` is 12, and the number with `S(π) ≥ 62` is 37. -/
theorem U1 :
    (Finset.univ.filter (fun π : Equiv.Perm (Fin 6) => S π ≥ 64)).card = 21
      ∧ (Finset.univ.filter (fun π : Equiv.Perm (Fin 6) => S π ≥ 66)).card = 12
      ∧ (Finset.univ.filter (fun π : Equiv.Perm (Fin 6) => S π ≥ 62)).card = 37 :=
  ⟨U1_64, U1_66, U1_62⟩

/-- **U2.** `1 − 64/35 = −29/35`, and `−29/35 > −829/1000`. -/
theorem U2 : (1 : ℚ) - (64 : ℚ) / 35 = -29 / 35 ∧ (-29 : ℚ) / 35 > -829 / 1000 := by
  norm_num

end Dobson.Cutoff

#print axioms Dobson.Cutoff.U1_64
#print axioms Dobson.Cutoff.U1_66
#print axioms Dobson.Cutoff.U1_62
#print axioms Dobson.Cutoff.U1
#print axioms Dobson.Cutoff.U2
