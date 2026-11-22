"""Two of secondary's illustrations recomputed for a host that enters two steps of the cycle with opposite signs.

secondary's reading of the transmission biology (verification/review-draft-5.md, section 2): in Borrelia, mouse abundance
raises the nymph-to-mouse step (kappa between 0 and 1) and lowers the per-mouse larval burden on the mouse-to-larva
step (kappa between -1 and 0); in Echinococcus multilocularis, vole abundance multiplies the egg-to-vole step
(kappa = 1) and, when fox predation saturates, divides the vole-to-fox step (kappa between -1 and 0). The table in
the manuscript took kappa = 1 on one step and 0 on the other. This script gives the second-order loss per generation
from Proposition 4a (notes/proofs.md, section 11; function closed_form of checks/one_host_two_transfers.py) for
kappa on the second step equal to 0, -0.5 and -1, with secondary's discrete parameterisation (time step = shortest
stay). Published inputs measured for other purposes; nothing is fitted; these are illustrations.

Claims:
  Z1  With kappa = 0 on the second step the function reproduces secondary's discrete values with memory
      (Table A of verification/systems/SYSTEMS.md: 0.10431, 0.39368, 0.49344) to 4 significant figures.
  Z2  (reported, no criterion) the values at kappa = -0.5 and -1.

Run: python3 -I checks/real_systems_two_transfers.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
spec = importlib.util.spec_from_file_location("w", HERE / "one_host_two_transfers.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

SYSTEMS = [  # name, stays in steps, index of the step with kappa = +1, index of the step with kappa <= 0, sigma^2, phi, secondary's value
    ("B. burgdorferi, oak forest (mouse 1, tick 2.273 steps of 4.4 months)", [1.0, 2.273], 1, 0, 0.4462, 0.30, 0.10431),
    ("B. burgdorferi, northern edge", [1.0, 2.273], 1, 0, 1.4161, 0.50, 0.39368),
    ("E. multilocularis (fox 1, egg 1.25, vole 2 steps of 2 months)", [1.0, 1.25, 2.0], 1, 2, 1.3783, 0.75, 0.49344),
]


def main():
    fails = 0
    for name, stays, i_pos, i_neg, s2, phi, secondary in SYSTEMS:
        T = sum(stays)
        vals = [s2 * T * w.closed_form(stays, i_pos, i_neg, 1.0, k, phi) for k in (0.0, -0.5, -1.0)]
        ok = abs(vals[0] / secondary - 1) < 5e-4
        fails += 0 if ok else 1
        print(f"{name}: loss per generation with the second step at kappa = 0: {vals[0]:.4f} (secondary {secondary:.4f}) [{'ok' if ok else 'FAIL'} Z1]; "
              f"at -0.5: {vals[1]:.4f}; at -1: {vals[2]:.4f}")
    print("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
