#!/usr/bin/env python3
"""High-precision reference D for the six Gate (ii) R16 cells (auxiliary ground-truth check).

Does NOT replace the registered 32-path Gate (ii) run in `run_gates.py`; provides
tight Monte Carlo reference values (512 independent paths, mean-zero host control
variate where applicable) alongside analytical second-order D2.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

sys.dont_write_bytecode = True
HERE = (__import__('pathlib').Path(__file__).resolve().parent / '.').resolve()
sys.path.insert(0, str(HERE))

from run_gates import R16_CSV, compile_r16_world_model, r16_analytical_D2
from sim_world import generate_bounded_refresh_hosts, solve_threshold_D


def main() -> None:
    with R16_CSV.open(newline="", encoding="utf-8") as fh:
        archive = {r["id"]: r for r in csv.DictReader(fh)}

    ids = ["0c-00000", "0c-01000", "0c-00100", "0c-00010", "0c-00001", "0f-10000"]
    out: dict[str, dict[str, float]] = {}
    for idx, cid in enumerate(ids):
        row = archive[cid]
        mdl = compile_r16_world_model(row)
        fine = row["fine"] == "True"
        dt = 0.25 if fine else 1.0
        sig = float(row["sigma"])
        phi_step = float(row["phi"]) ** dt
        d2 = r16_analytical_D2(mdl, sig, phi_step)

        n_reps = 256 if fine else 512
        rec = 32_768 * (4 if fine else 1)
        burn = 2_048 * (4 if fine else 1)
        h = generate_bounded_refresh_hosts(
            n_steps=rec + burn,
            n_reps=n_reps,
            sigma=sig,
            phi=phi_step,
            seed=359000 + idx,
            n_drivers=mdl.n_drivers,
            n_patches=mdl.n_patches,
            low_level=0.05,
        )
        # Every bounded channel has exact stationary mean 1.0, so any constant cv_weight is unbiased
        _, v0, w0 = mdl.perron_at_mean_host(mdl.critical_tilt_mean_host())
        b_step = float(w0 @ (mdl.U1[0] + pow(2.718281828459045, mdl.critical_tilt_mean_host()) * mdl.F1[0]) @ v0)
        res = solve_threshold_D(mdl, h, burn_steps=burn, cv_weight=b_step, init_offset=d2, tol=1e-10)
        out[cid] = {
            "D_highprec": float(res["D"]),
            "SE_highprec": float(res["se"]),
            "D2_analytical": d2,
            "n_paths": n_reps,
        }
        print(f"{cid}: D_highprec = {res['D']:.8f} +/- {res['se']:.8f}  (D2 = {d2:.8f})", flush=True)

    (HERE / "r16_highprec_reference.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
