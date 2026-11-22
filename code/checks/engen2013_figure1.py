"""Our closed form against a published figure: Engen et al. (2013), Methods in Ecology and Evolution 4:573-584, Fig. 1.

Claim checked: for the two models of that figure whose parameters are printed in full (the model of their Fig. 2,
solid line; and the seven-class model, dotted line), the closed form of `notes/general-ring-v2.md`, B1, gives the
published variance term (the horizontal lines) and the published autocorrelation term -tau/lambda (the curves).

Their model is an age-structured loop: survival p_i from class i to class i+1, a terminal class with annual survival
p_a, and recruitment F_t from the terminal class into class 1, F_t a first-order autoregressive series with mean f,
variance sigma_f^2 and autocorrelation beta = exp(-1/T_f). In our terms: a ring in which only the last stage has
staying, and one fluctuating transfer with relative variance sigma_f^2/f^2.

The published values are READ FROM THE FIGURE BY EYE (tertiary, 14 November 2026) on a rendering of the page at
220 dots per inch, at the points where the published curves have their kinks (T_f = 1 and the half-integers), to
about +-0.0001 on an axis whose labelled ticks are 0.001 apart. Tolerance, stated beforehand: 0.0002. This is a
comparison with a picture, not with numbers supplied by the authors, and it carries that weight.

History, for honesty: a first version of this script compared at T_f = 2, 5 and 10 with "readings" that I wrote down
after seeing our own values. I then rendered the figure properly, read it at its kinks before looking at our values
at those points, and replaced the readings with the ones below.

Run: python3 -I checks/engen2013_figure1.py
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import theory  # noqa: E402

MODELS = {
    "four classes (their Figs 1 solid, 2)": dict(s=[0, 0, 0, 0.7], g=[0.7, 0.9, 0.8, 0.5], var=0.03, flat=0.0014,
                                                 curve={1: 0.0015, 1.5: 0.0022, 2.5: 0.0033, 3.5: 0.0039, 4.5: 0.00425, 5.5: 0.0045, 9.5: 0.0050, 19.5: 0.0053}),
    "seven classes (their Fig. 1 dotted)": dict(s=[0] * 6 + [0.7], g=[0.8, 0.95, 0.95, 0.95, 0.9, 0.9, 0.5], var=0.03, flat=0.00068,
                                                curve={1: 0.00095, 2.5: 0.00225, 3.5: 0.0029, 4.5: 0.0033, 5.5: 0.0036, 9.5: 0.00415, 19.5: 0.0046}),
}


def main():
    worst = 0.0
    for name, mdl in MODELS.items():
        s, g = mdl["s"], mdl["g"]
        n = len(s)
        lam, m, tau, T = theory.ring_quantities(s, g)
        rel = mdl["var"] / g[-1] ** 2
        sig = [0.0] * (n - 1) + [rel]
        flat = rel / (2 * T * T)
        print(f"{name}: lambda = {lam:.4f}, generation time {T:.3f}")
        print(f"   variance term: closed form {flat:.5f}; read from the figure {mdl['flat']:.4f}")
        worst = max(worst, abs(flat - mdl["flat"]))
        for tf, seen in mdl["curve"].items():
            ours = float(theory.cost_transfer(s, g, sig, math.exp(-1 / tf))[n - 1]) - flat
            print(f"   T_f = {tf:4.1f}: autocorrelation term, closed form {ours:.5f}; read from the figure {seen:.5f}; difference {ours - seen:+.5f}")
            worst = max(worst, abs(ours - seen))
    ok = worst <= 0.0002
    print(f"\nlargest difference from the values read off the figure: {worst:.5f} (tolerance 0.0002) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
