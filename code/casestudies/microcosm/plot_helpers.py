"""Pure-Python SVG and pgfplots (.tex + .csv) figure writers for `microcosm`.

Matplotlib is broken on this machine (missing dateutil; CASESTUDY_SPEC.md Rule 7).
This helper uses only the Python standard library and writes publication-grade
vector `.svg` files alongside `.csv` data tables and standalone `pgfplots` `.tex` files.
"""
from __future__ import annotations

import csv
import math
import os
from html import escape


def write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


class SvgCanvas:
    def __init__(self, width: int = 860, height: int = 400, bg: str = "#ffffff") -> None:
        self.width = width
        self.height = height
        self.elems: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>',
        ]

    def line(self, x1: float, y1: float, x2: float, y2: float,
             stroke: str = "#222222", width: float = 1.2, dash: str = "") -> None:
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.elems.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}"{da}/>'
        )

    def rect(self, x: float, y: float, w: float, h: float,
             fill: str = "none", stroke: str = "#222222", width: float = 1.0, opacity: float = 1.0) -> None:
        op = f' fill-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
        self.elems.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{op}/>'
        )

    def circle(self, cx: float, cy: float, r: float = 3.8,
               fill: str = "#1f77b4", stroke: str = "#ffffff", width: float = 0.8, opacity: float = 1.0) -> None:
        op = f' fill-opacity="{opacity:.2f}"' if opacity < 1.0 else ""
        self.elems.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{op}/>'
        )

    def polyline(self, pts: list[tuple[float, float]],
                 stroke: str = "#1f77b4", width: float = 2.0, dash: str = "", fill: str = "none") -> None:
        if not pts:
            return
        pstr = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.elems.append(
            f'<polyline points="{pstr}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{da}/>'
        )

    def polygon(self, pts: list[tuple[float, float]], fill: str = "#1f77b4", opacity: float = 0.18) -> None:
        if not pts:
            return
        pstr = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.elems.append(
            f'<polygon points="{pstr}" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="none"/>'
        )

    def text(self, x: float, y: float, txt: str, size: float = 11.5,
             anchor: str = "middle", weight: str = "normal", fill: str = "#111111", rotate: float = 0.0) -> None:
        tr = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if abs(rotate) > 1e-6 else ""
        self.elems.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size:.1f}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{tr}>{escape(txt)}</text>'
        )

    def axes(self, x0: float, y0: float, w: float, h: float,
             xmin: float, xmax: float, ymin: float, ymax: float,
             xticks: list[tuple[float, str]], yticks: list[tuple[float, str]],
             xlabel: str, ylabel: str, title: str) -> tuple[object, object]:
        def sx(val: float) -> float:
            return x0 + (val - xmin) / (xmax - xmin) * w

        def sy(val: float) -> float:
            return y0 + h - (val - ymin) / (ymax - ymin) * h

        # Subtle gridlines
        for yv, _ in yticks:
            yy = sy(yv)
            self.line(x0, yy, x0 + w, yy, stroke="#eeeeee", width=1.0)
        self.rect(x0, y0, w, h, fill="none", stroke="#333333", width=1.2)
        self.text(x0 + w / 2.0, y0 - 12.0, title, size=12.5, weight="bold")
        self.text(x0 + w / 2.0, y0 + h + 36.0, xlabel, size=11.5)
        self.text(x0 - 46.0, y0 + h / 2.0, ylabel, size=11.5, rotate=-90.0)

        for xv, xlbl in xticks:
            xx = sx(xv)
            self.line(xx, y0 + h, xx, y0 + h + 4.5, stroke="#333333", width=1.1)
            self.text(xx, y0 + h + 17.5, xlbl, size=10.0)
        for yv, ylbl in yticks:
            yy = sy(yv)
            self.line(x0 - 4.5, yy, x0, yy, stroke="#333333", width=1.1)
            self.text(x0 - 8.0, yy + 3.5, ylbl, size=10.0, anchor="end")
        return sx, sy

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.elems + ["</svg>"]) + "\n")


def make_fig1_ferris(fig_dir: str, summary_rows: list[dict], rep_rows: list[dict]) -> None:
    """Figure 1: Ferris et al. (2020) host density swings, Floquet D predictions, and phage persistence/density."""
    csv_path = os.path.join(fig_dir, "fig1_ferris_phage.csv")
    write_csv(
        csv_path,
        ["amplitude_kb", "label", "mean_H_e9", "high_H_e9", "low_H_e9", "sigma_lnH",
         "D_waste", "D_wait01", "D_wait003", "mean_H_shift", "surv_frac_T24", "mean_surv_transfer", "early_lnPFU_mean", "early_lnPFU_se"],
        [
            [
                f"{r['amp']:.2f}", r["label"], f"{r['mean_H']/1e9:.4f}", f"{r['high_H']/1e9:.4f}",
                f"{r['low_H']/1e9:.4f}", f"{r['sigma_lnH']:.4f}", f"{r['D_waste']:.5f}",
                f"{r['D_wait01']:.5f}", f"{r['D_wait003']:.5f}", f"{r['mean_H_shift']:.4f}",
                f"{r['surv_frac']:.4f}", f"{r['mean_surv_tr']:.2f}",
                f"{r['early_lnPFU_mean']:.4f}", f"{r['early_lnPFU_se']:.4f}"
            ]
            for r in summary_rows
        ],
    )

    svg = SvgCanvas(width=880, height=400)
    # Left panel: Predicted Floquet loss D vs resource oscillation amplitude
    sx1, sy1 = svg.axes(
        68, 42, 340, 295, -0.04, 1.04, -0.08, 0.50,
        [(0.0, "0"), (0.50, "0.50"), (0.80, "0.80"), (0.98, "0.98")],
        [(-0.05, "-0.05"), (0.0, "0.00"), (0.1, "0.10"), (0.2, "0.20"), (0.3, "0.30"), (0.4, "0.40"), (0.5, "0.50")],
        "Resource oscillation amplitude (delta KB around 0.50 KB)",
        "Predicted ln R0 loss (D or mean-host shift)",
        "A. Phage theory: near one-stage zero-loss limit (S1)"
    )
    svg.line(sx1(-0.04), sy1(0.0), sx1(1.04), sy1(0.0), stroke="#888888", width=1.0, dash="3,3")
    pts_hshift = [(sx1(r["amp"]), sy1(min(0.49, max(-0.075, r["mean_H_shift"])))) for r in summary_rows]
    pts_dwaste = [(sx1(r["amp"]), sy1(r["D_waste"])) for r in summary_rows]
    pts_dwait = [(sx1(r["amp"]), sy1(r["D_wait01"])) for r in summary_rows]
    svg.polyline(pts_hshift, stroke="#d95f02", width=2.0, dash="5,3")
    svg.polyline(pts_dwaste, stroke="#1f77b4", width=2.2)
    svg.polyline(pts_dwait, stroke="#2ca02c", width=2.2)
    for x, y in pts_hshift:
        svg.circle(x, y, r=4.0, fill="#d95f02")
    for x, y in pts_dwaste:
        svg.circle(x, y, r=4.0, fill="#1f77b4")
    for x, y in pts_dwait:
        svg.circle(x, y, r=4.0, fill="#2ca02c")
    # Legend left
    svg.line(82, 62, 106, 62, stroke="#d95f02", width=2.0, dash="5,3")
    svg.text(112, 66, "Mean-host shift ln(H_const / H_trt)", size=9.8, anchor="start")
    svg.line(82, 80, 106, 80, stroke="#1f77b4", width=2.2)
    svg.text(112, 84, "Floquet D: gain-only waste (d = 1)", size=9.8, anchor="start")
    svg.line(82, 98, 106, 98, stroke="#2ca02c", width=2.2)
    svg.text(112, 102, "Floquet D: contact-limited (d = 0.1)", size=9.8, anchor="start")

    # Right panel: Observed early two-phase mean ln(PFU+1) & terminal survival transfer
    sx2, sy2 = svg.axes(
        495, 42, 345, 295, -0.04, 1.04, 0.0, 26.0,
        [(0.0, "0"), (0.50, "0.50"), (0.80, "0.80"), (0.98, "0.98")],
        [(0.0, "0"), (5.0, "5"), (10.0, "10"), (15.0, "15"), (20.0, "20"), (25.0, "25")],
        "Resource oscillation amplitude (delta KB around 0.50 KB)",
        "Early ln(PFU/ml + 1) or terminal transfer",
        "B. Observed phage density & persistence (N = 30)"
    )
    # Individual replicate points (jittered slightly)
    for idx, r in enumerate(rep_rows):
        jit = ((idx % 6) - 2.5) * 0.008
        svg.circle(sx2(r["amp"] + jit), sy2(r["early_lnPFU"]), r=2.8, fill="#1f77b4", opacity=0.45)
        svg.circle(sx2(r["amp"] + jit), sy2(r["surv_tr"]), r=2.8, fill="#7570b3", opacity=0.45)

    pts_pfu = [(sx2(r["amp"]), sy2(r["early_lnPFU_mean"])) for r in summary_rows]
    pts_surv = [(sx2(r["amp"]), sy2(r["mean_surv_tr"])) for r in summary_rows]
    svg.polyline(pts_pfu, stroke="#1f77b4", width=2.2)
    svg.polyline(pts_surv, stroke="#7570b3", width=2.2, dash="4,3")
    for r in summary_rows:
        xx = sx2(r["amp"])
        ym = sy2(r["early_lnPFU_mean"])
        ylo = sy2(max(0.0, r["early_lnPFU_mean"] - r["early_lnPFU_se"]))
        yhi = sy2(min(26.0, r["early_lnPFU_mean"] + r["early_lnPFU_se"]))
        svg.line(xx, ylo, xx, yhi, stroke="#1f77b4", width=1.4)
        svg.circle(xx, ym, r=4.2, fill="#1f77b4")
        svg.circle(xx, sy2(r["mean_surv_tr"]), r=4.2, fill="#7570b3")
    svg.line(510, 62, 534, 62, stroke="#1f77b4", width=2.2)
    svg.text(540, 66, "Early 2-phase mean ln(PFU+1) +/- SE", size=9.8, anchor="start")
    svg.line(510, 80, 534, 80, stroke="#7570b3", width=2.2, dash="4,3")
    svg.text(540, 84, "Mean last surviving transfer (max 24)", size=9.8, anchor="start")

    svg.save(os.path.join(fig_dir, "fig1_ferris_phage.svg"))

    tex_str = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.47\textwidth, height=0.36\textwidth,
    xlabel={Resource oscillation amplitude ($\Delta$ KB)},
    ylabel={Predicted $\ln R_0$ loss $D$},
    title={A. Phage Floquet predictions},
    legend pos=north west, font=\small
]
\addplot[orange!80!black, dashed, thick, mark=*] table[x=amplitude_kb, y=mean_H_shift, col sep=comma]{fig1_ferris_phage.csv};
\addlegendentry{Mean-host shift}
\addplot[blue!75!black, thick, mark=*] table[x=amplitude_kb, y=D_waste, col sep=comma]{fig1_ferris_phage.csv};
\addlegendentry{Floquet $D$ ($d=1$)}
\addplot[green!60!black, thick, mark=*] table[x=amplitude_kb, y=D_wait01, col sep=comma]{fig1_ferris_phage.csv};
\addlegendentry{Floquet $D$ ($d=0.1$)}
\end{axis}
\end{tikzpicture}
"""
    with open(os.path.join(fig_dir, "fig1_ferris_phage.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)


def make_fig2_duncan(fig_dir: str, traj_rows: list[dict], cell_summary: list[dict]) -> None:
    """Figure 2: Duncan et al. (2013) Paramecium-Holospora prevalence trajectories and early vs late contrasts."""
    csv_path = os.path.join(fig_dir, "fig2_duncan_holospora.csv")
    write_csv(
        csv_path,
        ["day", "group", "mean_prev", "se_prev", "mean_host", "se_host"],
        [
            [
                r["day"], r["group"], f"{r['mean_prev']:.5f}", f"{r['se_prev']:.5f}",
                f"{r['mean_host']:.2f}", f"{r['se_host']:.2f}"
            ]
            for r in traj_rows
        ],
    )
    csv2_path = os.path.join(fig_dir, "fig2b_duncan_cells.csv")
    write_csv(
        csv2_path,
        ["cell", "r_prev_0_16_mean", "r_prev_0_16_se", "prev_day56_mean", "prev_day56_se", "pred_D_engine", "pred_D_arrest"],
        [
            [
                c["cell"], f"{c['r016_mean']:.5f}", f"{c['r016_se']:.5f}",
                f"{c['p56_mean']:.4f}", f"{c['p56_se']:.4f}",
                f"{c['D_engine']:.4f}", f"{c['D_arrest']:.4f}"
            ]
            for c in cell_summary
        ],
    )

    svg = SvgCanvas(width=880, height=400)
    # Panel A: Prevalence trajectories Day 0..56 across Const 23C, Const 5C, White, Red
    sx1, sy1 = svg.axes(
        68, 42, 350, 295, 0.0, 58.0, 0.0, 0.60,
        [(0, "0"), (8, "8"), (16, "16"), (24, "24"), (32, "32"), (48, "48"), (56, "56")],
        [(0.0, "0.00"), (0.15, "0.15"), (0.30, "0.30"), (0.45, "0.45"), (0.60, "0.60")],
        "Time (days)",
        "Mean Holospora infection prevalence",
        "A. Epidemic trajectories: rare invasion vs late turnover (S2)"
    )
    # Highlight rare-invasion window Days 0..16
    svg.rect(sx1(0.0), 42, sx1(16.0) - sx1(0.0), 295, fill="#f0f4f8", stroke="none", opacity=0.7)
    svg.line(sx1(16.0), 42, sx1(16.0), 337, stroke="#999999", width=1.0, dash="4,3")
    svg.text(sx1(8.0), 328, "Rare window", size=9.0, fill="#555555")

    styles = {
        "Const_23C": ("#444444", "2,2", "Constant 23 C (n=8)"),
        "White": ("#1f77b4", "", "White noise phi=0 (n=16)"),
        "Red": ("#d62728", "", "Red noise phi=0.8 (n=16)"),
        "Const_5C": ("#17becf", "4,2", "Constant 5 C (n=8)"),
    }
    ly = 60
    for grp, (col, dsh, lbl) in styles.items():
        grows = sorted([r for r in traj_rows if r["group"] == grp], key=lambda z: z["day"])
        pts = [(sx1(r["day"]), sy1(r["mean_prev"])) for r in grows]
        svg.polyline(pts, stroke=col, width=2.2, dash=dsh)
        for r in grows:
            xx = sx1(r["day"])
            ym = sy1(r["mean_prev"])
            ylo = sy1(max(0.0, r["mean_prev"] - r["se_prev"]))
            yhi = sy1(min(0.60, r["mean_prev"] + r["se_prev"]))
            svg.line(xx, ylo, xx, yhi, stroke=col, width=1.2)
            svg.circle(xx, ym, r=3.5, fill=col)
        svg.line(195, ly - 3, 218, ly - 3, stroke=col, width=2.2, dash=dsh)
        svg.text(224, ly, lbl, size=9.5, anchor="start")
        ly += 16

    # Panel B: Early rare-growth rate r_{0->16} vs Day-56 prevalence across the 4 Noise x Synchrony cells
    sx2, sy2 = svg.axes(
        505, 42, 335, 295, 0.4, 4.6, -0.03, 0.26,
        [(1, "White Sync"), (2, "White Async"), (3, "Red Sync"), (4, "Red Async")],
        [(0.0, "0.00"), (0.05, "0.05"), (0.10, "0.10"), (0.15, "0.15"), (0.20, "0.20"), (0.25, "0.25")],
        "Variable-temperature treatment cell (n = 8 each)",
        "Early growth r_{0->16} (/d) or Day-56 prev (/2)",
        "B. Early rare growth vs Day-56 prevalence by cell"
    )
    svg.line(sx2(0.4), sy2(0.0), sx2(4.6), sy2(0.0), stroke="#888888", width=1.0, dash="3,3")
    for idx, c in enumerate(cell_summary, 1):
        # Early growth r016
        x_l = sx2(idx - 0.14)
        ym1 = sy2(c["r016_mean"])
        svg.line(x_l, sy2(c["r016_mean"] - c["r016_se"]), x_l, sy2(c["r016_mean"] + c["r016_se"]), stroke="#1f77b4", width=2.0)
        svg.circle(x_l, ym1, r=4.8, fill="#1f77b4")
        # Day-56 prevalence scaled by 0.5 to fit axis cleanly
        x_r = sx2(idx + 0.14)
        p_sc = 0.5 * c["p56_mean"]
        se_sc = 0.5 * c["p56_se"]
        svg.line(x_r, sy2(p_sc - se_sc), x_r, sy2(p_sc + se_sc), stroke="#d62728", width=2.0)
        svg.circle(x_r, sy2(p_sc), r=4.8, fill="#d62728")
    svg.circle(525, 60, r=4.5, fill="#1f77b4")
    svg.text(535, 64, "Early rare-growth rate r_{0->16} (per day)", size=9.6, anchor="start")
    svg.circle(525, 78, r=4.5, fill="#d62728")
    svg.text(535, 82, "0.5 x Final Day-56 prevalence (post-peak)", size=9.6, anchor="start")

    svg.save(os.path.join(fig_dir, "fig2_duncan_holospora.svg"))

    tex_str = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.55\textwidth, height=0.38\textwidth,
    xlabel={Time (days)},
    ylabel={Mean \emph{Holospora} prevalence},
    title={Epidemic trajectories under white vs.\ red noise},
    legend pos=north west, font=\small
]
\addplot[blue!75!black, thick, mark=*] table[x=day, y=mean_prev, col sep=comma]
  {fig2_duncan_holospora.csv};
\end{axis}
\end{tikzpicture}
"""
    with open(os.path.join(fig_dir, "fig2_duncan_holospora.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)


def make_fig3_daphnia_calibration(fig_dir: str, thresh_rows: list[dict], surv_rows: list[dict], pred_curve: list[dict]) -> None:
    """Figure 3: Daphnia-Metschnikowia empirical invasion threshold, spore survival, and calibrated D predictions."""
    csv_a = os.path.join(fig_dir, "fig3a_dallas_threshold.csv")
    write_csv(
        csv_a,
        ["init_H_per_L", "mean_H_d11_per_L", "primary_inv_frac", "secondary_inv_frac", "mean_auc", "d_contact_full", "d_contact_inf"],
        [
            [
                r["H0_L"], f"{r['H_d11_L']:.2f}", f"{r['prim_frac']:.4f}", f"{r['sec_frac']:.4f}",
                f"{r['mean_auc']:.4f}", f"{r['d_full']:.4f}", f"{r['d_inf']:.4f}"
            ]
            for r in thresh_rows
        ],
    )
    csv_b = os.path.join(fig_dir, "fig3b_calibrated_predictions.csv")
    write_csv(
        csv_b,
        ["period_or_dwell_days", "D_floq_waste", "D_floq_wait_full", "D_tele_waste", "D_tele_wait_full"],
        [
            [
                f"{r['scale_d']:.2f}", f"{r['D_floq_waste']:.5f}", f"{r['D_floq_wait']:.5f}",
                f"{r['D_tele_waste']:.5f}", f"{r['D_tele_wait']:.5f}"
            ]
            for r in pred_curve
        ],
    )

    svg = SvgCanvas(width=880, height=400)
    # Panel A: Empirical invasion threshold & contact death fraction d(H) vs log2(H0)
    sx1, sy1 = svg.axes(
        68, 42, 345, 295, 4.0, 9.6, 0.0, 1.05,
        [(math.log2(20), "20"), (math.log2(40), "40"), (math.log2(80), "80"),
         (math.log2(160), "160"), (math.log2(320), "320"), (math.log2(640), "640")],
        [(0.0, "0.0"), (0.25, "0.25"), (0.50, "0.50"), (0.75, "0.75"), (1.00, "1.0")],
        "Initial Daphnia dentifera density (hosts / L, log scale)",
        "Invasion probability or waiting-death share d(H)",
        "A. Metschnikowia threshold & waiting mortality d(H)"
    )
    pts_prim = [(sx1(math.log2(r["H0_L"])), sy1(r["prim_frac"])) for r in thresh_rows]
    pts_sec = [(sx1(math.log2(r["H0_L"])), sy1(r["sec_frac"])) for r in thresh_rows]
    pts_dfull = [(sx1(math.log2(r["H0_L"])), sy1(r["d_full"])) for r in thresh_rows]
    svg.polyline(pts_prim, stroke="#1f77b4", width=2.2)
    svg.polyline(pts_sec, stroke="#2ca02c", width=2.2, dash="5,3")
    svg.polyline(pts_dfull, stroke="#d95f02", width=2.2)
    for x, y in pts_prim:
        svg.circle(x, y, r=4.0, fill="#1f77b4")
    for x, y in pts_sec:
        svg.circle(x, y, r=4.0, fill="#2ca02c")
    for x, y in pts_dfull:
        svg.circle(x, y, r=4.0, fill="#d95f02")
    svg.line(80, 62, 102, 62, stroke="#1f77b4", width=2.2)
    svg.text(108, 66, "Primary invasion (day <= 11, S3)", size=9.5, anchor="start")
    svg.line(80, 79, 102, 79, stroke="#2ca02c", width=2.2, dash="5,3")
    svg.text(108, 83, "Secondary epidemic (day >= 21, S3)", size=9.5, anchor="start")
    svg.line(80, 96, 102, 96, stroke="#d95f02", width=2.2)
    svg.text(108, 100, "Calibrated d(H) = mu_out/(mu_out + F*H) (S4)", size=9.5, anchor="start")

    # Panel B: Empirically calibrated Daphnia-Metschnikowia loss D across schedule timescales
    sx2, sy2 = svg.axes(
        500, 42, 340, 295, 2.0, 125.0, -0.08, 0.35,
        [(7, "7"), (14, "14"), (28, "28"), (56, "56"), (112, "112")],
        [(-0.05, "-0.05"), (0.0, "0.00"), (0.10, "0.10"), (0.20, "0.20"), (0.30, "0.30")],
        "Schedule period P or mean dwell m (days; +/-80% host swing)",
        "Predicted threshold loss D of ln R0",
        "B. Calibrated Metschnikowia predictions (S3 + S4)"
    )
    svg.line(sx2(2.0), sy2(0.0), sx2(125.0), sy2(0.0), stroke="#888888", width=1.0, dash="3,3")
    p_tw = [(sx2(r["scale_d"]), sy2(r["D_tele_waste"])) for r in pred_curve]
    p_tc = [(sx2(r["scale_d"]), sy2(r["D_tele_wait"])) for r in pred_curve]
    p_fw = [(sx2(r["scale_d"]), sy2(r["D_floq_waste"])) for r in pred_curve]
    p_fc = [(sx2(r["scale_d"]), sy2(r["D_floq_wait"])) for r in pred_curve]
    svg.polyline(p_tw, stroke="#1f77b4", width=2.2)
    svg.polyline(p_tc, stroke="#2ca02c", width=2.2)
    svg.polyline(p_fw, stroke="#1f77b4", width=1.8, dash="4,3")
    svg.polyline(p_fc, stroke="#2ca02c", width=1.8, dash="4,3")
    for x, y in p_tw:
        svg.circle(x, y, r=3.8, fill="#1f77b4")
    for x, y in p_tc:
        svg.circle(x, y, r=3.8, fill="#2ca02c")
    for x, y in p_fw:
        svg.circle(x, y, r=3.3, fill="#1f77b4")
    for x, y in p_fc:
        svg.circle(x, y, r=3.3, fill="#2ca02c")
    svg.line(512, 62, 534, 62, stroke="#1f77b4", width=2.2)
    svg.text(540, 66, "Telegraph: waste (d=1, spore bank)", size=9.4, anchor="start")
    svg.line(512, 79, 534, 79, stroke="#2ca02c", width=2.2)
    svg.text(540, 83, "Telegraph: contact-limited (calibrated d)", size=9.4, anchor="start")
    svg.line(512, 96, 534, 96, stroke="#1f77b4", width=1.8, dash="4,3")
    svg.text(540, 100, "Square-wave Floquet (dashed: resonance dip)", size=9.4, anchor="start")

    svg.save(os.path.join(fig_dir, "fig3_daphnia_calibration.svg"))

    tex_str = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.55\textwidth, height=0.38\textwidth,
    xlabel={Schedule period $P$ or mean dwell $m$ (days)},
    ylabel={Predicted threshold loss $D$},
    title={Calibrated \emph{Daphnia--Metschnikowia} predictions},
    legend pos=north west, font=\small
]
\addplot[blue!75!black, thick, mark=*] table[x=period_or_dwell_days, y=D_tele_waste, col sep=comma]{fig3b_calibrated_predictions.csv};
\addlegendentry{Telegraph ($d=1$)}
\addplot[green!60!black, thick, mark=*] table[x=period_or_dwell_days, y=D_tele_wait_full, col sep=comma]{fig3b_calibrated_predictions.csv};
\addlegendentry{Telegraph (calibrated $d$)}
\end{axis}
\end{tikzpicture}
"""
    with open(os.path.join(fig_dir, "fig3_daphnia_calibration.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)


def make_fig4_weather_lakes(fig_dir: str, lake_wx_series: list[dict], outbreaks_wx: list[dict], tadiri_curve: list[dict]) -> None:
    """Figure 4: S6 Michigan lakes 2021 Daymet/GHCN weather, outbreak lags, and S5/W2 thermal stage-duration input."""
    write_csv(
        os.path.join(fig_dir, "fig4a_michigan_weather_2021.csv"),
        ["doy", "daymet_6lake_tmean_14d", "ghcnd_a2_tmean_14d", "daymet_srad_14d"],
        [
            [r["doy"], f"{r['dm_t14']:.3f}", f"{r['ghcn_t14']:.3f}", f"{r['dm_srad14']:.2f}"]
            for r in lake_wx_series
        ],
    )
    write_csv(
        os.path.join(fig_dir, "fig4b_outbreaks_and_thermal_D.csv"),
        ["lake", "parasite", "host", "start_doy", "lag_days", "t14_at_start_C", "pred_tau_in_d", "pred_D_waste", "pred_D_wait05"],
        [
            [
                r["lake"], r["parasite"], r["host"], r["start_doy"], r["lag_days"],
                f"{r['t14_start']:.2f}", f"{r['tau_in_T']:.2f}",
                f"{r['D_waste_T']:.5f}", f"{r['D_wait_T']:.5f}"
            ]
            for r in outbreaks_wx
        ],
    )

    svg = SvgCanvas(width=880, height=400)
    # Panel A: 2021 SE Michigan lake Daymet vs NOAA GHCN-Daily 14-d mean temperature + 13 outbreak start dates
    sx1, sy1 = svg.axes(
        68, 42, 350, 295, 145.0, 325.0, 0.0, 28.0,
        [(152, "Jun 1"), (182, "Jul 1"), (213, "Aug 1"), (244, "Sep 1"), (274, "Oct 1"), (305, "Nov 1")],
        [(0, "0"), (5, "5"), (10, "10"), (15, "15"), (20, "20"), (25, "25")],
        "2021 Day of year (SE Michigan lakes, S6)",
        "14-d trailing air temperature (deg C) or lag/2 (d)",
        "A. 2021 Daymet (1 km) & GHCN weather + 13 outbreaks"
    )
    pts_dm = [(sx1(r["doy"]), sy1(r["dm_t14"])) for r in lake_wx_series]
    pts_gh = [(sx1(r["doy"]), sy1(r["ghcn_t14"])) for r in lake_wx_series if not math.isnan(r["ghcn_t14"])]
    svg.polyline(pts_dm, stroke="#d95f02", width=2.2)
    svg.polyline(pts_gh, stroke="#555555", width=1.6, dash="4,3")
    for ob in outbreaks_wx:
        xx = sx1(ob["start_doy"])
        yy = sy1(min(27.5, ob["lag_days"] * 0.5))
        svg.circle(xx, yy, r=4.2, fill="#1f77b4")
    svg.line(80, 60, 102, 60, stroke="#d95f02", width=2.2)
    svg.text(108, 64, "Daymet 1 km 6-lake 14-d T_mean (C)", size=9.4, anchor="start")
    svg.line(80, 77, 102, 77, stroke="#555555", width=1.6, dash="4,3")
    svg.text(108, 81, "NOAA GHCN-D Ann Arbor 14-d T_mean (C)", size=9.4, anchor="start")
    svg.circle(91, 91, r=4.0, fill="#1f77b4")
    svg.text(108, 95, "13 S6 outbreaks: 0.5 x |spore-to-outbreak lag| (d)", size=9.4, anchor="start")

    # Panel B: S5 Tadiri empirical Pasteuria tti(T) & W2 predicted thermal modulation of tau_in(T) and D(T)
    sx2, sy2 = svg.axes(
        505, 42, 335, 295, 9.0, 25.0, 0.0, 48.0,
        [(10, "10"), (12.5, "12.5"), (15, "15"), (17.5, "17.5"), (20, "20"), (24, "24")],
        [(0, "0"), (10, "10"), (20, "20"), (30, "30"), (40, "40")],
        "Temperature T (deg C)",
        "Pasteuria time to infection tau_in (d) or 400 x D",
        "B. Thermal development (S5) & W2 loss D(T)"
    )
    pts_tti = [(sx2(r["temp_c"]), sy2(r["mean_tti"])) for r in tadiri_curve]
    pts_dw = [(sx2(r["temp_c"]), sy2(400.0 * r["D_waste"])) for r in tadiri_curve]
    pts_dc = [(sx2(r["temp_c"]), sy2(400.0 * r["D_wait05"])) for r in tadiri_curve]
    svg.polyline(pts_tti, stroke="#7570b3", width=2.2)
    svg.polyline(pts_dw, stroke="#1f77b4", width=2.0, dash="5,3")
    svg.polyline(pts_dc, stroke="#2ca02c", width=2.0, dash="2,2")
    for r in tadiri_curve:
        xx = sx2(r["temp_c"])
        ym = sy2(r["mean_tti"])
        if r.get("se_tti", 0.0) > 0:
            svg.line(xx, sy2(r["mean_tti"] - r["se_tti"]), xx, sy2(r["mean_tti"] + r["se_tti"]), stroke="#7570b3", width=1.5)
        svg.circle(xx, ym, r=4.2, fill="#7570b3")
        svg.circle(xx, sy2(400.0 * r["D_waste"]), r=3.5, fill="#1f77b4")
        svg.circle(xx, sy2(400.0 * r["D_wait05"]), r=3.5, fill="#2ca02c")
    svg.line(520, 60, 542, 60, stroke="#7570b3", width=2.2)
    svg.text(548, 64, "Observed Pasteuria tti +/- SE (S5 Tadiri)", size=9.4, anchor="start")
    svg.line(520, 77, 542, 77, stroke="#1f77b4", width=2.0, dash="5,3")
    svg.text(548, 81, "400 x predicted loss D(T), waste d=1", size=9.4, anchor="start")
    svg.line(520, 94, 542, 94, stroke="#2ca02c", width=2.0, dash="2,2")
    svg.text(548, 98, "400 x predicted loss D(T), wait d=0.5", size=9.4, anchor="start")

    svg.save(os.path.join(fig_dir, "fig4_s6_weather_lakes.svg"))

    tex_str = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.55\textwidth, height=0.38\textwidth,
    xlabel={2021 Day of year},
    ylabel={14-day trailing temperature ($^\circ$C)},
    title={SE Michigan lake weather (Daymet 1 km vs.\ NOAA GHCN-D)},
    legend pos=south west, font=\small
]
\addplot[orange!85!black, thick] table[x=doy, y=daymet_6lake_tmean_14d, col sep=comma]{fig4a_michigan_weather_2021.csv};
\addlegendentry{Daymet 1 km (6-lake mean)}
\addplot[black!65, dashed, thick] table[x=doy, y=ghcnd_a2_tmean_14d, col sep=comma]{fig4a_michigan_weather_2021.csv};
\addlegendentry{NOAA GHCN-D (Ann Arbor)}
\end{axis}
\end{tikzpicture}
"""
    with open(os.path.join(fig_dir, "fig4_s6_weather_lakes.tex"), "w", encoding="utf-8") as f:
        f.write(tex_str)

