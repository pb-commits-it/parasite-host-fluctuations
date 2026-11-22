#!/usr/bin/env python3
"""Vector SVG and pgfplots (.tex + .csv) figure generator for study `chemostat`.

Matplotlib is intentionally not imported (`CASESTUDY_SPEC.md` Rule 7). Writes clean,
self-contained SVG figures and companion pgfplots `.tex` + `.csv` data files directly
from computed numerical arrays.
"""
from __future__ import annotations

import csv
import html
import math
import os


PALETTE = {
    "blue": "#1f5a96",
    "orange": "#d95f02",
    "green": "#1b9e77",
    "purple": "#7570b3",
    "red": "#c0392b",
    "grey": "#666666",
    "lightgrey": "#e0e0e0",
    "dark": "#222222",
}


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_fig1_larsen_chemostat(fig_dir: str, d1_units: list, d1_results: list[dict]) -> None:
    """Figure 1: Larsen et al. (2019) continuous chemostats (N- vs P-limited trajectories, resistance, and D breakdown)."""
    # Export CSV data for trajectories
    traj_rows = []
    for u in d1_units:
        for idx, t in enumerate(u.times_d):
            h = u.host_density[idx]
            p = u.phage_density[idx]
            traj_rows.append({
                "unit_id": u.unit_id,
                "treatment": u.treatment,
                "infected": int(u.infected),
                "day": f"{t:.1f}",
                "log10_host": f"{math.log10(h):.4f}" if (not math.isnan(h) and h > 0) else "",
                "log10_phage": f"{math.log10(p):.4f}" if (not math.isnan(p) and p > 0) else "",
            })
    write_csv(
        os.path.join(fig_dir, "fig1_larsen_trajectories.csv"),
        ["unit_id", "treatment", "infected", "day", "log10_host", "log10_phage"],
        traj_rows,
    )

    # SVG layout: 860 x 500, 2x2 panels:
    # (A) N-limited host & phage trajectories (NL2, NL3, NL5 + controls NL1, NL4)
    # (B) P-limited host & phage trajectories (PL2, PL4, PL5 + controls PL1, PL3)
    # (C) Assayed host resistance fraction over time (N vs P)
    # (D) Magnitude comparison in ln R0 units per infected chemostat (ln R0, waste D, contact D, resistance loss)
    W, H = 880, 540
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="440" y="24" text-anchor="middle" font-size="14" font-weight="bold" fill="#222222">'
        'D1 (Larsen et al. 2019): Synechococcus–Cyanophage Continuous Chemostats (ω = 1.0 d⁻¹)</text>',
    ]

    def draw_ts_panel(x0: int, y0: int, pw: int, ph: int, title: str, lim_prefix: str) -> None:
        lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#888888" stroke-width="1"/>')
        lines.append(f'<text x="{x0 + pw/2:.1f}" y="{y0 - 7}" text-anchor="middle" font-size="12" font-weight="bold" fill="#222222">{_esc(title)}</text>')
        # Axes: x in [0, 170], y in [4.0, 9.5] (log10 mL^-1)
        xmin, xmax = 0.0, 170.0
        ymin, ymax = 4.0, 9.6
        for ytick in [4, 5, 6, 7, 8, 9]:
            yy = y0 + ph - (ytick - ymin) / (ymax - ymin) * ph
            lines.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#eeeeee" stroke-width="1"/>')
            lines.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="#444444">10^{ytick}</text>')
        for xtick in [0, 40, 80, 120, 160]:
            xx = x0 + (xtick - xmin) / (xmax - xmin) * pw
            lines.append(f'<line x1="{xx:.1f}" y1="{y0+ph}" x2="{xx:.1f}" y2="{y0+ph+4}" stroke="#444444"/>')
            lines.append(f'<text x="{xx:.1f}" y="{y0+ph+16}" text-anchor="middle" font-size="10" fill="#444444">{xtick}</text>')
        lines.append(f'<text x="{x0 + pw/2:.1f}" y="{y0+ph+30}" text-anchor="middle" font-size="10" fill="#222222">Time since phage addition (days)</text>')
        lines.append(f'<text x="{x0 - 38}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="10" fill="#222222" transform="rotate(-90 {x0-38} {y0+ph/2:.1f})">Density (mL⁻¹)</text>')

        for u in d1_units:
            if not u.unit_id.startswith(lim_prefix):
                continue
            pts_h = []
            pts_p = []
            for idx, t in enumerate(u.times_d):
                if xmin <= t <= xmax:
                    xx = x0 + (t - xmin) / (xmax - xmin) * pw
                    hv = u.host_density[idx]
                    pv = u.phage_density[idx]
                    if not math.isnan(hv) and hv > 0:
                        lh = min(max(math.log10(hv), ymin), ymax)
                        yy = y0 + ph - (lh - ymin) / (ymax - ymin) * ph
                        pts_h.append(f"{xx:.1f},{yy:.1f}")
                    if u.infected and not math.isnan(pv) and pv > 0:
                        lp = min(max(math.log10(pv), ymin), ymax)
                        yy = y0 + ph - (lp - ymin) / (ymax - ymin) * ph
                        pts_p.append(f"{xx:.1f},{yy:.1f}")
            if not u.infected and pts_h:
                lines.append(f'<polyline fill="none" stroke="#888888" stroke-width="1.2" stroke-dasharray="4,3" points="{" ".join(pts_h)}"/>')
            elif u.infected:
                if pts_h:
                    lines.append(f'<polyline fill="none" stroke="{PALETTE["green"]}" stroke-width="1.5" opacity="0.85" points="{" ".join(pts_h)}"/>')
                if pts_p:
                    lines.append(f'<polyline fill="none" stroke="{PALETTE["orange"]}" stroke-width="1.5" opacity="0.85" points="{" ".join(pts_p)}"/>')

        # Legend inside top-right
        lx, ly = x0 + pw - 148, y0 + 14
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+16}" y2="{ly}" stroke="{PALETTE["green"]}" stroke-width="2"/>')
        lines.append(f'<text x="{lx+20}" y="{ly+3}" font-size="9.5" fill="#222222">Infected host H(t)</text>')
        lines.append(f'<line x1="{lx}" y1="{ly+13}" x2="{lx+16}" y2="{ly+13}" stroke="{PALETTE["orange"]}" stroke-width="2"/>')
        lines.append(f'<text x="{lx+20}" y="{ly+16}" font-size="9.5" fill="#222222">Free phage P(t)</text>')
        lines.append(f'<line x1="{lx}" y1="{ly+26}" x2="{lx+16}" y2="{ly+26}" stroke="#888888" stroke-width="1.4" stroke-dasharray="4,3"/>')
        lines.append(f'<text x="{lx+20}" y="{ly+29}" font-size="9.5" fill="#222222">Control host (no phage)</text>')

    draw_ts_panel(62, 52, 355, 185, "(A) N-limited chemostats (large phage-driven host crashes)", "NL")
    draw_ts_panel(490, 52, 355, 185, "(B) P-limited chemostats (damped slow host decline)", "PL")

    # Panel (C): Resistance sweep over time
    x0, y0, pw, ph = 62, 300, 355, 185
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#888888" stroke-width="1"/>')
    lines.append(f'<text x="{x0 + pw/2:.1f}" y="{y0 - 7}" text-anchor="middle" font-size="12" font-weight="bold" fill="#222222">(C) Host resistance sweep (inf-mat.csv)</text>')
    xmin, xmax = -10.0, 170.0
    for ytick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        yy = y0 + ph - ytick * ph
        lines.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        lines.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="#444444">{ytick:.2f}</text>')
    for xtick in [0, 40, 80, 120, 160]:
        xx = x0 + (xtick - xmin) / (xmax - xmin) * pw
        lines.append(f'<line x1="{xx:.1f}" y1="{y0+ph}" x2="{xx:.1f}" y2="{y0+ph+4}" stroke="#444444"/>')
        lines.append(f'<text x="{xx:.1f}" y="{y0+ph+16}" text-anchor="middle" font-size="10" fill="#444444">{xtick}</text>')
    lines.append(f'<text x="{x0 + pw/2:.1f}" y="{y0+ph+30}" text-anchor="middle" font-size="10" fill="#222222">Isolation day</text>')
    lines.append(f'<text x="{x0 - 38}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="10" fill="#222222" transform="rotate(-90 {x0-38} {y0+ph/2:.1f})">Fraction resistant</text>')

    for u in d1_units:
        if not u.infected or len(u.res_times_d) == 0:
            continue
        col = PALETTE["blue"] if u.unit_id.startswith("NL") else PALETTE["purple"]
        pts = []
        for rt, rf in zip(u.res_times_d, u.res_frac_anc):
            xx = x0 + (rt - xmin) / (xmax - xmin) * pw
            yy = y0 + ph - rf * ph
            pts.append(f"{xx:.1f},{yy:.1f}")
            lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.8" fill="{col}"/>')
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{" ".join(pts)}"/>')
    lines.append(f'<line x1="{x0+185}" y1="{y0+145}" x2="{x0+203}" y2="{y0+145}" stroke="{PALETTE["blue"]}" stroke-width="2"/>')
    lines.append(f'<text x="{x0+208}" y="{y0+148}" font-size="9.5" fill="#222222">N-limited (NL2, NL3, NL5)</text>')
    lines.append(f'<line x1="{x0+185}" y1="{y0+160}" x2="{x0+203}" y2="{y0+160}" stroke="{PALETTE["purple"]}" stroke-width="2"/>')
    lines.append(f'<text x="{x0+208}" y="{y0+163}" font-size="9.5" fill="#222222">P-limited (PL2, PL4, PL5)</text>')

    # Panel (D): Grouped bars comparing D_sim(contact), Gamma_c(2nd-order), D_sim(waste d=1), and Resistance loss
    x0, y0, pw, ph = 490, 300, 355, 185
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#888888" stroke-width="1"/>')
    lines.append(f'<text x="{x0 + pw/2:.1f}" y="{y0 - 7}" text-anchor="middle" font-size="12" font-weight="bold" fill="#222222">(D) Fluctuation vs resistance loss (ln R₀ units)</text>')
    # Log10 scale on y from 1e-3 to 10.0 so both 0.03-0.30 and 1.5-2.5 are cleanly visible
    ymin_l, ymax_l = -3.0, 0.8
    for ltick, lbl in [(-3, "0.001"), (-2, "0.01"), (-1, "0.1"), (0, "1.0")]:
        yy = y0 + ph - (ltick - ymin_l) / (ymax_l - ymin_l) * ph
        lines.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        lines.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="9.5" fill="#444444">{lbl}</text>')
    lines.append(f'<text x="{x0 - 38}" y="{y0 + ph/2:.1f}" text-anchor="middle" font-size="10" fill="#222222" transform="rotate(-90 {x0-38} {y0+ph/2:.1f})">Loss in ln R₀ (log scale)</text>')

    n_bars = len(d1_results)
    gw = pw / max(n_bars, 1)
    for idx, res in enumerate(d1_results):
        gx = x0 + idx * gw + gw * 0.14
        bw = gw * 0.18
        vals = [
            (max(res["D_sim_contact"], 1e-3), PALETTE["blue"]),
            (max(abs(res["Gamma_c_2nd"]), 1e-3), PALETTE["green"]),
            (max(res["D_sim_waste"], 1e-3), PALETTE["orange"]),
            (max(res.get("res_loss_lnR0", 1e-3), 1e-3), PALETTE["red"]),
        ]
        for b_i, (val, col) in enumerate(vals):
            lv = min(max(math.log10(val), ymin_l), ymax_l)
            bh = (lv - ymin_l) / (ymax_l - ymin_l) * ph
            bx = gx + b_i * (bw + 1.5)
            by = y0 + ph - bh
            lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}"/>')
        lines.append(f'<text x="{x0 + idx*gw + gw/2:.1f}" y="{y0+ph+15}" text-anchor="middle" font-size="10" fill="#222222">{_esc(res["unit_id"])}</text>')

    # Legend for Panel D
    leg_items = [
        ("D_sim (contact)", PALETTE["blue"]),
        ("Γ_c (2nd order)", PALETTE["green"]),
        ("D_sim (waste d=1)", PALETTE["orange"]),
        ("Resistance Δln R₀", PALETTE["red"]),
    ]
    for li, (lbl, col) in enumerate(leg_items):
        lx = x0 + 10 + (li % 2) * 172
        ly = y0 + 10 + (li // 2) * 14
        lines.append(f'<rect x="{lx}" y="{ly-7}" width="9" height="9" fill="{col}"/>')
        lines.append(f'<text x="{lx+13}" y="{ly+1}" font-size="9" fill="#222222">{_esc(lbl)}</text>')

    lines.append("</svg>")
    with open(os.path.join(fig_dir, "fig1_larsen_chemostat.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Companion pgfplots .tex
    tex = r"""\begin{tikzpicture}
\begin{axis}[
    width=11cm, height=6.5cm,
    xlabel={Chemostat unit (Larsen et al.\ 2019)},
    ylabel={Loss in $\ln R_0$ (log scale)},
    ymode=log, ymin=1e-3, ymax=6,
    symbolic x coords={""" + ",".join(r["unit_id"] for r in d1_results) + r"""},
    xtick=data, ybar=2pt, bar width=6pt,
    legend style={at={(0.02,0.98)}, anchor=north west, font=\footnotesize}
]
\addplot coordinates {""" + " ".join(f"({r['unit_id']},{max(r['D_sim_contact'],1e-4):.5f})" for r in d1_results) + r"""};
\addplot coordinates {""" + " ".join(f"({r['unit_id']},{max(abs(r['Gamma_c_2nd']),1e-4):.5f})" for r in d1_results) + r"""};
\addplot coordinates {""" + " ".join(f"({r['unit_id']},{max(r['D_sim_waste'],1e-4):.5f})" for r in d1_results) + r"""};
\addplot coordinates {""" + " ".join(f"({r['unit_id']},{max(r.get('res_loss_lnR0',1e-4),1e-4):.5f})" for r in d1_results) + r"""};
\legend{$D_{\mathrm{sim}}$ (contact), $\Gamma_c$ (2nd order), $D_{\mathrm{sim}}$ (waste $d=1$), Resistance $\Delta\ln R_0$}
\end{axis}
\end{tikzpicture}
"""
    with open(os.path.join(fig_dir, "fig1_larsen_chemostat.tex"), "w", encoding="utf-8") as f:
        f.write(tex)


def make_fig2_cross_study_synthesis(fig_dir: str, all_inf_rows: list[dict], coupled_curve: list[dict]) -> None:
    """Figure 2: Cross-dataset synthesis (D_sim vs Gamma_c, empirical phage vs D_sim across D1-D5, and coupled dilution)."""
    write_csv(
        os.path.join(fig_dir, "fig2_coupled_dilution_curve.csv"),
        ["omega_bar_d", "d_threshold", "D_uniform_wash", "D_selective_neg_cov", "D_selective_const_omega", "PropB_bound_dt005"],
        coupled_curve,
    )

    W, H = 900, 340
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="450" y="22" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#222222">'
        'Cross-Dataset Theory Synthesis (D1–D5, n = 104 infected trajectories) &amp; Coupled Chemostat Dilution</text>',
    ]

    ds_colors = {
        "D1_larsen2019": PALETTE["blue"],
        "D2_reuter2026": PALETTE["green"],
        "D3_chaudhry2018": PALETTE["orange"],
        "D4_duncan2017": PALETTE["purple"],
        "D5_zhao2016": PALETTE["red"],
    }

    # Panel A: Second-order Gamma_c vs Exact D_sim (log-log scatter across all infected units with positive D_sim)
    x0, y0, pw, ph = 58, 52, 230, 225
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#888888"/>')
    lines.append(f'<text x="{x0+pw/2:.1f}" y="{y0-7}" text-anchor="middle" font-size="11.5" font-weight="bold">(A) 2nd-order Γ_c vs exact D_sim</text>')
    lmin, lmax = -5.0, 0.2
    # 1:1 diagonal
    lines.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0}" stroke="#999999" stroke-dasharray="4,3"/>')
    for lt in [-5, -4, -3, -2, -1, 0]:
        pos_x = x0 + (lt - lmin) / (lmax - lmin) * pw
        pos_y = y0 + ph - (lt - lmin) / (lmax - lmin) * ph
        lines.append(f'<line x1="{x0}" y1="{pos_y:.1f}" x2="{x0+pw}" y2="{pos_y:.1f}" stroke="#f0f0f0"/>')
        lines.append(f'<text x="{x0-4}" y="{pos_y+3:.1f}" text-anchor="end" font-size="9" fill="#444444">10^{lt}</text>')
        lines.append(f'<text x="{pos_x:.1f}" y="{y0+ph+14}" text-anchor="middle" font-size="9" fill="#444444">10^{lt}</text>')
    lines.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+28}" text-anchor="middle" font-size="10">Exact linear loss D_sim (ln R₀)</text>')
    lines.append(f'<text x="{x0-36}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="10" transform="rotate(-90 {x0-36} {y0+ph/2:.1f})">2nd-order |Γ_c| (ln R₀)</text>')

    for r in all_inf_rows:
        dx = float(r["D_sim_contact"])
        gy = abs(float(r["Gamma_c_2nd"]))
        if dx > 1e-6 and gy > 1e-6:
            lx = min(max(math.log10(dx), lmin), lmax)
            ly = min(max(math.log10(gy), lmin), lmax)
            px = x0 + (lx - lmin) / (lmax - lmin) * pw
            py = y0 + ph - (ly - lmin) / (lmax - lmin) * ph
            col = ds_colors.get(r["dataset_id"], "#333333")
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="{col}" opacity="0.8"/>')

    # Panel B: Waiting protection ratio D_sim(contact) / D_sim(waste) vs threshold d
    x0, y0, pw, ph = 345, 52, 230, 225
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#888888"/>')
    lines.append(f'<text x="{x0+pw/2:.1f}" y="{y0-7}" text-anchor="middle" font-size="11.5" font-weight="bold">(B) Contact waiting protection vs d</text>')
    for yt in [0.0, 0.25, 0.50, 0.75, 1.0, 1.25]:
        py = y0 + ph - (yt / 1.35) * ph
        lines.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#f0f0f0"/>')
        lines.append(f'<text x="{x0-4}" y="{py+3:.1f}" text-anchor="end" font-size="9" fill="#444444">{yt:.2f}</text>')
    for xt in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        px = x0 + xt * pw
        lines.append(f'<text x="{px:.1f}" y="{y0+ph+14}" text-anchor="middle" font-size="9" fill="#444444">{xt:.1f}</text>')
    # Reference d^2 curve and d curve
    pts_d2 = [f"{x0 + (i/50)*pw:.1f},{y0 + ph - ((i/50)**2 / 1.35)*ph:.1f}" for i in range(51)]
    pts_d1 = [f"{x0 + (i/50)*pw:.1f},{y0 + ph - ((i/50) / 1.35)*ph:.1f}" for i in range(51)]
    lines.append(f'<polyline fill="none" stroke="#aaaaaa" stroke-dasharray="3,3" points="{" ".join(pts_d1)}"/>')
    lines.append(f'<polyline fill="none" stroke="#777777" stroke-dasharray="5,3" points="{" ".join(pts_d2)}"/>')
    lines.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+28}" text-anchor="middle" font-size="10">Threshold waiting-loss fraction d</text>')
    lines.append(f'<text x="{x0-34}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="10" transform="rotate(-90 {x0-34} {y0+ph/2:.1f})">D_sim(contact) / D_sim(waste)</text>')

    for r in all_inf_rows:
        dw = float(r["D_sim_waste"])
        dc = float(r["D_sim_contact"])
        d_val = float(r["d_threshold"])
        if dw > 1e-5:
            rat = min(max(dc / dw, 0.0), 1.35)
            px = x0 + min(max(d_val, 0.0), 1.0) * pw
            py = y0 + ph - (rat / 1.35) * ph
            col = ds_colors.get(r["dataset_id"], "#333333")
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.0" fill="{col}" opacity="0.8"/>')

    # Legend inside Panel B
    for idx_l, (ds_k, lbl) in enumerate([
        ("D1_larsen2019", "D1 Chemostat"),
        ("D2_reuter2026", "D2 ΦX174"),
        ("D3_chaudhry2018", "D3 λVIR"),
        ("D4_duncan2017", "D4 SBW25 temp"),
        ("D5_zhao2016", "D5 SBW25 nut."),
    ]):
        lx, ly = x0 + 10, y0 + 14 + idx_l * 13
        lines.append(f'<circle cx="{lx+4}" cy="{ly-3}" r="3.2" fill="{ds_colors[ds_k]}"/>')
        lines.append(f'<text x="{lx+11}" y="{ly}" font-size="8.8" fill="#222222">{_esc(lbl)}</text>')

    # Panel C: Theoretical Coupled-Dilution Curve
    x0, y0, pw, ph = 635, 52, 235, 225
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#888888"/>')
    lines.append(f'<text x="{x0+pw/2:.1f}" y="{y0-7}" text-anchor="middle" font-size="11.5" font-weight="bold">(C) Coupled chemostat washout ω</text>')
    xmax_om = max(float(c["omega_bar_d"]) for c in coupled_curve)
    ymax_d = max(float(c["PropB_bound_dt005"]) for c in coupled_curve) * 1.08
    for i_t in range(5):
        yv = ymax_d * i_t / 4.0
        py = y0 + ph - (yv / ymax_d) * ph
        lines.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+pw}" y2="{py:.1f}" stroke="#f0f0f0"/>')
        lines.append(f'<text x="{x0-4}" y="{py+3:.1f}" text-anchor="end" font-size="9" fill="#444444">{yv:.2f}</text>')
    for xt in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        if xt <= xmax_om + 1e-9:
            px = x0 + (xt / xmax_om) * pw
            lines.append(f'<text x="{px:.1f}" y="{y0+ph+14}" text-anchor="middle" font-size="9" fill="#444444">{xt:.0f}</text>')
    lines.append(f'<text x="{x0+pw/2:.1f}" y="{y0+ph+28}" text-anchor="middle" font-size="10">Mean washout rate ω̄ (d⁻¹)</text>')
    lines.append(f'<text x="{x0-36}" y="{y0+ph/2:.1f}" text-anchor="middle" font-size="10" transform="rotate(-90 {x0-36} {y0+ph/2:.1f})">Loss D &amp; Prop. B bound (ln R₀)</text>')

    for key, col, dash, lbl, ly_off in [
        ("PropB_bound_dt005", PALETTE["grey"], 'stroke-dasharray="4,3"', "Prop. B bound (Δt=0.05d)", 14),
        ("D_selective_neg_cov", PALETTE["red"], "", "Selective ω_P, Cov(H,ω)<0", 27),
        ("D_uniform_wash", PALETTE["blue"], "", "Uniform ω_P=ω_I=ω(t)", 40),
    ]:
        pts = []
        for c in coupled_curve:
            px = x0 + (float(c["omega_bar_d"]) / xmax_om) * pw
            py = y0 + ph - (float(c[key]) / ymax_d) * ph
            pts.append(f"{px:.1f},{py:.1f}")
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" {dash} points="{" ".join(pts)}"/>')
        lines.append(f'<line x1="{x0+8}" y1="{y0+ly_off-3}" x2="{x0+24}" y2="{y0+ly_off-3}" stroke="{col}" stroke-width="1.8" {dash}/>')
        lines.append(f'<text x="{x0+28}" y="{y0+ly_off}" font-size="8.7" fill="#222222">{_esc(lbl)}</text>')

    lines.append("</svg>")
    with open(os.path.join(fig_dir, "fig2_cross_study_and_dilution.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    tex2 = r"""\begin{tikzpicture}
\begin{axis}[
    width=10cm, height=6.2cm,
    xlabel={Mean chemostat washout rate $\bar{\omega}$ ($\mathrm{d}^{-1}$)},
    ylabel={Threshold loss $D$ or bound ($\ln R_0$ units)},
    xmin=0, xmax=5, ymin=0,
    legend style={at={(0.03,0.97)}, anchor=north west, font=\footnotesize}
]
\addplot[thick, blue] table[x=omega_bar_d, y=D_uniform_wash, col sep=comma] {fig2_coupled_dilution_curve.csv};
\addplot[thick, red] table[x=omega_bar_d, y=D_selective_neg_cov, col sep=comma] {fig2_coupled_dilution_curve.csv};
\addplot[thick, dashed, gray] table[x=omega_bar_d, y=PropB_bound_dt005, col sep=comma] {fig2_coupled_dilution_curve.csv};
\legend{Uniform washout $\omega_P=\omega_I=\omega(t)$, Selective free-phage washout ($\mathrm{Cov}(H,\omega)<0$), Proposition B bound ($\Delta t=0.05\,\mathrm{d}$)}
\end{axis}
\end{tikzpicture}
"""
    with open(os.path.join(fig_dir, "fig2_cross_study_and_dilution.tex"), "w", encoding="utf-8") as f:
        f.write(tex2)
