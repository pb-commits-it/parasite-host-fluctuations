#!/usr/bin/env python3
"""Vector figure (.svg + pgfplots .tex) and CSV data generator for case study `voles`.

Per CASESTUDY_SPEC.md Rule 7, matplotlib is not used; scripts write clean CSV data
tables alongside standalone SVG vector figures and pgfplots LaTeX sources.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Any


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def generate_all_figures(fig_dir: str, results: dict[str, Any]) -> list[str]:
    os.makedirs(fig_dir, exist_ok=True)
    written: list[str] = []

    # -----------------------------------------------------------------------
    # FIGURE 1: Host Memory Regimes (Trajectories + 28-day ACF + Model Fits)
    # -----------------------------------------------------------------------
    mem = results["goal1_host_memory"]
    f02_win = results["_plot_data"]["f02_core_series"]
    k15_win = results["_plot_data"]["k15_site_series"]

    # Save CSV for Fig 1A (trajectories) and Fig 1B (ACFs)
    traj_rows = []
    for w in f02_win:
        traj_rows.append({
            "system": "Konnevesi_F02_core",
            "site": "core",
            "step_k": w["step_k"],
            "date": w["mean_date"],
            "H_index": round(w["H_index"], 4),
            "ln_H_centred": round(w["z_centred"], 4) if w["z_centred"] is not None else "",
            "T_28d_C": round(w["T_28d_C"], 2),
        })
    for site, s_list in sorted(k15_win.items()):
        for w in s_list:
            traj_rows.append({
                "system": "Kielder_K15",
                "site": site,
                "step_k": w["step_k"],
                "date": w["mean_date"],
                "H_index": round(w["H_index"], 4),
                "ln_H_centred": round(w["z_centred"], 4) if w["z_centred"] is not None else "",
                "T_28d_C": round(w["T_28d_C"], 2),
            })
    p1a_csv = os.path.join(fig_dir, "fig1a_host_trajectories.csv")
    _write_csv(p1a_csv, ["system", "site", "step_k", "date", "H_index", "ln_H_centred", "T_28d_C"], traj_rows)
    written.append(p1a_csv)

    acf_rows = []
    k15_acf = mem["k15_pooled"]["acf_0_to_12"]
    k15_ar1 = mem["k15_pooled"]["ar1_acf_0_to_12"]
    k15_best = mem["k15_pooled"]["best_model_acf_0_to_12"]
    f02_acf = mem["f02_core"]["acf_0_to_12"]
    f02_ar1 = mem["f02_core"]["ar1_acf_0_to_12"]
    f02_best = mem["f02_core"]["best_model_acf_0_to_12"]
    for lag in range(13):
        acf_rows.append({
            "lag_28d_steps": lag,
            "kielder_emp_acf": "" if k15_acf[lag] is None else round(k15_acf[lag], 4),
            "kielder_ar1_fit": round(k15_ar1[lag], 4),
            "kielder_best_fit": round(k15_best[lag], 4),
            "konnevesi_emp_acf": "" if f02_acf[lag] is None else round(f02_acf[lag], 4),
            "konnevesi_ar1_fit": round(f02_ar1[lag], 4),
            "konnevesi_best_fit": round(f02_best[lag], 4),
        })
    p1b_csv = os.path.join(fig_dir, "fig1b_host_acf_fits.csv")
    _write_csv(
        p1b_csv,
        [
            "lag_28d_steps",
            "kielder_emp_acf",
            "kielder_ar1_fit",
            "kielder_best_fit",
            "konnevesi_emp_acf",
            "konnevesi_ar1_fit",
            "konnevesi_best_fit",
        ],
        acf_rows,
    )
    written.append(p1b_csv)

    # Render Figure 1 SVG (920 x 540, 2x2 panels)
    svg1 = _build_fig1_svg(f02_win, k15_win, acf_rows, mem)
    p1_svg = os.path.join(fig_dir, "fig1_host_memory_regimes.svg")
    with open(p1_svg, "w", encoding="utf-8") as f:
        f.write(svg1)
    written.append(p1_svg)

    p1_tex = os.path.join(fig_dir, "fig1_host_memory_regimes.tex")
    with open(p1_tex, "w", encoding="utf-8") as f:
        f.write(_build_fig1_tex())
    written.append(p1_tex)

    # -----------------------------------------------------------------------
    # FIGURE 2: Pathogen Prevalence Tracking CCF(0..12) vs K9 Theory & Weather
    # -----------------------------------------------------------------------
    ccf_res = results["goal2_parasite_tracking"]["series_results"]
    ccf_csv_rows = []
    keys_fig2 = [
        ("K08_cowpox", "Cowpox serology (Kielder K08)"),
        ("K15_bartonella", "Bartonella spp. (Kielder K15)"),
        ("K15_babesia", "Babesia microti (Kielder K15)"),
        ("F02_puuv_core", "Puumala virus (Konnevesi F02)"),
    ]
    for lag in range(13):
        row_d: dict[str, Any] = {"lag_28d_steps": lag}
        for key, _ in keys_fig2:
            r = ccf_res[key]
            obs_v = r["ccf_obs_0_to_12"][lag]
            w_v = r["ccf_weather_partial_0_to_12"][lag]
            th_pre = r["k9_predata_ccf_0_to_12"][lag]
            th_emp = r["k9_empspec_ccf_0_to_12"][lag]
            row_d[f"{key}_obs"] = "" if obs_v is None else round(obs_v, 4)
            row_d[f"{key}_weather_adj"] = "" if w_v is None else round(w_v, 4)
            row_d[f"{key}_k9_predata"] = round(th_pre, 4)
            row_d[f"{key}_k9_empspec"] = round(th_emp, 4)
        ccf_csv_rows.append(row_d)

    p2_csv = os.path.join(fig_dir, "fig2_pathogen_ccf_tracking.csv")
    _write_csv(p2_csv, list(ccf_csv_rows[0].keys()), ccf_csv_rows)
    written.append(p2_csv)

    svg2 = _build_fig2_svg(ccf_res, keys_fig2)
    p2_svg = os.path.join(fig_dir, "fig2_pathogen_ccf_tracking.svg")
    with open(p2_svg, "w", encoding="utf-8") as f:
        f.write(svg2)
    written.append(p2_svg)

    p2_tex = os.path.join(fig_dir, "fig2_pathogen_ccf_tracking.tex")
    with open(p2_tex, "w", encoding="utf-8") as f:
        f.write(_build_fig2_tex())
    written.append(p2_tex)

    # -----------------------------------------------------------------------
    # FIGURE 3: Engine Sensitivity & Weather Thermal / Joint Loss Decomposition
    # -----------------------------------------------------------------------
    eng_rows = results["sensitivity_engine"]["rows"]
    therm_rows = results["weather_and_environment"]["thermal_stage_results"]

    p3a_csv = os.path.join(fig_dir, "fig3a_engine_mechanisms.csv")
    if eng_rows:
        _write_csv(p3a_csv, list(eng_rows[0].keys()), eng_rows)
        written.append(p3a_csv)

    p3b_csv = os.path.join(fig_dir, "fig3b_weather_thermal_losses.csv")
    if therm_rows:
        _write_csv(p3b_csv, list(therm_rows[0].keys()), therm_rows)
        written.append(p3b_csv)

    svg3 = _build_fig3_svg(eng_rows, therm_rows)
    p3_svg = os.path.join(fig_dir, "fig3_mechanisms_and_weather.svg")
    with open(p3_svg, "w", encoding="utf-8") as f:
        f.write(svg3)
    written.append(p3_svg)

    p3_tex = os.path.join(fig_dir, "fig3_mechanisms_and_weather.tex")
    with open(p3_tex, "w", encoding="utf-8") as f:
        f.write(_build_fig3_tex())
    written.append(p3_tex)

    return written


def _panel_axes(
    lines: list[str],
    x0: float,
    y0: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[float],
    yticks: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    zero_hline: bool = True,
) -> tuple[Any, Any]:
    def sx(v: float) -> float:
        return x0 + (v - xmin) / max(xmax - xmin, 1e-12) * w

    def sy(v: float) -> float:
        return y0 + h - (v - ymin) / max(ymax - ymin, 1e-12) * h

    lines.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#fafafa" stroke="#444" stroke-width="1"/>')
    if zero_hline and ymin < 0 < ymax:
        zy = sy(0.0)
        lines.append(f'<line x1="{x0:.1f}" y1="{zy:.1f}" x2="{x0+w:.1f}" y2="{zy:.1f}" stroke="#888" stroke-width="1" stroke-dasharray="4,3"/>')

    for xt in xticks:
        px = sx(xt)
        lines.append(f'<line x1="{px:.1f}" y1="{y0+h:.1f}" x2="{px:.1f}" y2="{y0+h+4:.1f}" stroke="#333"/>')
        lbl = f"{int(xt)}" if float(xt).is_integer() else f"{xt:.1f}"
        lines.append(f'<text x="{px:.1f}" y="{y0+h+16:.1f}" font-size="10" text-anchor="middle" fill="#222">{lbl}</text>')

    for yt in yticks:
        py = sy(yt)
        lines.append(f'<line x1="{x0-4:.1f}" y1="{py:.1f}" x2="{x0:.1f}" y2="{py:.1f}" stroke="#333"/>')
        lines.append(f'<line x1="{x0:.1f}" y1="{py:.1f}" x2="{x0+w:.1f}" y2="{py:.1f}" stroke="#eaeaea" stroke-width="0.7"/>')
        lbl = f"{yt:.1f}" if abs(yt) < 10 else f"{yt:.0f}"
        lines.append(f'<text x="{x0-7:.1f}" y="{py+3.5:.1f}" font-size="10" text-anchor="end" fill="#222">{lbl}</text>')

    lines.append(f'<text x="{x0 + 0.5*w:.1f}" y="{y0 - 8:.1f}" font-size="12" font-weight="bold" text-anchor="middle" fill="#111">{_esc(title)}</text>')
    lines.append(f'<text x="{x0 + 0.5*w:.1f}" y="{y0 + h + 31:.1f}" font-size="11" text-anchor="middle" fill="#222">{_esc(xlabel)}</text>')
    lines.append(
        f'<text x="{x0 - 36:.1f}" y="{y0 + 0.5*h:.1f}" font-size="11" text-anchor="middle" fill="#222" transform="rotate(-90 {x0 - 36:.1f} {y0 + 0.5*h:.1f})">{_esc(ylabel)}</text>'
    )
    return sx, sy


def _polyline(lines: list[str], pts: list[tuple[float, float]], color: str, width: float = 1.8, dash: str = "") -> None:
    if len(pts) < 2:
        return
    pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{d_attr} points="{pstr}"/>')


def _build_fig1_svg(
    f02_win: list[dict[str, Any]],
    k15_win: dict[str, list[dict[str, Any]]],
    acf_rows: list[dict[str, Any]],
    mem: dict[str, Any],
) -> str:
    W, H = 940, 560
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="470" y="22" font-size="14" font-weight="bold" text-anchor="middle" fill="#111">Figure 1. Empirical 28-Day Host Abundance &amp; Autocorrelation Structure (Kielder Field Voles &amp; Konnevesi Bank Voles)</text>',
    ]

    # Panel A: Kielder K15 trajectories (centred ln H across grids)
    sx, sy = _panel_axes(
        lines, 62, 52, 390, 195, 0, 35, -2.2, 2.2, [0, 7, 13, 20, 26, 33], [-2, -1, 0, 1, 2],
        "A. Kielder M. agrestis (2015-2017, 7 clear-cut grids)", "28-day step k from March 2015", "Centred ln H"
    )
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    for idx_s, (site, s_list) in enumerate(sorted(k15_win.items())):
        col = palette[idx_s % len(palette)]
        seg: list[tuple[float, float]] = []
        prev_k = None
        for w in s_list:
            if w["z_centred"] is None:
                continue
            k = w["step_k"]
            if prev_k is not None and k - prev_k > 3:
                _polyline(lines, seg, col, 1.4)
                seg = []
            seg.append((sx(k), sy(max(-2.15, min(2.15, w["z_centred"])))))
            lines.append(f'<circle cx="{sx(k):.1f}" cy="{sy(max(-2.15, min(2.15, w["z_centred"]))):.1f}" r="2.3" fill="{col}"/>')
            prev_k = k
        _polyline(lines, seg, col, 1.4)

    # Panel B: Konnevesi F02 core 7-year cyclic bank vole trajectory
    sx2, sy2 = _panel_axes(
        lines, 525, 52, 390, 195, 0, 92, -3.0, 2.5, [0, 13, 26, 39, 52, 65, 78, 91], [-3, -2, -1, 0, 1, 2],
        "B. Konnevesi M. glareolus core grid (2002-2009, 3 cycles)", "28-day step k from April 2002", "Centred ln H"
    )
    pts_f02 = []
    for w in f02_win:
        if w["z_centred"] is not None:
            px, py = sx2(w["step_k"]), sy2(max(-2.95, min(2.45, w["z_centred"])))
            pts_f02.append((px, py))
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="#0b559f"/>')
    _polyline(lines, pts_f02, "#0b559f", 1.7)

    # Panel C: Kielder K15 pooled ACF vs AR(1) vs Best model
    sx3, sy3 = _panel_axes(
        lines, 62, 315, 390, 195, 0, 12, -0.8, 1.05, list(range(0, 13, 2)), [-0.8, -0.4, 0.0, 0.4, 0.8, 1.0],
        f"C. Kielder 28-day ACF (dAICc vs AR1 = {mem['k15_pooled']['delta_aicc_beyond_ar1']:+.1f})",
        "Lag l (28-day steps; 6.5 steps = 6 months)", "Autocorrelation rho_H(l)"
    )
    pts_k_ar1 = [(sx3(r["lag_28d_steps"]), sy3(float(r["kielder_ar1_fit"]))) for r in acf_rows]
    pts_k_best = [(sx3(r["lag_28d_steps"]), sy3(float(r["kielder_best_fit"]))) for r in acf_rows]
    _polyline(lines, pts_k_ar1, "#888888", 1.8, "5,4")
    _polyline(lines, pts_k_best, "#d62728", 2.2)
    for r in acf_rows:
        if r["kielder_emp_acf"] != "":
            px, py = sx3(r["lag_28d_steps"]), sy3(max(-0.78, min(1.02, float(r["kielder_emp_acf"]))))
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="#111"/>')

    # Panel D: Konnevesi F02 core ACF vs AR(1) vs Best model
    sx4, sy4 = _panel_axes(
        lines, 525, 315, 390, 195, 0, 12, -0.8, 1.05, list(range(0, 13, 2)), [-0.8, -0.4, 0.0, 0.4, 0.8, 1.0],
        f"D. Konnevesi 28-day ACF (dAICc vs AR1 = {mem['f02_core']['delta_aicc_beyond_ar1']:+.1f})",
        "Lag l (28-day steps; 13 steps = 1 year)", "Autocorrelation rho_H(l)"
    )
    pts_f_ar1 = [(sx4(r["lag_28d_steps"]), sy4(float(r["konnevesi_ar1_fit"]))) for r in acf_rows]
    pts_f_best = [(sx4(r["lag_28d_steps"]), sy4(float(r["konnevesi_best_fit"]))) for r in acf_rows]
    _polyline(lines, pts_f_ar1, "#888888", 1.8, "5,4")
    _polyline(lines, pts_f_best, "#d62728", 2.2)
    for r in acf_rows:
        if r["konnevesi_emp_acf"] != "":
            px, py = sx4(r["lag_28d_steps"]), sy4(max(-0.78, min(1.02, float(r["konnevesi_emp_acf"]))))
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="#111"/>')

    # Legend at bottom
    lines.append('<circle cx="230" cy="548" r="3.6" fill="#111"/><text x="240" y="552" font-size="11" fill="#111">Empirical 28-day ACF</text>')
    lines.append('<line x1="390" y1="548" x2="420" y2="548" stroke="#888" stroke-width="2" stroke-dasharray="5,4"/><text x="427" y="552" font-size="11" fill="#111">Fitted AR(1) (geometric)</text>')
    lines.append('<line x1="585" y1="548" x2="615" y2="548" stroke="#d62728" stroke-width="2.2"/><text x="622" y="552" font-size="11" fill="#111">Best Seasonal + AR/CAR(2) fit</text>')
    lines.append("</svg>\n")
    return "\n".join(lines)


def _build_fig2_svg(ccf_res: dict[str, Any], keys_fig2: list[tuple[str, str]]) -> str:
    W, H = 940, 560
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="470" y="22" font-size="14" font-weight="bold" text-anchor="middle" fill="#111">Figure 2. Observed vs K9 Predicted Cross-Correlation Corr(ln prevalence(t + l), ln host(t)) Across Lags 0..12 Steps</text>',
    ]
    coords = [(62, 52), (525, 52), (62, 312), (525, 312)]
    panel_lets = ["A", "B", "C", "D"]
    for idx_p, ((key, label), (x0, y0)) in enumerate(zip(keys_fig2, coords)):
        r = ccf_res[key]
        verd = r["verdict_T2_track"]
        pk_obs = r["peak_lag_0_to_8"]
        lcom = r["L_com_0_to_8"]
        title = f"{panel_lets[idx_p]}. {label} [{verd}; peak={pk_obs}, L_com={lcom:.2f}]"
        sx, sy = _panel_axes(
            lines, x0, y0, 390, 192, 0, 12, -0.85, 0.95, list(range(0, 13, 2)), [-0.8, -0.4, 0.0, 0.4, 0.8],
            title, "Lag l (28-day steps: pathogen after host)", "Cross-correlation CCF(l)"
        )
        # Null 95% cutoff horizontal dotted line
        cut = r["null_95pct_S_pos"]
        if -0.85 < cut < 0.95:
            cy = sy(cut)
            lines.append(f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x0+390:.1f}" y2="{cy:.1f}" stroke="#2ca02c" stroke-width="1.1" stroke-dasharray="2,3"/>')

        pts_pre = [(sx(l), sy(max(-0.83, min(0.93, r["k9_predata_ccf_0_to_12"][l])))) for l in range(13)]
        pts_emp = [(sx(l), sy(max(-0.83, min(0.93, r["k9_empspec_ccf_0_to_12"][l])))) for l in range(13)]
        _polyline(lines, pts_pre, "#888888", 1.6, "5,3")
        _polyline(lines, pts_emp, "#d62728", 2.0)

        pts_obs = []
        pts_w = []
        for l in range(13):
            ov = r["ccf_obs_0_to_12"][l]
            wv = r["ccf_weather_partial_0_to_12"][l]
            if ov is not None:
                px, py = sx(l), sy(max(-0.83, min(0.93, ov)))
                pts_obs.append((px, py))
                lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#0b559f"/>')
            if wv is not None:
                pxw, pyw = sx(l), sy(max(-0.83, min(0.93, wv)))
                pts_w.append((pxw, pyw))
                lines.append(f'<rect x="{pxw-2.5:.1f}" y="{pyw-2.5:.1f}" width="5" height="5" fill="none" stroke="#ff7f0e" stroke-width="1.3"/>')
        _polyline(lines, pts_obs, "#0b559f", 1.9)
        _polyline(lines, pts_w, "#ff7f0e", 1.3, "3,2")

    lines.append('<circle cx="95" cy="547" r="3.5" fill="#0b559f"/><text x="104" y="551" font-size="10.5" fill="#111">Observed CCF</text>')
    lines.append('<rect x="205" y="544" width="6" height="6" fill="none" stroke="#ff7f0e" stroke-width="1.4"/><text x="216" y="551" font-size="10.5" fill="#111">Weather-adjusted partial CCF (T, P)</text>')
    lines.append('<line x1="415" y1="547" x2="442" y2="547" stroke="#d62728" stroke-width="2"/><text x="448" y="551" font-size="10.5" fill="#111">K9 theory (empirical host PSD)</text>')
    lines.append('<line x1="625" y1="547" x2="652" y2="547" stroke="#888" stroke-width="1.7" stroke-dasharray="5,3"/><text x="658" y="551" font-size="10.5" fill="#111">K9 pre-data</text>')
    lines.append('<line x1="745" y1="547" x2="768" y2="547" stroke="#2ca02c" stroke-width="1.2" stroke-dasharray="2,3"/><text x="774" y="551" font-size="10.5" fill="#111">95% surrogate cutoff</text>')
    lines.append("</svg>\n")
    return "\n".join(lines)


def _build_fig3_svg(eng_rows: list[dict[str, Any]], therm_rows: list[dict[str, Any]]) -> str:
    W, H = 940, 430
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="470" y="22" font-size="14" font-weight="bold" text-anchor="middle" fill="#111">Figure 3. Lifecycle-Engine Mechanism Sensitivity (D) and Empirical Weather Thermal / Joint Loss Decomposition</text>',
    ]

    # Panel A: Horizontal dot/bar comparison of Engine D across mechanisms for Bartonella & PUUV
    x0, y0, pw, ph = 185, 55, 290, 320
    lines.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#444" stroke-width="1"/>')
    lines.append(f'<text x="{x0 + 0.5*pw:.1f}" y="{y0 - 8:.1f}" font-size="12" font-weight="bold" text-anchor="middle" fill="#111">A. lifecycle_engine.py loss D (ln R0 units)</text>')
    d_vals = [float(r["D_sim"]) for r in eng_rows] if eng_rows else [0.0, 0.2]
    dmin = min(-0.06, min(d_vals) - 0.02)
    dmax = max(0.22, max(d_vals) + 0.03)

    def sxa(v: float) -> float:
        return x0 + (v - dmin) / max(dmax - dmin, 1e-9) * pw

    z0 = sxa(0.0)
    lines.append(f'<line x1="{z0:.1f}" y1="{y0}" x2="{z0:.1f}" y2="{y0+ph}" stroke="#666" stroke-dasharray="4,3"/>')
    for xt in [-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        if dmin <= xt <= dmax:
            px = sxa(xt)
            lines.append(f'<line x1="{px:.1f}" y1="{y0+ph}" x2="{px:.1f}" y2="{y0+ph+4}" stroke="#333"/>')
            lines.append(f'<text x="{px:.1f}" y="{y0+ph+16}" font-size="9.5" text-anchor="middle" fill="#222">{xt:.2f}</text>')
    lines.append(f'<text x="{x0 + 0.5*pw:.1f}" y="{y0+ph+32}" font-size="11" text-anchor="middle" fill="#222">Threshold loss D (&lt;0 is fluctuation gain)</text>')

    n_r = max(len(eng_rows), 1)
    for i, r in enumerate(eng_rows):
        ry = y0 + (i + 0.6) * (ph / n_r)
        dv = float(r["D_sim"])
        d2v = float(r["D2_coeff"]) if r.get("D2_coeff") not in ("", None) else None
        lbl = f"{r['parasite_short']}: {r['variant_short']}"
        lines.append(f'<text x="{x0 - 6}" y="{ry + 3.5:.1f}" font-size="9.5" text-anchor="end" fill="#222">{_esc(lbl)}</text>')
        px = sxa(dv)
        col = "#2ca02c" if dv < 0 else "#0b559f"
        lines.append(f'<line x1="{z0:.1f}" y1="{ry:.1f}" x2="{px:.1f}" y2="{ry:.1f}" stroke="{col}" stroke-width="2.2"/>')
        lines.append(f'<circle cx="{px:.1f}" cy="{ry:.1f}" r="3.3" fill="{col}"/>')
        if d2v is not None and dmin <= d2v <= dmax:
            px2 = sxa(d2v)
            lines.append(f'<circle cx="{px2:.1f}" cy="{ry:.1f}" r="2.6" fill="none" stroke="#d62728" stroke-width="1.3"/>')

    # Panel B: Weather Thermal & Joint Loss Comparison
    x1, y1, pw2, ph2 = 630, 55, 280, 320
    lines.append(f'<rect x="{x1}" y="{y1}" width="{pw2}" height="{ph2}" fill="#fafafa" stroke="#444" stroke-width="1"/>')
    lines.append(f'<text x="{x1 + 0.5*pw2:.1f}" y="{y1 - 8:.1f}" font-size="12" font-weight="bold" text-anchor="middle" fill="#111">B. Empirical Weather: Host vs Joint (H, T) Loss</text>')
    t_vals = []
    for r in therm_rows:
        t_vals.extend([float(r["D_host_only"]), float(r["D_temp_only"]), float(r["D_joint_HT"]), float(r["J_temp_curvature"])])
    tmin = min(-0.08, (min(t_vals) - 0.02) if t_vals else -0.08)
    tmax = max(0.22, (max(t_vals) + 0.03) if t_vals else 0.22)

    def sxb(v: float) -> float:
        return x1 + (v - tmin) / max(tmax - tmin, 1e-9) * pw2

    zb0 = sxb(0.0)
    lines.append(f'<line x1="{zb0:.1f}" y1="{y1}" x2="{zb0:.1f}" y2="{y1+ph2}" stroke="#666" stroke-dasharray="4,3"/>')
    for xt in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        if tmin <= xt <= tmax:
            px = sxb(xt)
            lines.append(f'<line x1="{px:.1f}" y1="{y1+ph2}" x2="{px:.1f}" y2="{y1+ph2+4}" stroke="#333"/>')
            lines.append(f'<text x="{px:.1f}" y="{y1+ph2+16}" font-size="9.5" text-anchor="middle" fill="#222">{xt:.2f}</text>')
    lines.append(f'<text x="{x1 + 0.5*pw2:.1f}" y="{y1+ph2+32}" font-size="11" text-anchor="middle" fill="#222">ln R0 shift / loss along observed trajectories</text>')

    nt = max(len(therm_rows), 1)
    for i, r in enumerate(therm_rows):
        ry = y1 + (i + 0.55) * (ph2 / nt)
        lines.append(f'<text x="{x1 - 6}" y="{ry + 3.5:.1f}" font-size="9.5" text-anchor="end" fill="#222">{_esc(r["label_short"])}</text>')
        for off_y, key_m, col in [(-6.0, "D_host_only", "#0b559f"), (0.0, "D_temp_only", "#ff7f0e"), (6.0, "D_joint_HT", "#d62728")]:
            v = float(r[key_m])
            px = sxb(v)
            lines.append(f'<line x1="{zb0:.1f}" y1="{ry+off_y:.1f}" x2="{px:.1f}" y2="{ry+off_y:.1f}" stroke="{col}" stroke-width="2.3"/>')
            lines.append(f'<circle cx="{px:.1f}" cy="{ry+off_y:.1f}" r="2.8" fill="{col}"/>')

    lines.append('<circle cx="195" cy="416" r="3" fill="#0b559f"/><text x="203" y="420" font-size="10" fill="#111">Simulated D (open red circle = D2)</text>')
    lines.append('<circle cx="545" cy="416" r="3" fill="#0b559f"/><text x="553" y="420" font-size="10" fill="#111">D_host_only</text>')
    lines.append('<circle cx="645" cy="416" r="3" fill="#ff7f0e"/><text x="653" y="420" font-size="10" fill="#111">D_temp_only</text>')
    lines.append('<circle cx="745" cy="416" r="3" fill="#d62728"/><text x="753" y="420" font-size="10" fill="#111">D_joint(H, T)</text>')
    lines.append("</svg>\n")
    return "\n".join(lines)


def _build_fig1_tex() -> str:
    return r"""% pgfplots source for Figure 1 (reads fig1b_host_acf_fits.csv)
\begin{tikzpicture}
\begin{axis}[
    width=0.47\textwidth, height=0.34\textwidth,
    xlabel={Lag $\ell$ (28-day steps)}, ylabel={Autocorrelation $\rho_H(\ell)$},
    title={Kielder \emph{M.\ agrestis} 28-day ACF},
    ymin=-0.8, ymax=1.05, xmin=0, xmax=12, grid=major
]
\addplot[only marks, mark=*] table[col sep=comma, x=lag_28d_steps, y=kielder_emp_acf] {fig1b_host_acf_fits.csv};
\addplot[dashed, thick, gray] table[col sep=comma, x=lag_28d_steps, y=kielder_ar1_fit] {fig1b_host_acf_fits.csv};
\addplot[thick, red] table[col sep=comma, x=lag_28d_steps, y=kielder_best_fit] {fig1b_host_acf_fits.csv};
\end{axis}
\end{tikzpicture}
"""


def _build_fig2_tex() -> str:
    return r"""% pgfplots source for Figure 2 (reads fig2_pathogen_ccf_tracking.csv)
\begin{tikzpicture}
\begin{axis}[
    width=0.47\textwidth, height=0.34\textwidth,
    xlabel={Lag $\ell$ (28-day steps)}, ylabel={Cross-correlation $\mathrm{CCF}(\ell)$},
    title={Pathogen prevalence tracking vs.\ K9 prediction},
    ymin=-0.85, ymax=0.95, xmin=0, xmax=12, grid=major
]
\addplot[thick, blue, mark=*] table[col sep=comma, x=lag_28d_steps, y=F02_puuv_core_obs] {fig2_pathogen_ccf_tracking.csv};
\addplot[thick, red] table[col sep=comma, x=lag_28d_steps, y=F02_puuv_core_k9_empspec] {fig2_pathogen_ccf_tracking.csv};
\addplot[dashed, orange] table[col sep=comma, x=lag_28d_steps, y=F02_puuv_core_weather_adj] {fig2_pathogen_ccf_tracking.csv};
\end{axis}
\end{tikzpicture}
"""


def _build_fig3_tex() -> str:
    return r"""% pgfplots source for Figure 3 (reads fig3a_engine_mechanisms.csv and fig3b_weather_thermal_losses.csv)
\begin{tikzpicture}
\begin{axis}[
    width=0.65\textwidth, height=0.40\textwidth,
    xlabel={Threshold $\ln R_0$ loss $D$},
    symbolic y coords={0,1,2,3,4,5},
    grid=major
]
\addplot[only marks, mark=*, blue] table[col sep=comma, x=D_sim, y expr=\coordindex] {fig3a_engine_mechanisms.csv};
\end{axis}
\end{tikzpicture}
"""
