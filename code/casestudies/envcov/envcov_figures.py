"""Generate plain SVG figures, TSV data tables, and pgfplots .tex files for envcov.

Rule 7 of CASESTUDY_SPEC.md: matplotlib is broken on this machine; figures are written
directly as clean vector SVG files plus TSV data tables and pgfplots LaTeX sources.
"""
from __future__ import annotations

from html import escape
from pathlib import Path


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<style>',
        '  text { font-family: Helvetica, Arial, sans-serif; fill: #1f2933; }',
        '  .title { font-size: 13px; font-weight: bold; }',
        '  .axis-label { font-size: 11px; }',
        '  .tick { font-size: 10px; fill: #52606d; }',
        '  .site { font-size: 10px; font-weight: bold; }',
        '  .grid { stroke: #e4e7eb; stroke-width: 1; }',
        '  .axis { stroke: #3e4c59; stroke-width: 1.2; }',
        '  .zero { stroke: #7b8794; stroke-width: 1; stroke-dasharray: 4,3; }',
        '</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def write_figure1_neon_confounding(fig_dir: Path, neon_sites: list[dict], partial_rows: list[dict]) -> None:
    """Figure 1: Two-panel SVG + TSV + pgfplots .tex for NEON across-site confounding."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    tsv1 = fig_dir / "fig1a_neon_sites.tsv"
    lines = ["site\tsigma_site\tnip_site\tpc1_env\tdd6\twmin_c\tsnow_days"]
    for r in neon_sites:
        lines.append(
            f"{r['site']}\t{r['sigma']:.4f}\t{r['nip']:.4f}\t{r['PC1_env']:.4f}\t"
            f"{r['C1_dd6']:.1f}\t{r['C3_wmin']:.2f}\t{r['C4_snow']:.2f}"
        )
    tsv1.write_text("\n".join(lines) + "\n")

    tsv2 = fig_dir / "fig1b_neon_adjustments.tsv"
    lines2 = ["label\tr_stat\tp_neg\tp_pos"]
    for r in partial_rows:
        lines2.append(f"{r['label']}\t{r['r_stat']:.4f}\t{r['p_neg']:.4f}\t{r['p_pos']:.4f}")
    tsv2.write_text("\n".join(lines2) + "\n")

    w, h = 840, 360
    svg = _svg_header(w, height=h)

    # Panel A: NIP_site vs sigma_site coloured/annotated by PC1_env / winter cold & snow
    ax_x0, ax_y0, ax_w, ax_h = 65, 45, 305, 255
    svg.append(f'<text x="{ax_x0}" y="26" class="title">A. Six NEON sites: prevalence vs mouse variability</text>')
    x_min, x_max = 0.32, 1.08
    y_min, y_max = 0.00, 0.30

    def sx(v: float) -> float:
        return ax_x0 + (v - x_min) / (x_max - x_min) * ax_w

    def sy(v: float) -> float:
        return ax_y0 + ax_h - (v - y_min) / (y_max - y_min) * ax_h

    for xv in (0.4, 0.6, 0.8, 1.0):
        px = sx(xv)
        svg.append(f'<line x1="{px:.1f}" y1="{ax_y0}" x2="{px:.1f}" y2="{ax_y0 + ax_h}" class="grid"/>')
        svg.append(f'<text x="{px:.1f}" y="{ax_y0 + ax_h + 16}" text-anchor="middle" class="tick">{xv:.1f}</text>')
    for yv in (0.0, 0.1, 0.2, 0.3):
        py = sy(yv)
        svg.append(f'<line x1="{ax_x0}" y1="{py:.1f}" x2="{ax_x0 + ax_w}" y2="{py:.1f}" class="grid"/>')
        svg.append(f'<text x="{ax_x0 - 8}" y="{py + 4:.1f}" text-anchor="end" class="tick">{yv:.2f}</text>')

    svg.append(f'<line x1="{ax_x0}" y1="{ax_y0 + ax_h}" x2="{ax_x0 + ax_w}" y2="{ax_y0 + ax_h}" class="axis"/>')
    svg.append(f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y0 + ax_h}" class="axis"/>')
    svg.append(f'<text x="{ax_x0 + ax_w / 2:.1f}" y="{h - 15}" text-anchor="middle" class="axis-label">Mouse variability sigma_site (SD of ln Peromyscus index)</text>')
    svg.append(f'<text x="18" y="{ax_y0 + ax_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {ax_y0 + ax_h / 2:.1f})" class="axis-label">Nymphal prevalence NIP_site</text>')

    offsets = {
        "BLAN": (8, -6),
        "TREE": (8, 4),
        "ORNL": (8, -6),
        "SCBI": (8, 10),
        "SERC": (8, -7),
        "HARV": (-38, -8),
    }
    for r in neon_sites:
        px, py = sx(r["sigma"]), sy(r["nip"])
        # Warm sites (PC1_env > 0) in orange-red, cold/snowy sites (PC1_env <= 0) in steel-blue
        fill = "#d95f02" if r["PC1_env"] > 0 else "#2b6cb0"
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{fill}" stroke="#1f2933" stroke-width="1"/>')
        dx, dy = offsets.get(r["site"], (7, -5))
        svg.append(f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" class="site">{escape(r["site"])}</text>')

    # Legend inside panel A
    svg.append(f'<circle cx="{ax_x0 + 14}" cy="{ax_y0 + 16}" r="5" fill="#2b6cb0" stroke="#1f2933" stroke-width="0.8"/>')
    svg.append(f'<text x="{ax_x0 + 25}" y="{ax_y0 + 20}" class="tick">Cooler / snowy (PC1_env &lt;= 0)</text>')
    svg.append(f'<circle cx="{ax_x0 + 14}" cy="{ax_y0 + 32}" r="5" fill="#d95f02" stroke="#1f2933" stroke-width="0.8"/>')
    svg.append(f'<text x="{ax_x0 + 25}" y="{ax_y0 + 36}" class="tick">Warmer / longer season (PC1_env &gt; 0)</text>')

    # Panel B: Forest plot of unadjusted and single-covariate partial rank correlations
    bx0, by0, bw, bh = 520, 45, 270, 255
    svg.append(f'<text x="{bx0 - 95}" y="26" class="title">B. Rank association before and after 1-covariate adjustment (df = 3)</text>')

    def bx(v: float) -> float:
        return bx0 + (v - (-1.0)) / 2.0 * bw

    for rv in (-1.0, -0.5, 0.0, 0.5, 1.0):
        px = bx(rv)
        cls = "zero" if abs(rv) < 1e-9 else "grid"
        svg.append(f'<line x1="{px:.1f}" y1="{by0}" x2="{px:.1f}" y2="{by0 + bh}" class="{cls}"/>')
        svg.append(f'<text x="{px:.1f}" y="{by0 + bh + 16}" text-anchor="middle" class="tick">{rv:+.1f}</text>')

    svg.append(f'<line x1="{bx0}" y1="{by0 + bh}" x2="{bx0 + bw}" y2="{by0 + bh}" class="axis"/>')
    svg.append(f'<text x="{bx0 + bw / 2:.1f}" y="{h - 15}" text-anchor="middle" class="axis-label">Spearman / partial Spearman r(NIP_site, sigma_site | Z)</text>')

    n_rows = len(partial_rows)
    step_y = bh / max(n_rows, 1)
    z0_x = bx(0.0)
    for idx, r in enumerate(partial_rows):
        cy = by0 + (idx + 0.5) * step_y
        val_x = bx(max(-1.0, min(1.0, r["r_stat"])))
        bar_x = min(z0_x, val_x)
        bar_w = max(1.5, abs(val_x - z0_x))
        col = "#1b9e77" if r["r_stat"] < 0 else "#7570b3"
        if idx == 0:
            col = "#52606d"
        svg.append(f'<rect x="{bar_x:.1f}" y="{cy - 6:.1f}" width="{bar_w:.1f}" height="12" fill="{col}" opacity="0.85"/>')
        svg.append(f'<text x="{bx0 - 8}" y="{cy + 3.5:.1f}" text-anchor="end" class="tick">{escape(r["label"])}</text>')
        svg.append(f'<text x="{bx0 + bw + 5}" y="{cy + 3.5:.1f}" class="tick">{r["r_stat"]:+.2f}</text>')

    svg.append("</svg>")
    (fig_dir / "fig1_neon_confounding.svg").write_text("\n".join(svg) + "\n")

    # Companion pgfplots .tex
    tex = r"""\begin{tikzpicture}
\begin{axis}[
    width=7.5cm, height=6.0cm,
    xlabel={Mouse variability $\sigma_{\mathrm{site}}$},
    ylabel={Nymphal prevalence $\mathrm{NIP}_{\mathrm{site}}$},
    xmin=0.32, xmax=1.08, ymin=0.0, ymax=0.30,
    grid=major
]
\addplot[only marks, mark=*, mark size=2.5pt, nodes near coords, point meta=explicit symbolic]
    table[col sep=tab, x=sigma_site, y=nip_site, meta=site] {fig1a_neon_sites.tsv};
\end{axis}
\end{tikzpicture}
"""
    (fig_dir / "fig1_neon_confounding.tex").write_text(tex)


def write_figure2_cary_slopes(fig_dir: Path, cary_rows: list[dict]) -> None:
    """Figure 2: Forest plot of Cary within-plot lagged mouse slope under abiotic adjustments."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    tsv = fig_dir / "fig2_cary_slopes.tsv"
    lines = ["label\tslope\tse\tci_lo\tci_hi"]
    for r in cary_rows:
        lines.append(f"{r['label']}\t{r['b']:.4f}\t{r['se']:.4f}\t{r['b'] - 2 * r['se']:.4f}\t{r['b'] + 2 * r['se']:.4f}")
    tsv.write_text("\n".join(lines) + "\n")

    w, h = 750, 390
    svg = _svg_header(w, h)
    bx0, by0, bw, bh = 255, 45, 355, 290
    svg.append(f'<text x="30" y="26" class="title">Cary Institute (1995-2013, 102 plot-years): slope of ln NIP_(t+1) on ln mice_t (+- 2 SE)</text>')

    lo_all = min(-0.25, min(r["b"] - 2.0 * r["se"] for r in cary_rows) - 0.03)
    hi_all = max(0.35, max(r["b"] + 2.0 * r["se"] for r in cary_rows) + 0.03)
    x_min, x_max = round(lo_all, 2), round(hi_all, 2)

    def sx(v: float) -> float:
        return bx0 + (v - x_min) / (x_max - x_min) * bw

    for xv in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        px = sx(xv)
        cls = "zero" if abs(xv) < 1e-9 else "grid"
        svg.append(f'<line x1="{px:.1f}" y1="{by0}" x2="{px:.1f}" y2="{by0 + bh}" class="{cls}"/>')
        svg.append(f'<text x="{px:.1f}" y="{by0 + bh + 16}" text-anchor="middle" class="tick">{xv:+.2f}</text>')

    svg.append(f'<line x1="{bx0}" y1="{by0 + bh}" x2="{bx0 + bw}" y2="{by0 + bh}" class="axis"/>')
    svg.append(f'<text x="{bx0 + bw / 2:.1f}" y="{h - 14}" text-anchor="middle" class="axis-label">Within-plot slope b of ln NIP_(t+1) on ln mice_t</text>')

    step_y = bh / max(len(cary_rows), 1)
    for idx, r in enumerate(cary_rows):
        cy = by0 + (idx + 0.5) * step_y
        lo = max(x_min, r["b"] - 2.0 * r["se"])
        hi = min(x_max, r["b"] + 2.0 * r["se"])
        x_lo, x_hi, x_pt = sx(lo), sx(hi), sx(r["b"])
        col = "#1f2933" if idx == 0 else ("#2b6cb0" if "PC1" in r["label"] else "#52606d")
        svg.append(f'<line x1="{x_lo:.1f}" y1="{cy:.1f}" x2="{x_hi:.1f}" y2="{cy:.1f}" stroke="{col}" stroke-width="1.8"/>')
        svg.append(f'<circle cx="{x_pt:.1f}" cy="{cy:.1f}" r="4.2" fill="{col}"/>')
        svg.append(f'<text x="{bx0 - 10}" y="{cy + 3.5:.1f}" text-anchor="end" class="tick">{escape(r["label"])}</text>')
        svg.append(f'<text x="{bx0 + bw + 8}" y="{cy + 3.5:.1f}" class="tick">{r["b"]:+.3f} +- {r["se"]:.3f}</text>')

    svg.append("</svg>")
    (fig_dir / "fig2_cary_slopes.svg").write_text("\n".join(svg) + "\n")

    tex = r"""\begin{tikzpicture}
\begin{axis}[
    width=8.5cm, height=7.0cm,
    xlabel={Within-plot slope $b$ of $\ln\mathrm{NIP}_{t+1}$ on $\ln\mathrm{mice}_t$},
    ytick=data, yticklabels from table={[col sep=tab]fig2_cary_slopes.tsv}{label},
    grid=major
]
\addplot+[only marks, error bars/.cd, x dir=both, x explicit]
    table[col sep=tab, x=slope, y expr=\coordindex, x error expr=2*\thisrow{se}] {fig2_cary_slopes.tsv};
\end{axis}
\end{tikzpicture}
"""
    (fig_dir / "fig2_cary_slopes.tex").write_text(tex)


def write_figure3_durations_and_loss(fig_dir: Path, loss_rows: list[dict]) -> None:
    """Figure 3: Comparison of borrowed vs temperature-driven predicted loss D across sites."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    tsv = fig_dir / "fig3_predicted_loss_d.tsv"
    lines = ["site\td_r14_borrowed\td_fine_mem_phi0\td_phen_chain_phi0\td_phen_chain_phi03\tmean_stay_days\tcv_stay"]
    for r in loss_rows:
        lines.append(
            f"{r['site']}\t{r['d_m0']:.4f}\t{r['d_m3_mem_phi0']:.4f}\t"
            f"{r['d_m3_chain_phi0']:.4f}\t{r['d_m3_chain_phi03']:.4f}\t"
            f"{r['stay_days']:.1f}\t{r['stay_cv']:.4f}"
        )
    tsv.write_text("\n".join(lines) + "\n")

    w, h = 760, 360
    svg = _svg_header(w, h)
    ax0, ay0, aw, ah = 70, 48, 650, 245
    svg.append(f'<text x="{ax0}" y="26" class="title">Predicted fluctuation loss D of ln R0: borrowed fixed stay vs temperature-driven stay &amp; spread</text>')

    y_max = max(0.32, max(max(r["d_m0"], r["d_m3_chain_phi0"], r["d_m3_chain_phi03"]) for r in loss_rows) * 1.15)

    def sy(v: float) -> float:
        return ay0 + ah - (v / y_max) * ah

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = frac * y_max
        py = sy(yv)
        svg.append(f'<line x1="{ax0}" y1="{py:.1f}" x2="{ax0 + aw}" y2="{py:.1f}" class="grid"/>')
        svg.append(f'<text x="{ax0 - 8}" y="{py + 3.5:.1f}" text-anchor="end" class="tick">{yv:.2f}</text>')

    svg.append(f'<line x1="{ax0}" y1="{ay0 + ah}" x2="{ax0 + aw}" y2="{ay0 + ah}" class="axis"/>')
    svg.append(f'<line x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay0 + ah}" class="axis"/>')
    svg.append(f'<text x="20" y="{ay0 + ah / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {ay0 + ah / 2:.1f})" class="axis-label">Predicted loss D of ln R0 (kappa = 1)</text>')

    gw = aw / len(loss_rows)
    colors = [
        ("d_m0", "#7b8794", "M0: R12/R14 borrowed (10 mo, geometric, dt=4.4 mo, phi=0)"),
        ("d_m3_mem_phi0", "#2b6cb0", "M3a: Thermal stay, memoryless (dt=6 d, phi=0)"),
        ("d_m3_chain_phi0", "#1b9e77", "M3b: Thermal stay + empirical CV chain (dt=6 d, phi=0)"),
        ("d_m3_chain_phi03", "#d95f02", "M3c: Thermal stay + empirical CV chain (dt=6 d, phi_4.4mo=0.30)"),
    ]
    bw = gw * 0.17
    for i, r in enumerate(loss_rows):
        gx = ax0 + (i + 0.14) * gw
        for k_idx, (key, col, _lab) in enumerate(colors):
            val = r[key]
            py = sy(val)
            bh_val = max(1.0, ay0 + ah - py)
            bx = gx + k_idx * (bw + 2.0)
            svg.append(f'<rect x="{bx:.1f}" y="{py:.1f}" width="{bw:.1f}" height="{bh_val:.1f}" fill="{col}"/>')
        cx = gx + 2 * (bw + 2.0)
        svg.append(f'<text x="{cx:.1f}" y="{ay0 + ah + 16}" text-anchor="middle" class="site">{escape(r["site"])}</text>')
        svg.append(f'<text x="{cx:.1f}" y="{ay0 + ah + 29}" text-anchor="middle" class="tick">({r["stay_days"]:.0f}d, CV {r["stay_cv"]:.2f})</text>')

    # Legend
    for k_idx, (_key, col, lab) in enumerate(colors):
        lx = ax0 + 16 + (k_idx // 2) * 320
        ly = ay0 + 8 + (k_idx % 2) * 16
        svg.append(f'<rect x="{lx}" y="{ly - 8}" width="10" height="10" fill="{col}"/>')
        svg.append(f'<text x="{lx + 15}" y="{ly}" class="tick">{escape(lab)}</text>')

    svg.append("</svg>")
    (fig_dir / "fig3_predicted_loss_d.svg").write_text("\n".join(svg) + "\n")

    tex = r"""\begin{tikzpicture}
\begin{axis}[
    ybar, bar width=5pt, width=12cm, height=6.5cm,
    symbolic x coords={CARY,BLAN,HARV,ORNL,SCBI,SERC,TREE},
    xtick=data, ylabel={Predicted loss $D$ of $\ln R_0$},
    ymin=0, grid=major, legend pos=north east
]
\addplot table[col sep=tab, x=site, y=d_r14_borrowed] {fig3_predicted_loss_d.tsv};
\addplot table[col sep=tab, x=site, y=d_fine_mem_phi0] {fig3_predicted_loss_d.tsv};
\addplot table[col sep=tab, x=site, y=d_phen_chain_phi0] {fig3_predicted_loss_d.tsv};
\addplot table[col sep=tab, x=site, y=d_phen_chain_phi03] {fig3_predicted_loss_d.tsv};
\legend{M0 borrowed, M3a thermal mem ($\phi=0$), M3b thermal chain ($\phi=0$), M3c thermal chain ($\phi=0.3$)}
\end{axis}
\end{tikzpicture}
"""
    (fig_dir / "fig3_predicted_loss_d.tex").write_text(tex)


def write_figure4_masking_mimicking(fig_dir: Path, mask_rows: list[dict], n_sim: int = 2000) -> None:
    """Figure 4: Frequencies of theory sign (rho < 0) and registered support under weather regimes."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    tsv = fig_dir / "fig4_masking_mimicking.tsv"
    lines = ["scenario\tkappa\tp_sign_unadj\tp_sup_unadj\tp_pos49_unadj\tp_sign_adj\tp_sup_adj"]
    for r in mask_rows:
        lines.append(
            f"{r['scenario']}\t{r['kappa']:.2f}\t{r['p_sign_unadj']:.4f}\t"
            f"{r['p_sup_unadj']:.4f}\t{r['p_pos49_unadj']:.4f}\t"
            f"{r['p_sign_adj']:.4f}\t{r['p_sup_adj']:.4f}"
        )
    tsv.write_text("\n".join(lines) + "\n")

    w, h = 840, 360
    svg = _svg_header(w, h)
    ax0, ay0, aw, ah = 255, 48, 450, 255
    svg.append(f'<text x="28" y="26" class="title">Part 4 simulation (N = 6 sites): probability of negative rank sign and p &lt; 0.05 support</text>')

    def sx(v: float) -> float:
        return ax0 + v * aw

    for xv in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        px = sx(xv)
        cls = "zero" if abs(xv - 0.5) < 1e-9 else "grid"
        svg.append(f'<line x1="{px:.1f}" y1="{ay0}" x2="{px:.1f}" y2="{ay0 + ah}" class="{cls}"/>')
        svg.append(f'<text x="{px:.1f}" y="{ay0 + ah + 16}" text-anchor="middle" class="tick">{xv:.1f}</text>')

    svg.append(f'<line x1="{ax0}" y1="{ay0 + ah}" x2="{ax0 + aw}" y2="{ay0 + ah}" class="axis"/>')
    svg.append(f'<text x="{ax0 + aw / 2:.1f}" y="{h - 14}" text-anchor="middle" class="axis-label">Simulated frequency across {n_sim:,d} synthetic 6-site datasets</text>')

    step_y = ah / max(len(mask_rows), 1)
    for idx, r in enumerate(mask_rows):
        cy = ay0 + (idx + 0.5) * step_y
        lab = f"{r['scenario']} (kappa={r['kappa']:.1f})"
        svg.append(f'<text x="{ax0 - 8}" y="{cy + 3.5:.1f}" text-anchor="end" class="tick">{escape(lab)}</text>')
        w_un = max(1.5, r["p_sign_unadj"] * aw)
        w_ad = max(1.5, r["p_sign_adj"] * aw)
        svg.append(f'<rect x="{ax0}" y="{cy - 7:.1f}" width="{w_un:.1f}" height="6.5" fill="#7570b3" opacity="0.85"/>')
        svg.append(f'<rect x="{ax0}" y="{cy + 0.5:.1f}" width="{w_ad:.1f}" height="6.5" fill="#1b9e77" opacity="0.85"/>')
        svg.append(
            f'<text x="{ax0 + aw + 6}" y="{cy + 3.5:.1f}" class="tick">'
            f'sup: {100 * r["p_sup_unadj"]:.1f}% / {100 * r["p_sup_adj"]:.1f}%</text>'
        )

    svg.append(f'<rect x="{ax0 + 10}" y="{ay0 + 8}" width="10" height="7" fill="#7570b3"/>')
    svg.append(f'<text x="{ax0 + 25}" y="{ay0 + 15}" class="tick">Unadjusted P(rho &lt; 0)</text>')
    svg.append(f'<rect x="{ax0 + 175}" y="{ay0 + 8}" width="10" height="7" fill="#1b9e77"/>')
    svg.append(f'<text x="{ax0 + 190}" y="{ay0 + 15}" class="tick">PC1_env-adjusted P(r_part &lt; 0) [right text: P(p&lt;0.05)]</text>')

    svg.append("</svg>")
    (fig_dir / "fig4_masking_mimicking.svg").write_text("\n".join(svg) + "\n")

    tex = r"""\begin{tikzpicture}
\begin{axis}[
    xbar, bar width=4pt, width=9.5cm, height=7.0cm,
    xmin=0, xmax=1, xlabel={Simulated frequency},
    ytick=data, yticklabels from table={[col sep=tab]fig4_masking_mimicking.tsv}{scenario},
    grid=major
]
\addplot table[col sep=tab, y expr=\coordindex, x=p_sign_unadj] {fig4_masking_mimicking.tsv};
\addplot table[col sep=tab, y expr=\coordindex, x=p_sign_adj] {fig4_masking_mimicking.tsv};
\legend{Unadjusted $P(\rho<0)$, Adjusted $P(r_{\mathrm{part}}<0)$}
\end{axis}
\end{tikzpicture}
"""
    (fig_dir / "fig4_masking_mimicking.tex").write_text(tex)

