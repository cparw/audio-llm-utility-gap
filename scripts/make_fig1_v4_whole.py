#!/usr/bin/env python3
# make_fig1_v4_whole.py (23 Sep 2026): copy of
# scripts/make_fig1_v4.py in this repository
# (sha256 0806d85429c1d9c9b6178c84fdb08e83c33cc61cf10d9d28f54911f3fc8fe20d).
# The drawing code is v4's own. Changed, and nothing else in the drawing:
#   1. E-DAIC data paths -> PART 23 whole interview run, column auc_oof
#      (o25_whole_encoder_perlayer.csv, o25_whole_llm_perlayer.csv).
#   2. E-DAIC answer line -> 0.8366 (whole interview zero shot); it keeps v4's
#      E-DAIC style ls=":" (dotted).
#   3. Font sizes: the two 8 pt notes ("audio encoder", "language model") -> 9 pt.
#      Everything else was already 9 or 10 pt and is unchanged.
#   4. Font: STIXGeneral only (v4 listed Times New Roman first); pdf.fonttype 42 kept.
# Output handling only (no drawing change): default output is fig1_gap_whole.pdf in the
# current folder, an existing output is never overwritten, the png is
# 200 dpi, and the arrays each Line2D holds are dumped to <OUT>_plotted.csv.
"""Figure 1, v2 spec (22 Sep 2026).

One axes. Three datasets, one probe curve each, drawn across the full depth of
Qwen2.5-Omni: 32 audio encoder layers followed by 29 language model stages
(61 stages total). Each dataset also gets a horizontal reference line at its
zero-shot answer AUC, drawn in that dataset's own line style.

Dataset identity is carried by LINE STYLE and MARKER SHAPE, not colour, so the
figure survives a grayscale print:
    PC-GITA  solid   circle markers   every 4 stages
    Pitt     dashed  square markers   every 4 stages
    E-DAIC   dotted  triangle markers every 4 stages

E-DAIC uses the re-run window variants, not the paper's original first-30 s run:
    full    900 s cap        zero-shot answer 0.8285
    mid30   middle 30 s      zero-shot answer 0.7205
    mid300  middle 300 s     zero-shot answer 0.8122

All curve values are the auc_oof column (out-of-fold AUC, GroupKFold(5) by
speaker). The PC-GITA and Pitt mirrors carry that same column named plainly
"auc" - see release/overnight/scripts/part11_export_curves.py line 111.

usage:  make_fig1_v2.py [full|mid30|mid300] OUT.pdf
"""
import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["STIXGeneral"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 9
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 9
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# E-DAIC whole interview curves (PART 23) are in this repository. The PC-GITA and Pitt curves
# (pcgita_encoder.csv, pcgita_llm.csv, pitt_encoder.csv, pitt_llm.csv, column auc) are not:
# put them in a folder and point CURVES_DIR at it (default: curves/).
RERUN = os.path.join(REPO, "scores", "part23")
MIRROR = os.environ.get("CURVES_DIR", "curves")

# zero-shot answer AUCs. E-DAIC from the variant run json, PC-GITA and Pitt from
# release/master/master_lookup.csv (Qwen2.5-Omni, "zero shot answer").
EDAIC_ANSWER = {"whole": 0.8366}   # part23 A2_edaic_whole_zeroshot_perclip.csv, PART23_VERIFIED.tsv row 1_whole
EDAIC_WINDOW = {"whole": "whole interview (PART 23, 900 s cap on 89 of 275)"}
PCGITA_ANSWER = 0.5635
PITT_ANSWER = 0.6578


def column(path, name):
    with open(path) as fh:
        return [float(r[name]) for r in csv.DictReader(fh)]


def curve(enc_path, enc_col, llm_path, llm_col):
    """Encoder layers then language model stages, one continuous depth trace."""
    enc = column(enc_path, enc_col)
    llm = column(llm_path, llm_col)
    return enc, llm, enc + llm


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "whole"
    if variant not in EDAIC_ANSWER:
        sys.exit(f"variant must be one of {sorted(EDAIC_ANSWER)}")
    out = sys.argv[2] if len(sys.argv) > 2 else "fig1_gap_whole.pdf"
    png = out.replace(".pdf", ".png")
    plotted = out.replace(".pdf", "_plotted.csv")
    for p in (out, png, plotted):
        if os.path.exists(p):
            sys.exit(f"refusing to overwrite {p}")

    pc_enc, pc_llm, pc = curve(f"{MIRROR}/pcgita_encoder.csv", "auc",
                               f"{MIRROR}/pcgita_llm.csv", "auc")
    pt_enc, pt_llm, pt = curve(f"{MIRROR}/pitt_encoder.csv", "auc",
                               f"{MIRROR}/pitt_llm.csv", "auc")
    ed_enc, ed_llm, ed = curve(f"{RERUN}/o25_{variant}_encoder_perlayer.csv", "auc_oof",
                               f"{RERUN}/o25_{variant}_llm_perlayer.csv", "auc_oof")

    n_enc, n_llm = len(pc_enc), len(pc_llm)
    assert len(pt_enc) == len(ed_enc) == n_enc
    assert len(pt_llm) == len(ed_llm) == n_llm
    n = n_enc + n_llm
    x = list(range(n))

    series = [
        dict(key="PC-GITA", y=pc, ans=PCGITA_ANSWER, ls="-",  marker="o", color="#1f4e9a"),
        dict(key="Pitt",    y=pt, ans=PITT_ANSWER,   ls="--", marker="s", color="#d9531e"),
        dict(key="E-DAIC",  y=ed, ans=EDAIC_ANSWER[variant], ls=":", marker="^", color="#2e7d4f"),
    ]

    # single column: \columnwidth = 3.39 in, nothing scaled at include time
    fig, ax = plt.subplots(figsize=(3.39, 2.0), dpi=300)

    for s in series:                      # answer lines, thinner, dataset style
        s["ansline"] = ax.axhline(s["ans"], color=s["color"], ls=s["ls"], lw=0.8,
                                  alpha=0.9, zorder=2)
    for s in series:                      # probe curves
        s["line"], = ax.plot(x, s["y"], color=s["color"], ls=s["ls"], lw=1.3,
                             marker=s["marker"], markevery=4, ms=3.0,
                             markerfacecolor="white", markeredgewidth=0.9,
                             markeredgecolor=s["color"], zorder=4)

    ax.axvline(n_enc - 0.5, color="#bbbbbb", lw=0.7, zorder=1)
    ax.set_xlim(-1, n)
    ax.set_ylim(0.45, 1.00)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ticks = [0, 16, n_enc, n_enc + 14, n_enc + n_llm - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "16", "0", "14", str(n_llm - 1)])
    ax.set_xlabel("layer", labelpad=1.5)
    ax.set_ylabel("AUC", labelpad=1.5)

    # dataset names carried in-plot, 9 pt, each in its own colour
    # dataset names carried in-plot, 9 pt, white backing so they never sit on a line
    lab = {
        "PC-GITA": (n_enc * 0.38, max(pc) + 0.014, "center", "bottom"),
        "Pitt":    (n_enc + 3,    pt[n_enc + 3] + 0.030, "center", "bottom"),
        "E-DAIC":  (n_enc * 0.60, min(ed) + 0.005, "center", "top"),
    }
    for s in series:
        tx, ty, ha, va = lab[s["key"]]
        ax.text(tx, ty, s["key"], color=s["color"], fontsize=9,
                ha=ha, va=va, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="square,pad=0.12", alpha=0.95))

    ax.text((n_enc - 1) / 2.0, 0.462, "audio encoder", ha="center", va="bottom",
            fontsize=9, color="#666666")
    ax.text(n_enc + (n_llm - 1) / 2.0, 0.462, "language model", ha="center",
            va="bottom", fontsize=9, color="#666666")

    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.0, pad=1.5)

    # legend: one row BELOW the axes, style only, dataset names are in-plot
    from matplotlib.lines import Line2D
    keys = [Line2D([0], [0], color="#444444", lw=1.3, label="probe"),
            Line2D([0], [0], color="#444444", lw=0.8, label="zero-shot answer")]
    ax.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=9, frameon=False, handlelength=2.4,
              columnspacing=1.6, borderpad=0.0, handletextpad=0.6)

    fig.tight_layout(pad=0.35)
    fig.subplots_adjust(bottom=0.32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(png, dpi=200)

    # arrays each Line2D actually holds, after drawing
    with open(plotted, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "stream", "index", "x_plot", "auc_plotted"])
        for s in series:
            xs = [float(v) for v in s["line"].get_xdata()]
            ys = [float(v) for v in s["line"].get_ydata()]
            for i, (xx, yy) in enumerate(zip(xs, ys)):
                stream, idx = ("encoder", i) if i < n_enc else ("llm", i - n_enc)
                w.writerow([s["key"], stream, idx, repr(xx), repr(yy)])
            for yy in sorted(set(float(v) for v in s["ansline"].get_ydata())):
                w.writerow([s["key"], "answer_line", "", "", repr(yy)])

    # every visible text element, as matplotlib sizes it
    fig.canvas.draw()
    sizes = sorted((t.get_fontsize(), t.get_text())
                   for t in fig.findobj(matplotlib.text.Text)
                   if t.get_visible() and t.get_text().strip())
    print(f"text elements {len(sizes)}  min {sizes[0][0]} pt  max {sizes[-1][0]} pt")
    for fs, txt in sizes:
        print(f"  {fs:>5} pt  {txt!r}")

    print(f"variant {variant}  E-DAIC window {EDAIC_WINDOW[variant]}")
    for s in series:
        print(f"  {s['key']:<8} curve min {min(s['y']):.4f}  max {max(s['y']):.4f}  "
              f"answer line {s['ans']:.4f}")
    print(f"wrote {out}")


main()
