#!/usr/bin/env python3
"""Figure 1, whole interview version (PART 24 item 2, 23 Sep 2026).

Written beside make_fig1_v4.py. make_fig1_v4.py is not edited.

make_fig1_v4.py does NOT make the figure the final tex includes. The final tex
(\\includegraphics{fig1_gap.pdf}, graphicspath figures/) compiles on Overleaf
against figures/fig1_gap.pdf, which is option C of
scripts/fig_options.py: three stacked panels, a shaded
gap, a grey projector band, "proj." and "chance" notes, value labels and a
"gap" legend swatch. make_fig1_v4.py draws one axes with the three curves
overlaid and none of those elements. So this script has two layouts:

    overleaf   option C geometry, the drop in for the figure the tex includes
    v4         make_fig1_v4.py geometry, kept for the literal request

Changes against both parents, and nothing else:
  1. E-DAIC curve = PART 23 whole interview run, column auc_oof
     (o25_whole_encoder_perlayer.csv, 32 layers; o25_whole_llm_perlayer.csv,
     29 stages; 275 clips, 275 speakers).
  2. E-DAIC dotted answer line at 0.8366 (PART 23 whole interview zero shot,
     A2_edaic_whole_zeroshot_perclip.csv, verified row "1_whole").
  3. PC-GITA and Pitt unchanged: same mirror files, same answer AUCs.
  4. Every text element 9 pt: ticks, axis labels, panel labels, value labels,
     legend, "proj.", "chance", encoder / language model notes.
  5. The gap legend swatch is tinted: three side by side patches in the three
     panel fill colours at the fill alpha, instead of one grey patch.
  6. STIXGeneral, mathtext stix, pdf.fonttype 42.
  7. Figure width 86 mm = the spconf column width ((178 mm - 6 mm) / 2), so
     \\includegraphics[width=\\columnwidth] scales by exactly 1 and 9 pt stays
     9 pt in the paper. The parents used 3.39 in, which the paper shrinks by
     0.99878 (9 pt would land at 8.99 pt).

The arrays each Line2D actually holds are read back after drawing and written
to <OUT>_plotted.csv, so a verifier can compare them with the pod csvs.

usage:  make_fig1_whole.py overleaf|v4 OUT.pdf
writes: OUT.pdf, OUT.png (200 dpi), OUT_plotted.csv, OUT.sidecar.json
"""
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

FS = 9                                    # every text element
FIG_W = 86.0 / 25.4                       # spconf \columnwidth, inches

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The E-DAIC whole interview curves (PART 23) are in this repository. The PC-GITA and Pitt curves
# (pcgita_encoder.csv, pcgita_llm.csv, pitt_encoder.csv, pitt_llm.csv, column auc) are not:
# put them in a folder and point CURVES_DIR at it (default: curves/).
MIRROR = os.environ.get("CURVES_DIR", "curves")
WHOLE = os.path.join(REPO, "scores", "part23")
SRC = {
    "PC-GITA": (f"{MIRROR}/pcgita_encoder.csv", f"{MIRROR}/pcgita_llm.csv", "auc"),
    "Pitt":    (f"{MIRROR}/pitt_encoder.csv", f"{MIRROR}/pitt_llm.csv", "auc"),
    "E-DAIC":  (f"{WHOLE}/o25_whole_encoder_perlayer.csv",
                f"{WHOLE}/o25_whole_llm_perlayer.csv", "auc_oof"),
}
ANSWER = {"PC-GITA": 0.5635, "Pitt": 0.6578, "E-DAIC": 0.8366}
ANSWER_SRC = {
    "PC-GITA": "lookup/master_lookup.csv, Qwen2.5-Omni zero shot answer (unchanged)",
    "Pitt": "lookup/master_lookup.csv, Qwen2.5-Omni zero shot answer (unchanged)",
    "E-DAIC": "part23 A2_edaic_whole_zeroshot_perclip.csv, roc_auc_score(label, p_yes_whole); "
              "part23/verify/PART23_VERIFIED.tsv row 1_whole",
}
ORDER = ("PC-GITA", "Pitt", "E-DAIC")
N_ENC, N_LM = 32, 29


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def read_curve(path, col):
    """Rows sorted by layer / stage, one float per row from column col."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    key = "layer" if "layer" in rows[0] else "stage"
    rows.sort(key=lambda r: int(r[key]))
    return np.array([float(r[col]) for r in rows])


def load():
    c = {}
    for k in ORDER:
        pe, pl, col = SRC[k]
        enc, lm = read_curve(pe, col), read_curve(pl, col)
        assert len(enc) == N_ENC and len(lm) == N_LM, (k, len(enc), len(lm))
        c[k] = (enc, lm)
    return c


# ---------------------------------------------------------------- overleaf
# geometry, colours, line styles and positions copied from fig_options.py
# option_C; only font sizes, E-DAIC data, legend swatch and width changed.
STYLE_C = {"PC-GITA": dict(color="#1f4e9c", ls="-", marker="o"),
           "Pitt":    dict(color="#b03a2e", ls=(0, (5, 2)), marker="s"),
           "E-DAIC":  dict(color="#c98a00", ls=(0, (4, 1.5, 1, 1.5)), marker="^")}
LABEL_C = {"PC-GITA": "PC-GITA (PD)", "Pitt": "Pitt (AD)", "E-DAIC": "E-DAIC (MDD)"}
FILL_ALPHA = 0.16


def layout_overleaf(c):
    matplotlib.rcParams.update({"axes.linewidth": 0.6, "xtick.major.width": 0.6,
                                "ytick.major.width": 0.6})
    X_ENC = np.arange(N_ENC)
    X_LM = np.arange(N_LM) + N_ENC + 2
    X_PROJ = N_ENC + 0.5
    drawn = {}
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 2.6), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.12, right=0.905, top=0.985, bottom=0.225, hspace=0.10)
    for ax, k in zip(axes, ORDER):
        enc, lm = c[k]
        st = STYLE_C[k]
        ax.axvspan(N_ENC - 0.5, N_ENC + 1.5, color="#ebebeb", lw=0, zorder=0)
        ax.axhline(0.5, color="#b0b0b0", lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.fill_between(X_ENC, enc, ANSWER[k], color=st["color"], alpha=FILL_ALPHA, lw=0, zorder=1)
        ax.fill_between(X_LM, lm, ANSWER[k], color=st["color"], alpha=FILL_ALPHA, lw=0, zorder=1)
        x = np.concatenate([X_ENC, X_LM])
        y = np.concatenate([enc, lm])
        le, = ax.plot(X_ENC, enc, ls=st["ls"], color=st["color"], lw=1.1, zorder=3,
                      solid_capstyle="round")
        ll, = ax.plot(X_LM, lm, ls=st["ls"], color=st["color"], lw=1.1, zorder=3,
                      solid_capstyle="round")
        ax.plot(x[4::8], y[4::8], ls="none", marker=st["marker"], ms=3.0, mfc="white",
                mec=st["color"], mew=0.9, zorder=4)
        la = ax.axhline(ANSWER[k], color=st["color"], lw=0.8, ls=(0, (1.2, 1.8)), zorder=2)
        drawn[k] = (le, ll, la)
        ax.plot([X_LM[-1] + 1.3], [ANSWER[k]], marker=st["marker"], ms=3.2, color=st["color"],
                clip_on=False, zorder=5)
        ax.text(X_LM[-1] + 2.6, ANSWER[k], f"{ANSWER[k]:.2f}", fontsize=FS, color=st["color"],
                va="center", ha="left", clip_on=False)
        if k == "PC-GITA":
            ax.text(0.3, 0.60, LABEL_C[k], fontsize=FS, color=st["color"], ha="left",
                    va="bottom", zorder=6)
        else:
            ax.text(0.3, 0.935, LABEL_C[k], fontsize=FS, color=st["color"], ha="left",
                    va="top", zorder=6)
        ax.set_ylim(0.48, 0.96)
        ax.set_yticks([0.5, 0.7, 0.9])
        ax.tick_params(axis="y", labelsize=FS, length=2, pad=1.5)
        ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
        ax.set_xlim(-0.8, X_LM[-1] + 0.8)
        ax.set_xticks([0, 16, 31, X_LM[0], X_LM[14], X_LM[28]])
        last = ax is axes[-1]
        ax.set_xticklabels(["0", "16", "31", "0", "14", "28"] if last else [""] * 6,
                           fontsize=FS)
        ax.tick_params(axis="both", labelsize=FS, length=2.5, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_ylabel("AUC", fontsize=FS, labelpad=2)
    axes[-1].text(X_PROJ, 0.485, "proj.", fontsize=FS, style="italic", color="#666666",
                  ha="center", va="bottom")
    axes[-1].text(X_LM[-1] + 0.6, 0.505, "chance", fontsize=FS, color="#9a9a9a",
                  ha="right", va="bottom")
    axes[-1].text(15.5, 0.37, "audio encoder layer", fontsize=FS, ha="center", va="top",
                  clip_on=False)
    axes[-1].text(X_LM[14], 0.37, "language model layer", fontsize=FS, ha="center",
                  va="top", clip_on=False)
    gap = tuple(Patch(facecolor=STYLE_C[k]["color"], alpha=FILL_ALPHA, edgecolor="none")
                for k in ORDER)
    fig.legend([Line2D([0], [0], color="#333333", lw=1.2),
                Line2D([0], [0], color="#333333", lw=0.8, ls=(0, (1.2, 1.8))), gap],
               ["linear probe", "zero-shot answer", "gap"], loc="lower center", ncol=3,
               frameon=False, fontsize=FS, bbox_to_anchor=(0.5, -0.005), handlelength=2.2,
               columnspacing=1.4, handletextpad=0.6,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})

    def split(k):
        le, ll, la = drawn[k]
        return (np.asarray(le.get_xdata(), float), np.asarray(le.get_ydata(), float),
                np.asarray(ll.get_xdata(), float), np.asarray(ll.get_ydata(), float),
                float(np.unique(la.get_ydata())[0]))
    return fig, {k: split(k) for k in ORDER}


# ---------------------------------------------------------------- v4
# make_fig1_v4.py main() geometry; only font sizes, font family, width and the
# E-DAIC data / answer changed.
def layout_v4(c):
    n_enc, n_llm = N_ENC, N_LM
    n = n_enc + n_llm
    x = list(range(n))
    cat = {k: list(c[k][0]) + list(c[k][1]) for k in ORDER}
    series = [
        dict(key="PC-GITA", y=cat["PC-GITA"], ans=ANSWER["PC-GITA"], ls="-", marker="o", color="#1f4e9a"),
        dict(key="Pitt",    y=cat["Pitt"],    ans=ANSWER["Pitt"],    ls="--", marker="s", color="#d9531e"),
        dict(key="E-DAIC",  y=cat["E-DAIC"],  ans=ANSWER["E-DAIC"],  ls=":", marker="^", color="#2e7d4f"),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.0), dpi=300)
    drawn = {}
    for s in series:
        la = ax.axhline(s["ans"], color=s["color"], ls=s["ls"], lw=0.8, alpha=0.9, zorder=2)
        drawn[s["key"]] = [la]
    for s in series:
        lp, = ax.plot(x, s["y"], color=s["color"], ls=s["ls"], lw=1.3,
                      marker=s["marker"], markevery=4, ms=3.0,
                      markerfacecolor="white", markeredgewidth=0.9,
                      markeredgecolor=s["color"], zorder=4)
        drawn[s["key"]].append(lp)
    ax.axvline(n_enc - 0.5, color="#bbbbbb", lw=0.7, zorder=1)
    ax.set_xlim(-1, n)
    ax.set_ylim(0.45, 1.00)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticks([0, 16, n_enc, n_enc + 14, n_enc + n_llm - 1])
    ax.set_xticklabels(["0", "16", "0", "14", str(n_llm - 1)])
    ax.set_xlabel("layer", labelpad=1.5, fontsize=10)   # v4: axes.labelsize 10
    ax.set_ylabel("AUC", labelpad=1.5, fontsize=10)
    pc, pt, ed = cat["PC-GITA"], cat["Pitt"], cat["E-DAIC"]
    lab = {
        "PC-GITA": (n_enc * 0.38, max(pc) + 0.014, "center", "bottom"),
        "Pitt":    (n_enc + 3,    pt[n_enc + 3] + 0.030, "center", "bottom"),
        "E-DAIC":  (n_enc * 0.60, min(ed) + 0.005, "center", "top"),
    }
    for s in series:
        tx, ty, ha, va = lab[s["key"]]
        ax.text(tx, ty, s["key"], color=s["color"], fontsize=FS, ha=ha, va=va, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="square,pad=0.12", alpha=0.95))
    ax.text((n_enc - 1) / 2.0, 0.462, "audio encoder", ha="center", va="bottom",
            fontsize=FS, color="#666666")
    ax.text(n_enc + (n_llm - 1) / 2.0, 0.462, "language model", ha="center",
            va="bottom", fontsize=FS, color="#666666")
    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.0, pad=1.5, labelsize=FS)
    keys = [Line2D([0], [0], color="#444444", lw=1.3, label="probe"),
            Line2D([0], [0], color="#444444", lw=0.8, label="zero-shot answer")]
    ax.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=FS, frameon=False, handlelength=2.4,
              columnspacing=1.6, borderpad=0.0, handletextpad=0.6)
    fig.tight_layout(pad=0.35)
    fig.subplots_adjust(bottom=0.32)

    def split(k):
        la, lp = drawn[k]
        xs = np.asarray(lp.get_xdata(), float)
        ys = np.asarray(lp.get_ydata(), float)
        return (xs[:n_enc], ys[:n_enc], xs[n_enc:], ys[n_enc:],
                float(np.unique(la.get_ydata())[0]))
    return fig, {k: split(k) for k in ORDER}


def text_audit(fig):
    """Every visible text: font size, and whether it leaves the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    sizes, outside = [], []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        sizes.append(t.get_fontsize())
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            outside.append((t.get_text(), round(bb.x0, 1), round(bb.y0, 1),
                            round(bb.x1, 1), round(bb.y1, 1)))
    return min(sizes), max(sizes), len(sizes), outside


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("overleaf", "v4"):
        sys.exit(__doc__)
    layout, out = sys.argv[1], sys.argv[2]
    assert out.endswith(".pdf")
    c = load()
    fig, arr = (layout_overleaf if layout == "overleaf" else layout_v4)(c)
    fmin, fmax, ntext, outside = text_audit(fig)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    png = out[:-4] + ".png"
    fig.savefig(png, dpi=200)

    plotted = out[:-4] + "_plotted.csv"
    with open(plotted, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "stream", "layer", "x_plot", "auc_plotted"])
        for k in ORDER:
            xe, ye, xl, yl, ans = arr[k]
            for i, (xx, yy) in enumerate(zip(xe, ye)):
                w.writerow([k, "encoder", i, repr(float(xx)), repr(float(yy))])
            for i, (xx, yy) in enumerate(zip(xl, yl)):
                w.writerow([k, "llm", i, repr(float(xx)), repr(float(yy))])
            w.writerow([k, "answer_line", "", "", repr(ans)])

    side = {
        "result": "Figure 1, whole interview version, layout " + layout,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": "python3 " + " ".join(os.path.abspath(a) if i == 0 else a
                                                          for i, a in enumerate(sys.argv)),
        "script": os.path.abspath(sys.argv[0]),
        "script_sha256": sha256(os.path.abspath(sys.argv[0])),
        "outputs": {"pdf": out, "png_200dpi": png, "plotted_csv": plotted},
        "model_id": "Qwen/Qwen2.5-Omni-7B (curves only; no model run here)",
        "seed": None, "gpu": None, "bootstrap_rule": "none (figure; no intervals drawn)",
        "edaic_n": 275, "edaic_n_speakers": 275,
        "edaic_column": "auc_oof (GroupKFold(5) by speaker, pooled OOF, from the pod)",
        "sources": {k: {"encoder": SRC[k][0], "encoder_sha256": sha256(SRC[k][0]),
                        "llm": SRC[k][1], "llm_sha256": sha256(SRC[k][1]),
                        "column": SRC[k][2]} for k in ORDER},
        "answer_lines": ANSWER, "answer_sources": ANSWER_SRC,
        "fig_size_in": [round(float(v), 6) for v in fig.get_size_inches()],
        "font": {"family": matplotlib.rcParams["font.serif"],
                 "mathtext": matplotlib.rcParams["mathtext.fontset"],
                 "pdf_fonttype": matplotlib.rcParams["pdf.fonttype"],
                 "matplotlib_text_min_pt": fmin, "matplotlib_text_max_pt": fmax,
                 "n_text": ntext},
        "text_outside_figure": outside,
        "library_versions": {"python": platform.python_version(),
                             "matplotlib": matplotlib.__version__,
                             "numpy": np.__version__, "platform": platform.platform()},
    }
    with open(out[:-4] + ".sidecar.json", "w") as fh:
        json.dump(side, fh, indent=1)

    print(f"layout {layout}  size {side['fig_size_in']} in  text {ntext} items, "
          f"{fmin}-{fmax} pt  outside figure {outside}")
    for k in ORDER:
        xe, ye, xl, yl, ans = arr[k]
        print(f"  {k:<8} encoder {len(ye)} layers  llm {len(yl)} stages  "
              f"min {min(ye.min(), yl.min()):.4f} max {max(ye.max(), yl.max()):.4f}  "
              f"answer {ans:.4f}")
    print("wrote", out, png, plotted)


main()
