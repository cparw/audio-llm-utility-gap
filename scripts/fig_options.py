#!/usr/bin/env python3
# Reads the Qwen2.5-Omni per-layer probe curves. E-DAIC comes from scores/edaic_windows/o25_full_{encoder,llm}_perlayer.csv
# in this repository. PC-GITA and Pitt come from pcgita_encoder.csv, pcgita_llm.csv, pitt_encoder.csv and pitt_llm.csv
# (column auc), which are not in this repository. Put them in a folder and point CURVES_DIR at it (default: curves/).
# With --uploads, all six files are read from the current folder.
"""Figure 1 of the paper (option C, the submitted figure).  usage: fig_options.py C OUT.pdf [--uploads]"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

UP = "--uploads" in sys.argv
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_C = "./" if UP else os.path.join(os.environ.get("CURVES_DIR", "curves"), "")
BASE_E = "./" if UP else os.path.join(REPO, "scores", "edaic_windows") + "/"
PATHS = {"PC-GITA": (BASE_C + "pcgita_encoder.csv", BASE_C + "pcgita_llm.csv"),
         "Pitt":    (BASE_C + "pitt_encoder.csv",   BASE_C + "pitt_llm.csv"),
         "E-DAIC":  (BASE_E + "o25_full_encoder_perlayer.csv", BASE_E + "o25_full_llm_perlayer.csv")}
ANSWER = {"PC-GITA": 0.5635, "Pitt": 0.6578, "E-DAIC": 0.8285}
LABEL = {"PC-GITA": "PC-GITA (PD)", "Pitt": "Pitt (AD)", "E-DAIC": "E-DAIC (MDD)"}
STYLE = {"PC-GITA": dict(color="#1f4e9c", ls="-", marker="o"),
         "Pitt":    dict(color="#b03a2e", ls=(0, (5, 2)), marker="s"),
         "E-DAIC":  dict(color="#c98a00", ls=(0, (4, 1.5, 1, 1.5)), marker="^")}
ORDER = ("PC-GITA", "Pitt", "E-DAIC")
N_ENC, N_LM = 32, 29
X_ENC = np.arange(N_ENC); X_LM = np.arange(N_LM) + N_ENC + 2; X_PROJ = N_ENC + 0.5

def read_curve(path):
    d = pd.read_csv(path)
    col = "auc_oof" if "auc_oof" in d.columns else "auc"
    for k in ("layer", "stage"):
        if k in d.columns: d = d.sort_values(k); break
    return d[col].astype(float).values

def load():
    c = {}
    for k, (pe, pl) in PATHS.items():
        enc, lm = read_curve(pe), read_curve(pl)
        c[k] = (enc[:N_ENC], lm[-N_LM:])
    return c

def curve(ax, k, enc, lm, lw=1.2, ms=3.4):
    st = STYLE[k]
    x = np.concatenate([X_ENC, X_LM]); y = np.concatenate([enc, lm])
    ax.plot(X_ENC, enc, ls=st["ls"], color=st["color"], lw=lw, zorder=3, solid_capstyle="round")
    ax.plot(X_LM, lm, ls=st["ls"], color=st["color"], lw=lw, zorder=3, solid_capstyle="round")
    ax.plot(x[4::8], y[4::8], ls="none", marker=st["marker"], ms=ms, mfc="white", mec=st["color"], mew=0.9, zorder=4)

def band_and_chance(ax, chance_label=True):
    ax.axvspan(N_ENC - 0.5, N_ENC + 1.5, color="#ebebeb", lw=0, zorder=0)
    ax.axhline(0.5, color="#b0b0b0", lw=0.5, ls=(0, (1, 2)), zorder=1)

def xaxis(ax, labels=True):
    ax.set_xlim(-0.8, X_LM[-1] + 0.8)
    ax.set_xticks([0, 16, 31, X_LM[0], X_LM[14], X_LM[28]])
    ax.set_xticklabels(["0", "16", "31", "0", "14", "28"] if labels else [""] * 6, fontsize=9)
    ax.tick_params(axis="both", labelsize=9, length=2.5, pad=1.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def option_C(c, out):
    fig, axes = plt.subplots(3, 1, figsize=(3.39, 2.6), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.12, right=0.905, top=0.985, bottom=0.225, hspace=0.10)
    for ax, k in zip(axes, ORDER):
        enc, lm = c[k]; st = STYLE[k]
        band_and_chance(ax)
        x = np.concatenate([X_ENC, X_LM]); y = np.concatenate([enc, lm])
        ax.fill_between(X_ENC, enc, ANSWER[k], color=st["color"], alpha=0.16, lw=0, zorder=1)
        ax.fill_between(X_LM, lm, ANSWER[k], color=st["color"], alpha=0.16, lw=0, zorder=1)
        curve(ax, k, enc, lm, lw=1.1, ms=3.0)
        ax.axhline(ANSWER[k], color=st["color"], lw=0.8, ls=(0, (1.2, 1.8)), zorder=2)
        ax.plot([X_LM[-1] + 1.3], [ANSWER[k]], marker=st["marker"], ms=3.2, color=st["color"], clip_on=False, zorder=5)
        ax.text(X_LM[-1] + 2.6, ANSWER[k], f"{ANSWER[k]:.2f}", fontsize=8.5, color=st["color"], va="center", ha="left", clip_on=False)
        if k == "PC-GITA":
            ax.text(0.3, 0.60, LABEL[k], fontsize=8.5, color=st["color"], ha="left", va="bottom", zorder=6)
        else:
            ax.text(0.3, 0.935, LABEL[k], fontsize=8.5, color=st["color"], ha="left", va="top", zorder=6)
        ax.set_ylim(0.48, 0.96); ax.set_yticks([0.5, 0.7, 0.9]); ax.tick_params(axis="y", labelsize=8.5, length=2, pad=1.5)
        ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
        xaxis(ax, labels=(ax is axes[-1]))
    axes[1].set_ylabel("AUC", fontsize=9, labelpad=2)
    axes[-1].text(X_PROJ, 0.485, "proj.", fontsize=7.5, style="italic", color="#666666", ha="center", va="bottom")
    axes[-1].text(X_LM[-1] + 0.6, 0.505, "chance", fontsize=7.5, color="#9a9a9a", ha="right", va="bottom")
    axes[-1].text(15.5, 0.37, "audio encoder layer", fontsize=9, ha="center", va="top", clip_on=False)
    axes[-1].text(X_LM[14], 0.37, "language model layer", fontsize=9, ha="center", va="top", clip_on=False)
    fig.legend([Line2D([0], [0], color="#333333", lw=1.2), Line2D([0], [0], color="#333333", lw=0.8, ls=(0, (1.2, 1.8))), Patch(facecolor="#999999", alpha=0.35, edgecolor="none")],
               ["linear probe", "zero-shot answer", "gap"], loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005), handlelength=2.2, columnspacing=1.4, handletextpad=0.6)
    fig.savefig(out)

if __name__ == "__main__":
    opt, out = sys.argv[1], sys.argv[2]
    {"C": option_C}[opt](load(), out)
    print("wrote", out)
