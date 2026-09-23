# Figure 1, fig_options.py option C with every text element at 9 pt, a tinted gap swatch (review rows 91, M24, m33)
# and the figure drawn at the spconf column width (3.345 in) so it is placed at scale 1.0.
# Same data files and answer lines as the shipped fig1_gap.pdf (300 s E-DAIC window). usage: make_fig1_9pt.py OUT.pdf
import sys, matplotlib
sys.path.insert(0, "scripts")
import fig_options as fo
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
W_IN, H_IN, FS = 3.345, float(sys.argv[2]) if len(sys.argv) > 2 else 2.45, 9
def option_C9(c, out):
    fig, axes = plt.subplots(3, 1, figsize=(W_IN, H_IN), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.125, right=0.895, top=0.985, bottom=0.235, hspace=0.10)
    for ax, k in zip(axes, fo.ORDER):
        enc, lm = c[k]; st = fo.STYLE[k]
        fo.band_and_chance(ax)
        ax.fill_between(fo.X_ENC, enc, fo.ANSWER[k], color=st["color"], alpha=0.16, lw=0, zorder=1)
        ax.fill_between(fo.X_LM, lm, fo.ANSWER[k], color=st["color"], alpha=0.16, lw=0, zorder=1)
        fo.curve(ax, k, enc, lm, lw=1.1, ms=3.0)
        ax.axhline(fo.ANSWER[k], color=st["color"], lw=0.8, ls=(0, (1.2, 1.8)), zorder=2)
        ax.plot([fo.X_LM[-1] + 1.3], [fo.ANSWER[k]], marker=st["marker"], ms=3.2, color=st["color"], clip_on=False, zorder=5)
        ax.text(fo.X_LM[-1] + 2.6, fo.ANSWER[k], f"{fo.ANSWER[k]:.2f}", fontsize=FS, color=st["color"], va="center", ha="left", clip_on=False)
        if k == "PC-GITA":
            ax.text(0.3, 0.60, fo.LABEL[k], fontsize=FS, color=st["color"], ha="left", va="bottom", zorder=6)
        else:
            ax.text(0.3, 0.935, fo.LABEL[k], fontsize=FS, color=st["color"], ha="left", va="top", zorder=6)
        ax.set_ylim(0.48, 0.96); ax.set_yticks([0.5, 0.7, 0.9]); ax.tick_params(axis="y", labelsize=FS, length=2, pad=1.5)
        ax.grid(axis="y", color="#f0f0f0", lw=0.5, zorder=0)
        fo.xaxis(ax, labels=(ax is axes[-1]))
    axes[1].set_ylabel("AUC", fontsize=FS, labelpad=2)
    axes[-1].text(fo.X_PROJ, 0.485, "proj.", fontsize=FS, style="italic", color="#666666", ha="center", va="bottom")
    axes[-1].text(fo.X_LM[-1] + 0.6, 0.505, "chance", fontsize=FS, color="#8a8a8a", ha="right", va="bottom")
    axes[-1].text(15.5, 0.36, "audio encoder layer", fontsize=FS, ha="center", va="top", clip_on=False)
    axes[-1].text(fo.X_LM[14], 0.36, "language model layer", fontsize=FS, ha="center", va="top", clip_on=False)
    fig.legend([Line2D([0], [0], color="#333333", lw=1.2), Line2D([0], [0], color="#333333", lw=0.8, ls=(0, (1.2, 1.8))),
                Patch(facecolor="#b03a2e", alpha=0.22, edgecolor="none")],
               ["linear probe", "zero-shot answer", "gap"], loc="lower center", ncol=3, frameon=False, fontsize=FS,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.2, columnspacing=1.2, handletextpad=0.6)
    fig.savefig(out)
if __name__ == "__main__":
    data = fo.load()
    for k in fo.ORDER: print(k, "enc", len(data[k][0]), "lm", len(data[k][1]), "answer", fo.ANSWER[k], "files", fo.PATHS[k])
    option_C9(data, sys.argv[1]); print("wrote", sys.argv[1])
