"""Figures for paper 1. Two-column ICASSP width, greyscale-safe, no colour dependence."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/paper1"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
COL = 3.35  # one ICASSP column, inches

# ---------------- FIGURE 1: the conflict inversion ----------------
models = ["salmonn","diva","omni","qwen3-omni","qwen2-audio","flamingo 3","mimo","kimi","flamingo 2"]
conflict  = [0.291,0.299,0.307,0.349,0.351,0.362,0.466,0.521,0.549]
agreement = [0.850,0.699,0.848,0.867,0.923,0.926,0.819,0.865,0.574]

fig, ax = plt.subplots(figsize=(COL, 2.5))
y = np.arange(len(models))
ax.barh(y-0.2, agreement, height=0.38, color="0.75", edgecolor="black", lw=0.5, label="words agree")
ax.barh(y+0.2, conflict,  height=0.38, color="0.30", edgecolor="black", lw=0.5, label="words disagree")
ax.axvline(0.5, color="black", ls="--", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(models)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.set_xlabel("AUC against the clinical label")
ax.legend(loc="lower right", frameon=False)
ax.text(0.505, len(models)-0.3, "chance", fontsize=6, va="center")
fig.savefig(f"{OUT}/fig1_conflict.pdf"); fig.savefig(f"{OUT}/fig1_conflict.png")
plt.close(fig)

# ---------------- FIGURE 2: the gap, and that nothing closes it ----------------
fig, ax = plt.subplots(figsize=(COL, 2.2))
labels = ["encoder\nprobe", "model's\nanswer", "+12\nprompts", "+layer\nmixer", "+text\nsubtract", "+encoder\nsteering"]
vals   = [0.92, 0.63, 0.63, 0.63, 0.63, 0.63]
probe  = [0.92, None, 0.92, 0.92, 0.92, 0.92]
x = np.arange(len(labels))
ax.bar(x, vals, color=["0.30"]+["0.75"]*5, edgecolor="black", lw=0.5)
for i,p in enumerate(probe):
    if p is not None and i>0:
        ax.plot([i-0.4,i+0.4],[p,p], color="black", lw=1.0, ls=":")
ax.axhline(0.5, color="black", ls="--", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("AUC")
ax.text(3.0, 0.935, "encoder probe, unchanged", fontsize=6, ha="center")
fig.savefig(f"{OUT}/fig2_gap.pdf"); fig.savefig(f"{OUT}/fig2_gap.png")
plt.close(fig)

# ---------------- FIGURE 3: lexical capture grid ----------------
path = "results/health/local/lexical_capture_grid.csv"
rows = list(csv.DictReader(open(path)))
mods = sorted(set(r["model"] for r in rows))
dss  = ["daic","italian","kcl","neurovoz","pc-gita"]
M = np.full((len(mods), len(dss)), np.nan)
for r in rows:
    if r["dataset"] in dss:
        M[mods.index(r["model"]), dss.index(r["dataset"])] = float(r["flip_rate"])

fig, ax = plt.subplots(figsize=(COL, 1.9))
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(dss))); ax.set_xticklabels(dss, rotation=30, ha="right")
ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods)
for i in range(len(mods)):
    for j in range(len(dss)):
        if not np.isnan(M[i,j]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if M[i,j]>0.5 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("flip rate", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.savefig(f"{OUT}/fig3_lexical_capture.pdf"); fig.savefig(f"{OUT}/fig3_lexical_capture.png")
plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)): print("  ", os.path.join(OUT, f))
