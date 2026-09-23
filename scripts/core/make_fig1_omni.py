"""Figure 1 on Qwen2.5-Omni: probe AUC at every depth, one panel per condition.
x axis: 32 audio encoder layers, the projector as the bridge point, then 29 language model stages.
usage: make_fig1_omni.py OUT.pdf key[,key...]      keys: pcgita neurovoz kcl edaic pitt adresso adress2020"""
import os, sys, csv, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/Volumes/G-Drive Pro/paper1_local_runs"; P = f"{R}/omni_final"
def rows(f): return list(csv.DictReader(open(f)))
def col(f): return [float(r["auc_oof"]) for r in rows(f)]
def auc(y, p):
    pos = [b for a, b in zip(y, p) if a == 1]; neg = [b for a, b in zip(y, p) if a == 0]
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))
def clip_auc(f):
    if not f or not os.path.exists(f): return None
    Rw = rows(f); return auc([int(r["label"]) for r in Rw], [float(r["p_yes"]) for r in Rw])
TITLE = {"pcgita": "Parkinson's, PC-GITA", "neurovoz": "Parkinson's, NeuroVoz", "kcl": "Parkinson's, MDVR-KCL",
         "edaic": "Depression, E-DAIC", "pitt": "Alzheimer's, Pitt", "adresso": "Alzheimer's, ADReSSo",
         "adress2020": "Alzheimer's, ADReSS-2020"}
TEXT = {"pitt": f"{R}/omni_pitt_text.csv", "edaic": f"{R}/omni_edaic30_text.csv",
        "adresso": f"{R}/omni_adresso_text.csv", "adress2020": f"{R}/omni_adress2020_text.csv"}
out, keys = sys.argv[1], sys.argv[2].split(",")
BLUE, ORANGE, GREEN, GREY = "#1f4e9a", "#d9531e", "#2e7d4f", "#555555"
n = len(keys)
fig, axes = plt.subplots(1, n, figsize=(3.4 if n == 1 else 7.16, 2.45 if n == 1 else 2.35),
                         sharey=True, squeeze=False, dpi=300)
axes = axes[0]
for ax, k in zip(axes, keys):
    enc = col(f"{P}/omni_{k}_encoder_perlayer.csv"); prj = col(f"{P}/omni_{k}_proj_perlayer.csv")
    llm = col(f"{P}/omni_{k}_llm_perlayer.csv"); ans = col(f"{P}/omni_{k}_ans_perlayer.csv")
    NE, NL = len(enc), len(llm); xp = NE + 1.5; x0 = NE + 4            # projector sits between the two blocks
    za = clip_auc(f"{P}/omni_{k}_zeroshot_scores.csv"); zt = clip_auc(TEXT.get(k))
    xmax = x0 + NL - 1
    ax.axvspan(-1, NE - 0.4, color="#f4f7fb", zorder=0)
    ax.axvspan(x0 - 0.6, xmax + 1, color="#fbf7f4", zorder=0)
    if za is not None:
        ax.hlines(za, -1, xmax + 1, color=GREY, lw=1.0, ls=(0, (5, 2)), zorder=2, label="zero-shot answer, audio")
    if zt is not None:
        ax.hlines(zt, -1, xmax + 1, color=GREY, lw=1.0, ls=(0, (1, 1.6)), zorder=2, label="zero-shot answer, transcript")
    ax.plot(range(NE), enc, color=BLUE, lw=1.5, zorder=4, label="probe, audio encoder layer")
    ax.plot([x0 + i for i in range(NL)], llm, color=BLUE, lw=1.5, ls="--", zorder=4, label="probe, language model stage")
    ax.plot([x0 + i for i in range(1, NL)], ans[1:], color=ORANGE, lw=1.5, zorder=5, label="probe, answer position")
    ax.plot([xp], prj, marker="D", ms=4.2, color=GREEN, mec="white", mew=0.7, zorder=6, ls="none", label="probe, projector")
    ax.plot([NE - 1, xp], [enc[-1], prj[0]], color=GREEN, lw=0.9, alpha=0.55, zorder=3)
    ax.plot([xp, x0], [prj[0], llm[0]], color=GREEN, lw=0.9, alpha=0.55, zorder=3)
    ax.set_title(TITLE[k], fontsize=8.5, pad=3.5)
    lo = min(min(enc), min(llm), min(ans[1:]), prj[0], za or 1) - 0.03
    ax.set_ylim(min(0.45, lo), 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticks([0, 16, NE - 1, x0, x0 + 14, x0 + NL - 1]); ax.set_xticklabels(["0", "16", str(NE - 1), "0", "14", str(NL - 1)], fontsize=7)
    ax.set_xlim(-1, xmax + 1)
    yb = ax.get_ylim()[0] + 0.012
    ax.text((NE - 1) / 2, yb, "audio encoder", ha="center", va="bottom", fontsize=6.8, color="#5b6b80")
    ax.text(x0 + (NL - 1) / 2, yb, "language model", ha="center", va="bottom", fontsize=6.8, color="#8a6a58")
    ax.set_xlabel("layer", fontsize=8, labelpad=1.5)
    ax.tick_params(labelsize=7, length=2.5); ax.grid(axis="y", color="#e9e9e9", lw=0.6, zorder=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    print(f"{k}: enc max {max(enc):.3f}@{enc.index(max(enc))} | proj {prj[0]:.3f} | llm max {max(llm):.3f}@{llm.index(max(llm))} "
          f"| ans max {max(ans[1:]):.3f} | zero-shot audio {za:.3f}" + (f" text {zt:.3f}" if zt else ""))
axes[0].set_ylabel("AUC", fontsize=8.5)
h, l = [], []
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
o = [l.index(x) for x in ["probe, audio encoder layer", "probe, projector", "probe, language model stage",
                          "probe, answer position", "zero-shot answer, audio"] if x in l]
if "zero-shot answer, transcript" in l: o.append(l.index("zero-shot answer, transcript"))
fig.legend([h[i] for i in o], [l[i] for i in o], loc="lower center", ncol=3, fontsize=6.9,
           frameon=False, bbox_to_anchor=(0.5, -0.045), handlelength=2.1, columnspacing=1.4)
fig.tight_layout(rect=(0, 0.14 if n > 1 else 0.2, 1, 1))
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print("wrote", out)
