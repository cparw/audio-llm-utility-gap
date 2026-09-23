"""Figure 1: probe AUC at every depth of Qwen2-Audio, one panel per condition.
Panels are drawn only for datasets whose per-layer CSVs exist; missing panels are skipped.
Usage: python make_fig1.py  -> writes fig1_gap.pdf (single column if 1 panel, full width if 3)."""
import os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/chaitanyaparwatkar/Desktop/Hard drive data for paper"
HERE = os.path.dirname(os.path.abspath(__file__))

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def col(rows, name):
    return [float(r[name]) for r in rows]

# ---- panel definitions: fill enc/llm/ans paths when the CARC files arrive ----
PANELS = [
    dict(key="pd",  title="Parkinson's, PC-GITA",
         enc=os.path.join(HERE, "pcgita_encoder_perlayer.csv"), enc_col="auc",
         llm=None, llm_col=None, ans=None, answer_audio=0.54, answer_text=None),
    dict(key="mdd", title="Depression, E-DAIC",
         enc=os.path.join(HERE, "edaic_encoder_perlayer.csv"), enc_col="auc",
         llm=None, llm_col=None, ans=None, answer_audio=0.65, answer_text=None),
    dict(key="pd_kcl", title="Parkinson's, MDVR-KCL",
         enc=os.path.join(HERE, "kcl_encoder_perlayer.csv"), enc_col="auc_oof",
         llm=os.path.join(HERE, "kcl_llm_perlayer.csv"), llm_col="auc_oof",
         ans=None, answer_audio=0.65, answer_text=None),
    dict(key="ad",  title="Alzheimer's, Pitt",
         enc=os.path.join(D, "DementiaBank/ad_encoder_perlayer.csv"), enc_col="auc_oof",
         llm=os.path.join(D, "DementiaBank/ad_llm_perlayer.csv"), llm_col="auc_mean",
         ans=os.path.join(D, "DementiaBank/ad_answer_position_probe.csv"), ans_col="auc_overall",
         answer_audio=0.62, answer_text=0.68),
]

BLUE, ORANGE, GREY = "#1f4e9a", "#d9531e", "#444444"
want = sys.argv[1:]  # panel keys to draw, in order; default: every panel with data except the KCL fallback
avail = [p for p in PANELS if p["enc"] and os.path.exists(p["enc"]) and ((p["key"] in want) if want else (p["key"] != "pd_kcl"))]
if want: avail = sorted(avail, key=lambda p: want.index(p["key"]))
if not avail:
    sys.exit("no panel data found")
n = len(avail)
single = (n == 1)
W = 3.4 if single else (7.1 if n >= 2 else 3.4)
fig, axes = plt.subplots(1, n, figsize=(W, 2.35 if single else 2.2), sharey=True, squeeze=False)
axes = axes[0]
N_ENC, N_LLM, GAP = 33, 33, 2   # x positions: encoder 0..32, gap, llm 35..67

for ax, p in zip(axes, avail):
    enc = col(read_csv(p["enc"]), p["enc_col"])
    xe = list(range(len(enc)))
    ax.plot(xe, enc, color=BLUE, lw=1.4, label="probe, encoder layer")
    has_llm = p["llm"] and os.path.exists(p["llm"])
    if has_llm:
        llm = col(read_csv(p["llm"]), p["llm_col"])
        xl = [N_ENC + GAP + i for i in range(len(llm))]
        ax.plot(xl, llm, color=BLUE, lw=1.4, ls="--", label="probe, language model stage")
        ax.axvline(N_ENC + GAP / 2 - 0.5, color="#bbbbbb", lw=0.8)
    if p["ans"] and os.path.exists(p["ans"]):
        rows = [r for r in read_csv(p["ans"]) if not (int(r["llm_layer"]) == 0 and float(r[p["ans_col"]]) == 0.5)]  # stage 0 row is a 0.5 placeholder, not a measurement
        ans = col(rows, p["ans_col"]); xa = [N_ENC + GAP + int(r["llm_layer"]) for r in rows]
        ax.plot(xa, ans, color=ORANGE, lw=1.4, label="probe, answer state")
    xmax = (N_ENC + GAP + N_LLM - 1) if has_llm else (N_ENC - 1)
    if p["answer_audio"] is not None:
        ax.hlines(p["answer_audio"], 0, xmax, color=GREY, lw=1.0, ls=(0, (4, 2)),
                  label="zero-shot answer, audio")
    if p["answer_text"] is not None:
        ax.hlines(p["answer_text"], 0, xmax, color=GREY, lw=1.0, ls=":", label="zero-shot answer, transcript")
    ax.set_title(p["title"], fontsize=8, pad=3)
    ax.set_ylim(0.5, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if has_llm:
        ax.set_xticks([0, 16, 32, N_ENC + GAP, N_ENC + GAP + 16, N_ENC + GAP + 32])
        ax.set_xticklabels(["0", "16", "32", "0", "16", "32"])
        ax.text(16, 0.505, "encoder", ha="center", va="bottom", fontsize=7, color="#666666")
        ax.text(N_ENC + GAP + 16, 0.505, "language model", ha="center", va="bottom", fontsize=7, color="#666666")
        ax.set_xlim(-1, xmax + 1)
    else:
        ax.set_xticks([0, 8, 16, 24, 32]); ax.set_xlim(-1, 33)
        ax.text(16, 0.505, "encoder", ha="center", va="bottom", fontsize=7, color="#666666")
    ax.set_xlabel("layer", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("AUC", fontsize=8, labelpad=2)

# one legend for the whole figure, below the panels
handles, labels = [], []
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels: handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=2 if single else 5, fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.14 if single else 0.12, 1, 1))
out = os.path.join(HERE, "fig1_gap.pdf" if not want else "fig1_gap_" + "_".join(want) + ".pdf")
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight")
print("panels:", [p["key"] for p in avail], "->", out)
