#!/usr/bin/env python3
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
"auc", see release/overnight/scripts/part11_export_curves.py line 111.

usage:  make_fig1_v2.py [full|mid30|mid300] OUT.pdf
"""
import os
import sys
import csv

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 9
matplotlib.rcParams["ytick.labelsize"] = 9
matplotlib.rcParams["legend.fontsize"] = 9
import matplotlib.pyplot as plt

RERUN = "scores/edaic_windows"
MIRROR = "<local data dir>/release/overnight/curves/omni"

# zero-shot answer AUCs. E-DAIC from the variant run json, PC-GITA and Pitt from
# release/master/master_lookup.csv (Qwen2.5-Omni, "zero shot answer").
EDAIC_ANSWER = {"full": 0.8285, "mid30": 0.7205, "mid300": 0.8122}
EDAIC_WINDOW = {"full": "900 s cap", "mid30": "middle 30 s", "mid300": "middle 300 s"}
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
    variant = sys.argv[1] if len(sys.argv) > 1 else "full"
    if variant not in EDAIC_ANSWER:
        sys.exit(f"variant must be one of {sorted(EDAIC_ANSWER)}")
    out = sys.argv[2] if len(sys.argv) > 2 else (
        "figures/fig1_gap.pdf")

    pc_enc, pc_llm, pc = curve(f"{MIRROR}/pcgita_encoder.csv", "auc",
                               f"{MIRROR}/pcgita_llm.csv", "auc")
    pt_enc, pt_llm, pt = curve(f"{MIRROR}/pitt_encoder.csv", "auc",
                               f"{MIRROR}/pitt_llm.csv", "auc")
    ed_enc, ed_llm, ed = curve(f"{RERUN}/o25_{variant}_encoder_perlayer.csv", "auc_oof",
                               f"{RERUN}/o25_{variant}_llm_perlayer.csv", "auc_oof")

    n_enc = len(pc_enc)
    n_llm = len(pc_llm)
    assert len(pt_enc) == len(ed_enc) == n_enc, "encoder length mismatch"
    assert len(pt_llm) == len(ed_llm) == n_llm, "lm stage length mismatch"
    n = n_enc + n_llm
    x = list(range(n))

    series = [
        dict(key="PC-GITA", label="PC-GITA", y=pc, ans=PCGITA_ANSWER,
             ls="-", marker="o", color="#1f4e9a"),
        dict(key="Pitt", label="Pitt", y=pt, ans=PITT_ANSWER,
             ls="--", marker="s", color="#d9531e"),
        dict(key="E-DAIC", label="E-DAIC", y=ed, ans=EDAIC_ANSWER[variant],
             ls=":", marker="^", color="#2e7d4f"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=300)

    # answer reference lines first, so the probe curves sit on top
    ans_handles = []
    for s in series:
        h = ax.axhline(s["ans"], color=s["color"], ls=s["ls"], lw=1.0,
                       alpha=0.85, zorder=2,
                       label=f"{s['label']}, zero-shot answer")
        ans_handles.append(h)

    probe_handles = []
    for s in series:
        h, = ax.plot(x, s["y"], color=s["color"], ls=s["ls"], lw=1.6,
                     marker=s["marker"], markevery=4, ms=4.0,
                     markerfacecolor="white", markeredgewidth=1.1,
                     markeredgecolor=s["color"], zorder=4,
                     label=f"{s['label']}, probe")
        probe_handles.append(h)

    # encoder / language model boundary
    ax.axvline(n_enc - 0.5, color="#bbbbbb", lw=0.8, zorder=1)

    ax.set_xlim(-1, n)
    # headroom above 0.9234 (the PC-GITA peak) so the upper-left legend never
    # sits on a curve; the y axis still only ticks up to 1.0
    ax.set_ylim(0.45, 1.12)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # the last encoder tick and the first language model tick would print on top
    # of each other at the divider, so neither block labels its own boundary
    ticks = [0, 8, 16, 24, n_enc, n_enc + 8, n_enc + 16, n_enc + n_llm - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "8", "16", "24",
                        "0", "8", "16", str(n_llm - 1)])
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC")

    ax.text((n_enc - 1) / 2.0, 0.466, "audio encoder", ha="center", va="bottom",
            fontsize=9, color="#555555")
    ax.text(n_enc + (n_llm - 1) / 2.0, 0.466, "language model", ha="center",
            va="bottom", fontsize=9, color="#555555")

    ax.grid(axis="y", color="#e9e9e9", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5)

    ax.legend(probe_handles + ans_handles,
              [h.get_label() for h in probe_handles + ans_handles],
              loc="upper left", ncol=2, fontsize=9, frameon=True,
              framealpha=1.0, facecolor="white", edgecolor="#dddddd",
              handlelength=2.6, borderpad=0.45, labelspacing=0.35,
              columnspacing=1.2).set_zorder(10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))

    print(f"variant {variant}  E-DAIC window {EDAIC_WINDOW[variant]}")
    print(f"stages per dataset: {n_enc} encoder + {n_llm} language model = {n}")
    for s in series:
        lo, hi = min(s["y"]), max(s["y"])
        print(f"  {s['key']:<8} curve min {lo:.4f}  max {hi:.4f}  "
              f"dashed answer {s['ans']:.4f}  "
              f"(min at stage {s['y'].index(lo)}, max at stage {s['y'].index(hi)})")
    print("wrote", out)


if __name__ == "__main__":
    main()
