"""Figure 1 from today's runs: per-layer probe AUC (encoder, LM stages at audio tokens, answer state) + zero-shot lines.
usage: make_fig1_v2.py OUT.pdf KEY[,KEY...]   keys: pitt pcgita edaic neurovoz kcl adresso adress2020"""
import os, sys, csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "<local data dir>/paper1_local_runs"; PR = f"{R}/probe2"
def rows(f): return list(csv.DictReader(open(f)))
def auc(y, p):
    pos=[b for a,b in zip(y,p) if a==1]; neg=[b for a,b in zip(y,p) if a==0]
    return sum(1.0 if a>b else 0.5 if a==b else 0.0 for a in pos for b in neg)/(len(pos)*len(neg))
def clip_auc(f):
    if not f or not os.path.exists(f): return None
    Rw=rows(f); return auc([int(r["label"]) for r in Rw],[float(r["p_yes"]) for r in Rw])
TITLE={"pitt":"Alzheimer's, Pitt","pcgita":"Parkinson's, PC-GITA","edaic":"Depression, E-DAIC","neurovoz":"Parkinson's, NeuroVoz","kcl":"Parkinson's, MDVR-KCL","adresso":"Alzheimer's, ADReSSo","adress2020":"Alzheimer's, ADReSS-2020"}
TEXT={"pitt":f"{R}/qwen2audio_pitt_text.csv","edaic":f"{R}/qwen2audio_edaic30_text.csv","adresso":f"{R}/qwen2audio_adresso_text.csv","adress2020":f"{R}/qwen2audio_adress2020_text.csv"}
out, keys = sys.argv[1], sys.argv[2].split(",")
BLUE, ORANGE, GREY = "#1f4e9a", "#d9531e", "#444444"; NE, NL, GAP = 33, 33, 2
fig, axes = plt.subplots(1, len(keys), figsize=(3.4 if len(keys)==1 else 7.1, 2.35 if len(keys)==1 else 2.2), sharey=True, squeeze=False); axes=axes[0]
for ax, k in zip(axes, keys):
    enc=[float(r["auc_oof"]) for r in rows(f"{PR}/{k}_encoder_perlayer.csv")]; llm=[float(r["auc_oof"]) for r in rows(f"{PR}/{k}_llm_perlayer.csv")]; ans=[float(r["auc_oof"]) for r in rows(f"{PR}/{k}_ans_perlayer.csv")]
    za=clip_auc(f"{PR}/{k}_zeroshot_scores.csv"); zt=clip_auc(TEXT.get(k))
    ax.plot(range(NE), enc, color=BLUE, lw=1.4, label="probe, encoder layer")
    xl=[NE+GAP+i for i in range(NL)]; ax.plot(xl, llm, color=BLUE, lw=1.4, ls="--", label="probe, language model stage")
    ax.plot(xl[1:], ans[1:], color=ORANGE, lw=1.4, label="probe, answer state")  # stage 0 is the embedding of the last prompt token, not a model state
    ax.axvline(NE+GAP/2-0.5, color="#bbbbbb", lw=0.8); xmax=NE+GAP+NL-1
    ax.hlines(za, 0, xmax, color=GREY, lw=1.0, ls=(0,(4,2)), label="zero-shot answer, audio")
    if zt is not None: ax.hlines(zt, 0, xmax, color=GREY, lw=1.0, ls=":", label="zero-shot answer, transcript")
    ax.set_title(TITLE[k], fontsize=8, pad=3); ax.set_ylim(0.5,1.0); ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_xticks([0,16,32,NE+GAP,NE+GAP+16,NE+GAP+32]); ax.set_xticklabels(["0","16","32","0","16","32"]); ax.set_xlim(-1,xmax+1)
    ax.text(16,0.505,"encoder",ha="center",va="bottom",fontsize=7,color="#666666"); ax.text(NE+GAP+16,0.505,"language model",ha="center",va="bottom",fontsize=7,color="#666666")
    ax.tick_params(labelsize=7); ax.grid(axis="y", color="#eeeeee", lw=0.6)
    print(f"{k}: enc max {max(enc):.3f} L{enc.index(max(enc))} | llm max {max(llm):.3f} S{llm.index(max(llm))} | ans max {max(ans[1:]):.3f} S{1+ans[1:].index(max(ans[1:]))} last {ans[-1]:.3f} | zero-shot audio {za:.3f} text {zt if zt is None else round(zt,3)}")
axes[0].set_ylabel("AUC", fontsize=8)
for ax in axes: ax.set_xlabel("layer", fontsize=8, labelpad=1)
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=2 if len(keys)==1 else 3,fontsize=6.5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.tight_layout(rect=(0,0.22 if len(keys)==1 else 0.18,1,1)); fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf",".png"), dpi=200, bbox_inches="tight"); print("wrote", out)
