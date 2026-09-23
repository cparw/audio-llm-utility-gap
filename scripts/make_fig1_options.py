#!/usr/bin/env python3
"""Figure 1 design options, all at ICASSP single column 3.39 in."""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({
 "font.family":"serif","font.serif":["Times New Roman","Times","STIXGeneral"],
 "mathtext.fontset":"stix","pdf.fonttype":42,"ps.fonttype":42,
 "font.size":9,"axes.labelsize":9,"xtick.labelsize":9,"ytick.labelsize":9,
 "legend.fontsize":9})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

RERUN="scores/edaic_windows"
MIRROR="<local data dir>/release/overnight/curves/omni"
OUT="figures/fig1_options"
def col(p,n):
    with open(p) as fh: return [float(r[n]) for r in csv.DictReader(fh)]
pc=col(f"{MIRROR}/pcgita_encoder.csv","auc")+col(f"{MIRROR}/pcgita_llm.csv","auc")
pt=col(f"{MIRROR}/pitt_encoder.csv","auc")+col(f"{MIRROR}/pitt_llm.csv","auc")
ed=col(f"{RERUN}/o25_full_encoder_perlayer.csv","auc_oof")+col(f"{RERUN}/o25_full_llm_perlayer.csv","auc_oof")
NE=len(col(f"{MIRROR}/pcgita_encoder.csv","auc")); N=len(pc); x=list(range(N))
S=[dict(k="PC-GITA",y=pc,a=0.5635,ls="-", m="o",c="#1f4e9a"),
   dict(k="Pitt",   y=pt,a=0.6578,ls="--",m="s",c="#d9531e"),
   dict(k="E-DAIC", y=ed,a=0.8285,ls=":", m="^",c="#2e7d4f")]

def save(fig,name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png",dpi=300); plt.close(fig)
    print("  wrote",name)

# ---- A: shaded gap band -------------------------------------------------
fig,ax=plt.subplots(figsize=(3.39,2.0),dpi=300)
for s in S:
    ax.fill_between(x,[s["a"]]*N,s["y"],color=s["c"],alpha=0.13,lw=0,zorder=1)
    ax.axhline(s["a"],color=s["c"],ls=s["ls"],lw=0.8,zorder=2)
    ax.plot(x,s["y"],color=s["c"],ls=s["ls"],lw=1.3,marker=s["m"],markevery=6,ms=2.8,
            mfc="white",mew=0.9,mec=s["c"],zorder=4)
ax.axvline(NE-0.5,color="#bbb",lw=0.7)
ax.set_xlim(-1,N); ax.set_ylim(0.45,1.0); ax.set_yticks([0.5,0.7,0.9])
ax.set_xticks([0,16,NE,NE+14,N-1]); ax.set_xticklabels(["0","16","0","14","28"])
ax.set_xlabel("layer",labelpad=1); ax.set_ylabel("AUC",labelpad=1)
ax.text(NE*0.38,0.945,"PC-GITA",color=S[0]["c"],fontsize=9,ha="center",
        bbox=dict(fc="white",ec="none",pad=0.1))
ax.text(NE+4,0.855,"Pitt",color=S[1]["c"],fontsize=9,ha="center",
        bbox=dict(fc="white",ec="none",pad=0.1))
ax.text(NE*0.55,0.525,"E-DAIC",color=S[2]["c"],fontsize=9,ha="center",
        bbox=dict(fc="white",ec="none",pad=0.1))
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y",color="#eee",lw=0.5); ax.tick_params(length=2,pad=1.5)
ax.legend(handles=[Line2D([0],[0],color="#444",lw=1.3,label="probe"),
                   Line2D([0],[0],color="#444",lw=0.8,label="zero-shot answer")],
          loc="upper center",bbox_to_anchor=(0.5,-0.30),ncol=2,frameon=False,
          handlelength=2.2,columnspacing=1.4,handletextpad=0.5)
fig.tight_layout(pad=0.3); fig.subplots_adjust(bottom=0.32); save(fig,"optA_shaded_gap")

# ---- B: three stacked mini panels ---------------------------------------
fig,axes=plt.subplots(3,1,figsize=(3.39,2.6),dpi=300,sharex=True)
for ax,s in zip(axes,S):
    ax.fill_between(x,[s["a"]]*N,s["y"],color=s["c"],alpha=0.16,lw=0)
    ax.axhline(s["a"],color=s["c"],ls="--",lw=0.8)
    ax.plot(x,s["y"],color=s["c"],lw=1.2)
    ax.axvline(NE-0.5,color="#ccc",lw=0.6)
    ax.set_ylim(0.45,1.0); ax.set_yticks([0.6,0.9]); ax.set_xlim(-1,N)
    ax.text(0.98,0.88,s["k"],transform=ax.transAxes,ha="right",va="top",
            fontsize=9,color=s["c"])
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2,pad=1.2)
axes[-1].set_xticks([0,16,NE,NE+14,N-1]); axes[-1].set_xticklabels(["0","16","0","14","28"])
axes[-1].set_xlabel("layer",labelpad=1); axes[1].set_ylabel("AUC",labelpad=2)
fig.tight_layout(pad=0.3,h_pad=0.2); save(fig,"optB_stacked")

# ---- C: the gap itself, probe minus answer ------------------------------
fig,ax=plt.subplots(figsize=(3.39,2.0),dpi=300)
for s in S:
    g=[v-s["a"] for v in s["y"]]
    ax.plot(x,g,color=s["c"],ls=s["ls"],lw=1.3,marker=s["m"],markevery=6,ms=2.8,
            mfc="white",mew=0.9,mec=s["c"])
ax.axhline(0,color="#444",lw=0.9)
ax.axvline(NE-0.5,color="#bbb",lw=0.7)
ax.set_xlim(-1,N); ax.set_ylim(-0.30,0.42)
ax.set_xticks([0,16,NE,NE+14,N-1]); ax.set_xticklabels(["0","16","0","14","28"])
ax.set_xlabel("layer",labelpad=1); ax.set_ylabel("probe $-$ answer",labelpad=1)
ax.text(NE*0.40,0.36,"PC-GITA",color=S[0]["c"],fontsize=9,ha="center")
ax.text(NE+6,0.20,"Pitt",color=S[1]["c"],fontsize=9,ha="center")
ax.text(NE*0.50,-0.235,"E-DAIC",color=S[2]["c"],fontsize=9,ha="center")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y",color="#eee",lw=0.5); ax.tick_params(length=2,pad=1.5)
fig.tight_layout(pad=0.3); save(fig,"optC_gap_curve")

# ---- D: dumbbell summary, best probe vs answer --------------------------
fig,ax=plt.subplots(figsize=(3.39,1.7),dpi=300)
ys=[2,1,0]
for s,yy in zip(S,ys):
    best=max(s["y"])
    ax.plot([s["a"],best],[yy,yy],color=s["c"],lw=1.4,zorder=2)
    ax.scatter([s["a"]],[yy],s=26,facecolor="white",edgecolor=s["c"],lw=1.2,zorder=3)
    ax.scatter([best],[yy],s=26,color=s["c"],zorder=3)
    ax.text(best+0.012,yy,f"{best:.2f}",va="center",fontsize=8,color=s["c"])
    ax.text(s["a"]-0.012,yy,f"{s['a']:.2f}",va="center",ha="right",fontsize=8,color=s["c"])
ax.set_yticks(ys); ax.set_yticklabels([s["k"] for s in S])
ax.set_xlim(0.45,1.02); ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0])
ax.set_xlabel("AUC",labelpad=1)
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
ax.grid(axis="x",color="#eee",lw=0.5); ax.tick_params(length=2,pad=1.5)
ax.legend(handles=[Line2D([0],[0],marker="o",color="w",mfc="white",mec="#444",ms=5,label="zero-shot answer"),
                   Line2D([0],[0],marker="o",color="w",mfc="#444",ms=5,label="best probe")],
          loc="upper center",bbox_to_anchor=(0.5,-0.38),ncol=2,frameon=False,handletextpad=0.3)
fig.tight_layout(pad=0.3); fig.subplots_adjust(bottom=0.36); save(fig,"optD_dumbbell")
print("done")
