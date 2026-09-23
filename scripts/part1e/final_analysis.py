import os,sys,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_fp import decode,mfcc
S=os.path.dirname(os.path.abspath(__file__))
L=[]
def P(s=""): print(s); L.append(s)

# ---------- intersections, two thresholds ----------
cw=pd.read_csv(f"{S}/ad2020_to_pitt.csv"); ao=pd.read_csv(f"{S}/adresso_xcorr.csv")
cw["spk"]=cw.best.str.split(":").str[0]+":"+cw.best.str.split(":").str[1].str.split("-").str[0]
pm=pd.read_csv("/Volumes/G-Drive Pro/paper1_local_runs/pitt_manifest.csv")
def norm(s):
    s=str(s)
    return ("Control:"+s[7:]) if s.startswith("Control") else (("Dementia:"+s[8:]) if s.startswith("Dementia") else s)
Pset=set(norm(s) for s in pm.speaker.unique())
A=set(cw.spk); Af=set(cw.best)
for name,thr in [("strict margin>=0.5030",0.5030),("relaxed margin>=0.20",0.20)]:
    k=ao[ao.margin>=thr].copy()
    k["spk"]=k.best_pitt.str.split(":").str[0]+":"+k.best_pitt.str.split(":").str[1].str.split("-").str[0]
    B=set(k.spk); Bf=set(k.best_pitt)
    P(f"[{name}] ADReSSo recordings accepted {len(k)}/237 -> {len(B)} distinct Pitt speakers")
    P(f"   ADReSS-2020 speakers={len(A)}  ADReSSo speakers={len(B)}  INTERSECT={len(A&B)}  same Pitt recording in both={len(Af&Bf)}")
    P(f"   Pitt-probe(228) & ADReSS-2020={len(Pset&A)}   Pitt-probe & ADReSSo={len(Pset&B)}   Pitt-probe & (union)={len(Pset&(A|B))}   three-way={len(Pset&A&B)}")

# ---------- B3 top pairs with sequence confirmation ----------
d=pd.read_csv(f"{S}/fp.csv"); d=d[d.dur_s>0].reset_index(drop=True)
M=np.stack([np.fromstring(s,sep=";") for s in d.mfcc])
def seqvec(p,dur=10):
    x=decode(p,dur=dur); m=mfcc(x)[:,1:]
    m=m-m.mean(0); s=m.std(0); s[s<1e-6]=1e-6; m=m/s
    v=m.ravel(); v=v-v.mean(); n=np.linalg.norm(v)
    return v/n if n>0 else v
P(); P("== top-5 nearest within-set pairs (c0-dropped z-scored cosine) + first-10s MFCC-sequence r ==")
for ds,g in d.groupby("dataset",sort=True):
    idx=g.index.to_numpy(); X=M[idx][:,1:]
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1e-9
    Z=(X-mu)/sd; Zn=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    C=Zn@Zn.T; iu=np.triu_indices(len(idx),1); sim=C[iu]
    o=np.argsort(-sim)[:5]
    P(f"[{ds}]  n={len(idx)}  pairs={len(sim)}  max_cos_z={sim.max():.6f}")
    for k in o:
        a=int(idx[iu[0][k]]); b=int(idx[iu[1][k]])
        try: r=float(np.dot(seqvec(d.path[a]),seqvec(d.path[b])))
        except Exception: r=float("nan")
        P("    cos_z=%.6f seq_r=%.4f  %s (%.4fs) <-> %s (%.4fs)"%(sim[k],r,os.path.basename(d.path[a]),d.dur_s[a],os.path.basename(d.path[b]),d.dur_s[b]))

# ---------- neurovoz twin trees ----------
P(); P("== neurovoz zenodo_upload vs zenodo_upload_silencemode_1 (same basenames) ==")
a=d[d.dataset=="neurovoz"].copy(); b=d[d.dataset=="neurovoz_silencemode1"].copy()
a["bn"]=a.path.map(os.path.basename); b["bn"]=b.path.map(os.path.basename)
ai=a.set_index("bn"); bi=b.set_index("bn")
common=sorted(set(ai.index)&set(bi.index))
P(f"shared basenames {len(common)} of {len(a)} / {len(b)}")
P(f"md5 identical: {int((ai.loc[common,'md5'].values==bi.loc[common,'md5'].values).sum())} / {len(common)}")
Ai=np.stack([M[i] for i in ai.loc[common].index.map(lambda x:0)]) if False else None
Aidx=[ai.index.get_indexer([c])[0] for c in common[:0]]
A2=np.stack([M[a.index[a.bn==c][0]][1:] for c in common])
B2=np.stack([M[b.index[b.bn==c][0]][1:] for c in common])
mu=np.vstack([A2,B2]).mean(0); sd=np.vstack([A2,B2]).std(0); sd[sd<1e-9]=1e-9
Za=(A2-mu)/sd; Zb=(B2-mu)/sd
Za/=np.maximum(np.linalg.norm(Za,axis=1,keepdims=True),1e-12); Zb/=np.maximum(np.linalg.norm(Zb,axis=1,keepdims=True),1e-12)
cos=(Za*Zb).sum(1)
dd=(ai.loc[common,'dur_s'].values-bi.loc[common,'dur_s'].values)
P(f"paired cos_z: median={np.median(cos):.6f}  >0.999: {int((cos>0.999).sum())}/{len(common)}  >0.99: {int((cos>0.99).sum())}")
P(f"paired duration diff s: median={np.median(dd):.4f}  mean={dd.mean():.4f}  max|diff|={np.abs(dd).max():.4f}  identical: {int((np.abs(dd)<1e-9).sum())}")

# ---------- PART 3 silence ----------
P(); P("== PART 3: clips that are mostly silence ==")
P("   floor := 5th percentile of 25ms/10ms frame RMS in dBFS; metric := fraction of frames < floor+6 dB; flag if > 0.8")
for ds,g in d.groupby("dataset",sort=True):
    f=g.frac_below_floor6
    P(f"{ds:24s} n={len(g):5d}  flagged(>0.8)={int((f>0.8).sum()):4d} ({100*float((f>0.8).mean()):6.2f}%)  median={f.median():.4f} p95={f.quantile(.95):.4f} max={f.max():.4f} | frac<-50dBFS: median={g.frac_below_m50.median():.4f}, n>0.8={int((g.frac_below_m50>0.8).sum())}")
P(); P("   worst offenders (top 5 per dataset with flagged clips):")
for ds,g in d.groupby("dataset",sort=True):
    fl=g[g.frac_below_floor6>0.8].sort_values("frac_below_floor6",ascending=False)
    if len(fl)==0: continue
    P(f"   [{ds}] {len(fl)} flagged")
    for _,r in fl.head(5).iterrows():
        P("      frac=%.4f dur=%.4fs db_p5=%.4f db_med=%.4f db_max=%.4f  %s"%(r.frac_below_floor6,r.dur_s,r.db_p5,r.db_med,r.db_max,r.path))
P(); P("   top 3 by frac_below_floor6 in every dataset (even if below the 0.8 flag):")
for ds,g in d.groupby("dataset",sort=True):
    t=g.sort_values("frac_below_floor6",ascending=False).head(3)
    for _,r in t.iterrows():
        P("      [%s] frac=%.4f dur=%.4fs db_med=%.4f  %s"%(ds,r.frac_below_floor6,r.dur_s,r.db_med,os.path.basename(r.path)))
open(f"{S}/part3_out.txt","w").write("\n".join(L))
