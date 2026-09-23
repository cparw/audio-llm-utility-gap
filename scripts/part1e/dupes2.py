import os,sys,numpy as np,pandas as pd
S=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(f"{S}/fp.csv"); d=d[d.dur_s>0].reset_index(drop=True)
M=np.stack([np.fromstring(s,sep=";") for s in d.mfcc])
L=[]
def P(s=""): print(s); L.append(s)
P("== A. byte-identical files (md5 over raw file bytes) ==")
for ds,g in d.groupby("dataset",sort=True):
    dups=g[g.duplicated("md5",keep=False)]
    P(f"{ds:24s} files={len(g):5d}  md5-identical groups={dups.md5.nunique():3d}  files_in_groups={len(dups):4d}")
for ds,g in d.groupby("dataset",sort=True):
    dups=g[g.duplicated("md5",keep=False)]
    for m5,gg in dups.groupby("md5"):
        P(f"   [{ds}] md5={m5}  n={len(gg)}")
        for p in gg.path.tolist(): P("      "+p)
P()
P("== B1. LITERAL SPEC: cosine on 13-dim MFCC mean (c0 included), threshold 0.999 ==")
for ds,g in d.groupby("dataset",sort=True):
    X=M[g.index.to_numpy()]
    Xn=X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    C=Xn@Xn.T; iu=np.triu_indices(len(g),1); sim=C[iu]
    npair=len(sim)
    P(f"{ds:24s} n={len(g):5d} pairs={npair:9d}  cos>0.999={int((sim>0.999).sum()):7d} ({100*float((sim>0.999).mean()):7.3f}%)  min_cos={sim.min():.4f} median={np.median(sim):.6f}")
P("   NOTE: c0 (log-energy) dominates the 13-dim mean vector, so every pair is already near-collinear.")
P("   The 0.999 threshold on this vector is not a duplicate test. B2 is the corrected test.")
P()
P("== B2. CORRECTED: c0 dropped, per-dataset z-scored coefficients, cosine > 0.999 ==")
cands={}
for ds,g in d.groupby("dataset",sort=True):
    idx=g.index.to_numpy(); X=M[idx][:,1:]
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1e-9
    Z=(X-mu)/sd
    Zn=Z/np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    C=Zn@Zn.T; np.fill_diagonal(C,-9)
    iu=np.triu_indices(len(idx),1); sim=C[iu]
    hit=np.where(sim>0.999)[0]
    pr=sorted([(float(sim[k]),int(idx[iu[0][k]]),int(idx[iu[1][k]])) for k in hit],reverse=True)
    cands[ds]=pr
    P(f"{ds:24s} n={len(idx):5d} pairs={len(sim):9d}  cos_z>0.999={len(pr):7d} ({100*len(pr)/max(len(sim),1):7.4f}%)  max={sim.max():.6f} median={np.median(sim):.4f}")
open(f"{S}/part2_A_B.txt","w").write("\n".join(L))
import pickle; pickle.dump({k:v[:4000] for k,v in cands.items()},open(f"{S}/cands.pkl","wb"))
d[["dataset","path","md5","dur_s"]].to_csv(f"{S}/fp_index.csv",index=False)
