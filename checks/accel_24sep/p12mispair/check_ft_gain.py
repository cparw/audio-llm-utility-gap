#!/usr/local/bin/python3
# five of six PD/AD sets: projector FT minus zero-shot, paired speaker bootstrap. Independent code.
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = os.path.expanduser("<local data dir>/release")
def auc(y,p): return float(roc_auc_score(np.asarray(y), np.asarray(p,float)))
for ds in ["pcgita","neurovoz","kcl","pitt","adresso","adress2020"]:
    ft = pd.read_csv(f"{R}/omni_final/omnisft_{ds}_oof.csv"); ft["clip"] = ft.path.map(os.path.basename)
    zs = pd.read_csv(f"{R}/omni_final/omni_{ds}_zeroshot_scores.csv")
    m = ft[["clip","p_yes","label","speaker"]].merge(zs[["clip","p_yes","label"]].rename(columns={"p_yes":"p_zs","label":"l2"}), on="clip")
    assert len(m)==len(ft)==len(zs) and (m.label==m.l2).all(), (ds, len(m), len(ft), len(zs))
    y=m.label.values; a=m.p_yes.values; b=m.p_zs.values
    spk=m.speaker.astype(str).values; u,inv=np.unique(spk,return_inverse=True); g=[np.where(inv==k)[0] for k in range(len(u))]
    rng=np.random.default_rng(0); v=[]
    for _ in range(2000):
        ii=np.concatenate([g[k] for k in rng.integers(0,len(g),len(g))])
        if len(np.unique(y[ii]))<2: continue
        v.append(auc(y[ii],a[ii])-auc(y[ii],b[ii]))
    lo,hi=np.percentile(v,[2.5,97.5]); d=auc(y,a)-auc(y,b)
    print(f"{ds}: FT {auc(y,a):.4f} ZS {auc(y,b):.4f} gain {d:+.4f} [{lo:+.4f}, {hi:+.4f}] n={len(m)} spk={len(u)} excludes0={lo>0 or hi<0}")
