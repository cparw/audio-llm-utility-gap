#!/usr/bin/env python3
"""Per-layer encoder AUC sweep for each POD4B arm: plain GroupKFold(5)-by-speaker
OOF AUC at every one of the 32 encoder layers, recomputed with the rank formula.
This is the curve the nested probe selects a layer from."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np, sys, csv, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
def auc_rank(y,s):
    y=np.asarray(y,float); s=np.asarray(s,float)
    o=np.argsort(s,kind="mergesort"); ss=s[o]; rr=np.arange(1,len(s)+1,dtype=float); i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        rr[i:j+1]=rr[i:j+1].mean(); i=j+1
    r=np.empty(len(s)); r[o]=rr
    n1=(y==1).sum(); n0=(y==0).sum()
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def one(X,y,g,l):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        sc=StandardScaler().fit(X[tr][:,l])
        oof[te]=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr][:,l]),y[tr]).predict_proba(sc.transform(X[te][:,l]))[:,1]
    return float(auc_rank(y,oof))
for CORPUS in ("adress2020","adresso"):
    out=[]
    for arm in ("orig","paronly","durmatch"):
        z=np.load(f"/workspace/p16pod4b/out/p16_{CORPUS}_{arm}_states.npz",allow_pickle=True)
        y=z["label"].astype(int); spk=z["spk"].astype(str); X=z["enc"].astype(np.float32)
        seen={}
        for s in spk:
            if s not in seen: seen[s]=len(seen)
        g=np.array([seen[s] for s in spk])
        a=Parallel(n_jobs=32)(delayed(one)(X,y,g,l) for l in range(X.shape[1]))
        for l,v in enumerate(a): out.append((arm,l,round(v,6)))
        print(f"{CORPUS} {arm}: best layer {int(np.argmax(a))} AUC {max(a):.4f}  layer0 {a[0]:.4f} last {a[-1]:.4f}",flush=True)
    with open(f"/workspace/p16pod4b/out/p16_{CORPUS}_enc_perlayer.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["arm","enc_layer","oof_auc"]); w.writerows(out)
    print(f"wrote p16_{CORPUS}_enc_perlayer.csv",flush=True)
