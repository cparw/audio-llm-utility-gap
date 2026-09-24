"""Re-derive the inner layer choice for all 25 outer folds (5 seeds x 5 folds) on the Mac with my own inner fill."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import csv, json, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
CELL="scores/part26/Qwen3-Omni-30B-A3B_NeuroVoz"
R=list(csv.DictReader(open(f"{CELL}/per_clip/Qwen3-Omni-30B-A3B_NeuroVoz_perclip.csv")))
S=np.load(f"{CELL}/stage/q3o_neurovoz_encstates.npz"); X=S["enc"]; y=S["label"].astype(int); spk=S["spk"].astype(str)
FO=np.array([[int(r[f"fold_seed{s}"]) for r in R] for s in range(5)])
J=json.load(open(f"{CELL}/pull/p26-q3o-neurovoz/out/p26_q3o_neurovoz_enc_nested5.json"))
def auc(yy,s):
    r=rankdata(s); n1=int(yy.sum()); n0=len(yy)-n1; return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
def fill(groups,k):
    uu,inv=np.unique(groups,return_inverse=True); c=np.bincount(inv); o=np.argsort(c,kind="stable")[::-1]
    load=np.zeros(k); fg=np.empty(len(uu),int)
    for gi in o:
        j=int(np.argmin(load)); fg[gi]=j; load[j]+=c[gi]
    return fg[inv]
def fp(tr,te,l):
    sc=StandardScaler().fit(X[tr,l])
    return LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr,l]),y[tr]).predict_proba(sc.transform(X[te,l]))[:,1]
def one(s,k,l):
    rng=np.random.default_rng(s); u=np.unique(spk); perm={a:i for i,a in enumerate(rng.permutation(u))}; g=np.array([perm[a] for a in spk])
    tr=np.flatnonzero(FO[s]!=k); f4=fill(g[tr],4); v=0.0
    for j in range(4):
        a,b=tr[f4!=j],tr[f4==j]; v+=auc(y[b],fp(a,b,l))
    return v/4
jobs=[(s,k,l) for s in range(5) for k in range(5) for l in range(32)]
res=Parallel(n_jobs=14)(delayed(one)(*j) for j in jobs)
out={"folds":[]}; mism=0
for s in range(5):
    for k in range(5):
        v=np.array([res[(s*5+k)*32+l] for l in range(32)]); pod=np.array(J["tags"][f"seed{s}"]["inner"][k])
        mine=int(np.argmax(v)); claimed=J["tags"][f"seed{s}"]["layers"][k]
        srt=np.sort(pod)[::-1]
        rec={"seed":s,"fold":k,"mac_layer":mine,"claimed_layer":claimed,"max_abs_inner_diff_vs_pod":float(np.max(np.abs(v-pod))),
             "pod_margin_top1_top2":float(srt[0]-srt[1]),"mac_score_at_claimed_minus_mac_best":float(v[claimed]-v[mine])}
        mism+=mine!=claimed; out["folds"].append(rec); print(rec,flush=True)
out["n_layer_mismatch"]=int(mism); out["max_abs_inner_diff"]=max(r["max_abs_inner_diff_vs_pod"] for r in out["folds"])
json.dump(out,open(sys.argv[1],"w"),indent=1); print("MISMATCH",mism,"MAXDIFF",out["max_abs_inner_diff"])
