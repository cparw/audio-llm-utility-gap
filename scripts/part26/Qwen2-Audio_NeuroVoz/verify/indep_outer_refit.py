# Own outer refit at the pod-chosen layers, Mac sklearn (not a gate, a sanity check that OOF rows are held-out predictions)
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import csv, json, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
P="scores/part26/Qwen2-Audio_NeuroVoz"
z=np.load(P+"/stage/q2a_neurovoz_encstates.npz"); X=z["enc"]; y=z["label"].astype(int); names=z["name"].astype(str)
pc=list(csv.DictReader(open(P+"/per_clip/p26_q2a_neurovoz_perclip.csv"))); assert [r["clip"] for r in pc]==list(names)
L={"seed0":[24,25,18,24,20],"seed1":[20,20,30,22,25],"seed2":[17,21,25,19,30],"seed3":[20,25,20,17,20],"seed4":[20,30,24,21,28]}
def job(t,k):
    f=np.array([int(r["fold_"+t]) for r in pc]); tr=np.where(f!=k)[0]; te=np.where(f==k)[0]; l=L[t][k]
    sc=StandardScaler().fit(X[tr,l]); m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr,l]),y[tr])
    return t,k,te,m.predict_proba(sc.transform(X[te,l]))[:,1]
res=Parallel(n_jobs=12)(delayed(job)(t,k) for t in L for k in range(5))
out={}
for t in L:
    p=np.full(len(y),np.nan)
    for tt,k,te,pr in res:
        if tt==t: p[te]=pr
    orig=np.array([float(r["p_probe_"+t]) for r in pc])
    # same model at a WRONG layer ordering check is not needed; compare to OOF
    out[t]={"max_abs":float(np.max(np.abs(p-orig))),"auc_mac":float(roc_auc_score(y,p)),"auc_pod":float(roc_auc_score(y,orig)),
            "corr":float(np.corrcoef(p,orig)[0,1])}
    print(t,out[t],flush=True)
json.dump(out,open(sys.argv[1],"w"),indent=1)
