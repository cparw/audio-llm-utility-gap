"""Verifier twin for the 200-partition spread, with per-partition output (lo-pod3b-4).
Code path is the verifier's (pod3b_verify.py: perm_folds, fits, mannwhitneyu AUC, logsumexp readout),
not pod3b_gap.py. Input: exact float32 slice ans[:,-1,:] of q3o_pitt_states.npz (sha256 of the bytes logged)."""
import os, json, csv, hashlib, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
YES=[7414,9454,9693,9834,14004]; NO=[902,2152,2308,2753,8996]
z=np.load("/workspace/q3o_pitt_ansfinal.npz",allow_pickle=True)
Xf=z["X"]; X_all=Xf.astype(np.float64); y_all=z["label"].astype(int); spk_all=z["spk"].astype(str); nm=z["name"].astype(str)
arm_all=np.char.partition(nm,"_")[:,0]
W=np.load("/workspace/lmh/q3o_lm_head.npy").astype(np.float16).astype(np.float32)
def auc(y,s):
    U=mannwhitneyu(s[y==1],s[y==0],alternative="two-sided",method="asymptotic").statistic; return float(U/((y==1).sum()*(y==0).sum()))
def readout(X):
    L=X@W.T; P=np.exp(L-logsumexp(L,axis=1,keepdims=True)); a=P[:,YES].mean(0); b=P[:,NO].mean(0); a/=a.sum(); b/=b.sum()
    dv=W[YES].astype(np.float64).T@a-W[NO].astype(np.float64).T@b; return dv/np.linalg.norm(dv)
def perm_folds(spk,seed,k=5):
    u,inv=np.unique(spk,return_inverse=True); c=np.bincount(inv); pr=np.random.default_rng(seed).permutation(len(u))
    order=pr[np.argsort(-c[pr],kind="stable")]; load=np.zeros(k); gf=np.zeros(len(u),int)
    for gi in order: j=int(np.argmin(load)); load[j]+=c[gi]; gf[gi]=j
    return gf[inv]
def fits(X,y,fold):
    oof=np.zeros(len(y)); ws=[]
    for k in range(5):
        tr=np.flatnonzero(fold!=k); te=np.flatnonzero(fold==k); sc=StandardScaler().fit(X[tr])
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr]); oof[te]=m.predict_proba(sc.transform(X[te]))[:,1]; ws.append(m.coef_[0]/sc.scale_)
    w=np.mean(ws,0); return oof, w/np.linalg.norm(w)
res={"X_sha256":hashlib.sha256(np.ascontiguousarray(Xf).tobytes()).hexdigest()}
for arm in ("all","conflict","agreement"):
    m=np.ones(len(y_all),bool) if arm=="all" else arm_all==arm
    X=X_all[m]; y=y_all[m]; spk=spk_all[m]; d=readout(X)
    def one(s):
        o,w=fits(X,y,perm_folds(spk,s)); return s,auc(y,o),float(w@d)
    C=Parallel(n_jobs=32)(delayed(one)(s) for s in range(200))
    with open(f"/workspace/out/verify_spread_{arm}.csv","w",newline="") as fh:
        wr=csv.writer(fh); wr.writerow(["perm_seed","auc_probe","cosine"]); [wr.writerow([s,repr(a),repr(c)]) for s,a,c in C]
    a=np.array([c[1] for c in C]); cc=np.array([c[2] for c in C])
    res[arm]={"auc_mean":float(a.mean()),"auc_sd":float(a.std(ddof=1)),"auc_min":float(a.min()),"auc_max":float(a.max()),
              "auc_p2_5":float(np.percentile(a,2.5)),"auc_p97_5":float(np.percentile(a,97.5)),"cos_mean":float(cc.mean()),"cos_min":float(cc.min()),"cos_max":float(cc.max())}
    print(arm,res[arm],flush=True)
json.dump(res,open("/workspace/out/verify_spread.json","w"),indent=1); print("DONE")
