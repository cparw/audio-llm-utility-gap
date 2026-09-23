"""Verifier twin (lo-pod3b-6) for the 200-partition spread of probe and probe-without-d AUC, per-partition output.
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
def fits(X,y,fold,dvec):
    oof=np.zeros(len(y)); pt=np.zeros(len(y)); pz=np.zeros(len(y))
    for k in range(5):
        tr=np.flatnonzero(fold!=k); te=np.flatnonzero(fold==k)
        sc=StandardScaler().fit(X[tr]); A=sc.transform(X[tr]); B=sc.transform(X[te])
        m=LogisticRegression(max_iter=2000,class_weight="balanced").fit(A,y[tr]); oof[te]=m.predict_proba(B)[:,1]
        w=m.coef_[0]; wp=w-np.dot(w,dvec)/np.dot(dvec,dvec)*dvec; a=A@wp; b=B@wp
        pt[te]=(b-a.mean())/(a.std()+1e-12); pz[te]=(b-b.mean())/b.std()
    return oof,pt,pz
res={"X_sha256":hashlib.sha256(np.ascontiguousarray(Xf).tobytes()).hexdigest()}
for arm in ("all","conflict","agreement"):
    m=np.ones(len(y_all),bool) if arm=="all" else arm_all==arm
    X=X_all[m]; y=y_all[m]; spk=spk_all[m]; d=readout(X)
    def one(s):
        o,pt,pz=fits(X,y,perm_folds(spk,s),d); return s,auc(y,o),auc(y,pt),auc(y,pz)
    C=Parallel(n_jobs=32)(delayed(one)(s) for s in range(200))
    with open(f"/workspace/out/verify_spread_wo_{arm}.csv","w",newline="") as fh:
        wr=csv.writer(fh); wr.writerow(["perm_seed","auc_probe","auc_wo_trainz","auc_wo_foldz"]); [wr.writerow([s,repr(a),repr(b),repr(c)]) for s,a,b,c in C]
    A=np.array([c[1:] for c in C]); res[arm]={"probe_mean":float(A[:,0].mean()),"wo_trainz_mean":float(A[:,1].mean()),"wo_foldz_mean":float(A[:,2].mean())}
    print(arm,res[arm],flush=True)
json.dump(res,open("/workspace/out/verify_spread_wo.json","w"),indent=1); print("DONE")
