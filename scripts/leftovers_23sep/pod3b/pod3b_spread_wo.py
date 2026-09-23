"""PRIMARY (lo-pod3b-5): over the same 200 tie-order partitions as pod3b_gap.py (greedy_folds seeds 0..199),
the probe AUC and the probe AUC with the readout direction removed (train-fold z as shipped, and held-out-fold z,
canonical), per direction_by_arm.py's per-fold recipe. Input: exact float32 slice ans[:,-1,:]."""
import os, sys, json, csv
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
YES=[7414,9454,9693,9834,14004]; NO=[902,2152,2308,2753,8996]
z=np.load("/workspace/q3o_pitt_ansfinal.npz",allow_pickle=True)
X_all=z["X"].astype(np.float64); y_all=z["label"].astype(int); spk_all=z["spk"].astype(str); nm=z["name"].astype(str)
arms_all=np.array([n.split("_")[0] for n in nm])
W=np.load("/workspace/lmh/q3o_lm_head.npy").astype(np.float16).astype(np.float32)
def rank_auc(yv,s):
    yv=np.asarray(yv); s=np.asarray(s,float); n1=int((yv==1).sum()); n0=int((yv==0).sum())
    o=np.argsort(s,kind="mergesort"); ss=s[o]; rk=np.empty(len(s)); i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        rk[o[i:j+1]]=(i+j)/2.0+1.0; i=j+1
    return float((rk[yv==1].sum()-n1*(n1+1)/2.0)/(n1*n0))
def direction(X):
    lg=X@W.T; mm=lg.max(axis=-1,keepdims=True); ee=np.exp(lg-mm); s=ee/ee.sum(axis=-1,keepdims=True)
    wy=s[:,YES].mean(0); wn=s[:,NO].mean(0); wy=wy/max(wy.sum(),1e-12); wn=wn/max(wn.sum(),1e-12)
    dv=(W[YES].T.astype(np.float64)@wy)-(W[NO].T.astype(np.float64)@wn); return dv/np.linalg.norm(dv)
def greedy_folds(spk,perm_seed):
    u,gi=np.unique(spk,return_inverse=True); cnt=np.bincount(gi)
    pr=np.random.default_rng(perm_seed).permutation(len(u)); order=pr[np.argsort(-cnt[pr],kind="stable")]
    load=np.zeros(5); g2f=np.zeros(len(u),int)
    for gidx in order:
        k=int(np.argmin(load)); load[k]+=cnt[gidx]; g2f[gidx]=k
    return g2f[gi]
def one(X,y,spk,dvec,seed):
    f=greedy_folds(spk,seed); oof=np.zeros(len(y)); oof2=np.zeros(len(y)); pz=np.zeros(len(y))
    for k in range(5):
        tr=np.where(f!=k)[0]; te=np.where(f==k)[0]
        ss=StandardScaler().fit(X[tr]); Xt=ss.transform(X[tr]); Xe=ss.transform(X[te])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xt,y[tr])
        oof[te]=lr.predict_proba(Xe)[:,1]
        wf=lr.coef_[0]; wp=wf-(wf@dvec)/(dvec@dvec)*dvec
        oof2[te]=(Xe@wp-(Xt@wp).mean())/((Xt@wp).std()+1e-12); b=Xe@wp; pz[te]=(b-b.mean())/b.std()
    return seed,rank_auc(y,oof),rank_auc(y,oof2),rank_auc(y,pz)
res={}
for arm in ("all","conflict","agreement"):
    m=np.ones(len(y_all),bool) if arm=="all" else arms_all==arm
    X=X_all[m]; y=y_all[m]; spk=spk_all[m]; d=direction(X)
    C=Parallel(n_jobs=32)(delayed(one)(X,y,spk,d,s) for s in range(200))
    with open(f"/workspace/out/spread_wo_{arm}.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["perm_seed","auc_probe","auc_wo_trainz","auc_wo_foldz"]); [w.writerow([s,repr(a),repr(b),repr(c)]) for s,a,b,c in C]
    A=np.array([c[1:] for c in C]); res[arm]={"probe_mean":float(A[:,0].mean()),"wo_trainz_mean":float(A[:,1].mean()),"wo_foldz_mean":float(A[:,2].mean())}
    print(arm,res[arm],flush=True)
json.dump(res,open("/workspace/out/spread_wo.json","w"),indent=1); print("DONE")
