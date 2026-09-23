# Verifier shared definitions (own code). Definitions follow direction_by_arm.py; implementation is independent.
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
YES=np.array([7414,9454,9693,9834,14004]); NO=np.array([902,2152,2308,2753,8996])
def readout_dir(X,W):
    """X float64 [n,2048], W float32 [V,2048]. Mass-weighted Yes minus No lm_head direction, unit norm."""
    L=X@W.T.astype(np.float64) if W.dtype!=np.float64 else X@W.T
    lz=logsumexp(L,axis=1,keepdims=True)
    py=np.exp(L[:,YES]-lz); pn=np.exp(L[:,NO]-lz)
    a=py.mean(0); b=pn.mean(0); a=a/a.sum(); b=b/b.sum()
    Wy=W[YES].astype(np.float64); Wn=W[NO].astype(np.float64)
    d=a@Wy-b@Wn
    return d/np.sqrt(d@d), (py.sum(1)/(py.sum(1)+pn.sum(1)))
def greedy_groupkfold(spk,k=5,tie_rank=None):
    """sklearn-style greedy: groups by size descending into the lightest fold.
    tie_rank None -> sklearn>=1.9 order (stable ascending sort reversed, i.e. ties by descending group index).
    tie_rank array -> ties by ascending tie_rank."""
    u,gi=np.unique(spk,return_inverse=True); c=np.bincount(gi)
    if tie_rank is None:
        order=np.lexsort((-np.arange(len(u)),-c))
    else:
        order=np.lexsort((tie_rank,-c))
    load=np.zeros(k,np.int64); g2f=np.empty(len(u),np.int64)
    for g in order:
        f=int(np.argmin(load)); g2f[g]=f; load[f]+=c[g]
    return g2f[gi]
def fit_folds(X,y,fold,d):
    """Per fold: scaler on train, balanced LR. Returns oof prob, unit mean raw-space weight, wo_trainz, wo_foldz."""
    n=len(y); oof=np.zeros(n); wt=np.zeros(n); wf=np.zeros(n); ws=[]
    for k in np.unique(fold):
        te=np.where(fold==k)[0]; tr=np.where(fold!=k)[0]
        sc=StandardScaler().fit(X[tr]); Xt=sc.transform(X[tr]); Xe=sc.transform(X[te])
        lr=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Xt,y[tr])
        oof[te]=lr.predict_proba(Xe)[:,1]
        c=lr.coef_[0]; ws.append(c/sc.scale_)
        cp=c-(c@d)/(d@d)*d
        st=Xt@cp; se=Xe@cp
        wt[te]=(se-st.mean())/(st.std()+1e-12)
        wf[te]=(se-se.mean())/se.std()
    w=np.mean(ws,axis=0); w=w/np.sqrt(w@w)
    return oof,w,wt,wf
def auc(y,s): return float(roc_auc_score(y,s))
def boot(y,spk,cols,draws=2000):
    """Paired speaker bootstrap: fresh default_rng(0), idx=rng.choice(N,N) over sorted unique speakers."""
    u=np.unique(spk); pos={s:np.where(spk==s)[0] for s in u}; rng=np.random.default_rng(0)
    out={k:[] for k in cols}
    for _ in range(draws):
        idx=rng.choice(len(u),size=len(u),replace=True)
        ii=np.concatenate([pos[u[j]] for j in idx]); yy=y[ii]
        if yy.min()==yy.max(): continue
        for k,v in cols.items(): out[k].append(roc_auc_score(yy,v[ii]))
    return {k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5)),len(v)] for k,v in out.items()}
def rand_floor(w,seed=0,n=2000):
    rng=np.random.default_rng(seed); R=np.array([rng.normal(size=len(w)) for _ in range(n)])
    R/=np.linalg.norm(R,axis=1,keepdims=True); a=np.abs(R@w)
    return float(a.mean()),float(np.percentile(a,2.5)),float(np.percentile(a,97.5))
