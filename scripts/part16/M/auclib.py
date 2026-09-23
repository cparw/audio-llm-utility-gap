import numpy as np, pandas as pd

def auc_rank(y, s):
    """AUC from ranks (Mann-Whitney), ties averaged."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    assert set(np.unique(y)) <= {0,1}
    n1 = int((y==1).sum()); n0 = int((y==0).sum())
    if n1==0 or n0==0: return float('nan')
    r = pd.Series(s).rank(method='average').to_numpy()
    return float((r[y==1].sum() - n1*(n1+1)/2.0) / (n1*n0))

def boot_one(y, s, spk, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    y=np.asarray(y).astype(int); s=np.asarray(s,float); spk=np.asarray(spk).astype(str)
    u=np.unique(spk); idx={k:np.where(spk==k)[0] for k in u}
    out=[]
    for _ in range(B):
        pick=rng.choice(u, size=len(u), replace=True)
        ii=np.concatenate([idx[k] for k in pick])
        if len(np.unique(y[ii]))<2: continue
        out.append(auc_rank(y[ii], s[ii]))
    out=np.array(out)
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5)), len(out)

def boot_paired(y, sa, sb, spk, B=2000, seed=0):
    """One speaker draw per replicate, both AUCs + difference inside it."""
    rng = np.random.default_rng(seed)
    y=np.asarray(y).astype(int); sa=np.asarray(sa,float); sb=np.asarray(sb,float); spk=np.asarray(spk).astype(str)
    u=np.unique(spk); idx={k:np.where(spk==k)[0] for k in u}
    da=[]; db=[]; dd=[]
    for _ in range(B):
        pick=rng.choice(u, size=len(u), replace=True)
        ii=np.concatenate([idx[k] for k in pick])
        if len(np.unique(y[ii]))<2: continue
        a=auc_rank(y[ii], sa[ii]); b=auc_rank(y[ii], sb[ii])
        da.append(a); db.append(b); dd.append(a-b)
    da=np.array(da); db=np.array(db); dd=np.array(dd)
    return dict(a_lo=float(np.percentile(da,2.5)), a_hi=float(np.percentile(da,97.5)),
                b_lo=float(np.percentile(db,2.5)), b_hi=float(np.percentile(db,97.5)),
                d_lo=float(np.percentile(dd,2.5)), d_hi=float(np.percentile(dd,97.5)),
                usable=int(len(dd)))

def sha1mb(p):
    import hashlib
    h=hashlib.sha256()
    with open(p,'rb') as f: h.update(f.read(1024*1024))
    return h.hexdigest()
