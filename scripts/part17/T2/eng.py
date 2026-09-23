import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from gkf import gkf_folds
NJ = 14

def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]

def _outer_one(X, y, fold, k, l):
    tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
    return te, fitpred(X[tr][:, l], y[tr], X[te][:, l])

def outer_at_layers(X, y, fold, layers):
    oof = np.zeros(len(y))
    for te, p in Parallel(n_jobs=5)(delayed(_outer_one)(X, y, fold, k, layers[k]) for k in range(5)):
        oof[te] = p
    return oof

def _inner_one(X, y, groups, fold, kind, k, l):
    tr = np.where(fold != k)[0]
    ifold = gkf_folds(groups[tr], kind, n_splits=4)
    s = 0.0
    for j in range(4):
        itr = np.where(ifold != j)[0]; ite = np.where(ifold == j)[0]
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return k, l, s / 4

def inner_select(X, y, groups, fold, kind):
    nl = X.shape[1]
    if nl == 1: return [0]*5
    res = Parallel(n_jobs=NJ)(delayed(_inner_one)(X, y, groups, fold, kind, k, l) for k in range(5) for l in range(nl))
    S = np.zeros((5, nl))
    for k, l, s in res: S[k, l] = s
    return [int(np.argmax(S[k])) for k in range(5)]

def _curve_one(X, y, fold, l):
    oof = np.zeros(len(y))
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        oof[te] = fitpred(X[tr][:, l], y[tr], X[te][:, l])
    return l, oof

def layer_oofs(X, y, fold, layers=None):
    layers = range(X.shape[1]) if layers is None else layers
    return dict(Parallel(n_jobs=NJ)(delayed(_curve_one)(X, y, fold, l) for l in layers))

def rank_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    o = np.argsort(s, kind="mergesort"); ss = s[o]; r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j+1] == ss[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

def perm_codes(spk, seed):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}
    return np.array([perm[s] for s in spk])
