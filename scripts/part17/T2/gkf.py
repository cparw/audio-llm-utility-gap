"""GroupKFold(n_splits=5, shuffle=False) re-implemented with a selectable argsort tie-break.
kind='mac'    : np.argsort default (what sklearn does on this Mac)  -> must equal sklearn output here
kind='stable' : np.argsort(kind='stable'), then reversed exactly as sklearn does
Returns fold index per sample (0..4)."""
import numpy as np
from sklearn.model_selection import GroupKFold

def gkf_folds(groups, kind="mac", n_splits=5):
    groups = np.asarray(groups)
    ug, gidx = np.unique(groups, return_inverse=True)
    cnt = np.bincount(gidx)
    if kind == "mac":
        order = np.argsort(cnt)[::-1]
    elif kind == "stable":
        order = np.argsort(cnt, kind="stable")[::-1]
    else:
        raise ValueError(kind)
    w = cnt[order]; per = np.zeros(n_splits); g2f = np.zeros(len(ug), int)
    for i, wt in enumerate(w):
        f = int(np.argmin(per)); per[f] += wt; g2f[order[i]] = f
    return g2f[gidx]

def sk_folds(groups, n_splits=5):
    groups = np.asarray(groups); f = np.full(len(groups), -1)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)):
        f[te] = k
    return f
