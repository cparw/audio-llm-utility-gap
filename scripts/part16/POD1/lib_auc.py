import numpy as np

def auc_rank(y, s):
    """AUC via the rank formula, with tie handling (average ranks)."""
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[y == 1].sum()
    return (r1 - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)

def boot_auc(y, s, spk, n_draws=2000, seed=0):
    """Speaker-level bootstrap CI for a single AUC. Returns (lo, hi, usable)."""
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk)
    uspk = np.unique(spk)
    idx_by = {u: np.where(spk == u)[0] for u in uspk}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        pick = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        a = auc_rank(y[ii], s[ii])
        if np.isfinite(a):
            vals.append(a)
    vals = np.asarray(vals)
    if len(vals) == 0:
        return float('nan'), float('nan'), 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)

def boot_paired(y, s, spk, arm, n_draws=2000, seed=0):
    """PAIRED speaker bootstrap: one speaker draw per replicate, both arm AUCs inside it."""
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk); arm = np.asarray(arm)
    uspk = np.unique(spk)
    idx_by = {u: np.where(spk == u)[0] for u in uspk}
    rng = np.random.default_rng(seed)
    vc, va = [], []
    for _ in range(n_draws):
        pick = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        yy, sy, aa = y[ii], s[ii], arm[ii]
        mc = aa == 'conflict'; ma = aa == 'agreement'
        ac = auc_rank(yy[mc], sy[mc]) if mc.sum() else float('nan')
        ag = auc_rank(yy[ma], sy[ma]) if ma.sum() else float('nan')
        if np.isfinite(ac) and np.isfinite(ag):
            vc.append(ac); va.append(ag)
    vc = np.asarray(vc); va = np.asarray(va)
    out = {}
    for nm, v in (('conflict', vc), ('agreement', va)):
        if len(v):
            out[nm] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        else:
            out[nm] = (float('nan'), float('nan'))
    out['usable'] = int(len(vc))
    return out
