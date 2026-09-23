import numpy as np
def auc_rank(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float('nan')
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s), float)
    ss = s[order]; i = 0; r = np.empty(len(s), float)
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j+1] == ss[i]: j += 1
        r[i:j+1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks[order] = r
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def boot_spk(y, s, spk, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk)
    u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}
    out = []
    for _ in range(n):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        a = auc_rank(y[ii], s[ii])
        if not np.isnan(a): out.append(a)
    out = np.array(out)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out))

def boot_paired_diff(y, s, spk, mask_a, mask_b, n=2000, seed=0):
    """paired: draw speaker list once per replicate, recompute both arms inside it"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); spk = np.asarray(spk)
    mask_a = np.asarray(mask_a, bool); mask_b = np.asarray(mask_b, bool)
    u = np.unique(spk); idx = {g: np.where(spk == g)[0] for g in u}
    da = []
    for _ in range(n):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        ia = ii[mask_a[ii]]; ib = ii[mask_b[ii]]
        if len(ia) == 0 or len(ib) == 0: continue
        aa = auc_rank(y[ia], s[ia]); bb = auc_rank(y[ib], s[ib])
        if np.isnan(aa) or np.isnan(bb): continue
        da.append(aa - bb)
    da = np.array(da)
    if len(da) == 0: return (float('nan'), float('nan'), 0)
    return (float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5)), len(da))
