# Independent verifier core for m7m11. Own code: weighted Mann-Whitney AUC computed from speaker draw COUNTS
# (a speaker drawn k times gets weight k on each of its clips), not from concatenated rows.
import numpy as np

NB = 2000

class WAUC:
    """AUC of score s for labels y under per-row weights, ties get half credit. Sort once, reuse for every draw."""
    def __init__(self, y, s):
        y = np.asarray(y, int); s = np.asarray(s, float)
        order = np.argsort(s, kind="stable"); ss = s[order]
        self.gid = np.empty(len(s), int)
        self.gid[order] = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])])
        self.ng = int(self.gid.max()) + 1
        self.pos = (y == 1); self.neg = (y == 0)
    def __call__(self, w):
        wp = np.bincount(self.gid, weights=w * self.pos, minlength=self.ng)
        wn = np.bincount(self.gid, weights=w * self.neg, minlength=self.ng)
        P, N = wp.sum(), wn.sum()
        if P == 0 or N == 0:
            return np.nan
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (P * N))

def speaker_draw_weights(spk_of_row, spk_list, n_draws=NB, seed=0):
    """yield per-row weights for each draw: idx = rng.choice(N, N, replace=True) over spk_list (already sorted)"""
    rng = np.random.default_rng(seed)
    code = {s: i for i, s in enumerate(spk_list)}
    rc = np.array([code[s] for s in spk_of_row])
    N = len(spk_list)
    for _ in range(n_draws):
        idx = rng.choice(N, size=N, replace=True)
        cnt = np.bincount(idx, minlength=N).astype(float)
        yield cnt[rc], idx

def pct(a):
    a = np.asarray(a, float)
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))
