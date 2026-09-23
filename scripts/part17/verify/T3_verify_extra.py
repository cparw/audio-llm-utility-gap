"""T3 verifier extra: (1) the exact first-pass removal (d removed from the STANDARDISED-space coefficient, training-fold z)
to explain 0.8327/0.6326/0.9070; (2) paired interval for the 227-participant regroup. Own AUC + bootstrap."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import csv, json, warnings, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
def auc(y, s):
    p, n = s[y == 1], s[y == 0]; d = p[:, None] - n[None, :]
    return (np.count_nonzero(d > 0) + 0.5 * np.count_nonzero(d == 0)) / d.size
S = np.load("T3_verify_scores.npz", allow_pickle=True)
z = np.load("<local data dir>/paper work/paper1_local_runs/probe2/pitt_states.npz", allow_pickle=True)
X = z["ans"][:, -1, :].astype(np.float64); y = z["label"].astype(int); spk = z["spk"].astype(str)
d = S["d"]; isC = S["arm"] == "conflict"
num = np.array(["".join(c for c in s if c.isdigit()) for s in spk])
fp = np.zeros(len(y)); rg = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, groups=spk):
    ss = StandardScaler().fit(X[tr]); Zt, Ze = ss.transform(X[tr]), ss.transform(X[te])
    c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Zt, y[tr]).coef_[0]
    wp = c - (c @ d) / (d @ d) * d
    fp[te] = (Ze @ wp - (Zt @ wp).mean()) / (Zt @ wp).std()
print("first-pass exact (std-space removal, training-fold z): all %.4f conflict %.4f agreement %.4f  (all full %.6f)"
      % (auc(y, fp), auc(y[isC], fp[isC]), auc(y[~isC], fp[~isC]), auc(y, fp)))
for tr, te in GroupKFold(5).split(X, y, groups=num):
    ss = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(ss.transform(X[tr]), y[tr])
    wf = lr.coef_[0] / ss.scale_; b = X[te] @ (wf - (wf @ d) / (d @ d) * d); rg[te] = (b - b.mean()) / b.std()
pt = auc(y[isC], rg[isC]) - auc(y[~isC], rg[~isC])
for unit_name, unit in (("227 participants", num), ("228 npz speakers", spk)):
    u = np.unique(unit); rows = {s: np.nonzero(unit == s)[0] for s in u}; rng = np.random.default_rng(0); v = []
    for _ in range(2000):
        r = np.concatenate([rows[s] for s in rng.choice(u, size=len(u), replace=True)]); rc, ra = r[isC[r]], r[~isC[r]]
        v.append(auc(y[rc], rg[rc]) - auc(y[ra], rg[ra]))
    print("regroup paired perp %.4f [%.4f, %.4f] resampling %s" % (pt, np.percentile(v, 2.5), np.percentile(v, 97.5), unit_name))
