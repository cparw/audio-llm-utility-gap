"""p16twins: independent refit of PART 16 M5 (Qwen2-Audio readout-direction test, Pitt 468) with the estimator AS SHIPPED
(the first-pass estimator the fragment rows carry), to twin the 12 M5 rows the audit found disagreeing with the M5 verifier.
 X = ans[:, -1, :] float64 (final LM stage at the first answer position), arms from the conflict manifest (set column).
 probe: GroupKFold(5) on speaker strings in npz order, StandardScaler on train fold, LogisticRegression(max_iter=2000, balanced);
   probe direction = unit(mean over folds of coef/scale). Random floor: mean |w.r| over 2000 unit normals, fresh default_rng(0).
 d = mass-weighted Yes minus No lm_head rows (softmax over the full vocabulary), unit length.
 without_d (coef projection, TRAIN-fold standardisation as shipped) and Xproj (project d out of X, rerun the probe).
 cells: all; conflict_refit / agreement_refit (whole pipeline inside the arm, arm's own d); *_restricted = all-cell scores on the arm.
Written from scratch. usage: python3 twin_m5.py STATES.npz MANIFEST.csv LMHEAD.npy OUT.json"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, json, time
import numpy as np, pandas as pd, scipy, sklearn
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
np.seterr(all="ignore")
T0 = time.time()
NPZ, MAN, LMH, OUTJ = sys.argv[1:5]
YES = [7414, 9454, 9693, 9834, 14004, 14080]; NO = [902, 2152, 2308, 2753, 5664, 8996]   # from the M5 run log
z = np.load(NPZ, allow_pickle=True)
X = z["ans"][:, -1, :].astype(np.float64); y = z["label"].astype(int); spk = z["spk"].astype(str); name = z["name"].astype(str)
man = pd.read_csv(MAN, usecols=["segment_path", "set", "label"])
arm_of = {os.path.basename(p): s for p, s in zip(man.segment_path, man.set)}
arm = np.array([arm_of[n] for n in name])
if LMH.endswith(".safetensors"):
    from safetensors import safe_open
    with safe_open(LMH, "pt") as f:
        W = f.get_tensor("language_model.lm_head.weight").float().numpy()
else:
    W = np.load(LMH, mmap_mode="r")
print("X", X.shape, "W", W.shape, W.dtype, {a: int((arm == a).sum()) for a in np.unique(arm)},
      "sklearn", sklearn.__version__, "numpy", np.__version__, flush=True)

def auc(yv, s):
    yv = np.asarray(yv); n1 = int(yv.sum()); n0 = len(yv) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# softmax mass of the Yes / No ids over the full vocabulary, per clip (log-sum-exp in blocks)
lse = np.full(len(X), -np.inf)
for a0 in range(0, W.shape[0], 8192):
    blk = X @ np.asarray(W[a0:a0 + 8192], dtype=np.float64).T
    mx = blk.max(1); lse = np.logaddexp(lse, mx + np.log(np.exp(blk - mx[:, None]).sum(1)))
WY = np.asarray(W[YES], dtype=np.float64); WN = np.asarray(W[NO], dtype=np.float64)
PY = np.exp(X @ WY.T - lse[:, None]); PN = np.exp(X @ WN.T - lse[:, None])
def dvec(mask):
    wy = PY[mask].mean(0); wn = PN[mask].mean(0); wy /= max(wy.sum(), 1e-12); wn /= max(wn.sum(), 1e-12)
    d = WY.T @ wy - WN.T @ wn; return d / np.linalg.norm(d)

def probe(Xm, ym, sm):
    oof = np.zeros(len(ym)); ws = []
    for tr, te in GroupKFold(n_splits=5).split(Xm, ym, groups=sm):
        sc = StandardScaler().fit(Xm[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xm[tr]), ym[tr])
        oof[te] = lr.predict_proba(sc.transform(Xm[te]))[:, 1]; ws.append(lr.coef_[0] / sc.scale_)
    w = np.mean(ws, 0); return oof, w / np.linalg.norm(w)

def wo_d(Xm, ym, sm, d):
    o = np.zeros(len(ym))
    for tr, te in GroupKFold(n_splits=5).split(Xm, ym, groups=sm):
        sc = StandardScaler().fit(Xm[tr]); Xt = sc.transform(Xm[tr]); Xe = sc.transform(Xm[te])
        c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, ym[tr]).coef_[0]
        wp = c - (c @ d) / (d @ d) * d; st = Xt @ wp
        o[te] = (Xe @ wp - st.mean()) / (st.std() + 1e-12)
    return o

def floor(w):
    g = np.random.default_rng(0); v = np.empty(2000)
    for k in range(2000):
        r = g.normal(size=len(w)); r /= np.linalg.norm(r); v[k] = abs(float(w @ r))
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def ci(yv, s, sm):
    us = np.unique(sm); rows = [np.flatnonzero(sm == u) for u in us]; g = np.random.default_rng(0); v = []
    for _ in range(2000):
        ii = np.concatenate([rows[k] for k in g.choice(len(us), size=len(us), replace=True)]); a = auc(yv[ii], s[ii])
        if a == a: v.append(a)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

res = {}; scores = {}
for tag, m in (("all", np.ones(len(y), bool)), ("conflict_refit", arm == "conflict"), ("agreement_refit", arm == "agreement")):
    Xm, ym, sm = X[m], y[m], spk[m]; d = dvec(m)
    oof, w = probe(Xm, ym, sm); o_wo = wo_d(Xm, ym, sm, d); o_x = probe(Xm - np.outer(Xm @ d, d), ym, sm)[0]
    fm, flo, fhi = floor(w)
    scores[tag] = dict(oof=oof, wo=o_wo, xp=o_x, mask=m)
    res[tag] = {"n": int(m.sum()), "cos": float(w @ d), "randfloor": fm, "randfloor_ci": [flo, fhi],
                "auc_probe": auc(ym, oof), "auc_without_d": auc(ym, o_wo), "auc_without_d_ci": list(ci(ym, o_wo, sm)),
                "auc_without_d_Xproj": auc(ym, o_x), "auc_without_d_Xproj_ci": list(ci(ym, o_x, sm))}
    print(tag, json.dumps(res[tag]), flush=True)
S = scores["all"]
for a in ("conflict", "agreement"):
    m = arm == a
    res[a + "_restricted"] = {"auc_without_d": auc(y[m], S["wo"][m]), "auc_without_d_ci": list(ci(y[m], S["wo"][m], spk[m])),
                              "auc_without_d_Xproj": auc(y[m], S["xp"][m]), "auc_without_d_Xproj_ci": list(ci(y[m], S["xp"][m], spk[m]))}
    print(a + "_restricted", json.dumps(res[a + "_restricted"]), flush=True)
# paired conflict minus agreement on the all-cell without_d score
us = np.unique(spk); rows = [np.flatnonzero(spk == u) for u in us]; g = np.random.default_rng(0); v = []
mc, ma = arm == "conflict", arm == "agreement"
for _ in range(2000):
    ii = np.concatenate([rows[k] for k in g.choice(len(us), size=len(us), replace=True)])
    c = ii[mc[ii]]; a_ = ii[ma[ii]]
    x1, x2 = auc(y[c], S["wo"][c]), auc(y[a_], S["wo"][a_])
    if x1 == x1 and x2 == x2: v.append(x1 - x2)
res["paired_without_d"] = {"diff": auc(y[mc], S["wo"][mc]) - auc(y[ma], S["wo"][ma]),
                           "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "draws": len(v)}
print("paired", json.dumps(res["paired_without_d"]), flush=True)
res["env"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}
res["runtime_s"] = round(time.time() - T0, 1)
json.dump(res, open(OUTJ, "w"), indent=1)
np.savez_compressed(OUTJ.replace(".json", "_scores.npz"), name=name, arm=arm,
                    **{f"{t}_{k}": scores[t][k] for t in scores for k in ("oof", "wo", "xp")})
print("DONE", res["runtime_s"], flush=True)
