"""p16twins: independent recompute of the POD3 3b readout-direction values (Qwen3-Omni, Pitt 468),
all clips and by arm (probe refit inside the arm), with the estimator exactly as shipped in POD3:
 X = final thinker stage at the first answer position (float64).
 direction d = unit( W[YES]^T wy - W[NO]^T wn ), wy / wn = mean softmax mass of each Yes / No id over the cell's clips,
   softmax over the full lm_head vocabulary; ids YES [7414,9454,9693,9834,14004], NO [902,2152,2308,2753,8996].
 probe: GroupKFold(5) on speaker ids (string groups, npz order), StandardScaler on the training fold,
   LogisticRegression(max_iter=2000, class_weight=balanced); probe direction = unit(mean over folds of coef/scale).
 cosine = probe direction . d ; random floor = mean |probe . r| over 2000 unit normals from a fresh default_rng(0).
 auc_d = AUC of X.d ; auc_without_d (as shipped, TRAINZ) = held-out (Xe.wp - mean(Xt.wp)) / std(Xt.wp),
   wp = fold coef minus its d component; also the canonical fold-z variant (standardise within the held-out fold).
 CIs: paired speaker bootstrap, 2000 draws, fresh default_rng(0), idx = rng.choice(N, N) over sorted unique speakers.
Written from scratch; does not import the POD3 scripts. usage: python3 twin_direction.py"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import json, sys, time
import numpy as np, scipy, sklearn
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

T0 = time.time()
YES = [7414, 9454, 9693, 9834, 14004]; NO = [902, 2152, 2308, 2753, 8996]
X_all = np.load("/workspace/in/pitt_ansfinal.npy").astype(np.float64)
m = np.load("/workspace/in/pitt_meta.npz", allow_pickle=True)
y_all = m["label"].astype(int); spk_all = m["spk"].astype(str); name_all = m["name"].astype(str)
W = np.load("/workspace/in/q3o_lm_head.npy").astype(np.float32)
arm_all = np.array([n.split("_", 1)[0] for n in name_all])
print("X", X_all.shape, "W", W.shape, {a: int((arm_all == a).sum()) for a in np.unique(arm_all)},
      "sklearn", sklearn.__version__, "numpy", np.__version__, "scipy", scipy.__version__, flush=True)

def auc(yv, s):
    yv = np.asarray(yv); n1 = int(yv.sum()); n0 = len(yv) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def direction(X):
    z = X @ W.T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    wy = p[:, YES].mean(axis=0); wn = p[:, NO].mean(axis=0)
    wy = wy / max(wy.sum(), 1e-12); wn = wn / max(wn.sum(), 1e-12)
    d = W[YES].T.astype(np.float64) @ wy - W[NO].T.astype(np.float64) @ wn
    return d / np.linalg.norm(d), float(p[:, YES].sum(1).mean()), float(p[:, NO].sum(1).mean())

def cell(X, y, spk, tag):
    d, my, mn = direction(X)
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=spk))
    oof = np.zeros(len(y)); oof_tz = np.zeros(len(y)); oof_fz = np.zeros(len(y)); coefs = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr]); Xt = sc.transform(X[tr]); Xe = sc.transform(X[te])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, y[tr])
        oof[te] = lr.predict_proba(Xe)[:, 1]
        coefs.append(lr.coef_[0] / sc.scale_)
        w = lr.coef_[0]; wp = w - (w @ d) / (d @ d) * d
        st, se = Xt @ wp, Xe @ wp
        oof_tz[te] = (se - st.mean()) / (st.std() + 1e-12)
        oof_fz[te] = (se - se.mean()) / (se.std() + 1e-12)
    pdir = np.mean(coefs, axis=0); pdir /= np.linalg.norm(pdir)
    cos = float(pdir @ d)
    g = np.random.default_rng(0); rnd = np.empty(2000)
    for k in range(2000):
        r = g.normal(size=len(d)); r /= np.linalg.norm(r); rnd[k] = abs(float(pdir @ r))
    s_d = X @ d
    point = {"probe": auc(y, oof), "d": auc(y, s_d), "wo_d_trainz": auc(y, oof_tz), "wo_d_foldz": auc(y, oof_fz)}
    us = np.unique(spk); rows = [np.flatnonzero(spk == s) for s in us]
    g = np.random.default_rng(0); B = {k: [] for k in point}
    for _ in range(2000):
        idx = g.choice(len(us), size=len(us), replace=True)
        ii = np.concatenate([rows[k] for k in idx]); yy = y[ii]
        if yy.min() == yy.max(): continue
        B["probe"].append(auc(yy, oof[ii])); B["d"].append(auc(yy, s_d[ii]))
        B["wo_d_trainz"].append(auc(yy, oof_tz[ii])); B["wo_d_foldz"].append(auc(yy, oof_fz[ii]))
    res = {"arm": tag, "n": int(len(y)), "n_speakers": int(len(us)), "n_positive": int(y.sum()),
           "yes_mass": my, "no_mass": mn, "cosine": cos,
           "rand_mean_abs": float(rnd.mean()), "rand_ci": [float(np.percentile(rnd, 2.5)), float(np.percentile(rnd, 97.5))]}
    for k, v in point.items():
        res["auc_" + k] = v
        res["auc_" + k + "_ci"] = [float(np.percentile(B[k], 2.5)), float(np.percentile(B[k], 97.5))]
    res["boot_usable"] = len(B["probe"])
    np.savez_compressed(f"/workspace/out/twin_dir_{tag}_perclip.npz", spk=spk, label=y,
                        oof=oof, s_d=s_d, oof_trainz=oof_tz, oof_foldz=oof_fz, d=d, probe_dir=pdir)
    print(tag, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    return res

out = [cell(X_all, y_all, spk_all, "all")]
for a in ("conflict", "agreement"):
    k = arm_all == a
    out.append(cell(X_all[k], y_all[k], spk_all[k], a))
json.dump({"cells": out, "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
           "python": sys.version.split()[0], "runtime_s": round(time.time() - T0, 1)},
          open("/workspace/out/twin_direction.json", "w"), indent=1)
print("DONE", round(time.time() - T0, 1), flush=True)
