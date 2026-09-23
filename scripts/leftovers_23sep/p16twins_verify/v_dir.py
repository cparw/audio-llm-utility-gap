"""Verifier (p16twins): POD3 3b readout-direction test, all clips and each arm, written from scratch.
Direction d: per cell, softmax over the full thinker vocabulary of X W^T (float64), average probability of each Yes id
and each No id over the cell's clips, normalise each set to sum 1, d = unit(W_yes^T a_yes - W_no^T a_no).
Probe: GroupKFold(5) by speaker, StandardScaler on the training fold, LogisticRegression(max_iter=2000,
class_weight='balanced'); probe direction = unit(mean over folds of coef/scale); cosine = probe dir . d.
Without-d (as shipped): per fold, remove the d component from the standardised-space coef, score held-out
clips with it and standardise by the TRAINING fold projection mean and std.
CIs: 2000 paired speaker draws, fresh default_rng(0) per cell, idx = rng.choice(N, N, replace=True)."""
import os, sys, json, time, platform
import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
t0 = time.time()
z = np.load("/workspace/in/pitt_ans_last.npz", allow_pickle=True)
X0 = z["X"].astype(np.float64); y0 = z["label"].astype(int); s0 = z["spk"].astype(str); arm0 = z["arm"].astype(str)
W = np.load("/workspace/in/lmh.npy")
YES = [7414, 9454, 9693, 9834, 14004]; NO = [902, 2152, 2308, 2753, 8996]

def auc_pairs(lab, sc):
    lab = np.asarray(lab); sc = np.asarray(sc, float)
    p = sc[lab == 1][:, None]; q = sc[lab == 0][None, :]
    return float(((p > q).sum() + 0.5 * (p == q).sum()) / (p.size * q.size))

def cell(X, y, spk):
    lg = X @ W.T.astype(np.float64)
    lp = lg - logsumexp(lg, axis=1, keepdims=True)
    ay = np.exp(lp[:, YES]).mean(0); an = np.exp(lp[:, NO]).mean(0)
    ay = ay / ay.sum(); an = an / an.sum()
    d = W[YES].astype(np.float64).T @ ay - W[NO].astype(np.float64).T @ an
    d = d / np.sqrt((d * d).sum())
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=spk))
    oof = np.zeros(len(y)); wo = np.zeros(len(y)); dirs = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xt = sc.transform(X[tr]); Xe = sc.transform(X[te])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xt, y[tr])
        oof[te] = m.predict_proba(Xe)[:, 1]
        c = m.coef_.ravel()
        dirs.append(c / sc.scale_)
        cp = c - (c @ d) / (d @ d) * d
        pt = Xt @ cp
        wo[te] = (Xe @ cp - pt.mean()) / (pt.std() + 1e-12)
    w = np.mean(dirs, axis=0); w = w / np.linalg.norm(w)
    cos = float(w @ d)
    sd = X @ d
    u = np.unique(spk); rows = {k: np.flatnonzero(spk == k) for k in u}
    rng = np.random.default_rng(0); B = []
    for _ in range(2000):
        idx = rng.choice(len(u), size=len(u), replace=True)
        ii = np.concatenate([rows[u[k]] for k in idx])
        if y[ii].min() == y[ii].max():
            continue
        B.append((auc_pairs(y[ii], oof[ii]), auc_pairs(y[ii], sd[ii]), auc_pairs(y[ii], wo[ii])))
    B = np.array(B)
    ci = lambda j: [float(v) for v in np.percentile(B[:, j], [2.5, 97.5])]
    return {"n": int(len(y)), "n_speakers": int(len(u)), "n_pos": int(y.sum()), "cosine": cos,
            "auc_probe": auc_pairs(y, oof), "auc_probe_ci": ci(0), "auc_d": auc_pairs(y, sd), "auc_d_ci": ci(1),
            "auc_wo_d_trainz": auc_pairs(y, wo), "auc_wo_d_trainz_ci": ci(2), "boot_usable": int(len(B)),
            "fold_test_sizes": [int(len(te)) for tr, te in folds]}, oof, wo, sd

out = {}
for a in ("all", "conflict", "agreement"):
    m = np.ones(len(y0), bool) if a == "all" else (arm0 == a)
    r, oof, wo, sd = cell(X0[m], y0[m], s0[m])
    out[a] = r
    np.savez_compressed(f"/workspace/out/v_dir_{a}_perclip.npz", oof=oof, wo=wo, sd=sd, y=y0[m], spk=s0[m])
    print(a, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("_ci")}),
          "ci", [round(x, 4) for x in r["auc_probe_ci"]], [round(x, 4) for x in r["auc_wo_d_trainz_ci"]], flush=True)
import sklearn, scipy
out["env"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
              "python": platform.python_version(), "seconds": round(time.time() - t0, 1)}
json.dump(out, open("/workspace/out/v_dir.json", "w"), indent=1)
print("DONE", round(time.time() - t0, 1), flush=True)
