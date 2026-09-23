# Independent verifier refit (m7m11 VAL_adress2020_pod4b). Protocol from release/scripts/nested_repeats_all.py:
# per seed r in 0..4: groups = position of each speaker in default_rng(r).permutation(sorted unique speakers);
# outer GroupKFold(5); per outer fold, every layer scored by mean AUC over inner GroupKFold(4) on the outer-train rows
# (0.5 for a one-class inner test fold); first argmax layer; Pipeline(StandardScaler, LogisticRegression(max_iter=2000,
# class_weight=balanced)) refit on outer train, predict outer test. Writes per-clip OOF per seed and fold ids.
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, time, platform, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sklearn, scipy
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

D = np.load(sys.argv[1], allow_pickle=True)
X = D["X"].astype(np.float32); Y = D["y"].astype(int); S = D["s"].astype(str); C = D["c"].astype(str)
L = X.shape[1]

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))

def layer_score(args):
    rows_tr, grp, lay = args
    Xl = X[rows_tr, lay]; yl = Y[rows_tr]; gl = grp[rows_tr]
    tot = 0.0
    for a, b in GroupKFold(n_splits=4).split(Xl, yl, groups=gl):
        if np.unique(yl[b]).size < 2:
            tot += 0.5; continue
        pr = model().fit(Xl[a], yl[a]).predict_proba(Xl[b])[:, 1]
        tot += roc_auc_score(yl[b], pr)
    return tot / 4.0

if __name__ == "__main__":
    t0 = time.time(); out = pd.DataFrame({"clip": C, "speaker": S, "label": Y}); meta = {"per_seed_auc": [], "layers": {}}
    with ProcessPoolExecutor(max_workers=28) as ex:
        for r in range(5):
            order = np.random.default_rng(r).permutation(np.unique(S))
            pos = {sp: i for i, sp in enumerate(order)}
            grp = np.array([pos[sp] for sp in S])
            pred = np.full(len(Y), np.nan); fid = np.full(len(Y), -1); chosen = []
            for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, Y, groups=grp)):
                sc = list(ex.map(layer_score, [(tr, grp, l) for l in range(L)]))
                best = int(np.argmax(sc)); chosen.append(best)
                pred[te] = model().fit(X[tr, best], Y[tr]).predict_proba(X[te, best])[:, 1]; fid[te] = k
            assert not np.isnan(pred).any()
            out[f"v_p{r}"] = pred; out[f"v_fold{r}"] = fid
            a = float(roc_auc_score(Y, pred)); meta["per_seed_auc"].append(a); meta["layers"][str(r)] = chosen
            print(f"seed {r} auc {a:.6f} layers {chosen} t={time.time()-t0:.0f}s", flush=True)
    meta["mean5"] = float(np.mean(meta["per_seed_auc"]))
    meta["env"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(), "machine": platform.machine()}
    out.to_csv("out/vrefit_adress2020_pod4b_oof.csv", index=False)
    json.dump(meta, open("out/vrefit_adress2020_pod4b.json", "w"), indent=1)
    print("mean5", meta["mean5"], flush=True)
