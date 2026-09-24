"""PART 25 track C. Refit the Table-style five-repeat nested probe for one (model, dataset, stream) cell and SAVE
the per-clip out-of-fold score vector of every repeat, the fold ids and the chosen layers.
Same code path as release/scripts/nested_repeats_all.py (the script that wrote every *_nested_repeats.json):
speakers renumbered with numpy default_rng(seed).permutation(unique speakers), seeds 0..4,
outer GroupKFold(5), inner GroupKFold(4) on the training speakers picks the layer by mean inner AUC,
StandardScaler + LogisticRegression(max_iter=2000, class_weight="balanced"), refit on the outer training fold.
usage: python3 rep5_refit.py TAG STREAM     reads in/TAG_meta.npz (label, spk, name) and in/TAG_STREAM.npy
writes out/TAG_STREAM_rep5_oof.csv (clip, speaker, label, p_seed0..4, fold_seed0..4) and out/TAG_STREAM_rep5.json"""
import os; os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np, sys, json, time, warnings; warnings.filterwarnings("ignore")
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import sklearn, scipy
TAG, ST = sys.argv[1:3]; R = 5
m = np.load(f"in/{TAG}_meta.npz", allow_pickle=True); y = m["label"].astype(int); spk = m["spk"].astype(str); name = m["name"].astype(str)
X = np.load(f"in/{TAG}_{ST}.npy").astype(np.float32); nl = X.shape[1]
NJ = int(os.environ.get("NJ", max(2, (os.cpu_count() or 8) - 4)))
def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
def inner(X, tr, l, g):
    s = 0.0
    for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], groups=g[tr]):
        p = fitpred(X[tr][itr, l], y[tr][itr], X[tr][ite, l])
        s += roc_auc_score(y[tr][ite], p) if len(set(y[tr][ite])) > 1 else 0.5
    return s / 4
t0 = time.time(); res = {"tag": TAG, "stream": ST, "per_repeat": [], "chosen_layers": {}, "n": int(len(y)), "n_points": int(nl), "n_speakers": int(len(np.unique(spk)))}
df = pd.DataFrame({"clip": name, "speaker": spk, "label": y})
for seed in range(R):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
    oof = np.zeros(len(y)); fold = np.full(len(y), -1); layers = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
        best = 0 if nl == 1 else int(np.argmax(Parallel(n_jobs=min(nl, NJ))(delayed(inner)(X, tr, l, g) for l in range(nl))))
        oof[te] = fitpred(X[tr][:, best], y[tr], X[te][:, best]); fold[te] = k; layers.append(best)
    a = float(roc_auc_score(y, oof)); res["per_repeat"].append(a); res["chosen_layers"][f"seed{seed}"] = layers
    df[f"p_seed{seed}"] = oof; df[f"fold_seed{seed}"] = fold
    print(f"{TAG} {ST} seed {seed}: AUC {a:.6f} layers {layers}  {time.time()-t0:.0f}s", flush=True)
res["mean5"] = float(np.mean(res["per_repeat"]))
res["per_repeat_4dp"] = [round(v, 4) for v in res["per_repeat"]]; res["mean5_4dp"] = round(res["mean5"], 4)
res["versions"] = {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]}
try:
    res["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    res["cpu_avx512f"] = "avx512f" in open("/proc/cpuinfo").read()
except Exception: pass
res["OPENBLAS_CORETYPE"] = os.environ.get("OPENBLAS_CORETYPE", "")
res["runtime_s"] = round(time.time()-t0, 1)
df.to_csv(f"out/{TAG}_{ST}_rep5_oof.csv", index=False, float_format="%.10g")
json.dump(res, open(f"out/{TAG}_{ST}_rep5.json", "w"), indent=1)
print(f"{TAG} {ST} mean5 {res['mean5']:.6f} per_repeat {res['per_repeat_4dp']}", flush=True)
