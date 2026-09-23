"""PART17 T2: eGeMAPS single-split probe (overnight/scripts/part5a_egemaps.py probe()) refit under both tie rules."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v,"1")
import json, hashlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from gkf import gkf_folds
from eng import rank_auc
R = "<local data dir>/release/overnight"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "egemaps_results.jsonl")
def sha1mb(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read(1 << 20)); return h.hexdigest()
open(OUT, "w").close()
for ds in ["pitt", "adresso", "adress2020", "pcgita", "neurovoz", "kcl", "edaic"]:
    npz = f"{R}/egemaps_{ds}.npz"; oofp = f"{R}/perclip/egemaps_{ds}_oof.csv"
    z = np.load(npz, allow_pickle=True); X = np.asarray(z["X"], dtype=np.float64); y = z["y"].astype(int); g = np.asarray(z["groups"])
    X[~np.isfinite(X)] = np.nan
    s = pd.read_csv(oofp, dtype={"speaker": str}); saved = s["p_egemaps"].values
    rec = dict(run=f"egemaps_{ds}", model="eGeMAPSv02 88-feature baseline", dataset=ds, stream="egemaps", states=npz, oof_file=oofp,
               script=f"{R}/scripts/part5a_egemaps.py", n=int(len(y)), n_spk=int(len(set(map(str, g)))), sha_states=sha1mb(npz), sha_oof=sha1mb(oofp),
               align_names=bool((s["clip"].astype(str).values == np.array([str(c) for c in z["clips"]])).all()),
               align_spk=bool((s["speaker"].astype(str).values == np.array([str(v) for v in g])).all()),
               align_label=bool((s["label"].values == y).all()), groups_dtype=str(g.dtype), saved_auc_rank=rank_auc(y, saved))
    cands = {}
    for kind in ("mac", "stable"):
        f = gkf_folds(g, kind); o = np.zeros(len(y))
        for k in range(5):
            tr = np.where(f != k)[0]; te = np.where(f == k)[0]
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[tr], y[tr])
            o[te] = pipe.predict_proba(X[te])[:, 1]
        d = np.abs(o - saved)
        cands[kind] = dict(maxabs=float(d.max()), medabs=float(np.median(d)), pearson=float(np.corrcoef(o, saved)[0, 1]), auc=rank_auc(y, o))
        if kind == "mac": fmac = f
        else: fst = f
    best = min(cands, key=lambda k: cands[k]["maxabs"])
    rec.update(candidates=cands, best_rule=best, fold=(fmac if best == "mac" else fst).tolist(), clip=[str(c) for c in z["clips"]], spk=[str(v) for v in g])
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(ds, "groups dtype", g.dtype, "align", rec["align_names"], rec["align_spk"], rec["align_label"], "best", best, {k: f"{v['maxabs']:.3e}" for k, v in cands.items()}, f"auc {rec['saved_auc_rank']:.4f}", flush=True)
