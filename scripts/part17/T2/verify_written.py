"""PART17 T2 final check, separate code path: read each WRITTEN fold csv from release/folds (not the in-memory folds),
join it to the saved run by clip id, refit the saved probe at the saved (or re-derived) layers with a Pipeline written
here, and report the max abs difference against the saved out-of-fold predictions. Also checks speaker disjointness."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(v, "1")
import json, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
HERE = os.path.dirname(os.path.abspath(__file__)); R = "<local data dir>/release"; FD = f"{R}/folds"
M = {f["file"]: f for f in json.load(open(f"{HERE}/files_manifest.json"))}
SINGLE = {(j["run"], j["stream"]): j for j in map(json.loads, open(f"{HERE}/single_results.jsonl"))}
EGE = {j["run"]: j for j in map(json.loads, open(f"{HERE}/egemaps_results.jsonl"))}
REPC = {(j["run"], j["stream"]): j for j in map(json.loads, open(f"{HERE}/repeats_perclip_results.jsonl"))}
OUT = f"{HERE}/written_check.jsonl"; open(OUT, "w").close()

def refit(X, y, fold, layers, egemaps=False):
    def one(k):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        A = X if egemaps else X[:, layers[k]]
        steps = [SimpleImputer(strategy="median")] if egemaps else []
        p = make_pipeline(*steps, StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(A[tr], y[tr])
        return te, p.predict_proba(A[te])[:, 1]
    o = np.zeros(len(y))
    for te, p in Parallel(n_jobs=5)(delayed(one)(k) for k in range(5)): o[te] = p
    return o

cache = {}
def states(p):
    if p not in cache: cache.clear(); cache[p] = np.load(p, allow_pickle=True)
    return cache[p]
for fname in sorted(os.listdir(FD)):
    if not fname.endswith(".csv"): continue
    F = pd.read_csv(f"{FD}/{fname}", dtype={"clip_id": str, "speaker_id": str}); m = M[fname]
    disjoint = bool((F.groupby("speaker_id")["fold"].nunique() == 1).all())
    fmap = dict(zip(F["clip_id"], F["fold"]))
    checks = []
    if m["family"] in ("single_mac", "single_pod"):
        for e in m["evidence"]:
            if e["run"].startswith("egemaps"):
                j = EGE[e["run"]]; z = states(j["states"]); X = np.asarray(z["X"], np.float64); X[~np.isfinite(X)] = np.nan
                y = z["y"].astype(int); fold = np.array([fmap[str(c)] for c in z["clips"]])
                o = refit(X, y, fold, None, egemaps=True); s = pd.read_csv(j["oof_file"])["p_egemaps"].values
                checks.append(dict(run=e["run"], stream="egemaps", maxabs=float(np.abs(o - s).max())))
                continue
            for st in e["streams"]:
                j = SINGLE[(e["run"], st["stream"])]
                if j["saved_layers"] is None: lay = j["rederived_layers"]
                else: lay = j["saved_layers"]
                z = states(j["states"]); y = z["label"].astype(int); nm = z["name"].astype(str)
                fold = np.array([fmap[c] for c in nm]); X = z[st["stream"]].astype(np.float32)
                o = refit(X, y, fold, lay); s = pd.read_csv(j["oof_file"])["p_probe"].values
                checks.append(dict(run=e["run"], stream=st["stream"], maxabs=float(np.abs(o - s).max())))
    elif m["family"] in ("seed_pod", "seed_mac", "seed_pod_pod3"):
        for p in m.get("perclip") or []:
            j = REPC[(p["run"], p["stream"])]; rr = [x for x in j["per_repeat"] if x["seed"] == m["seed"]][0]
            lay = rr["saved_layers"] if rr["saved_layers"] is not None else rr["rederived_layers"]
            z = states(j["states"]); y = z["label"].astype(int); nm = z["name"].astype(str)
            fold = np.array([fmap[c] for c in nm]); X = z[p["stream"]].astype(np.float32)
            o = refit(X, y, fold, lay)
            if j["oof_file"].endswith(".npz"): s = np.load(j["oof_file"], allow_pickle=True)["oof"][m["seed"]]
            else: s = pd.read_csv(j["oof_file"])[f"p_probe_rep{m['seed']}"].values
            checks.append(dict(run=p["run"], stream=p["stream"], maxabs=float(np.abs(o - s).max())))
    rec = dict(file=fname, status=m["status"], speakers_disjoint=disjoint, n=int(len(F)), checks=checks,
               max_over_checks=(max(c["maxabs"] for c in checks) if checks else None))
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(f"{fname:52s} {m['status']:10s} disjoint={disjoint} n={len(F)} checks={len(checks)} max={rec['max_over_checks']}", flush=True)
print("WRITTEN CHECK DONE")
