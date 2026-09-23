"""PART17 T2: does lbfgs converge (n_iter_ < max_iter=2000) in the outer refits? Non-converged fits stop on the
iteration cap, so their final coefficients depend on the optimisation path and differ between platforms."""
import json, numpy as np
from eng import *
from joblib import Parallel, delayed
OUT = "conv_results.jsonl"; open(OUT, "w").close()
def niter(X, y, fold, lay, k):
    tr = np.where(fold != k)[0]; A = X[:, lay[k]]
    sc = StandardScaler().fit(A[tr]); lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(A[tr]), y[tr])
    return int(lr.n_iter_[0])
cache = {}
for line in open("single_results.jsonl"):
    j = json.loads(line)
    if j["best_rule"] != "stable" or j["run"].startswith("q2a_probe2_podruns"): continue
    if j["states"] not in cache: cache.clear(); cache[j["states"]] = np.load(j["states"], allow_pickle=True)
    z = cache[j["states"]]; y = z["label"].astype(int); X = z[j["stream"]].astype(np.float32); f = np.array(j["fold"])
    it = Parallel(n_jobs=5)(delayed(niter)(X, y, f, j["saved_layers"], k) for k in range(5))
    rec = dict(run=j["run"], stream=j["stream"], n_iter=it, capped=[i >= 2000 for i in it], maxabs=j["candidates"]["stable"]["maxabs"])
    open(OUT, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
print("CONV DONE")
