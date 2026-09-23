"""PART17 T2: numerical noise floor for the cross-platform (pod-run) refits.
Same partition (stable), same saved layers, same data, but the feature columns permuted (a mathematically
identical problem whose floating-point summation order differs, as it does between BLAS builds) and,
separately, the features cast to float64 before the fit. The spread of these Mac-vs-Mac refits is the
size of difference that numerics alone produce. Compare with the pod-vs-Mac max abs diff."""
import json, os, numpy as np, pandas as pd
from eng import *
IN = "single_results.jsonl"; OUT = "noise_results.jsonl"
open(OUT, "w").close()
def outer_perm(X, y, fold, layers, perm=None, f64=False):
    oof = np.zeros(len(y))
    def one(k):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        A = X[:, layers[k]]
        if perm is not None: A = A[:, perm]
        if f64: A = A.astype(np.float64)
        return te, fitpred(A[tr], y[tr], A[te])
    for te, p in Parallel(n_jobs=5)(delayed(one)(k) for k in range(5)): oof[te] = p
    return oof
cache = {}
for line in open(IN):
    j = json.loads(line)
    if j["best_rule"] != "stable": continue
    if j["run"].startswith("q2a_probe2_podruns"): continue   # states differ from the pod's; not a numerics question
    if j["states"] not in cache:
        cache.clear(); cache[j["states"]] = np.load(j["states"], allow_pickle=True)
    z = cache[j["states"]]; y = z["label"].astype(int); X = z[j["stream"]].astype(np.float32)
    f = np.array(j["fold"]); lay = j["saved_layers"]
    saved = pd.read_csv(j["oof_file"])["p_probe"].values
    o0 = outer_perm(X, y, f, lay)
    rng = np.random.default_rng(0); perm = rng.permutation(X.shape[2])
    o1 = outer_perm(X, y, f, lay, perm=perm); o2 = outer_perm(X, y, f, lay, f64=True)
    rec = dict(run=j["run"], stream=j["stream"], pod_vs_mac=float(np.abs(o0 - saved).max()),
               mac_vs_mac_colperm=float(np.abs(o1 - o0).max()), mac_vs_mac_float64=float(np.abs(o2 - o0).max()),
               colperm_vs_saved=float(np.abs(o1 - saved).max()), f64_vs_saved=float(np.abs(o2 - saved).max()))
    open(OUT, "a").write(json.dumps(rec) + "\n")
    print(f"{j['run']:22s} {j['stream']:4s} pod-vs-mac {rec['pod_vs_mac']:.2e}  mac-vs-mac colperm {rec['mac_vs_mac_colperm']:.2e}  float64 {rec['mac_vs_mac_float64']:.2e}", flush=True)
print("NOISE DONE")
