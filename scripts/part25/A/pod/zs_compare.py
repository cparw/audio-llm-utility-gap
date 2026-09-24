"""compare this pod's zero-shot p_yes with the saved omni_final file, clip by clip (exact float equality and abs diff)"""
import csv, sys, json, numpy as np
new, old, out = sys.argv[1:4]
a = {r["clip"]: r for r in csv.DictReader(open(new))}; b = {r["clip"]: r for r in csv.DictReader(open(old))}
assert set(a) == set(b), "clip sets differ"
d = np.array([abs(float(a[c]["p_yes"]) - float(b[c]["p_yes"])) for c in b])
m = np.array([abs(float(a[c]["mass"]) - float(b[c]["mass"])) for c in b])
eq = int(sum(float(a[c]["p_yes"]) == float(b[c]["p_yes"]) for c in b))
res = {"n": len(b), "p_yes_exact_equal": eq, "p_yes_max_abs_diff": float(d.max()), "p_yes_median_abs_diff": float(np.median(d)),
       "n_below_1e-9": int((d < 1e-9).sum()), "n_below_1e-4": int((d < 1e-4).sum()), "mass_max_abs_diff": float(m.max()),
       "labels_equal": all(a[c]["label"] == b[c]["label"] for c in b), "speakers_equal": all(a[c]["speaker"] == b[c]["speaker"] for c in b)}
json.dump(res, open(out, "w"), indent=1); print("ZS_COMPARE", json.dumps(res), flush=True)
