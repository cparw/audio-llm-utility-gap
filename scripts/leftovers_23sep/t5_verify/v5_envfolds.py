"""Environment record plus fold check (own code). usage: v5_envfolds.py OUTJSON NPZ:DS [NPZ:DS ...]
For each dataset: sklearn GroupKFold(5) as installed, my stable reimplementation, my default-argsort reimplementation,
compared clip by clip with the released pod and Mac fold files (folds/<ds>_groupkfold5_{pod,mac}.csv)."""
import sys, os, json, csv, inspect, platform, numpy as np, sklearn, scipy
from sklearn.model_selection import GroupKFold
out = sys.argv[1]; res = {}
try:
    from threadpoolctl import threadpool_info; res["threadpool"] = threadpool_info()
except Exception as e: res["threadpool"] = str(e)
cpu = ""
try:
    for ln in open("/proc/cpuinfo"):
        if ln.startswith("model name"): cpu = ln.split(":", 1)[1].strip(); break
    res["avx512f"] = "avx512f" in open("/proc/cpuinfo").read()
except Exception:
    import subprocess; cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
res.update(cpu=cpu, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
src = inspect.getsource(GroupKFold._iter_test_indices)
res["groupkfold_argsort_lines"] = [l.strip() for l in src.splitlines() if "argsort" in l]
def mine(groups, kind):
    uniq, inv = np.unique(groups, return_inverse=True); cnt = np.bincount(inv)
    order = np.argsort(cnt, kind=kind)[::-1]; load = np.zeros(5); g2f = np.zeros(len(uniq), dtype=np.int64)
    for g in order:
        f = int(np.argmin(load)); load[f] += cnt[g]; g2f[g] = f
    return g2f[inv], cnt
FD = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "..", "folds")
for arg in sys.argv[2:]:
    npz, ds = arg.split(":"); z = np.load(npz); spk = z["spk"].astype(str); names = [str(v) for v in z["name"]]; n = len(names)
    sk = np.full(n, -1)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((n, 1)), z["label"], groups=spk)): sk[te] = f
    ms, cnt = mine(spk, "stable"); md, _ = mine(spk, "quicksort")
    r = dict(n=n, n_speakers=int(len(cnt)), argsort_default_equals_stable=bool(np.array_equal(np.argsort(cnt), np.argsort(cnt, kind="stable"))),
             sklearn_eq_mine_stable=int((sk == ms).sum()), sklearn_eq_mine_default=int((sk == md).sum()))
    for t in ("pod", "mac"):
        p = f"{FD}/{ds}_groupkfold5_{t}.csv"
        if not os.path.exists(p): r[f"{t}_file"] = "missing"; continue
        m = {row["clip_id"]: int(row["fold"]) for row in csv.DictReader(open(p))}
        ff = np.array([m.get(nm, -9) for nm in names])
        r[f"{t}_file"] = dict(n_file=len(m), missing=int((ff == -9).sum()), sklearn_here_eq=int((sk == ff).sum()),
                              mine_stable_eq=int((ms == ff).sum()), mine_default_eq=int((md == ff).sum()))
    res[ds] = r
json.dump(res, open(out, "w"), indent=1, default=str); print(json.dumps({k: v for k, v in res.items() if k != "threadpool"}, default=str))
