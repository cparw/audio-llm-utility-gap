"""PART20 POD5: load the SAVED outer folds (never recomputed) and derive the inner GroupKFold(4) splits with the
tie rule each file was made with (mac = numpy default argsort on this Mac, numpy 2.2.6 arm64; stable = kind='stable').
Checks each saved outer file equals the rule it claims before deriving anything."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "<local data dir>/release/edaic_rerun/part17/T2")
from gkf import gkf_folds, sk_folds
F = "folds"
def perm_codes(spk, seed):
    rng = np.random.default_rng(seed); u = np.unique(spk)
    perm = {s: i for i, s in enumerate(rng.permutation(u))}
    return np.array([perm[s] for s in spk])
base = pd.read_csv(f"{F}/pitt_groupkfold5_pod.csv"); clips = base.clip_id.astype(str).values; spk = base.speaker_id.astype(str).values
out = {"clip_id": clips.tolist(), "speaker_id": spk.tolist(), "sets": {}}
def add(name, fname, kind, seed):
    d = pd.read_csv(f"{F}/{fname}").set_index("clip_id").loc[clips]
    assert (d.speaker_id.astype(str).values == spk).all()
    outer = d.fold.values.astype(int)
    g = perm_codes(spk, seed) if seed is not None else spk
    chk = gkf_folds(g, kind)
    ok = bool((chk == outer).all())
    if kind == "mac": ok = ok and bool((sk_folds(g) == outer).all())
    inner = {}
    for k in range(5):
        tr = np.where(outer != k)[0]
        f4 = gkf_folds(g[tr], kind, n_splits=4)
        if kind == "mac": assert (f4 == sk_folds(g[tr], 4)).all()
        arr = np.full(len(clips), -1); arr[tr] = f4; inner[k] = arr.tolist()
    out["sets"][name] = {"file": f"{F}/{fname}", "kind": kind, "seed": seed, "outer": outer.tolist(), "inner": inner, "outer_matches_rule": ok}
    print(name, fname, kind, "outer file == rule:", ok, "fold sizes", np.bincount(outer).tolist(), flush=True)
for s in range(5):
    add(f"mac_seed{s}", f"pitt_groupkfold5_mac_seed{s}.csv", "mac", s)
    add(f"pod_seed{s}", f"pitt_groupkfold5_pod_seed{s}_UNVERIFIED.csv", "stable", s)
add("pod_single", "pitt_groupkfold5_pod.csv", "stable", None)
add("mac_single", "pitt_groupkfold5_mac.csv", "mac", None)
json.dump(out, open("pitt_folds_part20.json", "w"))
print("WROTE pitt_folds_part20.json")
