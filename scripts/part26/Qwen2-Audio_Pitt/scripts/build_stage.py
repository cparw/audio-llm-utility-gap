"""PART26 Qwen2-Audio Pitt: build the enc-only probe input from probe2/pitt_states.npz (read only) and pre-check
the saved fold files against it on the Mac. Arrays are copied byte-identically (checked)."""
import numpy as np, hashlib, json, csv, sys
SRC = "<local data dir>/paper1_local_runs/probe2/pitt_states.npz"
C = "scores/part26/Qwen2-Audio_Pitt"
OUT = f"{C}/stage/q2a_pitt_encstates.npz"
FD = "<local data dir>/release/folds"
ZS = "<local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
z = np.load(SRC, allow_pickle=True)
K = ("enc", "label", "spk", "p_yes", "name")
np.savez_compressed(OUT, **{k: z[k] for k in K})
z2 = np.load(OUT, allow_pickle=True)
assert all(np.array_equal(z[k], z2[k]) and z[k].dtype == z2[k].dtype for k in K)
enc = z2["enc"]; y = z2["label"].astype(int); spk = z2["spk"].astype(str); names = z2["name"].astype(str)
assert np.isfinite(enc).all() and len(set(names)) == len(names)
def stable_gkf(groups, k):
    uniq, inv = np.unique(groups, return_inverse=True); counts = np.bincount(inv)
    order = np.argsort(counts, kind="stable")[::-1]; fg = np.zeros(len(uniq), int); load = np.zeros(k)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += counts[gi]
    return fg[inv]
rep = {"source": SRC, "source_sha256": sha(SRC), "stage": OUT, "stage_sha256": sha(OUT),
       "keys": {k: [list(z2[k].shape), str(z2[k].dtype)] for k in K}, "arrays_equal_source": True,
       "n": int(len(y)), "n_pos": int(y.sum()), "n_speakers": int(len(np.unique(spk))), "folds": {}}
for tag in ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]:
    fn = "pitt_groupkfold5_pod.csv" if tag == "single" else f"pitt_groupkfold5_pod_{tag}_UNVERIFIED.csv"
    fr = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f"{FD}/{fn}"))}
    if tag == "single": g = spk
    else:
        rng = np.random.default_rng(int(tag[4:])); u = np.unique(spk)
        perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
    fs = np.array([fr[n][0] for n in names])
    rep["folds"][tag] = {"file": fn, "sha256": sha(f"{FD}/{fn}"), "clips_equal": set(fr) == set(names) and len(fr) == len(names),
                         "speakers_equal": all(fr[n][1] == s for n, s in zip(names, spk)),
                         "equals_stable_tie_rule": bool(np.array_equal(fs, stable_gkf(g, 5))),
                         "fold_sizes": np.bincount(fs).tolist()}
zs = {r["clip"]: r for r in csv.DictReader(open(ZS))}
rep["zeroshot"] = {"file": ZS, "sha256": sha(ZS), "clips_equal": set(zs) == set(names) and len(zs) == len(names),
                   "labels_equal": all(int(zs[n]["label"]) == l for n, l in zip(names, y)),
                   "speakers_equal": all(zs[n]["speaker"] == s for n, s in zip(names, spk)),
                   "p_yes_max_abs_diff_vs_states": float(np.max(np.abs(np.array([float(zs[n]["p_yes"]) for n in names]) - z2["p_yes"])))}
json.dump(rep, open(f"{C}/stage/stage_check.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
