"""PART26 Qwen2-Audio PC-GITA: check the saved inputs on the Mac and stage the probe input.
Checks: states enc shape/finite; clip set, order and speakers against the five repeat fold files and the single split
fold file; labels and p_yes against the zero-shot per-clip file; zero-shot AUC; fold files equal the pod tie rule.
Writes stage/q2a_pcgita_encstates.npz with the keys enc, label, spk, p_yes, name (arrays byte-identical to the
states file) and stage/stage_check.json.   usage: stage_inputs.py"""
import os, csv, json, hashlib, platform, numpy as np, sklearn
from sklearn.metrics import roc_auc_score
C = "scores/part26/Qwen2-Audio_PC-GITA"
ST = "<local data dir>/paper1_local_runs/probe2/pcgita_states.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/pcgita_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def stable_gkf(groups, k):
    uniq, inv = np.unique(groups, return_inverse=True); counts = np.bincount(inv)
    order = np.argsort(counts, kind="stable")[::-1]; fg = np.zeros(len(uniq), int); load = np.zeros(k)
    for gi in order:
        f = int(np.argmin(load)); fg[gi] = f; load[f] += counts[gi]
    return fg[inv]
z = np.load(ST, allow_pickle=True)
enc = z["enc"]; y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str); py = z["p_yes"]
rep = {"states": ST, "states_sha256": sha(ST), "keys": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
assert enc.shape == (1100, 33, 1280) and np.isfinite(enc).all()
assert len(set(names)) == 1100
rep.update(n=len(y), n_pos=int(y.sum()), n_speakers=int(len(np.unique(spk))), enc_all_finite=True)
zr = list(csv.DictReader(open(ZS)))
zmap = {r["clip"]: r for r in zr}
assert len(zr) == 1100 and set(zmap) == set(names)
rep["zeroshot_same_order_as_states"] = [r["clip"] for r in zr] == list(names)
rep["zeroshot_labels_equal"] = all(int(zmap[n]["label"]) == int(l) for n, l in zip(names, y))
zp = np.array([float(zmap[n]["p_yes"]) for n in names])
rep["zeroshot_p_yes_max_abs_diff_vs_states"] = float(np.max(np.abs(zp - py)))
rep["zeroshot_speaker_column_n_unique"] = len({r["speaker"] for r in zr})
rep["zeroshot_auc"] = float(roc_auc_score(y, zp))
rep["folds"] = {}
for tag in ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]:
    ff = f"{FD}/pcgita_groupkfold5_pod.csv" if tag == "single" else f"{FD}/pcgita_groupkfold5_pod_{tag}.csv"
    fr = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(ff))}
    if tag == "single": g = spk
    else:
        rng = np.random.default_rng(int(tag[4:])); u = np.unique(spk); perm = {s: i for i, s in enumerate(rng.permutation(u))}
        g = np.array([perm[s] for s in spk])
    fo = np.array([fr[n][0] for n in names])
    rep["folds"][tag] = {"file": ff, "sha256": sha(ff), "n": len(fr), "same_clip_set": set(fr) == set(names),
                         "speaker_ids_equal_states_spk": all(fr[n][1] == s for n, s in zip(names, spk)),
                         "equals_pod_tie_rule": bool(np.array_equal(fo, stable_gkf(g, 5))),
                         "speakers_cross_folds": int(sum(len(set(fo[spk == s])) > 1 for s in np.unique(spk))),
                         "fold_sizes": np.bincount(fo).tolist()}
out = f"{C}/stage/q2a_pcgita_encstates.npz"
np.savez_compressed(out, **{k: z[k] for k in ("enc", "label", "spk", "p_yes", "name")})
z2 = np.load(out, allow_pickle=True)
rep["staged_arrays_identical"] = all(np.array_equal(z[k], z2[k]) and z[k].dtype == z2[k].dtype for k in ("enc", "label", "spk", "p_yes", "name"))
rep["staged_file"] = out; rep["staged_sha256"] = sha(out)
rep["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
ok = (rep["zeroshot_labels_equal"] and rep["staged_arrays_identical"] and rep["zeroshot_p_yes_max_abs_diff_vs_states"] == 0.0
      and all(v["same_clip_set"] and v["speaker_ids_equal_states_spk"] and v["equals_pod_tie_rule"] and v["speakers_cross_folds"] == 0
              for v in rep["folds"].values()))
rep["PASS"] = bool(ok)
json.dump(rep, open(f"{C}/stage/stage_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in ("folds", "keys")}, indent=1))
for t, v in rep["folds"].items(): print(t, {k: v[k] for k in ("same_clip_set", "speaker_ids_equal_states_spk", "equals_pod_tie_rule", "speakers_cross_folds", "fold_sizes")})
