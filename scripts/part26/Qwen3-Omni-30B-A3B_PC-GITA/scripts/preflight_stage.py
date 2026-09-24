"""PART 26, Qwen3-Omni-30B-A3B on PC-GITA. Mac pre-flight and stage build (read only on every source).
1. Load the POD3 states (the master row 168/169 source), check n, keys, finite enc, 32 encoder layers.
2. Check the fold files (seed0..4 and the single split) have exactly the states clip set and the same speaker ids,
   and that the pod tie rule (stable argsort, greedy fill) on the renumbered speakers reproduces each saved file.
3. Check the zero-shot answer file (q3o_new) has the same clips, labels and speakers, and recompute its AUC.
4. Write stage/q3o_pcgita_encstates.npz with the arrays the probe reads (enc, label, spk, name, p_yes), byte-identical
   to the source arrays (checked by reloading), and record sha256 of the source file, the stage file and each array.
usage: /usr/local/bin/python3 preflight_stage.py"""
import os, csv, json, hashlib, platform, datetime, numpy as np, sklearn
from sklearn.metrics import roc_auc_score

W = "scores/part26/Qwen3-Omni-30B-A3B_PC-GITA"
SRC = "<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pcgita_states.npz"
ZS = "<local data dir>/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
OUT = f"{W}/stage/q3o_pcgita_encstates.npz"

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

rec = {"date_utc": datetime.datetime.utcnow().isoformat() + "Z", "source_states": SRC, "zeroshot_file": ZS,
       "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                    "platform": platform.platform()}}
z = np.load(SRC, allow_pickle=True)
rec["source_keys"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
enc = z["enc"]; y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str); pyes = z["p_yes"]
assert enc.shape == (1100, 32, 1280), enc.shape
assert np.isfinite(enc).all(), "non-finite enc"
assert len(set(names)) == 1100
rec.update(n=int(len(y)), n_pos=int(y.sum()), n_speakers=int(len(np.unique(spk))), enc_shape=list(enc.shape),
           enc_all_finite=True)
# folds
fchk = {}
for tag in ["single", "seed0", "seed1", "seed2", "seed3", "seed4"]:
    fn = "pcgita_groupkfold5_pod.csv" if tag == "single" else f"pcgita_groupkfold5_pod_{tag}.csv"
    fr = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(f"{FD}/{fn}"))}
    assert set(fr) == set(names) and len(fr) == len(names), f"{tag} clip set differs"
    assert all(fr[n][1] == s for n, s in zip(names, spk)), f"{tag} speaker ids differ"
    fs = np.array([fr[n][0] for n in names])
    if tag == "single": g = spk
    else:
        rng = np.random.default_rng(int(tag[4:])); u = np.unique(spk)
        perm = {s: i for i, s in enumerate(rng.permutation(u))}; g = np.array([perm[s] for s in spk])
    rule = stable_gkf(g, 5)
    # speaker-disjoint check
    disj = all(len(set(fs[spk == s])) == 1 for s in np.unique(spk))
    fchk[tag] = {"file": fn, "sha256": sha(f"{FD}/{fn}"), "clip_set_equal": True, "speaker_ids_equal": True,
                 "pod_tie_rule_equal": bool(np.array_equal(fs, rule)), "speaker_disjoint": bool(disj),
                 "fold_sizes": np.bincount(fs).tolist()}
    assert fchk[tag]["pod_tie_rule_equal"] and disj, (tag, fchk[tag])
rec["folds"] = fchk
# zero-shot
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zr) == set(names) and len(zr) == 1100, "zero-shot clip set differs"
assert all(int(zr[n]["label"]) == int(l) for n, l in zip(names, y)), "zero-shot labels differ"
spk_eq = all(zr[n]["speaker"] == s for n, s in zip(names, spk))
zs = np.array([float(zr[n]["p_yes"]) for n in names])
rec["zeroshot"] = {"sha256": sha(ZS), "n": len(zr), "labels_equal": True, "speakers_equal_states_spk": bool(spk_eq),
                   "auc": float(roc_auc_score(y, zs)), "auc_4dp": round(float(roc_auc_score(y, zs)), 4),
                   "max_abs_p_yes_vs_states_p_yes": float(np.max(np.abs(zs - pyes)))}
assert rec["zeroshot"]["auc_4dp"] == 0.5967, rec["zeroshot"]["auc"]
# stage file
if not os.path.exists(OUT):
    np.savez_compressed(OUT + ".tmp.npz", enc=enc, label=z["label"], spk=z["spk"], name=z["name"], p_yes=pyes)
    os.replace(OUT + ".tmp.npz", OUT)
z2 = np.load(OUT, allow_pickle=True)
for k in ("enc", "label", "spk", "name", "p_yes"):
    assert z2[k].dtype == z[k].dtype and np.array_equal(z2[k], z[k]), f"stage array {k} differs"
rec["stage"] = {"file": OUT, "sha256": sha(OUT), "arrays_identical_to_source": True,
                "array_sha256": {k: hashlib.sha256(np.ascontiguousarray(z2[k]).tobytes()).hexdigest()
                                 for k in ("enc", "label", "spk")}}
rec["source_sha256"] = sha(SRC)
json.dump(rec, open(f"{W}/verify/preflight.json", "w"), indent=1)
print("PREFLIGHT OK", json.dumps({k: rec[k] for k in ("n", "n_pos", "n_speakers", "enc_shape")}),
      "zs auc", rec["zeroshot"]["auc"], "spk_eq", spk_eq, "stage sha", rec["stage"]["sha256"][:16])
