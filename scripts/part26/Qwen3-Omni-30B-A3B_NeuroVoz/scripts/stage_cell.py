"""PART26 cell Qwen3-Omni-30B-A3B x NeuroVoz: stage the probe input and check it before any pod is used.

Reads (read only): the saved Qwen3-Omni NeuroVoz states, the five pod-order repeat fold files plus the single-split
pod fold file, and the q3o_new zero-shot per-clip file.
Writes: stage/q3o_neurovoz_encstates.npz (enc, label, spk, p_yes, name, byte-identical arrays) and
stage/stage_check.json. Stops if any check fails.
usage: /usr/local/bin/python3 stage_cell.py"""
import os, sys, csv, json, hashlib, datetime, numpy as np
from sklearn.metrics import roc_auc_score

C = "scores/part26/Qwen3-Omni-30B-A3B_NeuroVoz"
REL = "<local data dir>/release"
ST = f"{REL}/overnight2/q3o/q3o_neurovoz_states.npz"
ZS = f"{REL}/overnight2/q3o_new/q3o_neurovoz_zeroshot_scores.csv"
FD = f"{REL}/folds"
FOLDS = [f"neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)] + ["neurovoz_groupkfold5_pod.csv"]
OUT = f"{C}/stage/q3o_neurovoz_encstates.npz"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

z = np.load(ST, allow_pickle=True)
enc = z["enc"]; y = z["label"].astype(int); spk = z["spk"].astype(str); names = z["name"].astype(str); py = z["p_yes"]
rep = {"states": ST, "states_sha256": sha(ST), "keys": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
assert enc.shape == (1270, 32, 1280), enc.shape
assert np.isfinite(enc).all(), "non-finite enc"
assert len(set(names)) == 1270, "duplicate clip names"
rep["n"] = int(len(y)); rep["n_pos"] = int(y.sum()); rep["n_speakers"] = int(len(np.unique(spk)))
assert rep["n_speakers"] == 107 and rep["n_pos"] == 621, rep
# zero-shot file
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
assert set(zr) == set(names) and len(zr) == len(names), "zero-shot clip set differs"
assert all(int(zr[c]["label"]) == int(l) for c, l in zip(names, y)), "zero-shot labels differ"
zs = np.array([float(zr[c]["p_yes"]) for c in names])
rep["zeroshot_file"] = ZS; rep["zeroshot_sha256"] = sha(ZS)
rep["zeroshot_speaker_equals_states_spk"] = bool(all(str(zr[c]["speaker"]) == s for c, s in zip(names, spk)))
rep["max_abs_p_yes_states_vs_zeroshot"] = float(np.max(np.abs(py - zs)))
rep["zeroshot_auc"] = float(roc_auc_score(y, zs)); rep["zeroshot_auc_4dp"] = round(rep["zeroshot_auc"], 4)
assert rep["zeroshot_auc_4dp"] == 0.6393, rep["zeroshot_auc"]
# fold files
rep["folds"] = {}
for f in FOLDS:
    p = f"{FD}/{f}"; fr = {r["clip_id"]: r for r in csv.DictReader(open(p))}
    assert set(fr) == set(names) and len(fr) == len(names), f"{f}: clip set differs"
    assert all(fr[c]["speaker_id"] == s for c, s in zip(names, spk)), f"{f}: speaker ids differ"
    fo = np.array([int(fr[c]["fold"]) for c in names])
    rep["folds"][f] = {"sha256": sha(p), "fold_sizes": np.bincount(fo).tolist(),
                       "speakers_split_across_folds": int(sum(len(set(fo[spk == s])) > 1 for s in np.unique(spk)))}
    assert rep["folds"][f]["speakers_split_across_folds"] == 0, f
# enc-only npz, byte-identical arrays
if not os.path.exists(OUT):
    np.savez_compressed(OUT, **{k: z[k] for k in ("enc", "label", "spk", "p_yes", "name")})
z2 = np.load(OUT, allow_pickle=True)
assert all(np.array_equal(z[k], z2[k]) for k in ("enc", "label", "spk", "p_yes", "name")), "encstates differ"
rep["encstates"] = OUT; rep["encstates_sha256"] = sha(OUT); rep["encstates_arrays_identical"] = True
rep["date_utc"] = datetime.datetime.utcnow().isoformat() + "Z"; rep["command"] = "/usr/local/bin/python3 " + " ".join(sys.argv)
json.dump(rep, open(f"{C}/stage/stage_check.json", "w"), indent=1)
print("STAGE OK n", rep["n"], "spk", rep["n_speakers"], "pos", rep["n_pos"], "zs auc", round(rep["zeroshot_auc"], 6),
      "max p_yes diff", rep["max_abs_p_yes_states_vs_zeroshot"], "spk==zs speaker", rep["zeroshot_speaker_equals_states_spk"])
