"""PART26 Qwen2-Audio ADReSS-2020: build the enc-only probe input from the saved Qwen2-Audio states (read only) and
check it against the fold files and the zero-shot per-clip file before anything goes to a pod."""
import sys, csv, json, hashlib, numpy as np
C = "scores/part26/Qwen2-Audio_ADReSS-2020"
SRC = "<local data dir>/paper1_local_runs/probe2/adress2020_states.npz"
ZS = "<local data dir>/paper1_local_runs/probe2/adress2020_zeroshot_scores.csv"
FD = "<local data dir>/release/folds"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
z = np.load(SRC, allow_pickle=True)
K = ("enc", "label", "spk", "p_yes", "name")
out = f"{C}/stage/q2a_adress2020_encstates.npz"
np.savez_compressed(out, **{k: z[k] for k in K})
z2 = np.load(out, allow_pickle=True)
assert all(np.array_equal(z[k], z2[k]) and z[k].dtype == z2[k].dtype for k in K)
names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
assert len(set(names)) == len(names) == 156
rec = {"source_states": SRC, "source_sha256": sha(SRC), "stage_file": out, "stage_sha256": sha(out),
       "keys": {k: [list(z2[k].shape), str(z2[k].dtype)] for k in K}, "arrays_equal_source": True,
       "enc_finite": bool(np.isfinite(z["enc"]).all())}
zr = list(csv.DictReader(open(ZS))); zm = {r["clip"]: r for r in zr}
assert len(zr) == 156 and set(zm) == set(names)
rec["zeroshot_labels_equal"] = all(int(zm[n]["label"]) == int(l) for n, l in zip(names, y))
rec["zeroshot_speakers_equal"] = all(zm[n]["speaker"] == s for n, s in zip(names, spk))
rec["states_p_yes_vs_zeroshot_max_abs"] = float(max(abs(float(zm[n]["p_yes"]) - float(p)) for n, p in zip(names, z["p_yes"])))
ff = {}
for t in ["", "_seed0_UNVERIFIED", "_seed1_UNVERIFIED", "_seed2_UNVERIFIED", "_seed3_UNVERIFIED", "_seed4_UNVERIFIED"]:
    p = f"{FD}/adress2020_groupkfold5_pod{t}.csv"; fr = {r["clip_id"]: r for r in csv.DictReader(open(p))}
    ff[p] = {"sha256": sha(p), "n": len(fr), "clips_equal": set(fr) == set(names),
             "speakers_equal": all(fr[n]["speaker_id"] == s for n, s in zip(names, spk)),
             "fold_sizes": np.bincount([int(fr[n]["fold"]) for n in names]).tolist()}
rec["fold_files"] = ff
rec["n"] = 156; rec["n_pos"] = int(y.sum()); rec["n_speakers"] = int(len(np.unique(spk)))
json.dump(rec, open(f"{C}/stage/stage_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "fold_files"}, indent=1))
print(all(v["clips_equal"] and v["speakers_equal"] for v in ff.values()), [v["fold_sizes"] for v in ff.values()])
