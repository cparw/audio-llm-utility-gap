"""PART26 Qwen2-Audio NeuroVoz: write the enc-only probe input from the saved states (read only), byte-identical arrays.
Keys kept: enc, label, spk, p_yes, name (the only keys p25a_nested5.py and p25a_podverify.py read, plus p_yes).
Checks: every array equal to the source, clip set equal to the five fold files and the zero-shot file, speaker ids and
labels equal, all values finite. usage: stage_states.py SRC_NPZ OUT_NPZ"""
import sys, json, csv, hashlib, numpy as np
src, out = sys.argv[1:3]
F = "<local data dir>/release/folds"
ZS = "<local data dir>/paper1_local_runs/probe2/neurovoz_zeroshot_scores.csv"
z = np.load(src, allow_pickle=True)
keep = ("enc", "label", "spk", "p_yes", "name")
np.savez_compressed(out, **{k: z[k] for k in keep})
z2 = np.load(out, allow_pickle=True)
rep = {"src": src, "out": out, "keys_src": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
rep["arrays_equal"] = {k: bool(np.array_equal(z[k], z2[k]) and z[k].dtype == z2[k].dtype) for k in keep}
rep["array_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(z2[k]).tobytes()).hexdigest() for k in ("enc", "label", "p_yes")}
enc = z2["enc"]; rep["enc_finite"] = bool(np.isfinite(enc).all()); rep["enc_shape"] = list(enc.shape)
names = z2["name"].astype(str); spk = z2["spk"].astype(str); y = z2["label"].astype(int)
rep["n"] = len(names); rep["n_unique_names"] = len(set(names)); rep["n_speakers"] = len(set(spk)); rep["n_pos"] = int(y.sum())
fc = {}
for t in ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]:
    fn = f"{F}/neurovoz_groupkfold5_pod.csv" if t == "single" else f"{F}/neurovoz_groupkfold5_pod_{t}_UNVERIFIED.csv"
    fr = {r["clip_id"]: r for r in csv.DictReader(open(fn))}
    fc[t] = {"file": fn, "same_clip_set": set(fr) == set(names) and len(fr) == len(names),
             "speaker_equal": all(fr[n]["speaker_id"] == s for n, s in zip(names, spk)),
             "sha256": hashlib.sha256(open(fn, "rb").read()).hexdigest()}
rep["folds"] = fc
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
pz = z2["p_yes"].astype(float)
rep["zeroshot"] = {"file": ZS, "same_clip_set": set(zr) == set(names) and len(zr) == len(names),
                   "labels_equal": all(int(zr[n]["label"]) == int(l) for n, l in zip(names, y)),
                   "speaker_equal": all(zr[n]["speaker"] == s for n, s in zip(names, spk)),
                   "p_yes_max_abs_diff_vs_states": float(max(abs(float(zr[n]["p_yes"]) - p) for n, p in zip(names, pz)))}
ok = (all(rep["arrays_equal"].values()) and rep["enc_finite"] and rep["n"] == rep["n_unique_names"] == 1270
      and all(v["same_clip_set"] and v["speaker_equal"] for v in fc.values())
      and rep["zeroshot"]["same_clip_set"] and rep["zeroshot"]["labels_equal"] and rep["zeroshot"]["speaker_equal"])
rep["PASS"] = bool(ok)
json.dump(rep, open(out.replace(".npz", ".stage.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in ("folds", "keys_src")}, indent=1)); print("STAGE PASS", ok)
