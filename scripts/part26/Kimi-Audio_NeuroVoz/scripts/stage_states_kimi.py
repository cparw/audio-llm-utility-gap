"""PART26 Kimi-Audio NeuroVoz: check the pulled encoder states from the GPU pod and write the enc-only probe input.
Checks (all must hold for PASS):
  1 n = 1270 unique clips, row order = the Kimi zero-shot file overnight2/kimi_new/kimi_neurovoz_zeroshot_scores.csv;
  2 labels and speakers equal that file and the five fold files (and the single-split fold file) clip by clip;
  3 enc shape (1270, 32, 1280) float32, every value finite;
  4 every wav sha256 recorded on the pod equals the Mac stage list (stage/wav_sha256.txt);
  5 hook check: layer_norm(layer 31)[kept frames] mean vs the mean of history.continuous_feature, max abs diff < 1e-3;
  6 same audio and window as the zero-shot answer: p_yes of this run vs the zero-shot file, reported (max abs diff, AUC);
  7 determinism: the fresh-process rerun of the first 10 clips gives the same enc (max abs diff reported).
Keys written: enc, label, spk, p_yes, name (what p25a_nested5.py and p25a_podverify.py read, plus p_yes).
usage: stage_states_kimi.py PULLDIR"""
import sys, os, json, csv, hashlib, numpy as np
from sklearn.metrics import roc_auc_score
C = "scores/part26/Kimi-Audio_NeuroVoz"
PULL = sys.argv[1]
SRC = f"{PULL}/out/kimi_neurovoz_encstates.npz"; RR = f"{PULL}/out/rerun10_kimi_neurovoz_encstates.npz"
OUT = f"{C}/stage/kimi_neurovoz_encstates.npz"
F = "<local data dir>/release/folds"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_neurovoz_zeroshot_scores.csv"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
z = np.load(SRC, allow_pickle=True)
names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int); enc = z["enc"]; py = z["p_yes"].astype(float)
rep = {"src": SRC, "src_sha256": sha(SRC), "keys_src": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
zs = list(csv.DictReader(open(ZS)))
rep["n"] = len(names); rep["n_unique"] = len(set(names)); rep["n_speakers"] = len(set(spk)); rep["n_pos"] = int(y.sum())
rep["order_equals_zeroshot"] = [r["clip"] for r in zs] == names.tolist()
rep["labels_equal_zeroshot"] = all(int(r["label"]) == int(l) for r, l in zip(zs, y))
rep["speakers_equal_zeroshot"] = all(r["speaker"] == s for r, s in zip(zs, spk))
fc = {}
for t in ["seed0", "seed1", "seed2", "seed3", "seed4", "single"]:
    fn = f"{F}/neurovoz_groupkfold5_pod.csv" if t == "single" else f"{F}/neurovoz_groupkfold5_pod_{t}_UNVERIFIED.csv"
    fr = {r["clip_id"]: r for r in csv.DictReader(open(fn))}
    fc[t] = {"file": fn, "sha256": sha(fn), "same_clip_set": set(fr) == set(names) and len(fr) == len(names),
             "speaker_equal": all(fr[n]["speaker_id"] == s for n, s in zip(names, spk))}
rep["folds"] = fc
rep["enc_shape"] = list(enc.shape); rep["enc_dtype"] = str(enc.dtype); rep["enc_finite"] = bool(np.isfinite(enc).all())
mac = {}
for line in open(f"{C}/stage/wav_sha256.txt"):
    h, a = line.split(); mac[os.path.basename(a)] = h
rep["wav_sha256_equal_mac_stage"] = all(mac[n] == h for n, h in zip(names, z["wav_sha256"].astype(str)))
hk = z["hookcheck_maxdiff"].astype(float); rep["hookcheck_max"] = float(hk.max())
zmap = {r["clip"]: float(r["p_yes"]) for r in zs}
d = np.abs(np.array([zmap[n] for n in names]) - py)
rep["p_yes_vs_zeroshot"] = {"max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)), "n_over_0.01": int((d > 0.01).sum()),
                            "n_over_0.05": int((d > 0.05).sum()), "pearson_r": float(np.corrcoef([zmap[n] for n in names], py)[0, 1]),
                            "auc_this_run": float(roc_auc_score(y, py)), "auc_zeroshot_file": float(roc_auc_score(y, [zmap[n] for n in names]))}
kf = z["kept_frames"].astype(int); rep["kept_frames"] = {"min": int(kf.min()), "median": float(np.median(kf)), "max": int(kf.max())}
if os.path.exists(RR):
    r = np.load(RR, allow_pickle=True)
    rep["rerun10"] = {"names_equal": r["name"].astype(str).tolist() == names[:10].tolist(),
                      "enc_max_abs_diff": float(np.abs(r["enc"] - enc[:10]).max()),
                      "p_yes_max_abs_diff": float(np.abs(r["p_yes"].astype(float) - py[:10]).max())}
np.savez_compressed(OUT, enc=enc.astype(np.float32), label=z["label"], spk=z["spk"], p_yes=z["p_yes"], name=z["name"])
z2 = np.load(OUT, allow_pickle=True)
rep["staged"] = OUT; rep["staged_sha256"] = sha(OUT)
rep["staged_arrays_equal"] = {k: bool(np.array_equal(z[k], z2[k])) for k in ("enc", "label", "spk", "p_yes", "name")}
ok = (rep["n"] == rep["n_unique"] == 1270 and rep["n_speakers"] == 107 and rep["order_equals_zeroshot"] and rep["labels_equal_zeroshot"]
      and rep["speakers_equal_zeroshot"] and all(v["same_clip_set"] and v["speaker_equal"] for v in fc.values())
      and rep["enc_shape"] == [1270, 32, 1280] and rep["enc_finite"] and rep["wav_sha256_equal_mac_stage"] and rep["hookcheck_max"] < 1e-3
      and all(rep["staged_arrays_equal"].values()))
rep["PASS"] = bool(ok)
json.dump(rep, open(OUT.replace(".npz", ".stage.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in ("folds", "keys_src")}, indent=1)); print("STAGE PASS", ok)
