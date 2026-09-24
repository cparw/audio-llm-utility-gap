"""PART 26 Kimi-Audio ADReSSo: check the pulled encoder states before the probe, then write the slim npz the CPU pod reads.
Checks: every clip of the dataset list; same clip set as the five saved fold files and the zero-shot per-clip file;
speaker ids and labels equal clip by clip; 32 encoder layers x 1280; every value finite; pod sha256 of the npz equals
the Mac copy; rerun p_yes in the states file against the zero-shot file p_yes (the answer AUC used for the gap is the
zero-shot file's, not the rerun's). Writes podfiles/kimi_adresso_enc_states_slim.npz (keys enc, label, spk, name; enc
copied unchanged) and verify/p26_states_check.json.   usage: /usr/local/bin/python3 p26_states_check.py"""
import csv, json, hashlib, datetime, numpy as np
from sklearn.metrics import roc_auc_score
C = "scores/part26/Kimi-Audio_ADReSSo"
NPZ = f"{C}/pull/p26-kimi-adresso-gpu/out/kimi_adresso_enc_states.npz"
SJ = f"{C}/pull/p26-kimi-adresso-gpu/out/kimi_adresso_enc_states.json"
PODSHA = f"{C}/pull/p26-kimi-adresso-gpu/logs/out_sha256.txt"
MF = f"{C}/pull/p26-kimi-adresso-gpu/mf_adresso.csv"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_adresso_zeroshot_scores.csv"
FOLDS = "<local data dir>/release/folds"
SLIM = f"{C}/podfiles/kimi_adresso_enc_states_slim.npz"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "states": NPZ, "states_sha256": sha(NPZ)}
pod = dict(l.split()[::-1] for l in open(PODSHA) if l.strip())
out["states_sha256_equals_pod"] = pod["kimi_adresso_enc_states.npz"] == out["states_sha256"]
z = np.load(NPZ, allow_pickle=True)
out["keys"] = {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
E = z["enc"]; names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int)
out["enc_shape"] = list(E.shape); out["enc_all_finite"] = bool(np.isfinite(E).all())
out["n_unique_names"] = int(len(set(names))); out["n_unique_spk"] = int(len(set(spk)))
mf = list(csv.DictReader(open(MF)))
out["manifest_rows"] = len(mf)
out["manifest_order_equals_states"] = [r["path"].split("/")[-1] for r in mf] == names.tolist()
zr = {r["clip"]: r for r in csv.DictReader(open(ZS))}
out["zeroshot_file"] = ZS; out["zeroshot_sha256"] = sha(ZS); out["zeroshot_n"] = len(zr)
out["clipset_equals_zeroshot"] = set(zr) == set(names) and len(zr) == len(names)
out["labels_equal_zeroshot"] = all(int(zr[n]["label"]) == int(l) for n, l in zip(names, y))
out["speakers_equal_zeroshot"] = all(zr[n]["speaker"] == s for n, s in zip(names, spk))
fc = {}
for s in range(5):
    ff = f"{FOLDS}/adresso_groupkfold5_pod_seed{s}_UNVERIFIED.csv"
    fr = {r["clip_id"]: r for r in csv.DictReader(open(ff))}
    fc[f"seed{s}"] = {"file": ff, "sha256": sha(ff), "clipset_equal": set(fr) == set(names) and len(fr) == len(names),
                      "speakers_equal": all(fr[n]["speaker_id"] == sp for n, sp in zip(names, spk))}
ff = f"{FOLDS}/adresso_groupkfold5_pod.csv"; fr = {r["clip_id"]: r for r in csv.DictReader(open(ff))}
fc["single"] = {"file": ff, "sha256": sha(ff), "clipset_equal": set(fr) == set(names) and len(fr) == len(names),
                "speakers_equal": all(fr[n]["speaker_id"] == sp for n, sp in zip(names, spk))}
out["fold_files"] = fc
zp = np.array([float(zr[n]["p_yes"]) for n in names]); rp = z["p_yes"].astype(float)
out["zeroshot_auc"] = float(roc_auc_score(y, zp)); out["rerun_auc"] = float(roc_auc_score(y, rp))
out["rerun_vs_zeroshot_p_yes"] = {"pearson": float(np.corrcoef(zp, rp)[0, 1]), "max_abs": float(np.abs(zp - rp).max()),
                                  "median_abs": float(np.median(np.abs(zp - rp)))}
out["n_samples_min_max"] = [int(z["n_samples"].min()), int(z["n_samples"].max())]
out["enc_frames_min_max"] = [int(z["enc_frames"].min()), int(z["enc_frames"].max())]
out["layer_norms_first_last"] = [float(np.linalg.norm(E[:, 0], axis=1).mean()), float(np.linalg.norm(E[:, -1], axis=1).mean())]
out["layers_distinct"] = bool(all(not np.allclose(E[:, i], E[:, i + 1]) for i in range(E.shape[1] - 1)))
out["sidecar"] = json.load(open(SJ))
ok = (out["states_sha256_equals_pod"] and E.shape == (237, 32, 1280) and out["enc_all_finite"] and out["n_unique_names"] == 237
      and out["manifest_order_equals_states"] and out["clipset_equals_zeroshot"] and out["labels_equal_zeroshot"]
      and out["speakers_equal_zeroshot"] and all(v["clipset_equal"] and v["speakers_equal"] for v in fc.values())
      and out["layers_distinct"] and abs(out["zeroshot_auc"] - 0.7531004989308624) < 1e-9)
out["STATES_OK"] = bool(ok)
np.savez_compressed(SLIM, enc=E.astype(np.float32), label=y, spk=spk, name=names)
zs2 = np.load(SLIM, allow_pickle=True)
out["slim"] = {"file": SLIM, "sha256": sha(SLIM), "enc_equal_bitwise": bool(np.array_equal(zs2["enc"], E)),
               "keys": {k: [list(zs2[k].shape), str(zs2[k].dtype)] for k in zs2.files}}
json.dump(out, open(f"{C}/verify/p26_states_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("sidecar", "fold_files", "keys")}, indent=1))
print("fold files", {k: (v["clipset_equal"], v["speakers_equal"]) for k, v in fc.items()})
