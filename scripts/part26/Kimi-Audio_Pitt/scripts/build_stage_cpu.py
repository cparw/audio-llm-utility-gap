"""PART26 Kimi-Audio Pitt: build the enc-only probe input from the encoder states pulled from this cell's GPU pod
(p26-kimi-pitt-gpu, kimi_enc_extract.py) and check it against the fold files, the canonical zero-shot per-clip file,
the GPU clip list and the determinism rerun before anything goes to the CPU pod."""
import csv, json, hashlib, os, numpy as np
from sklearn.metrics import roc_auc_score
C = "scores/part26/Kimi-Audio_Pitt"
PO = f"{C}/pull/p26-kimi-pitt-gpu/out"
SRC = f"{PO}/kimi_pitt_encstates.npz"; RR = f"{PO}/rerun10_kimi_pitt_encstates.npz"; SM = f"{PO}/smoke_kimi_pitt_encstates.npz"
ZS = "<local data dir>/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv"
ZS_FULL = "<local data dir>/release/overnight/kimi/kimi_pitt_zeroshot_scores.csv"   # 18 Sep full-clip run, for contrast only
AUDSHA = f"{C}/gpu_stage/audio.sha256"
FD = f"{C}/stage/folds"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
z = np.load(SRC, allow_pickle=True)
names = z["name"].astype(str); spk = z["spk"].astype(str); y = z["label"].astype(int); E = z["enc"]
rec = {"source_states": SRC, "source_sha256": sha(SRC), "keys": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}}
zr = list(csv.DictReader(open(ZS))); zm = {r["clip"]: r for r in zr}
rec["n"] = int(len(names)); rec["n_unique"] = int(len(set(names)))
rec["order_equals_zeroshot_rows"] = list(names) == [r["clip"] for r in zr]
rec["zeroshot_labels_equal"] = all(int(zm[n]["label"]) == int(l) for n, l in zip(names, y))
rec["zeroshot_speakers_equal"] = all(zm[n]["speaker"] == s for n, s in zip(names, spk))
rec["enc_shape"] = list(E.shape); rec["enc_finite"] = bool(np.isfinite(E).all())
rec["kept_frames_all_1500"] = bool((z["kept_frames"] == 1500).all()); rec["samples_heard_all_480000"] = bool((z["samples_heard"] == 480000).all())
rec["enc_equals_enc_allframes"] = bool(np.array_equal(z["enc"], z["enc_allframes"]))
rec["hookcheck_max"] = float(z["hookcheck_maxdiff"].max())
aud = dict((l.split("  ")[1].strip().split("/")[-1], l.split("  ")[0]) for l in open(AUDSHA))
rec["wav_sha256_equal_mac"] = all(aud[n] == s for n, s in zip(names, z["wav_sha256"].astype(str)))
pz = z["p_yes"].astype(float); pc = np.array([float(zm[n]["p_yes"]) for n in names])
zf = {r["clip"]: float(r["p_yes"]) for r in csv.DictReader(open(ZS_FULL))}; pf = np.array([zf[n] for n in names])
d = np.abs(pz - pc); df = np.abs(pz - pf)
rec["p_yes_vs_canonical_30s"] = {"max_abs": float(d.max()), "median_abs": float(np.median(d)), "mean_abs": float(d.mean()),
                                 "pearson": float(np.corrcoef(pz, pc)[0, 1]), "auc_this_run": float(roc_auc_score(y, pz)),
                                 "auc_canonical": float(roc_auc_score(y, pc))}
rec["p_yes_vs_18sep_fullclip_contrast"] = {"file": ZS_FULL, "max_abs": float(df.max()), "median_abs": float(np.median(df)),
                                           "pearson": float(np.corrcoef(pz, pf)[0, 1]), "auc_fullclip": float(roc_auc_score(y, pf))}
r10 = np.load(RR, allow_pickle=True); sm = np.load(SM, allow_pickle=True)
rec["rerun10_enc_max_abs"] = float(np.abs(r10["enc"] - E[:10]).max()); rec["rerun10_p_yes_max_abs"] = float(np.abs(r10["p_yes"] - pz[:10]).max())
rec["rerun10_names_equal"] = list(r10["name"].astype(str)) == list(names[:10])
rec["smoke3_enc_max_abs"] = float(np.abs(sm["enc"] - E[:3]).max())
K = ("enc", "label", "spk", "p_yes", "name")
os.makedirs(f"{C}/stage/in", exist_ok=True)
out = f"{C}/stage/in/kimi_pitt_encstates.npz"
np.savez_compressed(out, **{k: z[k] for k in K})
z2 = np.load(out, allow_pickle=True)
assert all(np.array_equal(z[k], z2[k]) and z[k].dtype == z2[k].dtype for k in K)
rec["stage_file"] = out; rec["stage_sha256"] = sha(out); rec["stage_arrays_equal_source"] = True
ff = {}
for t in ["", "_seed0_UNVERIFIED", "_seed1_UNVERIFIED", "_seed2_UNVERIFIED", "_seed3_UNVERIFIED", "_seed4_UNVERIFIED"]:
    p = f"{FD}/pitt_groupkfold5_pod{t}.csv"; fr = {r["clip_id"]: r for r in csv.DictReader(open(p))}
    ff[os.path.basename(p)] = {"sha256": sha(p), "n": len(fr), "clips_equal": set(fr) == set(names),
             "speakers_equal": all(fr[n]["speaker_id"] == s for n, s in zip(names, spk)),
             "fold_sizes": np.bincount([int(fr[n]["fold"]) for n in names]).tolist()}
rec["fold_files"] = ff
rec["n_pos"] = int(y.sum()); rec["n_speakers"] = int(len(np.unique(spk)))
json.dump(rec, open(f"{C}/stage/stage_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k not in ("fold_files", "keys")}, indent=1))
print(all(v["clips_equal"] and v["speakers_equal"] for v in ff.values()))
