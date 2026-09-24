"""PART26 Kimi-Audio PC-GITA: check the pulled GPU-pod encoder states on the Mac and stage the probe input.
Checks: enc shape (1100, 32, 1280) and finite; no two layers identical; every clip's wav sha256 equals the staged upload
list (same audio); clip order, labels and speakers equal the Kimi zero-shot file and the five repeat fold files and the
single-split fold file; fold files equal the pod tie rule and no speaker crosses folds; per-clip hook check (layer 31 +
the encoder's layer_norm equals the continuous_feature Kimi passes to the LM); smoke (3 clips) and rerun (10 clips, fresh
process) equal the full run; p_yes of this run vs the canonical Kimi zero-shot file (same prompt, same 30 s cut, a
different pod: the answer side of the gap stays on the canonical file).
Writes stage/kimi_pcgita_encstates.npz (keys enc, label, spk, p_yes, name; arrays byte-identical to the pulled file)
and verify/stage_cpu_check.json.   usage: stage_cpu.py"""
import os, csv, json, hashlib, platform, numpy as np, sklearn
from sklearn.metrics import roc_auc_score
C = "scores/part26/Kimi-Audio_PC-GITA"
OUT = f"{C}/pull/p26-kimi-pcgita-gpu/out"
ST = f"{OUT}/kimi_pcgita_encstates.npz"
ZS = "<local data dir>/release/overnight2/part3/kimi_pcgita_zeroshot_scores.csv"
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
rep = {"states": ST, "states_sha256": sha(ST), "keys": {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files},
       "sidecar": json.load(open(f"{OUT}/kimi_pcgita_encstates.json"))}
rep["enc_shape_ok"] = enc.shape == (1100, 32, 1280); rep["enc_all_finite"] = bool(np.isfinite(enc).all())
rep["enc_layers_distinct"] = bool(all(not np.array_equal(enc[:, i], enc[:, j]) for i in range(32) for j in range(i + 1, 32)))
rep["enc_constant_columns_per_layer"] = [int((enc[:, l].std(axis=0) == 0).sum()) for l in range(32)]
rep["n_unique_names"] = len(set(names))
rep.update(n=len(y), n_pos=int(y.sum()), n_speakers=int(len(np.unique(spk))))
ws = {}
for line in open(f"{C}/stage/wav_sha256.txt"):
    h, a = line.split(); ws[os.path.basename(a)] = h
rep["wav_sha256_equal_upload_list"] = all(ws[n] == h for n, h in zip(names, z["wav_sha256"].astype(str)))
rep["hookcheck_maxdiff_max"] = float(z["hookcheck_maxdiff"].max())
kf = z["kept_frames"]; rep["kept_frames_min_max"] = [int(kf.min()), int(kf.max())]
rep["dur_file_s_min_median_max"] = [float(np.min(z["dur_file_s"])), float(np.median(z["dur_file_s"])), float(np.max(z["dur_file_s"]))]
rep["n_clips_over_30s"] = int((z["dur_file_s"] > 30).sum())
zr = list(csv.DictReader(open(ZS))); zmap = {r["clip"]: r for r in zr}
rep["zeroshot_same_order_as_states"] = [r["clip"] for r in zr] == list(names)
rep["zeroshot_labels_equal"] = all(int(zmap[n]["label"]) == int(l) for n, l in zip(names, y))
rep["zeroshot_speakers_equal"] = all(zmap[n]["speaker"] == s for n, s in zip(names, spk))
zp = np.array([float(zmap[n]["p_yes"]) for n in names]); d = np.abs(zp - py)
rep["p_yes_this_run_vs_canonical"] = {"max_abs": float(d.max()), "median_abs": float(np.median(d)), "n_over_1e-3": int((d > 1e-3).sum()),
                                      "pearson": float(np.corrcoef(zp, py)[0, 1]), "auc_canonical": float(roc_auc_score(y, zp)),
                                      "auc_this_run": float(roc_auc_score(y, py)),
                                      "note": "separate pod run of the same forward pass; part20 POD4d found the same kind of cross-pod drift for Kimi (median 0.010, max 0.20 on Pitt). The gap uses the canonical file."}
for tag, pre, k in (("smoke3", "smoke_kimi_pcgita", 3), ("rerun10", "rerun10_kimi_pcgita", 10)):
    p = f"{OUT}/{pre}_encstates.npz"
    if os.path.exists(p):
        s = np.load(p, allow_pickle=True)
        rep[f"{tag}_names_equal"] = list(s["name"].astype(str)) == list(names[:k])
        rep[f"{tag}_enc_max_abs_vs_full"] = float(np.abs(s["enc"] - enc[:k]).max())
        rep[f"{tag}_p_yes_max_abs_vs_full"] = float(np.abs(s["p_yes"] - py[:k]).max())
    else: rep[f"{tag}_missing"] = True
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
out = f"{C}/stage/kimi_pcgita_encstates.npz"
np.savez_compressed(out, **{k: z[k] for k in ("enc", "label", "spk", "p_yes", "name")})
z2 = np.load(out, allow_pickle=True)
rep["staged_arrays_identical"] = all(np.array_equal(z[k], z2[k]) and z[k].dtype == z2[k].dtype for k in ("enc", "label", "spk", "p_yes", "name"))
rep["staged_file"] = out; rep["staged_sha256"] = sha(out)
rep["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
ok = (rep["enc_shape_ok"] and rep["enc_all_finite"] and rep["enc_layers_distinct"] and rep["n_unique_names"] == 1100
      and rep["wav_sha256_equal_upload_list"] and rep["zeroshot_same_order_as_states"] and rep["zeroshot_labels_equal"]
      and rep["zeroshot_speakers_equal"] and rep["staged_arrays_identical"] and rep["hookcheck_maxdiff_max"] < 5e-2
      and rep.get("rerun10_enc_max_abs_vs_full", 1) < 1e-3 and rep.get("smoke3_enc_max_abs_vs_full", 1) < 1e-3
      and all(v["same_clip_set"] and v["speaker_ids_equal_states_spk"] and v["equals_pod_tie_rule"] and v["speakers_cross_folds"] == 0
              for v in rep["folds"].values()))
rep["PASS"] = bool(ok)
os.makedirs(f"{C}/verify", exist_ok=True)
json.dump(rep, open(f"{C}/verify/stage_cpu_check.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in rep.items() if k not in ("sidecar", "folds", "keys")}, indent=1, default=str))
print("folds", {t: (v["equals_pod_tie_rule"], v["speakers_cross_folds"], v["fold_sizes"]) for t, v in rep["folds"].items()})
print("STAGE PASS" if ok else "STAGE FAIL")
