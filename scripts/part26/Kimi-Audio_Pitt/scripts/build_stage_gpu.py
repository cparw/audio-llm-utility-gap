"""PART26 Kimi-Audio Pitt: build the GPU clip list (path,label,speaker) in the row order of the canonical Kimi Pitt
zero-shot file, and check it against the Pitt clip manifest, the audio files and the saved fold files before upload."""
import csv, json, hashlib, os, numpy as np, soundfile as sf
C = "scores/part26/Kimi-Audio_Pitt"
ZS = "<local data dir>/release/overnight2/part3/kimi_pitt_zeroshot_scores.csv"
AUD = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468"
MAN = "<local data dir>/paper1_local_runs/clips_local/clips_for_af3/pitt468_manifest.csv"
FD = "<local data dir>/release/folds"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
zr = list(csv.DictReader(open(ZS))); names = [r["clip"] for r in zr]
assert len(zr) == 468 and len(set(names)) == 468
mr = {r["clip_path"]: r for r in csv.DictReader(open(MAN))}
rec = {"zeroshot_file": ZS, "zeroshot_sha256": sha(ZS), "clip_manifest": MAN, "clip_manifest_sha256": sha(MAN), "n": len(zr)}
rec["manifest_clips_equal"] = set(mr) == set(names)
rec["manifest_speakers_equal"] = all(mr[r["clip"]]["speaker_id"] == r["speaker"] for r in zr)
rec["manifest_labels_equal"] = all(int(mr[r["clip"]]["label"]) == int(r["label"]) for r in zr)
info = [sf.info(os.path.join(AUD, n)) for n in names]
rec["audio_all_16k_mono_pcm16"] = all(i.samplerate == 16000 and i.channels == 1 and i.subtype == "PCM_16" for i in info)
rec["audio_min_dur_s"] = min(i.duration for i in info); rec["audio_max_dur_s"] = max(i.duration for i in info)
rec["audio_all_at_least_30s"] = all(i.frames >= 480000 for i in info)
ff = {}
for t in ["", "_seed0_UNVERIFIED", "_seed1_UNVERIFIED", "_seed2_UNVERIFIED", "_seed3_UNVERIFIED", "_seed4_UNVERIFIED"]:
    p = f"{FD}/pitt_groupkfold5_pod{t}.csv"; fr = {r["clip_id"]: r for r in csv.DictReader(open(p))}
    ff[os.path.basename(p)] = {"sha256": sha(p), "n": len(fr), "clips_equal": set(fr) == set(names),
             "speakers_equal": all(fr[r["clip"]]["speaker_id"] == r["speaker"] for r in zr),
             "fold_sizes": np.bincount([int(fr[n]["fold"]) for n in names]).tolist()}
rec["fold_files"] = ff
y = np.array([int(r["label"]) for r in zr]); spk = np.array([r["speaker"] for r in zr])
rec["n_pos"] = int(y.sum()); rec["n_speakers"] = int(len(np.unique(spk)))
os.makedirs(f"{C}/gpu_stage", exist_ok=True)
with open(f"{C}/gpu_stage/mf_pitt.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"])
    for r in zr: w.writerow([f"/workspace/data/pitt468/{r['clip']}", r["label"], r["speaker"]])
with open(f"{C}/gpu_stage/audio.sha256", "w") as fh:
    for n in names: fh.write(f"{sha(os.path.join(AUD, n))}  data/pitt468/{n}\n")
rec["gpu_manifest"] = f"{C}/gpu_stage/mf_pitt.csv"; rec["gpu_manifest_sha256"] = sha(rec["gpu_manifest"])
json.dump(rec, open(f"{C}/gpu_stage/stage_check.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "fold_files"}, indent=1))
print(all(v["clips_equal"] and v["speakers_equal"] for v in ff.values()), [v["fold_sizes"] for v in ff.values()])
