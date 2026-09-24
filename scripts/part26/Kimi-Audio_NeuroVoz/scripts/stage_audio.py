"""PART26 Kimi-Audio NeuroVoz: stage the 1270 read-task wavs and the pod clip list (adapted from the Kimi-Audio_PC-GITA
stage_audio.py). Clip list rule = podD2_final/build_manifests.py, the rule behind /workspace/mf_neurovoz.csv of the Kimi
zero-shot run: rows of drive_data/neurovoz.csv with task_type == "read", resolved by basename under
spanish_neurovoz/zenodo_upload (only audios/ holds wavs there, 2903 unique names, so the first-match choice cannot matter).
Row order = the Kimi zero-shot file overnight2/kimi_new/kimi_neurovoz_zeroshot_scores.csv. Labels and speakers must equal
that file and the five fold files clip by clip. Writes stage/files.txt (tar list), wav_sha256.txt, mf_neurovoz_pod.csv,
mf_neurovoz_mac.csv, stage_check.json. The tar itself is streamed to the pod by the upload step."""
import os, csv, hashlib, json, collections
import soundfile as sf, numpy as np
C = "scores/part26/Kimi-Audio_NeuroVoz"
ROOT = "<local data dir>/paper1_local_runs/drive_data/spanish_neurovoz/zenodo_upload"
AUD = ROOT + "/audios"
NVCSV = "<local data dir>/paper1_local_runs/drive_data/neurovoz.csv"
ZS = "<local data dir>/release/overnight2/kimi_new/kimi_neurovoz_zeroshot_scores.csv"
FOLDS = [f"<local data dir>/release/folds/neurovoz_groupkfold5_pod_seed{s}_UNVERIFIED.csv" for s in range(5)]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
wav_dirs = collections.Counter()
for d in os.listdir(ROOT):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p): wav_dirs[d] = sum(1 for f in os.listdir(p) if f.lower().endswith(".wav"))
have = set(f for f in os.listdir(AUD) if f.lower().endswith(".wav"))
man = {}; order = []
for r in csv.DictReader(open(NVCSV)):
    if r["task_type"] != "read": continue
    b = os.path.basename(r["filepath"])
    if b not in have: continue
    assert b not in man, f"duplicate basename {b}"
    man[b] = (os.path.join(AUD, b), r["label"], r["speaker_id"]); order.append(b)
zs = list(csv.DictReader(open(ZS)))
assert len(zs) == len(man) == 1270, (len(zs), len(man))
assert [z["clip"] for z in zs] == order, "row order differs from the build_manifests rule"
for z in zs:
    p, lab, spk = man[z["clip"]]
    assert int(lab) == int(z["label"]) and spk == z["speaker"], ("label/speaker differs", z["clip"])
for ff in FOLDS:
    fr = {r["clip_id"]: r["speaker_id"] for r in csv.DictReader(open(ff))}
    assert set(fr) == set(man) and all(fr[b] == man[b][2] for b in man), f"fold file differs: {ff}"
S = f"{C}/stage"; os.makedirs(S, exist_ok=True)
wavsha, durs = [], []
for z in zs:
    p = man[z["clip"]][0]; wavsha.append((sha(p), f"neurovoz/{z['clip']}")); durs.append(sf.info(p).duration)
with open(f"{S}/wav_sha256.txt", "w") as fh:
    for h, a in wavsha: fh.write(f"{h}  {a}\n")
with open(f"{S}/files.txt", "w") as fh:
    for z in zs: fh.write(z["clip"] + "\n")
with open(f"{S}/mf_neurovoz_pod.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"])
    for z in zs: w.writerow([f"/workspace/data/neurovoz/{z['clip']}", man[z["clip"]][1], man[z["clip"]][2]])
with open(f"{S}/mf_neurovoz_mac.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"])
    for z in zs: w.writerow(list(man[z["clip"]]))
durs = np.array(durs)
rep = {"n": len(zs), "n_speakers": len(set(m[2] for m in man.values())), "n_pos": sum(int(m[1]) for m in man.values()),
       "wav_counts_by_subfolder": dict(wav_dirs), "bytes": int(sum(os.path.getsize(man[z['clip']][0]) for z in zs)),
       "duration_s": {"min": float(durs.min()), "median": float(np.median(durs)), "max": float(durs.max()), "n_over_30": int((durs > 30).sum())},
       "order": "Kimi zero-shot file row order, equal to the build_manifests.py rule (neurovoz.csv read rows)",
       "checks": ["clip set and order == Kimi zero-shot file", "label and speaker == Kimi zero-shot file", "clip set and speaker == all 5 fold files"],
       "neurovoz_csv": NVCSV, "neurovoz_csv_sha256": sha(NVCSV), "zeroshot_file": ZS, "zeroshot_sha256": sha(ZS),
       "fold_files": {os.path.basename(f): sha(f) for f in FOLDS}}
json.dump(rep, open(f"{S}/stage_check.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
