"""PART26 Kimi-Audio PC-GITA: stage the 1100 read-text wavs and the pod clip list.
Clip list rule = podD2_final/build_manifests.py (the rule behind /workspace/mf_pcgita.csv of the Kimi zero-shot run):
rows of drive_data/pcgita_read.csv minus the "sobraron" leftovers, resolved by basename. Here the file is resolved by
its relative path under spanish_dataset (the csv path after .../spanish_dataset/), and every other copy with the same
basename is checked to be byte-identical, so the os.walk first-match choice on the pod cannot matter.
Row order = the Kimi zero-shot file overnight2/part3/kimi_pcgita_zeroshot_scores.csv. Labels and speakers must equal
that file and the five fold files clip by clip."""
import os, csv, sys, hashlib, tarfile, collections, json
C = "scores/part26/Kimi-Audio_PC-GITA"
ROOT = "<local data dir>/paper1_local_runs/clips_local/spanish_dataset"
READ = "<local data dir>/paper1_local_runs/drive_data/pcgita_read.csv"
ZS = "<local data dir>/release/overnight2/part3/kimi_pcgita_zeroshot_scores.csv"
FOLDS = [f"<local data dir>/release/folds/pcgita_groupkfold5_pod_seed{s}.csv" for s in range(5)]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
ix = collections.defaultdict(list)
for d, _, fs in os.walk(ROOT):
    for f in fs:
        if f.lower().endswith(".wav") and not f.startswith("._"): ix[f].append(os.path.join(d, f))
man = {}
dups = {}
for r in csv.DictReader(open(READ)):
    if "sobraron" in r["filepath"].lower(): continue
    b = os.path.basename(r["filepath"])
    rel = r["filepath"].split("/spanish_dataset/", 1)[1]
    p = os.path.join(ROOT, rel)
    assert os.path.exists(p), f"missing {p}"
    assert b not in man, f"duplicate basename in the clip list: {b}"
    c = ix[b]
    if len(c) > 1:
        hs = {q: sha(q) for q in c}
        dups[b] = {"n_copies": len(c), "all_identical": len(set(hs.values())) == 1}
        assert len(set(hs.values())) == 1, f"copies of {b} differ: {c}"
    man[b] = (p, r["label"], r["speaker_id"])
zs = list(csv.DictReader(open(ZS)))
assert len(zs) == len(man) == 1100, (len(zs), len(man))
assert [z["clip"] for z in zs] and set(z["clip"] for z in zs) == set(man), "clip sets differ vs Kimi zero-shot file"
for z in zs:
    p, lab, spk = man[z["clip"]]
    assert int(lab) == int(z["label"]) and spk == z["speaker"], ("label/speaker differs", z["clip"])
for ff in FOLDS:
    fr = {r["clip_id"]: r["speaker_id"] for r in csv.DictReader(open(ff))}
    assert set(fr) == set(man) and all(fr[b] == man[b][2] for b in man), f"fold file differs: {ff}"
os.makedirs(f"{C}/stage", exist_ok=True)
TAR = f"{C}/stage/pcgita1100.tar"
wavsha = []
with tarfile.open(TAR, "w") as t:
    for z in zs:
        p = man[z["clip"]][0]
        t.add(p, arcname=f"pcgita/{z['clip']}")
        wavsha.append((sha(p), f"pcgita/{z['clip']}"))
with open(f"{C}/stage/wav_sha256.txt", "w") as fh:
    for h, a in wavsha: fh.write(f"{h}  {a}\n")
with open(f"{C}/stage/mf_pcgita_pod.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"])
    for z in zs: w.writerow([f"/workspace/data/pcgita/{z['clip']}", man[z["clip"]][1], man[z["clip"]][2]])
with open(f"{C}/stage/mf_pcgita_mac.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"])
    for z in zs: w.writerow(list(man[z["clip"]]))
rep = {"n": len(zs), "n_speakers": len(set(m[2] for m in man.values())), "n_pos": sum(int(m[1]) for m in man.values()),
       "duplicate_basenames_in_tree": len(dups), "duplicates_all_identical": all(v["all_identical"] for v in dups.values()),
       "tar": TAR, "tar_sha256": sha(TAR), "tar_bytes": os.path.getsize(TAR), "order": "Kimi zero-shot file row order",
       "checks": ["clip set == Kimi zero-shot file", "label and speaker == Kimi zero-shot file", "clip set and speaker == all 5 fold files"],
       "read_csv": READ, "read_csv_sha256": sha(READ), "zeroshot_file": ZS, "zeroshot_sha256": sha(ZS)}
json.dump(rep, open(f"{C}/stage/stage_check.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
