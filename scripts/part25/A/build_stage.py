"""PART25 A: resolve the audio behind each Qwen2.5-Omni omni_final clip list, on the G-Drive, with the same rules as
release/scripts/build_manifests.py (basename index, first os.walk hit), in the exact row order of the saved
omni_<ds>_zeroshot_scores.csv. Writes stage/<ds>/mf_<ds>.csv (pod paths) and stage/<ds>/files.tsv (local path, sha256).
Reports any basename that resolves to more than one file with different content."""
import csv, os, sys, hashlib, collections, json
OF = "<local data dir>/release/omni_final"
G = "<local data dir>/paper work"
ST = f"{G}/release_from_mac/scores/part25/A/stage"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def index(root):
    ix = collections.defaultdict(list)
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".wav") and not f.startswith("._"): ix[f].append(os.path.join(d, f))
    return ix
ds = sys.argv[1]
zs = list(csv.DictReader(open(f"{OF}/omni_{ds}_zeroshot_scores.csv")))
want = [r["clip"] for r in zs]
src = {}
if ds == "pcgita":
    ix = index(f"{G}/paper1_local_runs/clips_local/spanish_dataset")
    for r in csv.DictReader(open(f"{G}/paper1_local_runs/drive_data/pcgita_read.csv")):
        if "sobraron" in r["filepath"].lower(): continue
        b = os.path.basename(r["filepath"]); c = ix.get(b)
        if c and b not in src: src[b] = (c, r["label"], r["speaker_id"])
elif ds == "neurovoz":
    ix = index(f"{G}/paper1_local_runs/drive_data/spanish_neurovoz/zenodo_upload")
    for r in csv.DictReader(open(f"{G}/paper1_local_runs/drive_data/neurovoz.csv")):
        if r["task_type"] != "read": continue
        b = os.path.basename(r["filepath"]); c = ix.get(b)
        if c and b not in src: src[b] = (c, r["label"], r["speaker_id"])
elif ds in ("adresso", "adress2020"):
    root = {"adresso": f"{G}/Hard drive data for paper/share_for_minoo/adresso_clips_patient_onset",
            "adress2020": f"{G}/Hard drive data for paper/share_for_minoo/adress2020_clips_patient_onset"}[ds]
    alt = f"{G}/Hard drive data for paper/{ds}/segments_patient"
    ix = index(root); ix2 = index(alt)
    for r in zs:
        b = r["clip"]; c = ix.get(b, []) + ix2.get(b, [])
        if c: src[b] = (c, r["label"], r["speaker"])
os.makedirs(f"{ST}/{ds}", exist_ok=True)
rep = {"dataset": ds, "n_list": len(want), "missing": [], "multi_same": 0, "multi_diff": [], "label_mismatch": 0, "speaker_mismatch": 0}
rows, files = [], []
for r in zs:
    b = r["clip"]
    if b not in src: rep["missing"].append(b); continue
    cands, lab, spk = src[b]
    if str(lab) != str(r["label"]): rep["label_mismatch"] += 1
    if str(spk) != str(r["speaker"]): rep["speaker_mismatch"] += 1
    hs = [sha(p) for p in cands]
    if len(cands) > 1:
        if len(set(hs)) == 1: rep["multi_same"] += 1
        else: rep["multi_diff"].append({"clip": b, "paths": cands, "sha": hs})
    rows.append((f"/workspace/data/{ds}/{b}", r["label"], r["speaker"]))
    files.append((cands[0], b, hs[0]))
with open(f"{ST}/{ds}/mf_{ds}.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "label", "speaker"]); w.writerows(rows)
with open(f"{ST}/{ds}/files.tsv", "w") as fh:
    for p, b, h in files: fh.write(f"{p}\t{b}\t{h}\n")
rep["n_resolved"] = len(rows); rep["n_multi_diff"] = len(rep["multi_diff"])
json.dump(rep, open(f"{ST}/{ds}/resolve_report.json", "w"), indent=1)
print(ds, {k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()})
