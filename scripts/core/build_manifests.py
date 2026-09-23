"""Build the seven clip lists on the pod, exactly the lists behind every Qwen2-Audio number in the paper.
Emits mf_<ds>.csv with path,label,speaker and, for Pitt, a set column from the conflict manifest."""
import csv, os, sys, collections
D = "/workspace/data"
def index(root):
    ix = collections.defaultdict(list)
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".wav"): ix[f].append(os.path.join(d, f))
    return ix
def write(name, rows):
    cols = ["path", "label", "speaker"] + (["set"] if rows and len(rows[0]) == 4 else [])
    with open(f"/workspace/mf_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
    print(f"mf_{name}.csv  n={len(rows)}  speakers={len(set(r[2] for r in rows))}  pos={sum(int(r[1]) for r in rows)}", flush=True)
# PC-GITA: read rows minus the "las que sobraron" leftovers
ix = index(f"{D}/spanish_dataset"); rows, miss = [], 0
for r in csv.DictReader(open(f"{D}/pcgita_read.csv")):
    if "sobraron" in r["filepath"].lower(): continue
    c = ix.get(os.path.basename(r["filepath"]))
    if not c: miss += 1; continue
    rows.append((c[0], r["label"], r["speaker_id"]))
print(f"pcgita: {len(rows)} resolved, {miss} missing", flush=True); write("pcgita", rows)
# NeuroVoz: the read rows
ix = index(f"{D}/spanish_neurovoz/zenodo_upload"); rows = []
for r in csv.DictReader(open(f"{D}/neurovoz.csv")):
    if r["task_type"] != "read": continue
    c = ix.get(os.path.basename(r["filepath"]))
    if c: rows.append((c[0], r["label"], r["speaker_id"]))
write("neurovoz", rows)
# MDVR-KCL: the recut list
ix = index(f"{D}/kcl_read30b"); rows = []
for r in csv.DictReader(open(f"{D}/kcl30b_manifest.csv")):
    p = next((v for k, v in r.items() if "path" in k), None)
    c = ix.get(os.path.basename(p))
    if c: rows.append((c[0], r["label"], r.get("speaker") or r.get("speaker_id")))
write("kcl", rows)
# E-DAIC
ix = index(f"{D}/dcaps_proc"); rows = []
for r in csv.DictReader(open(f"{D}/dcaps.csv")):
    p = next((v for k, v in r.items() if "path" in k or "file" in k), None)
    c = ix.get(os.path.basename(p))
    if c: rows.append((c[0], r["label"], r.get("speaker") or r.get("speaker_id") or os.path.basename(p).split(".")[0]))
write("edaic", rows)
# Pitt, with the conflict/agreement arms
ix = index(f"{D}/clips_for_af3/pitt468"); rows = []
for r in csv.DictReader(open(f"{D}/clips_for_af3/pitt468_manifest.csv")):
    p = next((v for k, v in r.items() if "path" in k), None)
    c = ix.get(os.path.basename(p))
    if not c: continue
    b = os.path.basename(p)
    arm = "conflict" if b.startswith("conflict") else ("agreement" if b.startswith("agreement") else r.get("set", ""))
    rows.append((c[0], r["label"], r.get("speaker") or r.get("speaker_id"), arm))
write("pitt", rows)
print("  pitt arms:", collections.Counter(r[3] for r in rows), flush=True)
# ADReSSo and ADReSS-2020
for name, root, man in (("adresso", f"{D}/adresso237", f"{D}/adresso237/adresso237_manifest.csv"),
                        ("adress2020", f"{D}/adress2020_for_af3", None)):
    man = man or next(f"{root}/{f}" for f in os.listdir(root) if f.endswith(".csv"))
    ix = index(root); rows = []
    for r in csv.DictReader(open(man)):
        p = next((v for k, v in r.items() if "path" in k), None)
        c = ix.get(os.path.basename(p))
        if c: rows.append((c[0], r["label"], r.get("speaker") or r.get("speaker_id")))
    write(name, rows)
