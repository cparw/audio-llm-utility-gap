"""ITALIAN CONFOUND step 2: design table. encoder toolchain x group x date x task."""
import csv, json, os, re
from collections import Counter, defaultdict
R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = f"{R}/final"

H = {r["path"]: r for r in csv.DictReader(open(f"{OUT}/italian_headers.csv"))}
M = list(csv.DictReader(open(f"{R}/manifests/italian.csv")))
print(f"manifest rows {len(M)}   header rows {len(H)}")
miss = [r for r in M if r["filepath"] not in H]
print(f"manifest rows with no header parse: {len(miss)}")

def enc_class(h):
    ch = h["chunks"]
    ex = h.get("extra_chunks", "")
    if "bext" in ch:
        return "BWF_bext(ProTools/Audition)"
    if "Mixcraft" in ex:
        return "Mixcraft_7.7"
    if "mp3cut.net" in ex:
        return "mp3cut.net(MP3 transcode)"
    if "id3" in ch or "LIST" in ch:
        return "LIST/id3_other"
    return "bare_fmt_data"

rows = []
for r in M:
    h = H.get(r["filepath"])
    if not h:
        continue
    d = dict(filepath=r["filepath"], label=r["label"], task=r["task_type"],
             spk=r["speaker_id"], sr=int(h["sample_rate"]),
             group=h["group_dir"], date=h["fn_date"], time=h["fn_time"],
             fntask=h["fn_task"], sex=h["fn_sex"], byy=h["fn_birthyy"],
             enc=enc_class(h), datefmt=h["fn_datefmt"],
             trailing=h["trailing_bytes"], subdir=h["subdir"])
    rows.append(d)
with open(f"{OUT}/italian_design.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def cross(a, b, sub=None, title=""):
    rr = [r for r in rows if sub is None or sub(r)]
    c = defaultdict(Counter)
    for r in rr:
        c[r[a]][r[b]] += 1
    cols = sorted({r[b] for r in rr}, key=str)
    print(f"\n### {title or (a+' x '+b)}   (n={len(rr)})")
    print(f"{'':40s}" + "".join(f"{str(x)[:13]:>15}" for x in cols))
    for k in sorted(c, key=str):
        print(f"{str(k)[:38]:40s}" + "".join(f"{c[k].get(x,0):>15}" for x in cols))

cross("enc", "label", title="ENCODER TOOLCHAIN x LABEL (0=control 1=patient)")
cross("enc", "group", title="ENCODER x GROUP DIR")
cross("date", "label", title="RECORDING DATE x LABEL")
cross("task", "label", title="TASK x LABEL")
cross("sr", "label", title="SAMPLE RATE x LABEL")
for t in ["read", "vowel", "ddk"]:
    cross("enc", "label", sub=lambda r, t=t: r["task"] == t, title=f"ENCODER x LABEL | task={t}")
    cross("sr", "label", sub=lambda r, t=t: r["task"] == t, title=f"SR x LABEL | task={t}")
cross("fntask", "task", title="FILENAME TASK CODE x MANIFEST TASK")

# speaker-level: is date perfectly nested in label?
sd = defaultdict(set)
for r in rows:
    sd[r["date"]].add(r["label"])
print("\n### is any recording DATE shared between patients and controls?")
for d in sorted(sd):
    print(f"   {d}   labels present: {sorted(sd[d])}")
mixed = [d for d in sd if len(sd[d]) > 1]
print(f"   DATES WITH BOTH LABELS: {mixed if mixed else 'NONE -- group is perfectly nested in session'}")

ed = defaultdict(set)
for r in rows:
    ed[r["enc"]].add(r["label"])
print("\n### is any ENCODER shared between patients and controls?")
for d in sorted(ed):
    print(f"   {d:34s} labels present: {sorted(ed[d])}")

# per-speaker table
spk = {}
for r in rows:
    spk.setdefault(r["spk"], dict(label=r["label"], dates=set(), enc=set(), sr=set(), n=0))
    s = spk[r["spk"]]; s["dates"].add(r["date"]); s["enc"].add(r["enc"]); s["sr"].add(r["sr"]); s["n"] += 1
print(f"\n### speakers: {len(spk)}  (patients {sum(1 for v in spk.values() if v['label']=='1')}, "
      f"controls {sum(1 for v in spk.values() if v['label']=='0')})")
multi = [k for k, v in spk.items() if len(v["dates"]) > 1]
print(f"   speakers recorded on >1 date: {len(multi)}")
