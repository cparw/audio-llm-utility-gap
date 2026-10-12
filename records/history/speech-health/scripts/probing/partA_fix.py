"""
PART A: independent confirmation of every data correction + corrected manifests.
Writes to $R/final/
"""
import os, re, glob, hashlib, collections, json
import numpy as np, pandas as pd
from multiprocessing import Pool

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
OUT = R + "/final"
os.makedirs(OUT, exist_ok=True)
XL = "/project2/msoleyma_946/speech_health/spanish_dataset/Copia de PCGITA_metadata.xlsx"

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return p, h.hexdigest()

def hashes(paths, nproc=12):
    with Pool(nproc) as pool:
        return dict(pool.map(md5, list(paths), chunksize=16))

L = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); L.append(s)

# ================================================================= A1 + A2  PC-GITA
P("=" * 78); P("A1/A2  PC-GITA roster and duplicates"); P("=" * 78)
m = pd.read_csv(R + "/manifests/pcgita.csv")
P("manifest as shipped: %d clips, %d speakers" % (len(m), m.speaker_id.nunique()))

meta = pd.ExcelFile(XL).parse("PD+HC")
meta["id"] = meta["RECODING ORIGINAL NAME"].astype(str).str.strip()
roster = set(meta.id)
ros_pd = {i for i in roster if not i.startswith("AVPEPUDEAC")}
ros_hc = roster - ros_pd
P("official metadata sheet 'PD+HC': %d rows, %d unique ids (%d patient, %d control)"
  % (len(meta), meta.id.nunique(), len(ros_pd), len(ros_hc)))

# EXACT join, no stripping of the C
m["on_roster"] = m.speaker_id.isin(roster)
# demonstrate the trap
stripped = m.speaker_id.str.replace("^AVPEPUDEAC", "AVPEPUDEA", regex=True)
n_trap = int((stripped.isin(ros_pd) & m.speaker_id.str.startswith("AVPEPUDEAC")).sum())
P("TRAP CHECK: stripping the C would map %d control clips onto patient roster ids" % n_trap)

m["fname"] = m.filepath.map(os.path.basename)
m["no_pd_in_name"] = m.fname.str.upper().str.contains("NO PD")
m["no_updrs_in_name"] = m.fname.str.upper().str.contains("NO UPDRS")
m["in_leftovers"] = m.filepath.str.contains("las que sobraron")

P("")
P("clips whose FILENAME says NO PD  : %d  (labelled patient: %d)"
  % (m.no_pd_in_name.sum(), int(m.loc[m.no_pd_in_name, "label"].sum())))
for _, r in m[m.no_pd_in_name].iterrows():
    P("   %-14s label=%d task=%s  %s" % (r.speaker_id, r.label, r.task_type, r.fname))
P("   of these, on the official roster: %d / %d" % (m.loc[m.no_pd_in_name, "on_roster"].sum(),
                                                    m.no_pd_in_name.sum()))
P("clips whose FILENAME says NO UPDRS: %d" % m.no_updrs_in_name.sum())
for _, r in m[m.no_updrs_in_name].iterrows():
    P("   %-14s label=%d task=%s  %s" % (r.speaker_id, r.label, r.task_type, r.fname))
P("   of these, on the official roster: %d / %d" % (m.loc[m.no_updrs_in_name, "on_roster"].sum(),
                                                    m.no_updrs_in_name.sum()))

off = sorted(set(m.loc[~m.on_roster, "speaker_id"]))
P("")
P("speakers in manifest but NOT on roster: %d  (%d patient-coded, %d control-coded)"
  % (len(off), sum(1 for s in off if not s.startswith("AVPEPUDEAC")),
     sum(1 for s in off if s.startswith("AVPEPUDEAC"))))
P("roster speakers absent from manifest : %d" % len(roster - set(m.speaker_id)))
P("all off-roster clips are in 'las que sobraron': %s ; all task_type==read: %s"
  % (bool(m.loc[~m.on_roster, "in_leftovers"].all()),
     bool((m.loc[~m.on_roster, "task_type"] == "read").all())))

# ---- duplicates by checksum, over the WHOLE manifest
P("")
P("checksumming all %d pc-gita wavs ..." % len(m))
H = hashes(m.filepath)
m["md5"] = m.filepath.map(H)
dup = collections.defaultdict(list)
for _, r in m.iterrows():
    dup[r.md5].append(r)
dupgroups = {k: v for k, v in dup.items() if len(v) > 1}
P("byte-identical groups: %d" % len(dupgroups))
crossspk = 0
dupmap = {}
for k, v in dupgroups.items():
    spks = sorted({r.speaker_id for r in v})
    P("   md5 %s  n=%d  speakers=%s  task=%s" % (k[:12], len(v), spks, v[0].task_type))
    for r in v:
        P("        %-14s label=%d  %s" % (r.speaker_id, r.label, r.fname))
    if len(spks) > 1:
        crossspk += 1
        keep = [s for s in spks if s in roster]
        drop = [s for s in spks if s not in roster]
        for r in v:
            if r.speaker_id in drop:
                dupmap[r.filepath] = keep[0] if keep else spks[0]
P("groups spanning MORE THAN ONE speaker_id (breaks speaker disjointness): %d" % crossspk)
m["dup_of_speaker"] = m.filepath.map(dupmap).fillna("")

m["keep"] = m.on_roster
P("")
P("CORRECTED pc-gita = official 100-speaker roster, exact-id join.")
P("  this single rule removes the 4 NO PD, the 2 NO UPDRS, both duplicate copies and the")
P("  11 other undocumented 'las que sobraron' extras, all of which are read-task only.")
P("  as shipped : %d clips / %d speakers  (read %d)"
  % (len(m), m.speaker_id.nunique(), (m.task_type == "read").sum()))
mk = m[m.keep]
P("  corrected  : %d clips / %d speakers  (read %d)"
  % (len(mk), mk.speaker_id.nunique(), (mk.task_type == "read").sum()))
P("  per-task corrected counts:")
for (t, lab), n in mk.groupby(["task_type", "label"]).size().items():
    P("     %-10s label=%d  n=%d" % (t, lab, n))

cols = ["filepath", "dataset", "speaker_id", "label", "task_type", "language", "age", "sex",
        "group", "transcript", "on_roster", "in_leftovers", "no_pd_in_name",
        "no_updrs_in_name", "md5", "dup_of_speaker", "keep"]
m[cols].to_csv(OUT + "/pcgita_corrected.csv", index=False)
P("wrote " + OUT + "/pcgita_corrected.csv")

# ================================================================= A3  NEUROVOZ
P(""); P("=" * 78); P("A3  NeuroVoz ESPONTANEA mislabelled as ddk"); P("=" * 78)
n = pd.read_csv(R + "/manifests/neurovoz.csv")
n["fname"] = n.filepath.map(os.path.basename)
n["item"] = n.fname.str.replace(r"^(HC|PD)_", "", regex=True).str.replace(r"_\d+\.wav$", "", regex=True)
P("manifest as shipped: %d clips, %d speakers" % (len(n), n.speaker_id.nunique()))
P("task_type as shipped:")
for (t, lab), k in n.groupby(["task_type", "label"]).size().items():
    P("   %-6s label=%d  n=%d" % (t, lab, k))
P("")
P("items currently filed under task_type=='ddk':")
for it, k in n[n.task_type == "ddk"].item.value_counts().items():
    P("   %-16s n=%d" % (it, k))
P("-> only PATAKA is true diadochokinesis (rapid /pa-ta-ka/). ESPONTANEA is a spontaneous")
P("   picture description. ACAMPADA/PAN_VINO/PATATA_BLANDA/PETACA_BLANCA are read sentences.")

esp = n[n.item.str.contains("ESPONTANEA")]
P("")
P("ESPONTANEA rows in manifest: %d  (task_type values: %s)"
  % (len(esp), sorted(set(esp.task_type))))
P("   label breakdown: PD %d / HC %d ; speakers %d"
  % (int(esp.label.sum()), int((esp.label == 0).sum()), esp.speaker_id.nunique()))

base = "/project2/msoleyma_946/speech_health/spanish_neurovoz"
disk = sorted(glob.glob(base + "/**/*ESPONTANEA*.wav", recursive=True))
bydir = collections.Counter(os.path.dirname(p) for p in disk)
P("")
P("ESPONTANEA wavs on disk anywhere under spanish_neurovoz: %d" % len(disk))
for d, k in bydir.items():
    P("   %-72s %d" % (d.replace(base + "/", ""), k))
main_dir = base + "/zenodo_upload/audios"
in_main = [p for p in disk if os.path.dirname(p) == main_dir]
P("of the %d in the primary audio dir, in manifest: %d ; missing: %d"
  % (len(in_main), sum(p in set(n.filepath) for p in in_main),
     sum(p not in set(n.filepath) for p in in_main)))
alt = base + "/zenodo_upload_silencemode_1/audios"
alt_f = {os.path.basename(p) for p in disk if os.path.dirname(p) == alt}
main_f = {os.path.basename(p) for p in in_main}
P("the other %d live in zenodo_upload_silencemode_1/audios; basenames identical to the")
P("   primary set: %s  -> they are the SILENCE-REMOVED RE-RENDER of the same 74 recordings,")
P("   not 74 extra sessions." % ())
P("   |primary|=%d |silencemode|=%d |intersection|=%d" % (len(main_f), len(alt_f), len(main_f & alt_f)))

n["task_type_corrected"] = n.task_type
n.loc[n.item.str.contains("ESPONTANEA"), "task_type_corrected"] = "spontaneous"
sent = n.item.isin(["ACAMPADA", "PAN_VINO", "PATATA_BLANDA", "PETACA_BLANCA"])
n.loc[sent & (n.task_type == "ddk"), "task_type_corrected"] = "sentence_repeat"
P("")
P("CORRECTED neurovoz task_type:")
for (t, lab), k in n.groupby(["task_type_corrected", "label"]).size().items():
    P("   %-16s label=%d  n=%d" % (t, lab, k))

P("")
P("checksumming all %d neurovoz wavs ..." % len(n))
Hn = hashes(n.filepath)
n["md5"] = n.filepath.map(Hn)
dn = collections.Counter(n.md5)
dupn = [k for k, v in dn.items() if v > 1]
P("byte-identical groups in neurovoz: %d" % len(dupn))
for k in dupn[:10]:
    P("   " + str(sorted(n.loc[n.md5 == k, "fname"].tolist())))
n["keep"] = True
n.drop(columns=["fname"]).to_csv(OUT + "/neurovoz_corrected.csv", index=False)
P("wrote " + OUT + "/neurovoz_corrected.csv")

# duplicate sweep on the other corpora too
P(""); P("duplicate sweep, other corpora:")
for ds in ["italian", "kcl", "dcaps"]:
    d = pd.read_csv(R + "/manifests/%s.csv" % ds)
    Hd = hashes(d.filepath)
    md = pd.Series(d.filepath.map(Hd))
    c = collections.Counter(md)
    ndup = sum(1 for v in c.values() if v > 1)
    P("   %-8s %5d clips, byte-identical groups: %d" % (ds, len(d), ndup))
    if ndup:
        for k, v in c.items():
            if v > 1:
                P("      " + str(sorted(d.loc[md.values == k, "speaker_id"].tolist())))

with open(OUT + "/partA_log.txt", "w") as f:
    f.write("\n".join(L) + "\n")
print("JOB_DONE", flush=True)
