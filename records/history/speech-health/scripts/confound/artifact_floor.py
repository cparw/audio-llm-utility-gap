"""
Artifact floor comparison.

For each dataset/task we now have two numbers:
  (a) the encoder probing AUC (what we reported as "disease detection")
  (b) the AUC obtainable from RECORDING PROPERTIES ALONE, no speech content
      (duration, loudness, spectral stats, SNR) and, where available, age/sex

(b) is an ARTIFACT FLOOR. Any honest claim of disease detection has to clear it.
This prints the gap, which is the part of the result that is plausibly the disease.
"""
import csv, json, os
R = "/project2/msoleyma_946/speech_health/results_chaitanya"

def peak(path):
    if not os.path.exists(path): return None
    rows = list(csv.DictReader(open(path)))
    if not rows: return None
    keys = rows[0].keys()
    col = next((c for c in ("auc_mean", "auc", "roc_auc") if c in keys), None)
    if col is None: return None
    best = max(rows, key=lambda r: float(r.get(col, "nan") or "nan"))
    return round(float(best[col]), 3)

ENC = {  # dataset/task -> probing csv produced earlier
    ("italian", "read"):  "italian_read_allHC.csv",
    ("kcl", "read"):      "kcl_encoder_read.csv",
    ("neurovoz", "read"): "neurovoz_read.csv",
    ("pcgita", "read"):   "pcgita_read.csv",
    ("italian", "vowel"): "italian_vowel_allHC.csv",
    ("neurovoz", "vowel"):"neurovoz_vowel.csv",
    ("pcgita", "vowel"):  "pcgita_vowel.csv",
}

rows_out = []
for task in ("read", "vowel"):
    ap = f"{R}/audit/confound_audit_{task}.json"
    if not os.path.exists(ap):
        print(f"[skip] missing {ap}"); continue
    audit = json.load(open(ap))
    for ds, a in audit.items():
        enc = peak(f"{R}/probing/{ENC.get((ds,task),'')}")
        floor_rec = a.get("AUC_from_recording_properties_only")
        floor_meta = a.get("AUC_from_metadata_only_age_sex")
        floor_all = a.get("AUC_recording_plus_metadata")
        floor = max([v for v in (floor_rec, floor_meta, floor_all)
                     if isinstance(v, (int, float)) and v == v] or [float("nan")])
        gap = round(enc - floor, 3) if (enc is not None and floor == floor) else None
        rows_out.append(dict(dataset=ds, task=task, encoder_AUC=enc,
                             artifact_floor=round(floor, 3) if floor == floor else None,
                             floor_recording=floor_rec, floor_age_sex=floor_meta,
                             gap_above_floor=gap,
                             shuffle_control=a.get("AUC_label_shuffled_control")))

hdr = ("dataset", "task", "encoder_AUC", "artifact_floor", "floor_recording",
       "floor_age_sex", "gap_above_floor", "shuffle_control")
print("=" * 104)
print("ENCODER PROBE vs ARTIFACT FLOOR  (floor = best AUC from recording properties / age-sex alone)")
print("=" * 104)
print("  " + "".join(f"{h:>17}" for h in hdr))
for r in rows_out:
    print("  " + "".join(f"{str(r[h]):>17}" for h in hdr))

os.makedirs(f"{R}/audit", exist_ok=True)
with open(f"{R}/audit/artifact_floor.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows_out)
print(f"\nwrote {R}/audit/artifact_floor.csv")
print("\nRead: gap_above_floor is the part of the 'disease' result NOT explained by how")
print("the recordings differ. A gap near zero means the number is an artifact.")
