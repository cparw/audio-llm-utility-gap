#!/usr/bin/env python3
"""Cut the conflict/agreement segments from the cookie audio.
Prefers the uncompressed wav where it exists, falls back to mp3.
Output: 16 kHz mono wav per segment, ready for the model scorers."""
import csv, os, subprocess

rows = list(csv.DictReader(open("pitt_conflict_manifest.csv")))
os.makedirs("segments", exist_ok=True)
done = fail = 0
for r in rows:
    grp, sess = r["grp"], r["session"]
    src = f"Pitt/{grp}/cookie/0wav/{sess}.wav"
    if not os.path.exists(src):
        src = f"Pitt/{grp}/cookie/{sess}.mp3"
    if not os.path.exists(src):
        fail += 1; continue
    a = int(r["start_ms"]) / 1000.0
    b = int(r["end_ms"]) / 1000.0
    out = f"segments/{r['set']}_{grp[:3]}_{sess}_{r['start_ms']}.wav"
    r["segment_path"] = out
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        done += 1; continue
    p = subprocess.run(["ffmpeg","-v","quiet","-y","-ss",f"{a:.2f}","-to",f"{b:.2f}",
        "-i",src,"-ac","1","-ar","16000",out], timeout=120)
    if os.path.exists(out) and os.path.getsize(out) > 10000: done += 1
    else: fail += 1
with open("pitt_conflict_manifest.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"cut {done} segments, {fail} failed")
