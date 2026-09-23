#!/usr/bin/env python3
"""Hash decoded PCM frames (plus rate/channels/width) of wav files. Usage: wavhash.py DIR OUT.csv [glob]"""
import sys, os, wave, hashlib, glob, csv
d, out = sys.argv[1], sys.argv[2]; pat = sys.argv[3] if len(sys.argv) > 3 else "*.wav"
rows = []
for f in sorted(glob.glob(os.path.join(d, pat))):
    w = wave.open(f); fr = w.readframes(w.getnframes())
    rows.append([os.path.basename(f), w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes(), hashlib.sha256(fr).hexdigest()])
with open(out, "w", newline="") as fh:
    cw = csv.writer(fh); cw.writerow(["file", "sr", "nch", "sw", "nframes", "sha256"]); cw.writerows(rows)
print(len(rows), "hashed")
