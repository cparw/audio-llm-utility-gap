#!/usr/bin/env python3
"""Emit PAR intervals as OFFSETS (ms) inside each existing segment wav.
segment wav = source[start_ms:end_ms]; the model hears its first 30 s.
So offset = par_start - start_ms, clipped to [0, 30000]."""
import csv, os, re, json, sys
ROOT = "<local data dir>/DementiaBank"
WIN_MS = 30000
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cha_overlap import parse_cha, merge, clip_to, total  # reuse the SAME parser

rows = list(csv.DictReader(open(f"{ROOT}/pitt_conflict_manifest.csv")))
cache = {}
out = []
for r in rows:
    grp, sess = r["grp"], r["session"]
    cha = f"{ROOT}/transcripts/{grp}/cookie/{sess}.cha"
    if cha not in cache: cache[cha] = parse_cha(cha)
    iv, _ = cache[cha]
    s, e = int(r["start_ms"]), int(r["end_ms"])
    we = min(e, s + WIN_MS)
    par = clip_to(merge(iv["PAR"]), s, we)
    inv = clip_to(merge(iv["INV"]), s, we)
    out.append(dict(
        clip=os.path.basename(r["segment_path"]),
        speaker=("Control" if grp == "Control" else "Dementia") + r["spk"],
        spk_raw=r["spk"], grp=grp, label=int(r["label"]), set=r["set"],
        seg_wav=os.path.basename(r["segment_path"]),
        win_ms=we - s,
        par_off=[[a - s, b - s] for a, b in par],
        inv_off=[[a - s, b - s] for a, b in inv],
        par_ms=total(par), inv_ms=total(inv),
    ))
json.dump(out, open(sys.argv[1], "w"))
print("clips", len(out))
print("total par ms", sum(o["par_ms"] for o in out))
print("clips with inv", sum(1 for o in out if o["inv_ms"] > 0))
print("speakers", len(set(o["speaker"] for o in out)))
