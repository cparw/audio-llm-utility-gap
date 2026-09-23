#!/usr/bin/env python3
"""4b: build the interviewer-free cuts of the Pitt 468.

arm A  orig      the existing segment wav, unchanged (model truncates to 30 s)
arm B  paronly   ONLY the *PAR intervals inside the heard 30 s, concatenated
arm C  durmatch  the first  len(armB)  seconds of arm A -- duration-matched to B
                 but still containing the interviewer and the gaps.
B vs C is the clean interviewer-removal contrast at matched duration;
A vs B confounds interviewer removal with the duration drop.

A 5 ms raised-cosine fade is applied at every concatenation boundary in arm B so
that splicing does not inject broadband clicks the encoder could read.
"""
import json, os, sys, csv
import numpy as np, soundfile as sf

SEG = "/workspace/p16pod4/pitt/segments"
OUT = "/workspace/p16pod4/pitt"
SR = 16000
FADE_MS = 5.0

meta = json.load(open("/workspace/p16pod4/par_intervals.json"))
for arm in ("paronly", "durmatch"):
    os.makedirs(f"{OUT}/{arm}", exist_ok=True)

fade_n = int(SR * FADE_MS / 1000.0)
ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n)))

rows = []
for m in meta:
    src = f"{SEG}/{m['seg_wav']}"
    x, sr = sf.read(src, dtype="float32")
    assert sr == SR, (src, sr)
    if x.ndim > 1: x = x.mean(axis=1)
    # arm B: concatenate PAR intervals
    pieces = []
    for a_ms, b_ms in m["par_off"]:
        a, b = int(a_ms * SR / 1000), int(b_ms * SR / 1000)
        a, b = max(0, a), min(len(x), b)
        if b - a < fade_n * 2: 
            if b > a: pieces.append(x[a:b].copy())
            continue
        seg = x[a:b].copy()
        seg[:fade_n] *= ramp
        seg[-fade_n:] *= ramp[::-1]
        pieces.append(seg)
    if not pieces:
        rows.append(dict(clip=m["clip"], ok=0, par_s=0.0, note="no PAR interval")); continue
    B = np.concatenate(pieces)
    # arm C: first len(B) samples of the ORIGINAL (still has interviewer+gaps)
    C = x[:len(B)].copy()
    sf.write(f"{OUT}/paronly/{m['clip']}", B, SR)
    sf.write(f"{OUT}/durmatch/{m['clip']}", C, SR)
    rows.append(dict(clip=m["clip"], ok=1, par_s=round(len(B)/SR, 3),
                     orig_s=round(len(x)/SR, 3), heard_s=round(min(len(x), 30*SR)/SR, 3),
                     inv_ms=m["inv_ms"], n_par=len(m["par_off"])))

ok = [r for r in rows if r["ok"]]
d = np.array([r["par_s"] for r in ok])
print(f"built {len(ok)}/{len(meta)} clips for arms paronly + durmatch")
print(f"  paronly duration: mean {d.mean():.2f}s median {np.median(d):.2f}s min {d.min():.2f}s max {d.max():.2f}s")
for t in (5, 10, 15):
    print(f"  under {t}s of participant speech: {(d<t).sum()} clips ({100*(d<t).mean():.1f}%)")

# manifests: path,label,speaker  (+ set for the arm split)
for arm, base in (("orig", SEG), ("paronly", f"{OUT}/paronly"), ("durmatch", f"{OUT}/durmatch")):
    with open(f"{OUT}/mf_{arm}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "speaker", "set", "clip"])
        for m in meta:
            p = f"{base}/{m['clip'] if arm!='orig' else m['seg_wav']}"
            if not os.path.exists(p): continue
            w.writerow([p, m["label"], m["speaker"], m["set"], m["clip"]])
    n = sum(1 for _ in open(f"{OUT}/mf_{arm}.csv")) - 1
    print(f"  manifest mf_{arm}.csv : {n} rows")

json.dump(rows, open(f"{OUT}/cut_report.json", "w"), indent=1)
print("CUT DONE")
