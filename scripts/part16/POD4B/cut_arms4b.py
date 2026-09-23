#!/usr/bin/env python3
"""POD4B: build the interviewer-free cuts for ADReSS-2020 and ADReSSo.

Exactly the Pitt 4b construction (POD4/cut_arms.py), re-pointed at a corpus:
  arm A  orig      the existing scored wav, unchanged (model truncates to 30 s)
  arm B  paronly   ONLY the participant intervals inside the scored window, concatenated
  arm C  durmatch  the first len(armB) seconds of arm A -- duration-matched to B but
                   still containing the interviewer and the gaps.
B vs C is the clean interviewer-removal contrast at matched duration; A vs B
confounds interviewer removal with the duration drop.

A 5 ms raised-cosine fade is applied at every concatenation boundary in arm B so
that splicing does not inject broadband clicks the encoder could read.

usage: cut_arms4b.py CORPUS   (CORPUS in {adress2020, adresso})
"""
import json, os, sys, csv
import numpy as np, soundfile as sf

CORPUS = sys.argv[1]
ROOT = "/workspace/p16pod4b"
SEG  = f"{ROOT}/src/{CORPUS}_orig"
OUT  = f"{ROOT}/cuts/{CORPUS}"
SR = 16000
FADE_MS = 5.0

meta = json.load(open(f"{ROOT}/par_intervals_{CORPUS}.json"))
for arm in ("paronly", "durmatch"):
    os.makedirs(f"{OUT}/{arm}", exist_ok=True)

fade_n = int(SR * FADE_MS / 1000.0)
ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n)))

rows, kept = [], []
for m in meta:
    src = f"{SEG}/{m['seg_wav']}"
    if not m["par_off"]:
        rows.append(dict(clip=m["clip"], ok=0, par_s=0.0, note="no participant interval in window")); continue
    if not os.path.exists(src):
        rows.append(dict(clip=m["clip"], ok=0, par_s=0.0, note="source wav missing")); continue
    x, sr = sf.read(src, dtype="float32")
    assert sr == SR, (src, sr)
    if x.ndim > 1: x = x.mean(axis=1)
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
        rows.append(dict(clip=m["clip"], ok=0, par_s=0.0, note="all intervals fell outside the clip")); continue
    B = np.concatenate(pieces)
    C = x[:len(B)].copy()                 # arm C: same length as B, interviewer KEPT
    sf.write(f"{OUT}/paronly/{m['clip']}", B, SR)
    sf.write(f"{OUT}/durmatch/{m['clip']}", C, SR)
    kept.append(m)
    rows.append(dict(clip=m["clip"], ok=1, par_s=round(len(B)/SR, 3),
                     orig_s=round(len(x)/SR, 3), heard_s=round(min(len(x), 30*SR)/SR, 3),
                     inv_ms=m["inv_ms"], n_par=len(m["par_off"])))

ok = [r for r in rows if r["ok"]]
d = np.array([r["par_s"] for r in ok]); o = np.array([r["orig_s"] for r in ok])
print(f"[{CORPUS}] built {len(ok)}/{len(meta)} clips for arms paronly + durmatch")
print(f"  arm A orig     duration: mean {o.mean():.2f}s median {np.median(o):.2f}s min {o.min():.2f}s max {o.max():.2f}s")
print(f"  arm B paronly  duration: mean {d.mean():.2f}s median {np.median(d):.2f}s min {d.min():.2f}s max {d.max():.2f}s")
print(f"  arm C durmatch duration: identical to arm B by construction")
print(f"  mean seconds removed A->B: {(o-d).mean():.2f}s  median {np.median(o-d):.2f}s")
for t in (5, 10, 15):
    print(f"  under {t}s of participant speech: {(d<t).sum()} clips ({100*(d<t).mean():.1f}%)")

for arm, base in (("orig", SEG), ("paronly", f"{OUT}/paronly"), ("durmatch", f"{OUT}/durmatch")):
    p_out = f"{ROOT}/mf_{CORPUS}_{arm}.csv"
    with open(p_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "speaker", "set", "clip"])
        for m in kept:
            p = f"{base}/{m['clip']}"
            if not os.path.exists(p): continue
            w.writerow([p, m["label"], m["speaker"], m["set"], m["clip"]])
    n = sum(1 for _ in open(p_out)) - 1
    print(f"  manifest mf_{CORPUS}_{arm}.csv : {n} rows")

json.dump(rows, open(f"{ROOT}/cut_report_{CORPUS}.json", "w"), indent=1)
print(f"CUT DONE {CORPUS}")
