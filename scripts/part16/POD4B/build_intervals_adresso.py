#!/usr/bin/env python3
"""POD4B step 1, ADReSSo: resolve the scored windows to the RELEASED segmentation.

Source of truth is the ADReSSo21 challenge distribution's own speaker-diarised
segment lists, ADReSSo21/diagnosis/train/segmentation/{ad,cn}/<spk>.csv, columns
"",speaker,begin,end with speaker in {PAR, INV} and begin/end in milliseconds.
These ship inside ADReSSo21-diagnosis-train.tgz. NO diariser is run here; if a
clip has no released segmentation file it is dropped, never imputed.

The scored clip (arm A) is segments_patient/<spk>.wav = ffmpeg -ss start_s -t 30
off the ADReSS-M English train mp3, so the window in SOURCE time is
[start_ms, start_ms + clip_ms] and an interval at source time t sits at
t - start_ms inside the clip.
"""
import csv, os, json, sys
import numpy as np, soundfile as sf

D    = "<local data dir>/paper work/Hard drive data for paper"
MF   = f"{D}/adresso/adresso_manifest_patient.csv"
SEG  = f"{D}/adresso/segments_patient"
SEGD = "<local data dir>/Desktop/release/edaic_rerun/part16/POD4B/adresso_seg/ADReSSo21/diagnosis/train/segmentation"

def seg_file(spk):
    for sub in ("ad", "cn"):
        p = f"{SEGD}/{sub}/{spk}.csv"
        if os.path.exists(p): return p
    return None

def read_seg(path):
    out = {"PAR": [], "INV": []}
    other = 0
    for r in csv.DictReader(open(path)):
        who = r["speaker"].strip().strip('"').upper()
        a, b = int(float(r["begin"])), int(float(r["end"]))
        if who in out: out[who].append((a, b))
        else: other += 1
    return out, other

def merge(iv):
    if not iv: return []
    iv = sorted(iv); m = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
        else: m.append([a, b])
    return [tuple(x) for x in m]

def clip_to(iv, lo, hi):
    out = []
    for a, b in iv:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2: out.append((a2, b2))
    return out

def total(iv): return sum(b - a for a, b in iv)

rows = list(csv.DictReader(open(MF)))
meta, rep, nofile, nowav = [], [], [], []
other_tiers = 0
for r in rows:
    spk, label = r["spk"], int(r["label"])
    sf_path, wav = seg_file(spk), f"{SEG}/{spk}.wav"
    if sf_path is None: nofile.append(spk); continue
    if not os.path.exists(wav): nowav.append(spk); continue
    iv, oth = read_seg(sf_path); other_tiers += oth
    info = sf.info(wav); clip_ms = int(round(info.frames / info.samplerate * 1000))
    start_ms = int(round(float(r["start_s"]) * 1000))
    end_ms = start_ms + clip_ms
    par_all, inv_all = merge(iv["PAR"]), merge(iv["INV"])
    par_w, inv_w = clip_to(par_all, start_ms, end_ms), clip_to(inv_all, start_ms, end_ms)
    span = end_ms - start_ms
    par_ms, inv_ms = total(par_w), total(inv_w)
    meta.append(dict(clip=f"{spk}.wav", seg_wav=f"{spk}.wav", speaker=spk, label=label,
                     set="adresso_train", start_ms=start_ms, end_ms=end_ms,
                     par_off=[[a - start_ms, b - start_ms] for a, b in par_w],
                     inv_ms=inv_ms, par_ms=par_ms, span_ms=span))
    rep.append(dict(clip=spk, label=label, start_ms=start_ms, clip_ms=clip_ms, span_ms=span,
                    seg_last_ms=max([b for _, b in par_all + inv_all] or [0]),
                    par_ms_win=par_ms, inv_ms_win=inv_ms, gap_ms_win=span - par_ms - inv_ms,
                    inv_share_win=round(inv_ms / span, 5) if span else 0.0,
                    par_share_win=round(par_ms / span, 5) if span else 0.0,
                    n_par_utt=len(par_w), n_inv_utt=len(inv_w),
                    n_par_all=len(par_all), n_inv_all=len(inv_all)))

print(f"manifest rows                  : {len(rows)}")
print(f"resolved to RELEASED segmentation: {len(meta)}")
print(f"no released segmentation file  : {len(nofile)}  (dropped, not imputed)")
print(f"no scored wav on disk          : {len(nowav)} {nowav[:5]}")
print(f"segment rows with a tier other than PAR/INV: {other_tiers}")
lab = {}
for m in meta: lab[m["label"]] = lab.get(m["label"], 0) + 1
print(f"label distribution of the resolved set: {lab}")
cm = np.array([x["clip_ms"] for x in rep], float)
sl = np.array([x["seg_last_ms"] for x in rep], float)
st = np.array([x["start_ms"] for x in rep], float)
print(f"alignment: window end (start+clip) vs last segmentation stamp -> "
      f"windows whose end exceeds the last stamp: {((st+cm) > sl).sum()}/{len(rep)}")
inv = np.array([x["inv_ms_win"] for x in rep], float)
par = np.array([x["par_ms_win"] for x in rep], float)
spn = np.array([x["span_ms"] for x in rep], float)
print()
print("=== interviewer inside the scored window (arm A) ===")
print(f"  INV s   : mean {inv.mean()/1000:.3f}  median {np.median(inv)/1000:.3f}  max {inv.max()/1000:.3f}")
print(f"  INV share: mean {(inv/spn).mean():.4f}  median {np.median(inv/spn):.4f}  max {(inv/spn).max():.4f}")
print(f"  windows with ANY INV speech : {(inv>0).sum()} / {len(rep)}  ({100*(inv>0).mean():.1f}%)")
print(f"  windows with >5 s INV       : {(inv>5000).sum()} / {len(rep)}")
print()
ps = par/1000.0
print("=== PAR-only concatenated duration (arm B) ===")
print(f"  mean {ps.mean():.2f}s  median {np.median(ps):.2f}s  min {ps.min():.2f}s  max {ps.max():.2f}s")
for t in (5,10,15,20):
    print(f"  < {t:2d} s of participant speech : {(ps<t).sum():3d} windows ({100*(ps<t).mean():.1f}%)")

json.dump(meta, open(sys.argv[1], "w"), indent=1)
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rep[0].keys())); w.writeheader(); w.writerows(rep)
print(f"\nwrote {sys.argv[1]} and {sys.argv[2]}")
