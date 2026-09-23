#!/usr/bin/env python3
"""POD4B step 1, ADReSS-2020: resolve the 156 scored windows to CHAT timings.

The scored clip (arm A) is segments_patient/<spk>.wav = ffmpeg -ss start_s -t 30
off the Full_wave_enhanced_audio source. So the window in SOURCE time is
[start_ms, start_ms + 30000], and an interval at source time t sits at
t - start_ms inside the clip. That offset form is what cut_arms.py consumes.

*PAR: participant, *INV: investigator. ADReSS-2020 is fully timed (PART 5: 0
untimed utterances in all 156 files) -- re-verified here, not assumed.
"""
import csv, os, re, json, sys
import numpy as np, soundfile as sf

D   = "<local data dir>/paper work/Hard drive data for paper"
B   = f"{D}/DementiaBank/challenges/ADReSS-2020"
MF  = f"{D}/adress2020/adress2020_manifest_patient.csv"
SEG = f"{D}/adress2020/segments_patient"
WIN_MS = 30000

def cha_path(spk, split, label):
    if split == "train":
        sub = "cd" if str(label) == "1" else "cc"
        return f"{B}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/transcription/{sub}/{spk}.cha"
    return f"{B}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/transcription/{spk}.cha"

def src_path(spk, split, label):
    if split == "train":
        sub = "cd" if str(label) == "1" else "cc"
        return f"{B}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/{sub}/{spk}.wav"
    return f"{B}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/Full_wave_enhanced_audio/{spk}.wav"

def parse_cha(path):
    """Same parser shape as the Pitt POD4 cha_overlap.py: join continuation lines,
    then read the \\x15start_end\\x15 bullet off each *PAR:/*INV: utterance."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out = {"PAR": [], "INV": []}
    untimed = {"PAR": 0, "INV": 0}
    lines, buf = [], None
    for ln in txt.split("\n"):
        if ln.startswith("*") or ln.startswith("%") or ln.startswith("@"):
            if buf is not None: lines.append(buf)
            buf = ln
        elif buf is not None and ln.startswith("\t"):
            buf += " " + ln.strip()
        else:
            if buf is not None: lines.append(buf); buf = None
    if buf is not None: lines.append(buf)
    for ln in lines:
        m = re.match(r"^\*(PAR|INV)\d*:", ln)
        if not m: continue
        who = m.group(1)
        tm = re.search(r"\x15(\d+)_(\d+)\x15", ln)
        if tm: out[who].append((int(tm.group(1)), int(tm.group(2))))
        else:  untimed[who] += 1
    return out, untimed

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
meta, rep, missing = [], [], []
tot_untimed = {"PAR": 0, "INV": 0}
align = []
for r in rows:
    spk, split, label = r["spk"], r["split"], int(r["label"])
    cp, sp = cha_path(spk, split, label), src_path(spk, split, label)
    wav = f"{SEG}/{spk}.wav"
    if not (os.path.exists(cp) and os.path.exists(wav)):
        missing.append((spk, os.path.exists(cp), os.path.exists(wav))); continue
    iv, unt = parse_cha(cp)
    tot_untimed["PAR"] += unt["PAR"]; tot_untimed["INV"] += unt["INV"]
    info = sf.info(wav); clip_ms = int(round(info.frames / info.samplerate * 1000))
    start_ms = int(round(float(r["start_s"]) * 1000))
    end_ms = start_ms + clip_ms            # the window the clip really spans
    par_all, inv_all = merge(iv["PAR"]), merge(iv["INV"])
    # alignment check: last CHAT timestamp vs source recording length
    if os.path.exists(sp):
        si = sf.info(sp); src_ms = int(round(si.frames / si.samplerate * 1000))
        last = max([b for _, b in par_all + inv_all] or [0])
        align.append((spk, src_ms, last, last - src_ms))
    par_w, inv_w = clip_to(par_all, start_ms, end_ms), clip_to(inv_all, start_ms, end_ms)
    span = end_ms - start_ms
    par_ms, inv_ms = total(par_w), total(inv_w)
    meta.append(dict(clip=f"{spk}.wav", seg_wav=f"{spk}.wav", speaker=spk, label=label,
                     set=split, start_ms=start_ms, end_ms=end_ms,
                     par_off=[[a - start_ms, b - start_ms] for a, b in par_w],
                     inv_ms=inv_ms, par_ms=par_ms, span_ms=span))
    rep.append(dict(clip=spk, split=split, label=label, start_ms=start_ms, clip_ms=clip_ms,
                    span_ms=span, par_ms_win=par_ms, inv_ms_win=inv_ms,
                    gap_ms_win=span - par_ms - inv_ms,
                    inv_share_win=round(inv_ms / span, 5) if span else 0.0,
                    par_share_win=round(par_ms / span, 5) if span else 0.0,
                    n_par_utt=len(par_w), n_inv_utt=len(inv_w),
                    untimed_par=unt["PAR"], untimed_inv=unt["INV"]))

print(f"manifest rows        : {len(rows)}")
print(f"resolved to CHAT+wav : {len(meta)}")
print(f"missing              : {len(missing)} {missing[:5]}")
print(f"untimed utterances across all resolved files: PAR {tot_untimed['PAR']}  INV {tot_untimed['INV']}")
d = np.array([a[3] for a in align], float)
print(f"CHAT-vs-source alignment (last bullet minus source length, ms): n={len(align)} "
      f"median {np.median(d):.0f}  min {d.min():.0f}  max {d.max():.0f}  |>2000ms| {(np.abs(d)>2000).sum()}")
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
