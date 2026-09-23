#!/usr/bin/env python3
"""4b step 1: resolve the 468 Pitt windows to CHAT timings and measure how much
of the audio the model actually hears is investigator (*INV) speech.

The original cut is [start_ms, end_ms] from build_ad_conflict.py, which merges
consecutive *PAR utterances -- but the audio between those utterances is kept,
and that is where *INV speaks. The model then sees only the FIRST 30 s
(extract_probe_layers.py: x = x[:16000*WIN], WIN=30).
"""
import csv, os, re, glob, json, sys
import numpy as np

ROOT = "<local data dir>/DementiaBank"
MF   = f"{ROOT}/pitt_conflict_manifest.csv"
WIN_MS = 30000

def parse_cha(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    out = {"PAR": [], "INV": []}
    untimed = {"PAR": 0, "INV": 0}
    # utterances can wrap onto continuation lines; join them first
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
        if tm:
            out[who].append((int(tm.group(1)), int(tm.group(2))))
        else:
            untimed[who] += 1
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
cache = {}
res = []
unresolved = []
for r in rows:
    grp, sess = r["grp"], r["session"]
    cha = f"{ROOT}/transcripts/{grp}/cookie/{sess}.cha"
    if cha not in cache:
        cache[cha] = parse_cha(cha) if os.path.exists(cha) else (None, None)
    iv, untimed = cache[cha]
    if iv is None:
        unresolved.append(f"{grp}/{sess}"); continue
    s, e = int(r["start_ms"]), int(r["end_ms"])
    par_all, inv_all = merge(iv["PAR"]), merge(iv["INV"])
    # effective window the model hears: first 30 s of the cut
    we = min(e, s + WIN_MS)
    par_w, inv_w = clip_to(par_all, s, we), clip_to(inv_all, s, we)
    span = we - s
    par_ms, inv_ms = total(par_w), total(inv_w)
    res.append(dict(
        clip=os.path.basename(r["segment_path"]), spk=r["spk"], grp=grp, sess=sess,
        label=int(r["label"]), set=r["set"], start_ms=s, end_ms=e,
        win_span_ms=span, par_ms_win=par_ms, inv_ms_win=inv_ms,
        gap_ms_win=span - par_ms - inv_ms,
        inv_share_win=round(inv_ms / span, 5) if span else 0.0,
        par_share_win=round(par_ms / span, 5) if span else 0.0,
        n_par_utt=len(par_w), n_inv_utt=len(inv_w),
        untimed_par=untimed["PAR"], untimed_inv=untimed["INV"],
    ))

print(f"windows in manifest : {len(rows)}")
print(f"resolved to CHAT    : {len(res)}")
print(f"unresolved          : {len(unresolved)} {unresolved[:10]}")
inv = np.array([x["inv_ms_win"] for x in res], float)
par = np.array([x["par_ms_win"] for x in res], float)
gap = np.array([x["gap_ms_win"] for x in res], float)
spn = np.array([x["win_span_ms"] for x in res], float)
print()
print("=== within the 30 s the model actually hears ===")
print(f"  INV ms  : mean {inv.mean():8.1f}  median {np.median(inv):8.1f}  max {inv.max():8.1f}")
print(f"  windows with ANY INV speech : {(inv>0).sum()} / {len(res)}  ({100*(inv>0).mean():.1f}%)")
print(f"  INV share: mean {(inv/spn).mean():.4f}  median {np.median(inv/spn):.4f}  max {(inv/spn).max():.4f}")
print(f"  PAR share: mean {(par/spn).mean():.4f}  median {np.median(par/spn):.4f}  min {(par/spn).min():.4f}")
print(f"  GAP (neither tier) share: mean {(gap/spn).mean():.4f} median {np.median(gap/spn):.4f} max {(gap/spn).max():.4f}")
print(f"  untimed INV utterances in these transcripts: {sum(x['untimed_inv'] for x in res)}")
print(f"  untimed PAR utterances in these transcripts: {sum(x['untimed_par'] for x in res)}")
print()
print("=== PAR-only concatenated duration (what 4b will feed the model) ===")
ps = par/1000.0
for t in (5,10,15,20,25):
    print(f"  < {t:2d} s of participant speech : {(ps<t).sum():3d} windows ({100*(ps<t).mean():.1f}%)")
print(f"  mean {ps.mean():.2f}s  median {np.median(ps):.2f}s  min {ps.min():.2f}s  max {ps.max():.2f}s")

with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
print(f"\nwrote {sys.argv[1]}")
