#!/usr/bin/env python3
"""Validity battery for the Alzheimer's conflict set, before any model runs.
The same checks that were run on the daic set: investigator overlap inside the
cut windows, duration and word-count confounds per arm, and whether the label
is separable from segment structure alone."""
import csv, glob, os, re
import numpy as np

# --- investigator overlap: how much INV speech falls inside each cut window ---
inv_map = {}
for cha in glob.glob("transcripts/*/cookie/*.cha"):
    base = os.path.basename(cha)[:-4]
    grp = cha.split(os.sep)[1]
    txt = open(cha, encoding="utf-8", errors="ignore").read()
    spans = []
    for mm in re.finditer(r"\*INV:.*?(\d+)_(\d+)\s*$", txt, re.M):
        spans.append((int(mm.group(1)), int(mm.group(2))))
    inv_map[(grp, base)] = spans

rows = list(csv.DictReader(open("pitt_conflict_manifest.csv")))
def overlap(a, b, spans):
    tot = 0
    for s, e in spans:
        lo, hi = max(a, s), min(b, e)
        if hi > lo: tot += hi - lo
    return tot

for r in rows:
    a, b = int(r["start_ms"]), int(r["end_ms"])
    spans = inv_map.get((r["grp"], r["session"]), [])
    r["inv_ms_in_window"] = overlap(a, b, spans)
    r["inv_share_window"] = round(r["inv_ms_in_window"] / max(b - a, 1), 4)

def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s)+1)
    n1 = (y==1).sum(); n0 = (y==0).sum()
    return (rk[y==1].sum() - n1*(n1+1)/2) / (n1*n0)

print("="*70)
print("1. INVESTIGATOR SPEECH INSIDE THE CUT WINDOWS")
print("="*70)
for arm in ("conflict", "agreement"):
    sub = [r for r in rows if r["set"]==arm]
    sh = np.array([float(r["inv_share_window"]) for r in sub])
    y  = np.array([int(r["label"]) for r in sub])
    print(f"  {arm}: mean inv share {sh.mean():.3f}, p90 {np.percentile(sh,90):.3f}, "
          f"windows with any INV {np.mean(sh>0):.1%}")
    if len(set(y))>1:
        print(f"      inv share alone vs label: AUC {auc(y, sh):.3f}")

print()
print("="*70)
print("2. STRUCTURE CONFOUNDS PER ARM (duration, words, rate vs label)")
print("="*70)
for arm in ("conflict", "agreement"):
    sub = [r for r in rows if r["set"]==arm]
    y   = np.array([int(r["label"]) for r in sub])
    if len(set(y)) < 2:
        print(f"  {arm}: single class, skip"); continue
    for k in ("dur", "nw", "rate"):
        v = np.array([float(r[k]) for r in sub])
        print(f"  {arm:10s} {k:5s} alone: AUC {auc(y,v):.3f}")

print()
print("="*70)
print("3. ARM COMPOSITION AND BALANCE")
print("="*70)
from collections import Counter
for arm in ("conflict","agreement"):
    sub=[r for r in rows if r["set"]==arm]
    lab=Counter(r["label"] for r in sub)
    spk=len(set(r["grp"]+r["spk"] for r in sub))
    dur=np.mean([float(r["dur"]) for r in sub])
    print(f"  {arm}: n={len(sub)}, labels {dict(lab)}, speakers {spk}, mean dur {dur:.1f}s")

print()
print("="*70)
print("4. MMSE vs THE TEXT SCORE (severity, within dementia speakers)")
print("="*70)
dem=[r for r in rows if r["label"]=="1" and str(r["mmse"]).isdigit()]
if dem:
    m=np.array([float(r["mmse"]) for r in dem]); p=np.array([float(r["p_text"]) for r in dem])
    # spearman via ranks
    def rank(x):
        o=np.argsort(x); rk=np.empty(len(x)); rk[o]=np.arange(len(x)); return rk
    rho=np.corrcoef(rank(m), rank(p))[0,1]
    print(f"  n={len(dem)} dementia segments with MMSE: spearman(text score, MMSE) = {rho:+.3f}")
    print("  (more impaired words should mean lower MMSE, so expect negative)")

with open("pitt_conflict_manifest.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nmanifest updated with inv_share_window")
