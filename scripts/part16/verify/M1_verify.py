#!/usr/bin/env python3
# INDEPENDENT verifier for PART16 M1.
# Nothing imported from the primary job's script. Sources read fresh from disk.
# AUC computed by brute-force Mann-Whitney U pair counting (ties = 0.5),
# cross-checked against a trapezoid-over-ROC implementation.

import csv, json, os
from collections import defaultdict
import numpy as np

TEXT  = "<local data dir>/paper1_local_runs/omni_pitt_text.csv"
AUDIO = "<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv"
ALT   = "<local data dir>/Desktop/release/overnight2/text_new/o25_pitt_text.csv"
MAN   = "manifests/pitt_conflict_manifest_468.csv"

def base(p):
    return os.path.basename(str(p).strip())

def load(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

# ---------- AUC route 1: brute force concordant-pair count, ties 0.5 ----------
def auc_pairs(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    # broadcast comparison over every (pos, neg) pair
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    return (wins + 0.5 * ties) / (len(pos) * len(neg))

# ---------- AUC route 2: trapezoid over the full ROC ----------
def auc_trapz(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0:
        return float('nan')
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]; ss = s[order]
    tps = np.cumsum(ys == 1); fps = np.cumsum(ys == 0)
    # keep last index of each distinct score so ties form one ROC vertex
    keep = np.r_[np.diff(ss) != 0, True]
    tpr = np.r_[0.0, tps[keep] / P]
    fpr = np.r_[0.0, fps[keep] / N]
    return float(np.trapz(tpr, fpr))

def auc(y, s):
    a = auc_pairs(y, s)
    b = auc_trapz(y, s)
    if not (np.isnan(a) and np.isnan(b)):
        assert abs(a - b) < 1e-12, f"internal AUC routes disagree {a} {b}"
    return a

# ---------- load + join ----------
man = load(MAN)
man_by_clip = {}
for r in man:
    man_by_clip[base(r['segment_path'])] = {
        'spk': r['grp'] + r['spk'],
        'label': int(r['label']),
        'arm': r['set'].strip(),
    }

txt = load(TEXT); aud = load(AUDIO); alt = load(ALT)
print(f"row counts  text={len(txt)}  audio={len(aud)}  alt={len(alt)}  manifest={len(man)}")

txt_by = {base(r['id']): r for r in txt}
aud_by = {base(r['clip']): r for r in aud}
alt_by = {base(r['id']): r for r in alt}
print(f"unique keys text={len(txt_by)} audio={len(aud_by)} alt={len(alt_by)} man={len(man_by_clip)}")

rows = []
miss = {'man': 0, 'aud': 0, 'alt': 0}
arm_mismatch = lab_mismatch_man = lab_mismatch_aud = spk_mismatch = 0
for k, r in txt_by.items():
    m = man_by_clip.get(k)
    if m is None:
        miss['man'] += 1; continue
    a = aud_by.get(k)
    if a is None:
        miss['aud'] += 1; continue
    al = alt_by.get(k)
    if al is None:
        miss['alt'] += 1
    # independent consistency checks
    if r['set'].strip() != m['arm']: arm_mismatch += 1
    if int(r['label']) != m['label']: lab_mismatch_man += 1
    if int(a['label']) != m['label']: lab_mismatch_aud += 1
    if a['speaker'].strip() != r['speaker'].strip(): spk_mismatch += 1
    rows.append({
        'clip': k, 'spk': r['speaker'].strip(), 'label': int(r['label']),
        'arm': m['arm'],
        'p_text': float(r['p_yes']), 'p_audio': float(a['p_yes']),
        'p_alt': float(al['p_yes']) if al else float('nan'),
    })

print(f"joined={len(rows)}  missing_from_manifest={miss['man']} missing_audio={miss['aud']} missing_alt={miss['alt']}")
print(f"arm_mismatch={arm_mismatch} label_mismatch_vs_manifest={lab_mismatch_man} "
      f"label_mismatch_vs_audio={lab_mismatch_aud} speaker_mismatch_text_vs_audio={spk_mismatch}")

# speaker naming cross-check against the manifest's own grp+spk
man_spk_mismatch = sum(1 for r in rows if man_by_clip[r['clip']]['spk'] != r['spk'])
print(f"speaker_mismatch_text_vs_manifest(grp+spk)={man_spk_mismatch}")

lab  = np.array([r['label'] for r in rows])
arm  = np.array([r['arm'] for r in rows])
spk  = np.array([r['spk'] for r in rows])
ptxt = np.array([r['p_text'] for r in rows])
paud = np.array([r['p_audio'] for r in rows])
palt = np.array([r['p_alt'] for r in rows])

masks = {
    'all468':    np.ones(len(rows), bool),
    'conflict':  arm == 'conflict',
    'agreement': arm == 'agreement',
}
for nm, m in masks.items():
    print(f"{nm}: n={m.sum()} n_spk={len(set(spk[m]))} pos={int(lab[m].sum())} neg={int((lab[m]==0).sum())}")

# ---------- bootstrap ----------
DRAWS = 2000
PCT = [2.5, 97.5]

def boot(fn, universe_mask):
    """fn(idx_array)->value or list of values. Speakers drawn from universe_mask."""
    rng = np.random.default_rng(0)
    uspk = np.array(sorted(set(spk[universe_mask])))
    idx_of = {s: np.where(spk == s)[0] for s in uspk}
    out = []
    for _ in range(DRAWS):
        pick = rng.choice(uspk, size=len(uspk), replace=True)
        idx = np.concatenate([idx_of[s] for s in pick])
        v = fn(idx)
        out.append(v)
    arr = np.asarray(out, dtype=float)
    return arr

def ci(arr):
    good = arr[~np.isnan(arr)]
    lo, hi = np.percentile(good, PCT)
    return lo, hi, len(good)

results = []
def record(idv, what, val, lo, hi, n, nspk, usable, src):
    results.append(dict(id=idv, what=what, value=val, lo=lo, hi=hi, n=int(n),
                        n_spk=int(nspk), usable=int(usable), file=src))
    print(f"{idv:38s} {val:.6f}  [{lo:.6f}, {hi:.6f}]  n={n} n_spk={nspk} usable={usable}/{DRAWS}")

# --- simple AUC cells ---
for stream, score, src in (('text', ptxt, TEXT), ('audio', paud, AUDIO), ('ALT_text', palt, ALT)):
    for nm, m in masks.items():
        val = auc(lab[m], score[m])
        sub = m
        arr = boot(lambda idx, sub=sub, score=score: auc(lab[idx][sub[idx]], score[idx][sub[idx]]),
                   universe_mask=sub)
        lo, hi, usable = ci(arr)
        record(f"M1_{stream}_{nm}", f"Pitt Qwen2.5-Omni {stream} AUC, {nm} arm",
               val, lo, hi, m.sum(), len(set(spk[m])), usable, src)

# --- paired conflict minus agreement, per stream, one speaker draw per replicate ---
mc, ma = masks['conflict'], masks['agreement']
for stream, score, src in (('text', ptxt, TEXT), ('audio', paud, AUDIO)):
    val = auc(lab[mc], score[mc]) - auc(lab[ma], score[ma])
    def f(idx, score=score):
        a = auc(lab[idx][mc[idx]], score[idx][mc[idx]])
        b = auc(lab[idx][ma[idx]], score[idx][ma[idx]])
        return a - b
    arr = boot(f, universe_mask=np.ones(len(rows), bool))
    lo, hi, usable = ci(arr)
    record(f"M1_{stream}_conflict_minus_agreement",
           f"Pitt Qwen2.5-Omni {stream} paired conflict minus agreement AUC",
           val, lo, hi, len(rows), len(set(spk)), usable, src)

# --- paired text minus audio, per arm ---
PAIRCSV = "scores/part16/M/M1_pitt_text_arms.csv"
for nm, m in masks.items():
    val = auc(lab[m], ptxt[m]) - auc(lab[m], paud[m])
    def f(idx, sub=m):
        s = sub[idx]
        return auc(lab[idx][s], ptxt[idx][s]) - auc(lab[idx][s], paud[idx][s])
    arr = boot(f, universe_mask=m)
    lo, hi, usable = ci(arr)
    record(f"M1_text_minus_audio_{nm}",
           f"Pitt Qwen2.5-Omni paired transcript minus audio AUC, {nm} arm",
           val, lo, hi, m.sum(), len(set(spk[m])), usable, PAIRCSV)

json.dump(results, open("reports/part16/verify/M1_verify.json","w"), indent=1)
print("\nwrote M1_verify.json")
