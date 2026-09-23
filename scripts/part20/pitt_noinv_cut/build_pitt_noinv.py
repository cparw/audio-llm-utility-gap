#!/usr/bin/env python3
"""Part 20: rebuild the Part 16 POD4 arm B (interviewer removed, PAR-only) Pitt cut on the Mac,
so every GPU pod reads the same clips.

Same logic as part16/POD4/make_intervals.py + cut_arms.py (arm B):
  heard window = segment wav's first min(end_ms-start_ms, 30000) ms
  keep only the timed *PAR intervals (merged) inside that window, as offsets into the segment wav
  concatenate, 5 ms raised-cosine fade in/out on every piece (pieces shorter than 2 fades kept unfaded)
  16 kHz mono, soundfile default subtype (PCM_16), as in cut_arms.py
The CHAT parser (parse_cha, merge, clip_to, total) is loaded verbatim from part16/POD4/cha_overlap.py via ast,
so no copy drift and no import side effects.
"""
import ast, csv, hashlib, json, os, sys
import numpy as np, soundfile as sf

ROOT = "<local data dir>/DementiaBank"
POD4 = "<local data dir>/release/edaic_rerun/part16/POD4"
FOLDS = "folds/pitt_groupkfold5_pod.csv"
OUT = "<local data dir>/release/scores/part20/pitt_noinv_cut"
SR, FADE_MS, WIN_MS = 16000, 5.0, 30000

src = open(f"{POD4}/cha_overlap.py").read()
tree = ast.parse(src)
fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("parse_cha", "merge", "clip_to", "total")]
assert len(fns) == 4
ns = {"re": __import__("re"), "os": os}
exec(compile(ast.Module(body=fns, type_ignores=[]), "cha_overlap.py", "exec"), ns)
parse_cha, merge, clip_to, total = ns["parse_cha"], ns["merge"], ns["clip_to"], ns["total"]

rows = list(csv.DictReader(open(f"{ROOT}/pitt_conflict_manifest.csv")))
folds = {r["clip_id"]: r for r in csv.DictReader(open(FOLDS))}
fade_n = int(SR * FADE_MS / 1000.0)
ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n)))

cache, man, rep, unresolved = {}, [], [], []
for r in rows:
    grp, sess = r["grp"], r["session"]
    cha = f"{ROOT}/transcripts/{grp}/cookie/{sess}.cha"
    if cha not in cache:
        cache[cha] = parse_cha(cha) if os.path.exists(cha) else (None, None)
    iv, untimed = cache[cha]
    clip = os.path.basename(r["segment_path"])
    if iv is None:
        unresolved.append(clip); continue
    s, e = int(r["start_ms"]), int(r["end_ms"])
    we = min(e, s + WIN_MS)
    par = clip_to(merge(iv["PAR"]), s, we)
    inv = clip_to(merge(iv["INV"]), s, we)
    par_off = [[a - s, b - s] for a, b in par]
    x, sr = sf.read(f"{ROOT}/{r['segment_path']}", dtype="float32")
    assert sr == SR, (clip, sr)
    if x.ndim > 1: x = x.mean(axis=1)
    pieces = []
    for a_ms, b_ms in par_off:
        a, b = int(a_ms * SR / 1000), int(b_ms * SR / 1000)
        a, b = max(0, a), min(len(x), b)
        if b - a < fade_n * 2:
            if b > a: pieces.append(x[a:b].copy())
            continue
        seg = x[a:b].copy()
        seg[:fade_n] *= ramp
        seg[-fade_n:] *= ramp[::-1]
        pieces.append(seg)
    assert pieces, f"empty clip {clip}"
    B = np.concatenate(pieces)
    outp = f"{OUT}/clips/noinv_{clip}"
    sf.write(outp, B, SR)
    y, _ = sf.read(outp, dtype="float32")
    assert len(y) == len(B) and np.abs(y).max() > 0, clip
    h = hashlib.sha256(open(outp, "rb").read()).hexdigest()
    f = folds[clip]
    span = we - s
    man.append(dict(clip=f"noinv_{clip}", orig_clip_id=clip, spk=f["speaker_id"], label=int(r["label"]),
                    arm="B_paronly_noinv", dur_par_s=round(len(B) / SR, 3), n_inv_utts_removed=len(inv), sha256=h))
    rep.append(dict(clip=clip, set=r["set"], par_ms_chat=total(par), inv_ms=total(inv), span_ms=span,
                    par_share=total(par) / span, n_par=len(par_off), dur_written_s=len(B) / SR, fold=f["fold"],
                    grp=grp, spk_raw=r["spk"]))

with open(f"{OUT}/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)
json.dump(dict(unresolved=unresolved, per_clip=rep), open(f"{OUT}/build_report.json", "w"), indent=1)
print("built", len(man), "of", len(rows), "unresolved", len(unresolved))
