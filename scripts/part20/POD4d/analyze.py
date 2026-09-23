"""PART20 POD4d analysis. AUC = rank formula (scipy.stats.rankdata, ties averaged).
Bootstrap: 2000 draws, fresh numpy.random.default_rng(0) per cell, speakers (fold-file speaker_id) resampled with replacement,
percentile 2.5/97.5; draws whose resampled clip set lacks a class are skipped (count reported).
PAIRED: one speaker draw per replicate, both AUCs recomputed on that draw, difference = original - interviewer-free.
usage: analyze.py NOINV_CSV ORIG_CSV OUTDIR SCORECOL COMMAND_STRING [MODE canon|podorig [CANON_CSV]]
  canon  : ORIG_CSV is the canonical paper file; emits noinv, canonical-original and paired original-minus-noinv cells
  podorig: ORIG_CSV is this pod's rerun of the 468 originals; emits pod-original, paired pod-original-minus-noinv,
           and paired canonical-minus-pod-original (environment drift) cells"""
import csv, sys, json, hashlib, os
import numpy as np
from scipy.stats import rankdata
NOINV, CANON, OUTD, COL, CMD = sys.argv[1:6]
MODE = sys.argv[6] if len(sys.argv) > 6 else "canon"
CANON0 = sys.argv[7] if len(sys.argv) > 7 else None
MC = "mass" if MODE == "canon" else "answer_mass"
NB = 2000
MODEL = "moonshotai/Kimi-Audio-7B-Instruct"
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
def sha1mb(p): return hashlib.sha256(open(p, "rb").read(1 << 20)).hexdigest()
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def boot(spk, y, s_list):
    rng = np.random.default_rng(0)
    u = np.unique(spk); idx = {sp: np.where(spk == sp)[0] for sp in u}
    vals = []
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        a = [auc(y[ii], s[ii]) for s in s_list]
        if any(np.isnan(x) for x in a): continue
        vals.append(a)
    return np.array(vals)
nz = {r["orig_clip_id"]: r for r in csv.DictReader(open(NOINV))}
cz = {r["clip"]: r for r in csv.DictReader(open(CANON))}
c0 = {r["clip"]: r for r in csv.DictReader(open(CANON0))} if CANON0 else None
assert all(k in cz for k in nz), "noinv clip without canonical original"
for k, r in nz.items():
    assert int(r["label"]) == int(cz[k]["label"]) and r["speaker"] == cz[k]["speaker"], k
os.makedirs(OUTD, exist_ok=True)
rows_out = []
def emit(rid, what, val, lo, hi, n, nspk, nuse, files, clip_rows, clip_cols, extra, diff=False):
    pc = os.path.join(OUTD, rid + ".csv"); sc = os.path.join(OUTD, rid + ".json")
    with open(pc, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(clip_cols); w.writerows(clip_rows)
    side = dict(id=rid, what=what, value=round(float(val), 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                n=n, n_speakers=nspk, usable_draws=nuse, bootstrap_draws=NB, seed=0,
                bootstrap="speakers resampled with replacement, fresh numpy.random.default_rng(0) per cell, percentile 2.5/97.5" + (", PAIRED: one speaker draw per replicate" if diff else ""),
                auc="rank formula, scipy.stats.rankdata ties averaged", score_column=COL,
                model_id=MODEL, prompt_verbatim=P, command=CMD, per_clip_csv=os.path.abspath(pc),
                sources=[dict(path=f, sha256_first_1MB=sha1mb(f)) for f in files], **extra)
    json.dump(side, open(sc, "w"), indent=1)
    two = f"{val:.2f} [{lo:.2f}, {hi:.2f}]"
    if diff: vs = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
    else: vs = "above 0.5" if lo > 0.5 else ("below 0.5" if hi < 0.5 else "interval includes 0.5")
    line = f"PASTE {rid} | {what} | {val:.4f} [{lo:.4f}, {hi:.4f}] n={n} n_spk={nspk} | {os.path.abspath(pc)} | {two} {vs}"
    print(line, flush=True)
    rows_out.append(dict(id=rid, what=what, value=f"{val:.4f}", lo=f"{lo:.4f}", hi=f"{hi:.4f}", n=n, n_spk=nspk,
                         file=os.path.abspath(pc), two_dp=two, vs_half=vs))
for arm in ("all", "conflict", "agreement"):
    keys = sorted(k for k in nz if arm == "all" or nz[k]["set"] == arm)
    spk = np.array([nz[k]["speaker"] for k in keys]); y = np.array([int(nz[k]["label"]) for k in keys])
    s_nz = np.array([float(nz[k][COL]) for k in keys]); s_or = np.array([float(cz[k]["p_yes"]) for k in keys])
    a_nz, a_or = auc(y, s_nz), auc(y, s_or)
    tag = "overall" if arm == "all" else arm
    if MODE == "canon":
        # cell 1: interviewer-free
        v = boot(spk, y, [s_nz])
        emit(f"kimi_pitt_noinv_{tag}", f"Kimi-Audio zero-shot Pitt interviewer-free (PAR-only) {tag} AUC", a_nz,
             np.percentile(v[:, 0], 2.5), np.percentile(v[:, 0], 97.5), len(keys), len(np.unique(spk)), len(v), [NOINV],
             [[k, nz[k]["clip"], nz[k]["speaker"], nz[k]["label"], nz[k]["set"], nz[k]["fold"], nz[k][COL], nz[k]["answer_mass"]] for k in keys],
             ["orig_clip_id", "clip", "speaker", "label", "set", "fold", "p_yes", "answer_mass"],
             dict(mean_answer_mass=float(np.mean([float(nz[k]["answer_mass"]) for k in keys])),
                  min_answer_mass=float(np.min([float(nz[k]["answer_mass"]) for k in keys]))))
        oid, owhat = f"kimi_pitt_orig_canonical_{tag}", f"Kimi-Audio zero-shot Pitt original windows {tag} AUC (canonical paper file, recomputed)"
        did, dwhat = f"kimi_pitt_orig_minus_noinv_{tag}", f"Kimi-Audio zero-shot Pitt {tag}: original (canonical file) minus interviewer-free AUC (paired)"
    else:
        oid, owhat = f"kimi_pitt_orig_podrerun_{tag}", f"Kimi-Audio zero-shot Pitt original windows {tag} AUC (this pod's rerun, same env as the cut)"
        did, dwhat = f"kimi_pitt_podorig_minus_noinv_{tag}", f"Kimi-Audio zero-shot Pitt {tag}: original (pod rerun) minus interviewer-free AUC (paired, same env)"
    # cell 2: original
    v = boot(spk, y, [s_or])
    emit(oid, owhat, a_or,
         np.percentile(v[:, 0], 2.5), np.percentile(v[:, 0], 97.5), len(keys), len(np.unique(spk)), len(v), [CANON],
         [[k, cz[k]["speaker"], cz[k]["label"], nz[k]["set"], nz[k]["fold"], cz[k]["p_yes"], cz[k][MC]] for k in keys],
         ["clip", "speaker", "label", "set", "fold", "p_yes", "answer_mass"], {})
    # cell 3: paired original - interviewer-free
    v = boot(spk, y, [s_or, s_nz]); d = v[:, 0] - v[:, 1]
    emit(did, dwhat, a_or - a_nz,
         np.percentile(d, 2.5), np.percentile(d, 97.5), len(keys), len(np.unique(spk)), len(v), [CANON, NOINV],
         [[k, nz[k]["clip"], nz[k]["speaker"], nz[k]["label"], nz[k]["set"], nz[k]["fold"], cz[k]["p_yes"], nz[k][COL]] for k in keys],
         ["orig_clip_id", "noinv_clip", "speaker", "label", "set", "fold", "p_yes_original", "p_yes_noinv"],
         dict(auc_original=round(float(a_or), 4), auc_noinv=round(float(a_nz), 4), original_source=CANON), diff=True)
    if MODE == "podorig" and c0 is not None:
        s_c0 = np.array([float(c0[k]["p_yes"]) for k in keys]); a_c0 = auc(y, s_c0)
        v = boot(spk, y, [s_c0, s_or]); d = v[:, 0] - v[:, 1]
        emit(f"kimi_pitt_canon_minus_podorig_{tag}", f"Kimi-Audio zero-shot Pitt {tag}: canonical file minus this pod's rerun of the same original windows (paired, environment drift)", a_c0 - a_or,
             np.percentile(d, 2.5), np.percentile(d, 97.5), len(keys), len(np.unique(spk)), len(v), [CANON0, CANON],
             [[k, cz[k]["speaker"], cz[k]["label"], nz[k]["set"], nz[k]["fold"], c0[k]["p_yes"], cz[k]["p_yes"], float(cz[k]["p_yes"]) - float(c0[k]["p_yes"])] for k in keys],
             ["clip", "speaker", "label", "set", "fold", "p_yes_canonical", "p_yes_pod_rerun", "pod_minus_canonical"],
             dict(auc_canonical=round(float(a_c0), 4), auc_pod_rerun=round(float(a_or), 4),
                  max_abs_clip_diff=float(np.max(np.abs(s_or - s_c0))), median_abs_clip_diff=float(np.median(np.abs(s_or - s_c0))),
                  n_clips_abs_diff_over_1e3=int((np.abs(s_or - s_c0) > 1e-3).sum())), diff=True)
json.dump(rows_out, open(os.path.join(OUTD, f"rows_{MODE}.json"), "w"), indent=1)
