#!/usr/bin/env python3
"""M3 - Speaker-level AUC beside clip-level, Qwen2.5-Omni zero shot, pitt / pcgita / neurovoz."""
import os, sys, json, hashlib, datetime
import numpy as np, pandas as pd
import scipy.stats as st

OUT = "<local data dir>/Desktop/release/edaic_rerun/part16"
ROWS = os.path.join(OUT, "rows")
DISC = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"
LOCAL = "<local data dir>/Desktop/release/omni_final"
GDRIVE = "<local data dir>/paper1_local_runs/omni_final"
MASTER = "lookup/master_lookup.csv"
INDEX = "<local data dir>/Desktop/release/overnight/perclip/INDEX.csv"
DATASETS = ["pitt", "pcgita", "neurovoz"]
NBOOT, SEED = 2000, 0
os.makedirs(ROWS, exist_ok=True)

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sha1mb(p):
    try:
        with open(p, 'rb') as f: return hashlib.sha256(f.read(1024*1024)).hexdigest()
    except Exception as e: return "ERR:"+str(e)

def disc(sev, msg):
    with open(DISC, 'a') as f:
        f.write("\n- **[%s] M3 (part16)** %s  _(%s)_\n" % (sev, msg, datetime.datetime.now().isoformat(timespec='seconds')))
    print("  DISCREPANCY [%s] %s" % (sev, msg))

lines, rows, spk_frames, sidecar_src = [], [], [], {}
master = pd.read_csv(MASTER)
idx = pd.read_csv(INDEX)

for ds in DATASETS:
    print("\n" + "="*78); print("DATASET:", ds); print("="*78)
    lp = os.path.join(LOCAL, "omni_%s_zeroshot_scores.csv" % ds)
    gp = os.path.join(GDRIVE, "omni_%s_zeroshot_scores.csv" % ds)
    if not os.path.exists(lp):
        disc("HIGH", "missing local score file %s" % lp); continue
    d = pd.read_csv(lp)
    sidecar_src.setdefault(ds, {})["local"] = {"path": lp, "sha256_first1mb": sha1mb(lp), "n_rows": int(len(d))}

    # --- 1. INDEX.csv cross-check + mirror identity ---
    ir = idx[(idx.experiment == "zeroshot_omni") & (idx.dataset == ds)]
    idx_src = ir.source_file.iloc[0] if len(ir) else None
    idx_auc = float(ir.auc_recomputed.iloc[0]) if len(ir) else float('nan')
    print("INDEX.csv row: source_file=%s  n_rows=%s  auc_recomputed=%s" %
          (idx_src, int(ir.n_rows.iloc[0]) if len(ir) else 'NA', idx_auc))
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        sidecar_src[ds]["gdrive"] = {"path": gp, "sha256_first1mb": sha1mb(gp), "n_rows": int(len(g))}
        if len(g) == len(d):
            m = d[["clip","p_yes"]].merge(g[["clip","p_yes"]], on="clip", suffixes=("_loc","_gd"))
            if len(m) == len(d):
                mx = float(np.abs(m.p_yes_loc - m.p_yes_gd).max())
                print("MIRROR CHECK: INDEX points at G-Drive %s; release-local mirror exists. max|dp_yes| = %.10g -> %s"
                      % (gp, mx, "IDENTICAL" if mx == 0 else "DIFFERS"))
                sidecar_src[ds]["mirror_max_abs_diff_p_yes"] = mx
                if mx > 0: disc("MEDIUM", "%s: G-Drive vs release-local zeroshot p_yes differ, max abs diff %.3g" % (ds, mx))
            else:
                disc("MEDIUM", "%s: clip names do not align between G-Drive and local mirror (%d matched of %d)" % (ds, len(m), len(d)))
        else:
            disc("MEDIUM", "%s: row count differs local %d vs G-Drive %d" % (ds, len(d), len(g)))
    else:
        print("MIRROR CHECK: G-Drive path not mounted/present (%s) - using release-local only" % gp)
        disc("LOW", "%s: G-Drive mirror %s not present, used release-local copy only" % (ds, gp))

    y = d.label.astype(int).values
    p = d.p_yes.astype(float).values
    spk = d.speaker.astype(str).values
    n, nspk = len(d), len(np.unique(spk))

    # --- 2. speaker labels consistent? ---
    g = pd.DataFrame({"spk": spk, "label": y, "p_yes": p})
    lab_n = g.groupby("spk").label.nunique()
    bad = lab_n[lab_n > 1].index.tolist()
    if bad:
        disc("HIGH", "%s: %d speaker(s) carry more than one clip label: %s" % (ds, len(bad), bad[:10]))
        print("SPEAKER LABEL CHECK: %d INCONSISTENT speakers -> %s" % (len(bad), bad[:10]))
    else:
        print("SPEAKER LABEL CHECK: all %d speakers carry a single consistent label" % nspk)
    sp = g.groupby("spk").agg(label=("label","max"), n_clips=("label","size"), mean_p_yes=("p_yes","mean")).reset_index()
    sp.insert(0, "dataset", ds)
    spk_frames.append(sp)

    clip_auc = auc(y, p)
    spk_auc = auc(sp.label.values, sp.mean_p_yes.values)
    diff = clip_auc - spk_auc

    # --- 3. paired bootstrap, unit = speaker ---
    rng = np.random.default_rng(SEED)
    uspk = sp.spk.values
    by_clip = {s: np.where(spk == s)[0] for s in uspk}
    spk_lab = sp.label.values; spk_mp = sp.mean_p_yes.values
    pos = {s: i for i, s in enumerate(uspk)}
    bc, bs, bd = [], [], []
    usable = 0
    for _ in range(NBOOT):
        draw = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.array([pos[s] for s in draw])
        sl, sm = spk_lab[ii], spk_mp[ii]
        ci = np.concatenate([by_clip[s] for s in draw])
        cl, cp = y[ci], p[ci]
        if len(np.unique(sl)) < 2 or len(np.unique(cl)) < 2: continue
        a_c, a_s = auc(cl, cp), auc(sl, sm)
        if np.isnan(a_c) or np.isnan(a_s): continue
        usable += 1; bc.append(a_c); bs.append(a_s); bd.append(a_c - a_s)
    bc, bs, bd = map(np.array, (bc, bs, bd))
    cl_lo, cl_hi = np.percentile(bc, [2.5, 97.5])
    sp_lo, sp_hi = np.percentile(bs, [2.5, 97.5])
    df_lo, df_hi = np.percentile(bd, [2.5, 97.5])

    # --- 4. master_lookup agreement on the clip-level value ---
    mr = master[(master.model == "Qwen2.5-Omni") & (master.dataset == ds) & (master.stream == "zero shot answer")]
    if len(mr):
        mv = float(mr.auc.iloc[0])
        ok = round(clip_auc, 4) == round(mv, 4)
        print("MASTER_LOOKUP: clip AUC recomputed %.4f vs master_lookup %.4f -> %s" % (clip_auc, mv, "MATCH" if ok else "MISMATCH"))
        if not ok: disc("MEDIUM", "%s: clip AUC recomputed %.4f != master_lookup %.4f" % (ds, clip_auc, mv))
    else:
        mv = float('nan'); disc("MEDIUM", "%s: no Qwen2.5-Omni 'zero shot answer' row in master_lookup.csv" % ds)
    if not np.isnan(idx_auc):
        iok = round(clip_auc, 4) == round(idx_auc, 4)
        print("INDEX.csv    : clip AUC recomputed %.4f vs INDEX auc_recomputed %.4f -> %s" % (clip_auc, idx_auc, "MATCH" if iok else "MISMATCH"))
        if not iok: disc("MEDIUM", "%s: clip AUC recomputed %.4f != INDEX.csv %.4f" % (ds, clip_auc, idx_auc))

    print("bootstrap usable draws: %d / %d" % (usable, NBOOT))
    L = [
      "M3 %-9s CLIP-level  AUC = %.4f [%.4f, %.4f]  n_clips=%d  n_speakers=%d  file=%s" % (ds, clip_auc, cl_lo, cl_hi, n, nspk, lp),
      "M3 %-9s SPEAKER-level AUC = %.4f [%.4f, %.4f]  n_speakers=%d (mean p_yes per speaker)  file=%s" % (ds, spk_auc, sp_lo, sp_hi, nspk, os.path.join(OUT,"M3_speaker_level.csv")),
      "M3 %-9s PAIRED clip minus speaker = %+.4f [%+.4f, %+.4f]  boot=%d/%d usable, unit=speaker, seed=0" % (ds, diff, df_lo, df_hi, usable, NBOOT),
    ]
    for l in L: print(l)
    lines += L
    f = os.path.join(OUT, "M3_speaker_level.csv")
    rows += [
      ("M3_%s_clip" % ds, "Omni zero shot %s clip-level AUC" % ds, clip_auc, cl_lo, cl_hi, n, nspk, lp),
      ("M3_%s_speaker" % ds, "Omni zero shot %s speaker-level AUC (mean p_yes)" % ds, spk_auc, sp_lo, sp_hi, nspk, nspk, f),
      ("M3_%s_clip_minus_speaker" % ds, "Omni zero shot %s paired clip minus speaker AUC" % ds, diff, df_lo, df_hi, n, nspk, f),
    ]
    sidecar_src[ds].update(dict(n_clips=n, n_speakers=nspk, clip_auc=clip_auc, clip_ci=[cl_lo, cl_hi],
                                speaker_auc=spk_auc, speaker_ci=[sp_lo, sp_hi], diff=diff, diff_ci=[df_lo, df_hi],
                                boot_usable=usable, master_lookup_auc=mv, index_auc=idx_auc,
                                inconsistent_label_speakers=bad,
                                prompt=str(d.prompt.iloc[0]) if "prompt" in d.columns else None))

# --- 5. write outputs ---
allsp = pd.concat(spk_frames, ignore_index=True)[["dataset","spk","label","n_clips","mean_p_yes"]]
allsp.to_csv(os.path.join(OUT, "M3_speaker_level.csv"), index=False)
with open(os.path.join(ROWS, "M3.tsv"), "w") as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows: f.write("%s\t%s\t%.4f\t%.4f\t%.4f\t%d\t%d\t%s\n" % r)
json.dump({"task": "M3 speaker-level AUC beside clip-level",
           "model_id": "Qwen/Qwen2.5-Omni-7B (zero shot, 30 s window) - scores read from disk, model not re-run",
           "seed": SEED, "n_boot": NBOOT, "bootstrap_unit": "speaker", "ci": "percentile 2.5/97.5",
           "paired": "speaker list drawn once per replicate, clip AUC and speaker AUC recomputed inside that draw",
           "auc": "rank formula, scipy.stats.rankdata, recomputed from per-clip p_yes",
           "folds": "not applicable - zero shot scores, no cross-validation",
           "shell_command": "/usr/local/bin/python3 %s" % os.path.abspath(__file__),
           "index_csv": {"path": INDEX, "sha256_first1mb": sha1mb(INDEX)},
           "master_lookup": {"path": MASTER, "sha256_first1mb": sha1mb(MASTER)},
           "datasets": sidecar_src,
           "outputs": [os.path.join(OUT,"M3_speaker_level.csv"), os.path.join(ROWS,"M3.tsv")],
           "written": datetime.datetime.now().isoformat(timespec='seconds')},
          open(os.path.join(OUT, "M3_speaker_level.json"), "w"), indent=2, default=str)
print("\n" + "="*78); print("PASTE LINES"); print("="*78)
for l in lines: print(l)
print("\nwrote %s (%d speakers)" % (os.path.join(OUT,"M3_speaker_level.csv"), len(allsp)))
print("wrote %s" % os.path.join(OUT,"M3_speaker_level.json"))
print("wrote %s" % os.path.join(ROWS,"M3.tsv"))
