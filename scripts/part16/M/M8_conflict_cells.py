#!/usr/bin/env python3
"""M8 - Are the conflict cells actually below chance?
Conflict-arm AUC with speaker bootstrap intervals for six audio LLMs on
Set A (E-DAIC Part 14, 483 pairs / 966 clips) and Set B (Pitt, 146 conflict / 322 agreement).
Every AUC is recomputed from per-clip scores with the rank formula. Nothing is copied from a json.
"""
import os, json, hashlib, subprocess, sys
import numpy as np, pandas as pd, scipy.stats as st

OUT   = "<local data dir>/Desktop/release/edaic_rerun/part16"
ROWS  = os.path.join(OUT, "rows")
DISC  = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"
os.makedirs(ROWS, exist_ok=True)
disc_lines = []

def discrep(sev, txt):
    disc_lines.append(f"- **[{sev}] M8 (part16)** {txt}")
    print(f"  !! {sev}: {txt}")

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float('nan')
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def sha1mb(p):
    with open(p, 'rb') as f: return hashlib.sha256(f.read(1024 * 1024)).hexdigest()

NDRAW = 2000

def boot_one(df, seed=0):
    """Cluster bootstrap of a single arm. Returns (lo, hi, n_usable)."""
    rng = np.random.default_rng(seed)
    spk = df['speaker'].to_numpy()
    y = df['label'].to_numpy().astype(int)
    s = df['p_yes'].to_numpy(float)
    uspk = np.unique(spk)
    idx_by = {u: np.where(spk == u)[0] for u in uspk}
    vals = []
    for _ in range(NDRAW):
        draw = rng.choice(uspk, size=len(uspk), replace=True)
        ii = np.concatenate([idx_by[u] for u in draw])
        a = auc(y[ii], s[ii])
        if not np.isnan(a): vals.append(a)
    vals = np.asarray(vals)
    if len(vals) == 0: return float('nan'), float('nan'), 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))

def boot_paired(dc, da, seed=0):
    """Paired bootstrap: ONE speaker draw per replicate, both arms recomputed inside it."""
    rng = np.random.default_rng(seed)
    uspk = np.unique(np.concatenate([dc['speaker'].unique(), da['speaker'].unique()]))
    cs = dc['speaker'].to_numpy(); as_ = da['speaker'].to_numpy()
    cy = dc['label'].to_numpy().astype(int); ay = da['label'].to_numpy().astype(int)
    cp = dc['p_yes'].to_numpy(float);        ap = da['p_yes'].to_numpy(float)
    ci_by = {u: np.where(cs == u)[0] for u in uspk}
    ai_by = {u: np.where(as_ == u)[0] for u in uspk}
    diffs = []
    for _ in range(NDRAW):
        draw = rng.choice(uspk, size=len(uspk), replace=True)
        ci = np.concatenate([ci_by[u] for u in draw]) if any(len(ci_by[u]) for u in draw) else np.array([], int)
        ai = np.concatenate([ai_by[u] for u in draw]) if any(len(ai_by[u]) for u in draw) else np.array([], int)
        if len(ci) == 0 or len(ai) == 0: continue
        a1 = auc(cy[ci], cp[ci]); a2 = auc(ay[ai], ap[ai])
        if np.isnan(a1) or np.isnan(a2): continue
        diffs.append(a1 - a2)
    diffs = np.asarray(diffs)
    if len(diffs) == 0: return float('nan'), float('nan'), float('nan'), 0
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), int(len(diffs))

# ---------------------------------------------------------------- sources
R = "<local data dir>/Desktop/release"
G = "<local data dir>/paper1_local_runs"
P14 = f"{R}/edaic_rerun/part14"
MAN_A = f"{R}/edaic_rerun/part14_manifest_new.csv"
MAN_B = "manifests/pitt_conflict_manifest_468.csv"

SET_A = [  # (model, tag, path, clip_col, spk_col)
 ("Qwen2.5-Omni",        "primary", f"{P14}/p14_o25_zeroshot_scores.csv",   "clip",      "speaker"),
 ("Qwen2-Audio",         "primary", f"{P14}/p14_q2a_zeroshot_scores.csv",   "clip",      "speaker"),
 ("Qwen3-Omni-30B-A3B",  "primary", f"{P14}/p14_q3o_zeroshot_scores.csv",   "clip",      "speaker"),
 ("Audio Flamingo 2",    "primary", f"{P14}/p14_af2.csv",                   "clip_path", "speaker_id"),
 ("Audio Flamingo 3",    "primary", f"{P14}/p14_af3.csv",                   "clip_path", "speaker_id"),
 ("Kimi-Audio",          "primary", f"{P14}/p14_kimi_zeroshot_scores.csv",  "clip",      "speaker"),
]
SET_B = [
 ("Qwen2.5-Omni",        "primary",            f"{R}/omni_final/omni_pitt_zeroshot_scores.csv",            "clip",      "speaker"),
 ("Qwen2-Audio",         "primary",            f"{G}/probe2/pitt_zeroshot_scores.csv",                     "clip",      "speaker"),
 ("Qwen3-Omni-30B-A3B",  "primary",            f"{R}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv",     "clip",      "speaker"),
 ("Audio Flamingo 2",    "af2_pitt468",        f"{G}/af2_results/af2_pitt468.csv",                         "clip_path", "speaker_id"),
 ("Audio Flamingo 2",    "af2_pitt_audio",     f"{G}/af2_pitt_audio.csv",                                  "path",      "speaker"),
 ("Audio Flamingo 3",    "primary",            f"{R}/overnight2/part3/af3_pitt.csv",                       "clip_path", "speaker_id"),
 ("Kimi-Audio",          "primary",            f"{R}/overnight2/part3/kimi_pitt_zeroshot_scores.csv",      "clip",      "speaker"),
 ("Kimi-Audio",          "alt_overnight_copy", f"{R}/overnight/kimi/kimi_pitt_zeroshot_scores.csv",        "clip",      "speaker"),
]

for m, t, p, *_ in SET_A + SET_B:
    if not os.path.exists(p): discrep("HIGH", f"source missing on disk: {p} ({m}/{t})")
missing = [p for _, _, p, *_ in SET_A + SET_B if not os.path.exists(p)]

# ---------------------------------------------------------------- manifests
mA = pd.read_csv(MAN_A)
mA['key'] = mA['set'] + '_' + mA['speaker_id'].astype(str) + '_' + mA['seg_uid'].astype(str)
mapA_arm = dict(zip(mA['key'], mA['set'])); mapA_lab = dict(zip(mA['key'], mA['label']))
mB = pd.read_csv(MAN_B)
mB['base'] = mB['segment_path'].str.split('/').str[-1]
mapB_arm = dict(zip(mB['base'], mB['set'])); mapB_lab = dict(zip(mB['base'], mB['label']))
print(f"Set A manifest {MAN_A}: n={len(mA)} arms={mA['set'].value_counts().to_dict()} spk={mA['speaker_id'].nunique()}")
print(f"Set B manifest {MAN_B}: n={len(mB)} arms={mB['set'].value_counts().to_dict()} spk={mB['spk'].nunique()}")

def load(setname, model, tag, path, clipcol, spkcol):
    d = pd.read_csv(path)
    d['clip'] = d[clipcol].astype(str).str.split('/').str[-1]
    d['speaker'] = d[spkcol].astype(str)
    if setname == 'A':
        k = d['clip'].str.replace('.wav', '', regex=False)
        d['arm'] = k.map(mapA_arm); d['man_label'] = k.map(mapA_lab)
    else:
        d['arm'] = d['clip'].map(mapB_arm); d['man_label'] = d['clip'].map(mapB_lab)
    nun = d['arm'].isna().sum()
    if nun: discrep("HIGH", f"Set {setname} {model} [{tag}]: {nun} of {len(d)} clips not found in the manifest ({path})")
    bad = int((d['label'].astype(int) != d['man_label'].fillna(-1).astype(int)).sum())
    if bad: discrep("HIGH", f"Set {setname} {model} [{tag}]: {bad} clips whose label disagrees with the manifest ({path})")
    d = d.dropna(subset=['arm'])
    d['model'] = model; d['tag'] = tag; d['set'] = 'A_edaic_part14' if setname == 'A' else 'B_pitt'
    d['source_file'] = path
    return d[['model', 'tag', 'set', 'clip', 'speaker', 'arm', 'label', 'p_yes', 'source_file']]

cells, perclip, tsv = [], [], []
for setname, spec in (('A', SET_A), ('B', SET_B)):
    for model, tag, path, cc, sc in spec:
        if not os.path.exists(path): continue
        d = load(setname, model, tag, path, cc, sc)
        perclip.append(d)
        setlabel = 'A_edaic_part14' if setname == 'A' else 'B_pitt'
        arms = {}
        for arm in ('conflict', 'agreement'):
            a = d[d['arm'] == arm].reset_index(drop=True)
            arms[arm] = a
            val = auc(a['label'], a['p_yes'])
            lo, hi, usable = boot_one(a, seed=0)
            verdict = ('below 0.5' if hi < 0.5 else 'above 0.5' if lo > 0.5 else 'straddles 0.5')
            cells.append(dict(model=model, source_tag=tag, set=setlabel, arm=arm,
                              n=len(a), n_spk=a['speaker'].nunique(), n_pos=int(a['label'].sum()),
                              auc=round(val, 4), lo=round(lo, 4), hi=round(hi, 4),
                              below_half=bool(hi < 0.5), verdict=verdict,
                              boot_usable=usable, source_file=path))
        dmean, dlo, dhi, dus = boot_paired(arms['conflict'], arms['agreement'], seed=0)
        dpoint = auc(arms['conflict']['label'], arms['conflict']['p_yes']) - auc(arms['agreement']['label'], arms['agreement']['p_yes'])
        cells.append(dict(model=model, source_tag=tag, set=setlabel, arm='conflict_minus_agreement',
                          n=len(d), n_spk=d['speaker'].nunique(), n_pos=int(d['label'].sum()),
                          auc=round(dpoint, 4), lo=round(dlo, 4), hi=round(dhi, 4),
                          below_half=bool(dhi < 0.0), verdict=('below 0' if dhi < 0 else 'above 0' if dlo > 0 else 'straddles 0'),
                          boot_usable=dus, source_file=path))
        print(f"{setlabel:16s} {model:20s} [{tag:18s}] conflict={cells[-3]['auc']:.4f} [{cells[-3]['lo']:.4f},{cells[-3]['hi']:.4f}] {cells[-3]['verdict']:14s} | agree={cells[-2]['auc']:.4f} [{cells[-2]['lo']:.4f},{cells[-2]['hi']:.4f}] | diff={dpoint:.4f} [{dlo:.4f},{dhi:.4f}]")

cells = pd.DataFrame(cells)
perclip = pd.concat(perclip, ignore_index=True)
cells_path = f"{OUT}/M8_conflict_cells.csv"
perclip_path = f"{OUT}/M8_perclip.csv"
cells.to_csv(cells_path, index=False)
perclip.to_csv(perclip_path, index=False)

for _, r in cells.iterrows():
    tsv.append(f"M8\t{r['model']} | {r['set']} | {r['arm']} | src={r['source_tag']}\t{r['auc']:.4f}\t{r['lo']:.4f}\t{r['hi']:.4f}\t{r['n']}\t{r['n_spk']}\t{r['source_file']}")
with open(f"{ROWS}/M8.tsv", 'w') as f:
    f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n" + "\n".join(tsv) + "\n")

side = dict(
    task="M8", question="Are the conflict cells actually below chance?",
    generated_by=os.path.abspath(__file__),
    shell_command="/usr/local/bin/python3 scripts/part16/M/M8_conflict_cells.py",
    python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, scipy=st.__name__,
    seed=0, n_bootstrap=NDRAW, bootstrap_unit="speaker (cluster), with replacement, percentile 2.5/97.5",
    paired_rule="one speaker draw per replicate, both arms recomputed inside that draw",
    auc_method="rank formula (scipy.stats.rankdata), recomputed from per-clip p_yes; no AUC copied from any json",
    model_ids_scored_elsewhere="zero-shot answer-token p_yes files produced by earlier parts; this task only re-reads and re-scores them",
    prompt_setA="Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
    prompt_setB="Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.",
    folds="not applicable - all quantities here are zero-shot per-clip scores, no cross-validated probe is fit in this task, so no saved fold file is used or recomputed",
    manifests={MAN_A: dict(n=len(mA), sha256_first1mb=sha1mb(MAN_A)), MAN_B: dict(n=len(mB), sha256_first1mb=sha1mb(MAN_B))},
    sources={p: dict(model=m, tag=t, set=('A_edaic_part14' if (m, t, p, cc, sc) in [(a, b, c, d2, e) for a, b, c, d2, e in SET_A] else 'B_pitt'),
                     n_rows=int(sum(1 for _ in open(p))) - 1, sha256_first1mb=sha1mb(p))
             for (m, t, p, cc, sc) in SET_A + SET_B if os.path.exists(p)},
    missing_sources=missing,
    outputs=[cells_path, perclip_path, f"{ROWS}/M8.tsv"],
)
with open(f"{OUT}/M8_conflict_cells.json", 'w') as f: json.dump(side, f, indent=1)

prim = cells[(cells.arm == 'conflict') & (cells.source_tag.isin(['primary', 'af2_pitt468']))]
nbelow = int(prim['below_half'].sum())
print(f"\nTWELVE PRIMARY CONFLICT CELLS: {nbelow} of {len(prim)} have an interval entirely below 0.5")
print(prim[['model', 'set', 'auc', 'lo', 'hi', 'verdict']].to_string(index=False))

if disc_lines:
    with open(DISC, 'a') as f:
        f.write("\n\n## M8 conflict-cell intervals (part16) - " + pd.Timestamp.now().isoformat(timespec='seconds') + "\n" + "\n".join(disc_lines) + "\n")
print("\nWROTE", cells_path, perclip_path, f"{ROWS}/M8.tsv", f"{OUT}/M8_conflict_cells.json", sep="\n")
