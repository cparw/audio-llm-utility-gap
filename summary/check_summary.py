#!/usr/bin/env python3
"""Independent check of summary_all.csv.

1. Row count against every source fragment (Parts 16, 17, 20, 22), missing and duplicated rows.
2. Every PAPER VALUE row plus a 10% random sample of the other rows (numpy default_rng(0)):
   open the cited file and confirm the value. Per-clip files: recompute the AUC with independent
   Mann-Whitney code, and the interval with independent speaker bootstrap (2000 draws, fresh
   numpy.random.default_rng(0) per cell, speakers resampled with replacement, percentile 2.5 / 97.5).
   Summary files (json, small csv tables, npz scalars): look the value up.
3. Every quoted paper_sentence against the pasted tex, unless it is a 23 Sep addition or a
   decision pending note.
Reads only. Writes check_summary.csv (and prints a report). Nothing is written under omni_final.
"""
import csv, glob, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

OUT = 'summary'
SUMMARY = OUT + '/summary_all.csv'
R = '<local data dir>/release'
TEX = R + '/edaic_rerun/part17/part12_prep/pasted_tex_dryrun.tex'
NB = 2000
TOL_V = 0.00005 + 1e-9     # value agrees at 4 dp
TOL_CI = 0.00015           # interval ends agree at 4 dp (both sides rounded)

assert 'omni_final' not in OUT

# =============================================================== 1. fragments
def read_tsv(path):
    with open(path, newline='') as f:
        lines = [l.rstrip('\r') for l in f.read().split('\n') if l.strip()]
    hdr = lines[0].split('\t')
    if hdr[0] != 'id':
        hdr = ['id', 'what', 'value', 'lo', 'hi', 'n', 'n_spk', 'file']; body = lines
    else:
        body = lines[1:]
    out = []
    for l in body:
        c = l.split('\t'); c += [''] * (len(hdr) - len(c))
        out.append(dict(zip(hdr, c)))
    return hdr, out

frag_rows = []   # (fragment path, id, value_verified)
frag_files = {}
for p in sorted(glob.glob(R + '/edaic_rerun/part16/rows/*.tsv')) + sorted(glob.glob(R + '/edaic_rerun/part17/rows/*.tsv')) \
        + sorted(x for x in glob.glob(R + '/scores/part20/rows/*.tsv') if 'provisional' not in x) \
        + [R + '/scores/part22/rows/P22_verified.tsv']:
    h, rs = read_tsv(p)
    frag_files[p] = len(rs)
    for r in rs:
        v = r['value_verified'] if 'value_verified' in h else r['value']
        frag_rows.append((p, r['id'], v.strip()))
# seed table in PART16_RESULTS.md (lines 11-27): independent parse of every markdown table row
md = open(R + '/edaic_rerun/part16/PART16_RESULTS.md').read().split('\n')
seed_src = R + '/edaic_rerun/part16/PART16_RESULTS.md (seed table, lines 11-27)'
seed = []
for l in md[10:27]:
    c = [x.strip() for x in l.strip().strip('|').split('|')]
    seed.append(c)
frag_files[seed_src] = len(seed)
for i, c in enumerate(seed):
    frag_rows.append((seed_src, 'SEED%02d_%s' % (i + 1, c[0]), c[2]))

# side checks on fragments the builder did not use
side = []
seedmd = [l for l in open(R + '/edaic_rerun/part16/SEED_ROWS.md').read().split('\n') if l.startswith('| ') and not l.startswith('| id') and not l.startswith('|--')]
side.append('SEED_ROWS.md has %d table rows; the PART16_RESULTS.md seed block has %d; identical: %s' % (
    len(seedmd), len(seed), [l.strip() for l in seedmd] == [l.strip() for l in md[10:27]]))
# every table row in PART16_RESULTS.md after the seed block should be in a part16 rows/*.tsv
p16ids = Counter(i for (p, i, v) in frag_rows if '/part16/rows/' in p)
mdrows = []
for l in md[27:]:
    if l.startswith('| ') and not l.startswith('| id') and not l.startswith('|--'):
        c = [x.strip() for x in l.strip().strip('|').split('|')]
        mdrows.append(c)
md_missing = [c[0] for c in mdrows if c[0] not in p16ids]
side.append('PART16_RESULTS.md has %d table rows after the seed block; ids not found in any part16 rows/*.tsv: %d %s' % (
    len(mdrows), len(md_missing), sorted(set(md_missing))[:20]))
h22, p22 = read_tsv(R + '/scores/part22/rows/P22.tsv'); h22v, p22v = read_tsv(R + '/scores/part22/rows/P22_verified.tsv')
side.append('P22.tsv %d rows vs P22_verified.tsv %d rows; same id set: %s (order differs: %s)' % (len(p22), len(p22v), sorted(r['id'] for r in p22) == sorted(r['id'] for r in p22v), [r['id'] for r in p22] != [r['id'] for r in p22v]))
for p in sorted(glob.glob(R + '/scores/part20/rows/*_provisional.tsv')):
    ver = p.replace('_provisional', '')
    _, a = read_tsv(p); _, b = read_tsv(ver)
    miss = sorted(set(r['id'] for r in a) - set(r['id'] for r in b))
    side.append('%s: provisional %d rows, verified %d rows (verified >= provisional: %s; %d provisional ids renamed or dropped in the verified file)' % (
        os.path.basename(ver), len(a), len(b), len(b) >= len(a), len(miss)))

S = list(csv.DictReader(open(SUMMARY, newline='')))
sum_keys = Counter((r['fragment'], r['id'].split('#')[0], r['value_verified'].strip()) for r in S)
frag_keys = Counter(frag_rows)
missing = frag_keys - sum_keys
extra = sum_keys - frag_keys
per_frag_sum = Counter(r['fragment'] for r in S)
frag_count_mismatch = {p: (n, per_frag_sum.get(p, 0)) for p, n in frag_files.items() if n != per_frag_sum.get(p, 0)}
dup_uid = [k for k, n in Counter((r['part'], r['task/pod'], r['id']) for r in S).items() if n > 1]
dup_frag_ids = [k for k, n in Counter((p, i) for (p, i, v) in frag_rows).items() if n > 1]

# =============================================================== helpers
def fnum(s):
    try:
        return float(str(s).strip().lstrip('+').replace('\u2212', '-'))
    except (ValueError, TypeError):
        return None

def lead_num(s):
    m = re.match(r'^\s*([+-]?\d[\d.]*(?:[eE][+-]?\d+)?)\s*(\[.*\])?\s*$', str(s))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def auc_mw(y, s):
    """Mann-Whitney AUC with mid-ranks for ties. y in {0,1}."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s); y = y[ok]; s = s[ok]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def auc_fast(y, s):
    """Same statistic via scipy-free vectorised mid-ranks (used inside the bootstrap)."""
    s = np.asarray(s, dtype=float); y = np.asarray(y)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    mid = csum - (cnt - 1) / 2.0
    r = mid[inv]
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def resolve(f):
    """Return a list of existing paths for one file field (may name several files)."""
    f = f.strip()
    m = re.search(r'\(collected as ([^)]+)\)', f)
    alts = []
    if m:
        alts.append(m.group(1).strip()); f = f[:m.start()].strip()
    parts = re.split(r'\s+MINUS\s+|\s+\|\s+|\s+\+\s+|\s*;\s*|,\s+', f)
    parts = [re.sub(r'\s+recomputed vs .*$', '', p).strip() for p in parts if p.strip()]
    parts = [re.sub(r'\s+vs\s+.*$', '', p).strip() for p in parts]
    out = []; stale = []
    for p in parts + alts:
        cands = []
        q = p
        if q.startswith('<local data dir>/paper1_local_runs/'):
            stale.append(q)
            q2 = q.replace('<local data dir>/paper1_local_runs/', '<local data dir>/paper1_local_runs/')
            cands += [q, q2]
        elif q.startswith('/'):
            cands.append(q)
        else:
            if q.startswith('release/'):
                cands.append('<local data dir>/' + q)
            for b in [R, R + '/edaic_rerun', R + '/edaic_rerun/part16', R + '/edaic_rerun/part17', R + '/scores/part22',
                      R + '/scores/part22/verify', R + '/scores/part22/verify/out', R + '/scores/part22/pod_final_sync',
                      R + '/edaic_rerun/part16/POD4B', R + '/edaic_rerun/part16/POD4', R + '/scores/part20']:
                cands.append(os.path.join(b, q))
        hit = next((c for c in cands if os.path.isfile(c)), None)
        if hit is None:
            mm_ = next((re.search(r'(\w*?)(\d)\.\.(\d)', c) for c in cands if re.search(r'\d\.\.\d', c)), None)
            if mm_:
                for c in cands:
                    if '..' in c:
                        for k_ in range(int(mm_.group(2)), int(mm_.group(3)) + 1):
                            c2 = re.sub(r'\d\.\.\d', str(k_), c)
                            if os.path.isfile(c2) and c2 not in out:
                                out.append(c2)
        if hit and hit not in out:
            out.append(hit)
    return out, stale

# --------------------------------------------------------------- per-clip frames
NAMECOLS = ['clip', 'name', 'clip_id', 'id', 'path', 'clip_path', 'orig_clip_id', 'key']
SPKCOLS = ['speaker', 'spk', 'speaker_id', 'pid']
EXCL = {'label', 'fold', 'pair_id', 'in_distinct_set', 'n_repeats_in_arm', 'first_manifest_row', 'n_audio_tokens', 'dur_s',
        'capped', 'sr', 'window_start_ms', 'window_span_ms', 'n', 'n_spk', 'fold_armrefit', 'nested_layer', 'n_par_intervals'}

_frames = {}
def load_frame(path):
    if path in _frames:
        return _frames[path]
    fr = None
    try:
        if path.endswith('.npz'):
            z = np.load(path, allow_pickle=True)
            if 'label' not in z.files:
                raise ValueError
            n_ = len(z['label']); d_ = {}
            for k_ in z.files:
                a_ = z[k_]
                if a_.ndim == 1 and len(a_) == n_:
                    d_[k_] = a_
                elif a_.ndim == 2 and a_.shape[1] == n_ and a_.dtype.kind == 'f':
                    for j_ in range(a_.shape[0]):
                        d_['%s_rep%d' % (k_, j_)] = a_[j_]
            df = pd.DataFrame(d_)
            if not any(c in df.columns for c in SPKCOLS):
                raise ValueError
        else:
            sep = '\t' if path.endswith('.tsv') else ','
            df = pd.read_csv(path, sep=sep, low_memory=False)
        if 'label' in df.columns and len(df) >= 20 and any(c in df.columns for c in SPKCOLS):
            fr = df
    except Exception:
        fr = None
    _frames[path] = fr
    return fr

def arm_series(df):
    for c in ('arm', 'set'):
        if c in df.columns and df[c].astype(str).str.contains('conflict|agreement').any():
            return df[c].astype(str).str.extract(r'(conflict|agreement)')[0].fillna('')
    for c in NAMECOLS:
        if c in df.columns:
            b = df[c].astype(str).str.replace(r'.*/', '', regex=True)
            if b.str.match(r'^(noinv_)?(conflict|agreement)_').any():
                return b.str.extract(r'(conflict|agreement)_')[0].fillna('')
    return None

def repeat_groups(cols):
    g = defaultdict(list)
    for c in cols:
        m = re.match(r'^(.*?)(\d)(\D*)$', c)
        if m:
            g[(m.group(1), m.group(3))].append(c)
    return {k: sorted(v) for k, v in g.items() if len(v) >= 3}

def frame_candidates(path):
    """All natural statistics of one per-clip file: dict key -> (value, spec)."""
    df = load_frame(path)
    if df is None:
        return None
    y_all = pd.to_numeric(df['label'], errors='coerce')
    spk_col = next(c for c in SPKCOLS if c in df.columns)
    arms = arm_series(df)
    num_cols = [c for c in df.columns if c not in EXCL and c != spk_col and pd.api.types.is_numeric_dtype(df[c])]
    groups = repeat_groups(num_cols)
    subsets = {'all': np.ones(len(df), bool)}
    if arms is not None:
        for a in ('conflict', 'agreement'):
            if (arms == a).any():
                subsets[a] = (arms == a).values
    if 'in_distinct_set' in df.columns:
        d = df['in_distinct_set'].astype(str).isin(['1', 'True', 'true'])
        if arms is not None and (arms == 'agreement').any():
            subsets['agreement_distinct'] = ((arms == 'agreement') & d).values
    for c in ('in_agreement311_distinct',):
        if c in df.columns:
            subsets['agreement_distinct'] = ((arms == 'agreement') & df[c].astype(str).isin(['1', 'True', 'true'])).values
    base_ok = y_all.notna().values
    cand = {}
    for sname, m in subsets.items():
        mm = m & base_ok
        y = y_all.values[mm].astype(int)
        for c in num_cols:
            s = pd.to_numeric(df[c], errors='coerce').values[mm]
            if np.isnan(s).all():
                continue
            a = auc_mw(y, s)
            if not np.isnan(a):
                cand[('auc', sname, c)] = a
                cand[('dirfree', sname, c)] = max(a, 1 - a)
        for k, cs in groups.items():
            aucs = []
            for c in cs:
                s = pd.to_numeric(df[c], errors='coerce').values[mm]
                aucs.append(auc_mw(y, s))
            if not np.isnan(aucs).any():
                cand[('mean5', sname, '|'.join(cs))] = float(np.mean(aucs))
                mcol = np.nanmean(np.vstack([pd.to_numeric(df[c], errors='coerce').values[mm] for c in cs]), axis=0)
                cand[('aucofmean', sname, '|'.join(cs))] = auc_mw(y, mcol)
    info = dict(df=df, y=y_all, spk=df[spk_col].astype(str).values, subsets=subsets, spk_col=spk_col)
    return cand, info

def boot_ci(info, spec, spec2=None, kind='single'):
    """Speaker bootstrap. spec = (stat, subset, cols); spec2 for paired differences."""
    df, spk, subsets = info['df'], info['spk'], info['subsets']
    y_all = info['y'].values
    if kind == 'armdiff':
        m = (subsets[spec[1]] | subsets[spec2[1]]) & ~np.isnan(y_all)
    elif kind == 'pair':
        m = subsets[spec[1]] & ~np.isnan(y_all)
    else:
        m = subsets[spec[1]] & ~np.isnan(y_all)
    idx_all = np.where(m)[0]
    uspk = np.unique(spk[idx_all])
    rows_of = defaultdict(list)
    for i in idx_all:
        rows_of[spk[i]].append(i)
    rows_of = {k: np.array(v) for k, v in rows_of.items()}

    colv = {c: pd.to_numeric(df[c], errors='coerce').values for sp in (spec, spec2) if sp for c in sp[2].split('|')}
    flip = {}
    for sp in (spec, spec2):
        if sp and sp[0] == 'dirfree':
            mm_ = subsets[sp[1]] & ~np.isnan(y_all)
            flip[sp] = auc_mw(y_all[mm_].astype(int), colv[sp[2]][mm_]) < 0.5

    def stat(sp, ii):
        st, sub, cols = sp
        sel = ii[subsets[sub][ii]]
        yy = y_all[sel].astype(int)
        if yy.sum() == 0 or yy.sum() == len(yy):
            return np.nan
        cs = cols.split('|')
        if st == 'aucofmean':
            return auc_fast(yy, np.nanmean(np.vstack([colv[c][sel] for c in cs]), axis=0))
        vals = [auc_fast(yy, colv[c][sel]) for c in cs]
        a = float(np.mean(vals))
        if st == 'dirfree' and flip.get(sp):
            a = 1 - a
        return a
    rng = np.random.default_rng(0)
    out = []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.concatenate([rows_of[uspk[p]] for p in pick])
        a = stat(spec, ii)
        if spec2 is not None:
            b = stat(spec2, ii)
            a = a - b
        if not np.isnan(a):
            out.append(a)
    if len(out) < 10:
        return None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)

# --------------------------------------------------------------- literal lookup
def flat_numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values(): flat_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj: flat_numbers(v, out)
    elif isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for t in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', obj):
            try: out.append(float(t))
            except ValueError: pass
    return out

_lit = {}
def literal_numbers(path):
    if path in _lit:
        return _lit[path]
    nums = []
    try:
        if path.endswith('.json'):
            nums = flat_numbers(json.load(open(path)), [])
        elif path.endswith('.npz'):
            z = np.load(path, allow_pickle=True)
            for k in z.files:
                a = z[k]
                if a.dtype.kind in 'fiub' and a.size <= 50:
                    nums += [float(x) for x in a.ravel()]
        else:
            txt = open(path, errors='replace').read()
            if len(txt) < 3_000_000:
                nums = flat_numbers(txt, [])
    except Exception:
        nums = []
    _lit[path] = np.array(nums, dtype=float) if nums else np.array([])
    return _lit[path]

def text_lookup(path, value):
    try:
        return value.strip() in open(path, errors='replace').read()
    except Exception:
        return False

def npz_candidates(path):
    z = np.load(path, allow_pickle=True)
    if 'label' not in z.files:
        return None
    y = z['label'].astype(int); cand = {}
    for k in z.files:
        a = z[k]
        if a.dtype.kind != 'f':
            continue
        if a.ndim == 2 and a.shape[1] == len(y):
            cand[('npz_mean5', 'all', k)] = float(np.mean([auc_mw(y, r) for r in a]))
        elif a.ndim == 1 and len(a) == len(y):
            cand[('npz_auc', 'all', k)] = auc_mw(y, a)
    spk = z['spk'].astype(str) if 'spk' in z.files else None
    return cand, dict(z=z, y=y, spk=spk)

def npz_boot(info, key):
    z, y, spk = info['z'], info['y'], info['spk']
    a = z[key[2]]
    vecs = [r for r in a] if a.ndim == 2 else [a]
    uspk = np.unique(spk)
    rows_of = {s: np.where(spk == s)[0] for s in uspk}
    rng = np.random.default_rng(0); out = []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.concatenate([rows_of[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out.append(float(np.mean([auc_fast(yy, v[ii]) for v in vecs])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)

# --------------------------------------------------------------- custom recomputes (own code)
E = R + '/edaic_rerun'
P14 = {'Qwen2.5-Omni': E + '/part14/p14_o25_zeroshot_scores.csv', 'Qwen2-Audio': E + '/part14/p14_q2a_zeroshot_scores.csv',
       'Qwen3-Omni-30B-A3B': E + '/part14/p14_q3o_zeroshot_scores.csv', 'AudioFlamingo2': E + '/part14/p14_af2.csv',
       'Audio Flamingo 2': E + '/part14/p14_af2.csv', 'AudioFlamingo3': E + '/part14/p14_af3.csv',
       'Audio Flamingo 3': E + '/part14/p14_af3.csv', 'Kimi-Audio': E + '/part14/p14_kimi_zeroshot_scores.csv'}

def pc(path, score='p_yes'):
    """Per-clip frame with key, spk, y, s, arm (own loader)."""
    d = pd.read_csv(path, low_memory=False)
    namec = next(c for c in NAMECOLS if c in d.columns)
    spkc = next(c for c in SPKCOLS if c in d.columns)
    key = d[namec].astype(str).str.replace(r'.*/', '', regex=True).str.replace(r'\.(wav|flac)$', '', regex=True).str.replace(r'^noinv_', '', regex=True)
    arm = key.str.extract(r'^(conflict|agreement)_')[0].fillna('')
    return pd.DataFrame(dict(key=key, spk=d[spkc].astype(str), y=pd.to_numeric(d['label']).astype(int),
                             s=pd.to_numeric(d[score], errors='coerce'), arm=arm))

def boot(spk, fn):
    uspk = np.unique(spk)
    rows_of = {u: np.where(spk == u)[0] for u in uspk}
    rng = np.random.default_rng(0); out = []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.concatenate([rows_of[uspk[p]] for p in pick])
        x = fn(ii)
        if x is not None and not np.isnan(x):
            out.append(x)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def two_class(y):
    return 0 < y.sum() < len(y)

def custom(r, v):
    i, w, t = r['id'].split('#')[0], r['what'], r['task/pod']
    # M4 yes rates, from the Part 14 per-clip files named in M4_yes_rates.py
    if t == 'M4' and w.startswith('yes_rate'):
        m = re.match(r'yes_rate_diff_conflict_minus_agreement_(.+)$', w)
        if m:
            d = pc(P14[m.group(1)]); yes = (d.s.values > 0.5).astype(float); arm = d.arm.values; spk = d.spk.values
            f = lambda ii: yes[ii][arm[ii] == 'conflict'].mean() - yes[ii][arm[ii] == 'agreement'].mean()
            lo_, hi_ = boot(spk, f)
            return dict(value=float(f(np.arange(len(d)))), lo=lo_, hi=hi_, config='yes = p_yes > 0.5, conflict minus agreement, paired speaker draw, ' + P14[m.group(1)])
        m = re.match(r'yes_rate_(.+)_(conflict|agreement)$', w)
        if m:
            d = pc(P14[m.group(1)]); d = d[d.arm == m.group(2)].reset_index(drop=True)
            yes = (d.s.values > 0.5).astype(float)
            lo_, hi_ = boot(d.spk.values, lambda ii: yes[ii].mean())
            return dict(value=float(yes.mean()), lo=lo_, hi=hi_, config='yes = p_yes > 0.5 in arm %s, %s' % (m.group(2), P14[m.group(1)]))
    # M7fix paired: mean-of-five encoder probe minus zero-shot, Pitt
    if t == 'M7fix' and 'paired' in w:
        z = np.load(R + '/overnight2/part10/pitt_enc_nested5_oof.npz')
        name = pd.Series(z['name']).str.replace(r'\.wav$', '', regex=True).values
        zs = pc(R + '/omni_final/omni_pitt_zeroshot_scores.csv').set_index('key').loc[name]
        y = z['label']; oof = z['oof']; s0 = zs.s.values; spk = z['spk'].astype(str)
        assert np.array_equal(zs.y.values, y)
        f = lambda ii: (np.mean([auc_fast(y[ii], o[ii]) for o in oof]) - auc_fast(y[ii], s0[ii])) if two_class(y[ii]) else np.nan
        lo_, hi_ = boot(spk, f)
        return dict(value=float(f(np.arange(len(y)))), lo=lo_, hi=hi_, config='mean of 5 per-repeat AUCs (pitt_enc_nested5_oof.npz) minus omni_pitt_zeroshot p_yes, paired',
                    note='cited file pitt_FIXED.csv holds only the averaged OOF; recomputed from the npz the M7fix row names')
    # E-DAIC projector fine-tune minus answer, full window
    if i == 'MEDAIC_full_sft_minus_answer':
        a = pc(E + '/variants/sft_full_oof.csv'); b = pc(E + '/variants/o25_full_zeroshot_scores.csv')
        a['k'] = a.spk; b['k'] = b.spk
        m_ = a.merge(b, on='k', suffixes=('_a', '_b'))
        assert (m_.y_a == m_.y_b).all()
        y = m_.y_a.values; sa = m_.s_a.values; sb = m_.s_b.values; spk = m_.k.values
        f = lambda ii: auc_fast(y[ii], sa[ii]) - auc_fast(y[ii], sb[ii]) if two_class(y[ii]) else np.nan
        lo_, hi_ = boot(spk, f)
        return dict(value=float(f(np.arange(len(y)))), lo=lo_, hi=hi_, config='sft_full_oof p_yes minus variants/o25_full_zeroshot p_yes (original full-window run), paired, n=%d' % len(y))
    # transcript vs audio agreement, E-DAIC pairs
    m = re.match(r'1a_(o25|q2a)_(pearson_text_audio|same_decision)$', i)
    if m or i in ('SEED07_1a', 'SEED08_1a'):
        mk, kind = (m.group(1), m.group(2)) if m else ('o25', 'pearson_text_audio' if i == 'SEED07_1a' else 'same_decision')
        d = pd.read_csv(E + '/part16/POD1/p14_%s_1a_joined.csv' % mk)
        tx, au = d.p_yes_text.values, d.p_yes_audio.values
        val = float(np.corrcoef(tx, au)[0, 1]) if kind.startswith('pearson') else float(np.mean((tx > 0.5) == (au > 0.5)))
        return dict(value=val, config='%s over %d rows of p14_%s_1a_joined.csv' % (kind, len(d), mk),
                    note='cited file for the SEED row (B_text_conflict.json) does not hold this number' if i.startswith('SEED') else '')
    m = re.match(r'1f_(agree_gen_logit|share_yesno)_(all|conflict|agreement)$', i)
    if m:
        d = pd.read_csv(E + '/part16/POD1/p14_o25_greedy.csv')
        if m.group(2) != 'all': d = d[d.arm == m.group(2)]
        if m.group(1) == 'share_yesno':
            val = float(d.is_yes_no.astype(int).mean())
        else:
            d = d[d.is_yes_no.astype(int) == 1]
            val = float((d.first_word.str.strip().str.lower() == d.logit_decision.str.strip().str.lower()).mean())
        return dict(value=val, config='%s on %d rows of p14_o25_greedy.csv (own comparison of first_word and logit_decision)' % (m.group(1), len(d)))
    m = re.match(r'1b_(p2|p3)_(conflict|agreement)_delta$', i)
    if m or i == '1b_max_abs_change':
        base = pc(P14['Qwen2.5-Omni'])
        def armauc(d, a): dd = d[d.arm == a]; return auc_mw(dd.y.values, dd.s.values)
        deltas = {}
        for pw in ('p2', 'p3'):
            d = pc(E + '/part16/POD1/p14_o25_audio_%s.csv' % pw)
            for a in ('conflict', 'agreement'):
                deltas[(pw, a)] = armauc(d, a) - armauc(base, a)
        val = deltas[(m.group(1), m.group(2))] if m else max(abs(x) for x in deltas.values())
        return dict(value=float(val), config='AUC(wording file, arm) minus AUC(part14 original wording, arm)')
    if i == '1e_conflict_auc_range':
        vals = []
        for f_ in (E + '/part16/POD1/p14_thr060_omni.csv', P14['Qwen2.5-Omni'], E + '/part16/POD1/p14_thr080_omni.csv'):
            d = pc(f_); dd = d[d.arm == 'conflict']; vals.append(auc_mw(dd.y.values, dd.s.values))
        return dict(value=float(max(vals) - min(vals)), config='conflict AUC at 0.60 / 0.70 (part14 file) / 0.80: ' + ' / '.join('%.4f' % x for x in vals))
    m = re.match(r'T5_(.+)_layers_(above_ci|above|strict)$', i)
    if m:
        d = pd.read_csv(R + '/edaic_rerun/part17/T5_omitted_curves.csv'); d = d[d.dataset == m.group(1)]
        if m.group(2) == 'above': val = int((d.probe_auc > d.answer_auc).sum()); how = 'probe_auc > answer_auc recomputed'
        elif m.group(2) == 'strict': val = int(((d.auc_mean - d.auc_std) > d.answer_auc).sum()); how = 'auc_mean - auc_std > answer_auc recomputed'
        else: val = int(d.above_answer_ci_hi.sum()); how = 'count of the above_answer_ci_hi flag (answer upper bound not in file)'
        return dict(value=float(val), config='%s over %d layer rows' % (how, len(d)))
    m = re.match(r'3a_(pitt|pcgita)_enc_nested$', i)
    if m:
        d = pd.read_csv(E + '/part16/POD3/q3o_%s_encoder_nested_oof.csv' % m.group(1))
        y = d.label.values.astype(int); reps = [d['p_probe_rep%d' % k].values for k in range(5)]; spk = d.speaker.astype(str).values
        f = lambda ii: np.mean([auc_fast(y[ii], o[ii]) for o in reps]) if two_class(y[ii]) else np.nan
        lo_, hi_ = boot(spk, f)
        return dict(value=float(f(np.arange(len(y)))), lo=lo_, hi=hi_, config='mean of 5 per-repeat AUCs, POD3/q3o_%s_encoder_nested_oof.csv' % m.group(1),
                    note='cited file is the per-layer csv, which does not hold this value; recomputed from the nested OOF file beside it')
    return None


def paired_files(fa, fb, sa='p_yes', sb='p_yes', key='key'):
    a = pc(fa, sa); b = pc(fb, sb)
    if key == 'spk':
        a['key'] = a.spk; b['key'] = b.spk
    m_ = a.merge(b, on='key', suffixes=('_a', '_b'))
    assert (m_.y_a == m_.y_b).all() and len(m_) == len(a) == len(b), (fa, fb, len(m_), len(a), len(b))
    y = m_.y_a.values; xa = m_.s_a.values; xb = m_.s_b.values; spk = m_.spk_a.values
    f = lambda ii: auc_fast(y[ii], xa[ii]) - auc_fast(y[ii], xb[ii]) if two_class(y[ii]) else np.nan
    lo_, hi_ = boot(spk, f)
    return float(f(np.arange(len(y)))), lo_, hi_, len(y)

def custom2(r, v):
    i, w, t = r['id'].split('#')[0], r['what'], r['task/pod']
    O = R + '/omni_final'
    m = re.match(r'P16\.POD2\.(D1|D2|D3)$', i)
    if m:
        L = E + '/part16/POD2/lora_pitt_oof.csv'; F = O + '/omnisft_pitt_oof.csv'; Z = O + '/omni_pitt_zeroshot_scores.csv'
        fa, fb = {'D1': (L, F), 'D2': (L, Z), 'D3': (F, Z)}[m.group(1)]
        val, lo_, hi_, nn = paired_files(fa, fb)
        return dict(value=val, lo=lo_, hi=hi_, config='AUC(%s) minus AUC(%s), clips aligned by name, paired speaker draw, n=%d' % (os.path.basename(fa), os.path.basename(fb), nn))
    if i == 'MEDAIC_w30_proj_minus_answer' or re.match(r'MEDAIC_w30_(enc|llm|ans|proj)_minus_answer$', i):
        st = re.match(r'MEDAIC_w30_(\w+?)_minus_answer', i).group(1)
        val, lo_, hi_, nn = paired_files(O + '/omni_edaic_%s_nested_oof.csv' % st, O + '/omni_edaic_zeroshot_scores.csv', 'p_probe', 'p_yes')
        return dict(value=val, lo=lo_, hi=hi_, config='AUC(omni_edaic_%s_nested_oof p_probe) minus AUC(omni_edaic_zeroshot p_yes), paired, n=%d' % (st, nn))
    m = re.match(r'T7_PROXY([DA])_(enc|proj|llm|ans)_meanof5_minus_answer$', i)
    if m:
        z = np.load(E + '/part17/T7_scratch/' + {'D': 'T7_oof_repeats_D_mac_podpicks.npz', 'A': 'T7_oof_repeats_A.npz'}[m.group(1)], allow_pickle=True)
        pcl = pd.read_csv(E + '/part16/EDAICFULL/edaic_full_perclip.csv')
        assert [c.replace('.wav', '') for c in z['clip']] == pcl.pid.astype(str).tolist()
        y = pcl.label.values; ans = pcl.p_yes_answer.values; reps = z['%s_oof_repeats' % m.group(2)]; spk = pcl.pid.astype(str).values
        f = lambda ii: np.mean([auc_fast(y[ii], o[ii]) for o in reps]) - auc_fast(y[ii], ans[ii]) if two_class(y[ii]) else np.nan
        lo_, hi_ = boot(spk, f)
        return dict(value=float(f(np.arange(len(y)))), lo=lo_, hi=hi_, config='mean of 5 per-repeat AUCs (npz) minus edaic_full_perclip p_yes_answer, paired')
    m = re.match(r'[A-Z]_(.+)_mean_(PD|HC)$', w)
    if t == 'M11' and m:
        d = pd.read_csv(E + '/part16/M11_kcl_wer.csv'); col = {'wer_vs_local_copy': 'wer_local'}.get(m.group(1), m.group(1))
        if col not in d.columns:
            return None
        d = d[d.label == (1 if m.group(2) == 'PD' else 0)]
        x = pd.to_numeric(d[col], errors='coerce').values; spk = d.spk.astype(str).values
        lo_, hi_ = boot(spk, lambda ii: np.nanmean(x[ii]))
        return dict(value=float(np.nanmean(x)), lo=lo_, hi=hi_, config='mean of %s over %s rows of M11_kcl_wer.csv' % (col, m.group(2)))
    if t == 'M9' and 'median answer mass' in w:
        paths, _ = resolve(r['file'])
        if not paths: return None
        d = pd.read_csv(paths[0]); mc = next(c for c in ('mass', 'answer_mass', 'mass_text') if c in d.columns)
        spkc = next(c for c in SPKCOLS if c in d.columns)
        x = pd.to_numeric(d[mc], errors='coerce').values; spk = d[spkc].astype(str).values
        lo_, hi_ = boot(spk, lambda ii: float(np.nanmedian(x[ii])))
        return dict(value=float(np.nanmedian(x)), lo=lo_, hi=hi_, config='median of %s, %s' % (mc, paths[0]))
    if i == 'POD1b_range_overall_seeds_1_2_3':
        a = []
        for k_ in (1, 2, 3):
            d = pc(R + '/scores/part20/POD1b/std_ft_seed%d_pitt_oof.csv' % k_); a.append(auc_mw(d.y.values, d.s.values))
        return dict(value=float(round(max(a) - min(a), 6)), lo=min(a), hi=max(a), config='max minus min of overall AUC over seed files 1-3: ' + ', '.join('%.4f' % x for x in a))
    m = re.match(r'POD4_4a_(nv|pg)_pd_(zs|enc)$', i)
    if m:
        ds = m.group(1); kind = m.group(2)
        if kind == 'zs':
            fb, fa, col = E + '/part16/POD4/p16_%s_before_zeroshot_scores.csv' % ds, E + '/part16/POD4/p16_%s_after_zeroshot_scores.csv' % ds, 'p_yes'
        else:
            fb, fa, col = E + '/part16/POD4/p16_%s_before_enc_nested_oof.csv' % ds, E + '/part16/POD4/p16_%s_after_enc_nested_oof.csv' % ds, 'p_probe'
        val, lo_, hi_, nn = paired_files(fa, fb, col, col)
        return dict(value=val, lo=lo_, hi=hi_, config='AUC(after) minus AUC(before), %s, paired, n=%d' % (col, nn))
    if i in ('SEED02_1a', 'SEED05_1a', 'SEED03_1a', 'SEED06_1a'):
        mk = 'o25' if i in ('SEED02_1a', 'SEED03_1a') else 'q2a'; arm = 'conflict' if i in ('SEED02_1a', 'SEED05_1a') else 'agreement'
        d = pd.read_csv(E + '/part16/POD1/p14_%s_1a_joined.csv' % mk); d = d[d.arm == arm]
        d = d.drop_duplicates('key') if arm == 'agreement' else d
        y = d.label.values.astype(int); x = d.p_yes_text.values; spk = d.speaker_id.astype(str).values
        lo_, hi_ = boot(spk, lambda ii: auc_fast(y[ii], x[ii]) if two_class(y[ii]) else np.nan)
        return dict(value=float(auc_mw(y, x)), lo=lo_, hi=hi_, config='transcript p_yes_text AUC, %s arm, p14_%s_1a_joined.csv, n=%d' % (arm, mk, len(y)),
                    note='cited file is the Part 15 summary; recomputed from the Part 16 joined per-clip file')
    m = re.match(r'T5_(\w+?)_probe_max$', i)
    if m:
        d = pd.read_csv(R + '/omni_final/omni_%s_encoder_perlayer.csv' % m.group(1))
        return dict(value=float(d.auc_oof.max()), config='max of auc_oof over %d layers, omni_%s_encoder_perlayer.csv (argmax layer %d)' % (len(d), m.group(1), int(d.layer[d.auc_oof.idxmax()])))
    if i == 'P22_S1_capped900':
        d = pd.read_csv(E + '/variants/windows_full.csv')
        return dict(value=float(int((d.capped == 1).sum())), config='count capped==1 in variants/windows_full.csv (%d windows; %d with src_dur_s > 900)' % (len(d), int((d.src_dur_s > 900).sum())))
    if i in ('P22_all275_lift', 'P22_auc_lifted_all275', 'P22_auc_lifted_all275_ruleids', 'P22_all275_diff', 'P22_auc_diff_all275'):
        d = pd.read_csv(R + '/scores/part22/edaic_lifted_perclip.csv')
        y = d.label.values.astype(int); a = d.p_yes_lifted.values; b = d.p_yes_default_ref.values; spk = d.pid.astype(str).values
        cut = int((d.audio_tok_default == 7500).sum()); cap = int(d.capped.sum())
        if 'diff' in i:
            f = lambda ii: auc_fast(y[ii], a[ii]) - auc_fast(y[ii], b[ii]) if two_class(y[ii]) else np.nan
        else:
            f = lambda ii: auc_fast(y[ii], a[ii]) if two_class(y[ii]) else np.nan
        lo_, hi_ = boot(spk, f)
        return dict(value=float(f(np.arange(len(y)))), lo=lo_, hi=hi_, config='edaic_lifted_perclip.csv p_yes_lifted%s, n=%d; windows at 7500 default tokens: %d; capped: %d' % (' minus p_yes_default_ref, paired' if 'diff' in i else '', len(y), cut, cap))
    if i == 'P22_s2_466_verdict':
        a = pd.read_csv(R + '/scores/part22/verify/out/s2_default.csv'); b = pd.read_csv(R + '/scores/part22/verify/out/s2_cut300.csv')
        a = a[a.pid == 466].iloc[0]; b = b[b.pid == 466].iloc[0]
        same = a.n_audio_tok == b.n_audio_tok == 7500 and a.p_yes_paper == b.p_yes_paper
        return dict(value='TRUNCATION PROVEN' if same else 'NOT PROVEN', config='pid 466: tokens %d vs %d, p_yes %.6f vs %.6f (full vs first 300 s)' % (a.n_audio_tok, b.n_audio_tok, a.p_yes_paper, b.p_yes_paper))
    if i == 'P22_s1_quotes':
        a = json.load(open(R + '/scores/part22/verify/out/step1_check_pod.json'))['summary']
        b = json.load(open(R + '/scores/part22/verify/out/step1_check_mac_negcontrol.json'))
        mac_ok = sum(1 for x in b['refs'] if x['status'] == 'OK' and x['ref'].startswith('edaic_rerun/'))
        ok = a['ok'] + mac_ok
        val = '%d/%d OK, %d mismatch' % (ok, a['n_refs'], a['mismatch'])
        return dict(value=val if val == r['value_verified'].strip() else val, config='pod %d OK / %d skipped / %d mismatch; Mac scripts OK %d' % (a['ok'], a['skipped'], a['mismatch'], mac_ok))
    return None

# =============================================================== 2. value checks
PV = 'PAPER VALUE'
others = [i for i, r in enumerate(S) if r['paper_role'] != PV]
k = int(round(0.10 * len(others)))
samp = set(np.random.default_rng(0).choice(others, size=k, replace=False).tolist())
EXTRA = {'P22_all275_lift', 'P22_all275_diff', 'P22_auc_diff_all275'}   # SUPPORT rows behind a contradicted sentence
to_check = [i for i, r in enumerate(S) if r['paper_role'] == PV or i in samp or r['id'] in EXTRA]

def is_diff(r):
    t = (r['id'] + ' ' + r['what']).lower()
    return any(w in t for w in ('minus', 'gap', 'diff', 'paired', 'drop', ' vs ', 'gain'))

results = []
for i in to_check:
    r = S[i]
    vv = r['value_verified'].strip()
    v = lead_num(vv)
    lo, hi = fnum(r['lo']), fnum(r['hi'])
    n = fnum(r['n'])
    paths, stale = resolve(r['file'])
    res = dict(row=i + 2, part=r['part'], task=r['task/pod'], id=r['id'], paper_role=r['paper_role'],
               sampled='paper value' if r['paper_role'] == PV else ('random 10%' if i in samp else 'extra: backs a contradicted sentence'), value_verified=vv,
               lo=r['lo'], hi=r['hi'], n=r['n'], file=r['file'], resolved=' ; '.join(paths),
               method='', config='', check_value='', check_lo='', check_hi='', value_status='', ci_status='', note='')
    if stale:
        res['note'] = 'cited path is stale (%s); resolved under <local data dir>/' % stale[0]
    if not paths:
        res['value_status'] = 'FILE NOT FOUND'
        results.append(res); continue
    cu = custom(r, v)
    if cu is None:
        cu = custom2(r, v)
    if cu is not None:
        res.update(method=cu.get('method', 'custom recompute from per-clip file (own code)'), config=cu['config'],
                   check_value='%.4f' % cu['value'] if isinstance(cu['value'], float) else str(cu['value']))
        cv = cu['value']
        okv = (abs(cv - v) <= TOL_V) if (isinstance(cv, float) and v is not None) else (str(cv) == vv)
        res['value_status'] = ('MATCH (recomputed)' if okv else 'MISMATCH (recomputed %s vs row %s)' % (res['check_value'], vv))
        if cu.get('lo') is not None and lo is not None and hi is not None:
            res['check_lo'], res['check_hi'] = '%.4f' % cu['lo'], '%.4f' % cu['hi']
            ok = abs(cu['lo'] - lo) <= TOL_CI and abs(cu['hi'] - hi) <= TOL_CI
            res['ci_status'] = 'CI MATCH' if ok else 'CI DIFF (row [%s, %s] vs indep [%.4f, %.4f])' % (r['lo'], r['hi'], cu['lo'], cu['hi'])
        if cu.get('note'):
            res['note'] = (res['note'] + '; ' if res['note'] else '') + cu['note']
        results.append(res)
        print('%4d %-6s %-8s %-45s %-10s %-22s %s' % (res['row'], r['part'], r['task/pod'], r['id'][:45], vv[:10], res['value_status'], res['ci_status']), flush=True)
        continue
    if v is None and re.match(r'^[A-Z]+_(-?\d+\.\d+)$', vv):
        v = float(vv.split('_')[1]); res['note'] = (res['note'] + '; ' if res['note'] else '') + 'numeric part of %s checked' % vv
    if v is None:
        found = any(text_lookup(p, vv) for p in paths)
        res['method'] = 'text lookup'
        res['value_status'] = 'FOUND IN FILE (text)' if found else 'TEXT NOT FOUND'
        results.append(res); continue

    done = False
    # ---- (a) per-clip recompute, every per-clip file named
    frames = []
    for p in paths:
        if p.endswith(('.csv', '.tsv')):
            fc = frame_candidates(p)
            if fc: frames.append((p, fc))
        elif p.endswith('.npz'):
            fc = frame_candidates(p)
            if fc: frames.append((p, fc))
    best = None
    diff_row = is_diff(r)
    PRI = {'auc': 0, 'mean5': 0, 'aucofmean': 1, 'dirfree': 2}
    def consider(score, p, k1, k2, a, info):
        global_best[0] = global_best[0] if (global_best[0] is not None and global_best[0][0] <= score) else (score, p, k1, k2, a, info)
    global_best = [None]
    for p, (cand, info) in frames:
        keys = list(cand.keys())
        for key, a in cand.items():
            if abs(a - v) <= TOL_V:
                sub_n = int(info['subsets'][key[1]].sum())
                consider((1 if diff_row else 0, 0 if (n is None or sub_n == n) else 1, PRI[key[0]]), p, key, None, a, info)
        if diff_row:
            for key in keys:
                if key[1] != 'conflict' or key[0] == 'dirfree':
                    continue
                for sub2 in [x for x in info['subsets'] if x.startswith('agreement')]:
                    k2 = (key[0], sub2, key[2])
                    if k2 in cand and abs(cand[key] - cand[k2] - v) <= TOL_V:
                        consider((0, 0, PRI[key[0]]), p, key, k2, cand[key] - cand[k2], info)
            for a_ in keys:
                for b_ in keys:
                    if a_ != b_ and a_[1] == b_[1] and 'dirfree' not in (a_[0], b_[0]) and abs(cand[a_] - cand[b_] - v) <= TOL_V:
                        consider((0, 1, PRI[a_[0]] + PRI[b_[0]]), p, a_, b_, cand[a_] - cand[b_], info)
    best = global_best[0]
    if best is not None:
        _, p, k1, k2, a, info = best
        res['method'] = 'per-clip recompute (own Mann-Whitney)'
        res['config'] = '%s %s [%s]' % (k1[0], k1[1], k1[2]) + ('' if k2 is None else ' MINUS %s %s [%s]' % (k2[0], k2[1], k2[2]))
        res['check_value'] = '%.4f' % a
        res['value_status'] = 'MATCH (recomputed)'
        done = True
        if lo is not None and hi is not None and info is not None:
            if k2 is None:
                ci = boot_ci(info, k1)
            elif k1[1] != k2[1]:
                ci = boot_ci(info, k1, k2, kind='armdiff')
            else:
                ci = boot_ci(info, k1, k2, kind='pair')
            if ci:
                res['check_lo'], res['check_hi'] = '%.4f' % ci[0], '%.4f' % ci[1]
                ok = abs(ci[0] - lo) <= TOL_CI and abs(ci[1] - hi) <= TOL_CI
                res['ci_status'] = 'CI MATCH' if ok else 'CI DIFF (row [%s, %s] vs indep [%.4f, %.4f])' % (r['lo'], r['hi'], ci[0], ci[1])
        elif lo is not None and hi is not None:
            res['ci_status'] = 'CI not recomputed (two-file difference)'
    # ---- (b) literal lookup in summary files
    if not done:
        hits = []
        for p in paths:
            if load_frame(p) is not None:
                continue   # per-clip file: a literal hit among thousands of cells proves nothing
            nums = literal_numbers(p)
            if nums.size and np.any(np.abs(nums - v) <= TOL_V):
                hits.append(p)
            elif nums.size and np.any(np.abs(np.abs(nums) - abs(v)) <= TOL_V) and 'cos' in (r['id'] + r['what']).lower():
                hits.append(p + ' (magnitude)')
        if hits:
            res['method'] = 'lookup in cited summary file'
            res['value_status'] = 'FOUND IN FILE'
            res['config'] = hits[0]
            if lo is not None and hi is not None:
                nums = literal_numbers(hits[0].replace(' (magnitude)', ''))
                okl = np.any(np.abs(nums - lo) <= TOL_V); okh = np.any(np.abs(nums - hi) <= TOL_V)
                res['ci_status'] = 'CI ends found in file' if (okl and okh) else 'CI ends NOT both in file (lo %s, hi %s)' % (okl, okh)
            done = True
    if not done:
        if frames:
            res['method'] = 'per-clip recompute (own Mann-Whitney)'
            near = []
            for p, (cand, info) in frames:
                for key, a in cand.items():
                    near.append((abs(a - v), a, key))
            near.sort(key=lambda x: x[0])
            res['value_status'] = 'NOT REPRODUCED'
            res['note'] = (res['note'] + '; ' if res['note'] else '') + 'closest natural statistic: ' + '; '.join(
                '%.4f (%s %s [%s])' % (a, k_[0], k_[1], k_[2]) for _, a, k_ in near[:2])
        else:
            res['method'] = 'lookup in cited file'
            res['value_status'] = 'NOT FOUND IN FILE'
    if res['value_status'] in ('NOT FOUND IN FILE', 'NOT REPRODUCED'):
        sib = []
        if r['task/pod'] == 'T2':
            for q in sorted(glob.glob(R + '/edaic_rerun/part17/T2/*')) + [R + '/edaic_rerun/part17/T2_folds.json']:
                if os.path.isfile(q) and os.path.getsize(q) < 3_000_000:
                    nums = literal_numbers(q)
                    if nums.size and np.any(np.abs(nums - v) <= max(TOL_V, abs(v) * 5e-4)):
                        sib.append(q)
        for p in paths:
            stem = re.sub(r'(_perclip|_summary|_oof|_perlayer)?\.(csv|tsv|json|npz|md)$', '', os.path.basename(p))
            stem = re.sub(r'_(encoder|enc)_?$', '', stem)
            for q in sorted(glob.glob(os.path.join(os.path.dirname(p), '*')), key=lambda q_: (not os.path.basename(q_).startswith(stem), q_)):
                if q == p or not q.endswith(('.json', '.csv', '.md', '.txt', '.stdout', '.log')):
                    continue
                if not os.path.basename(q).startswith(stem[:12]) or os.path.getsize(q) > 3_000_000:
                    continue
                nums = literal_numbers(q)
                if nums.size and np.any(np.abs(nums - v) <= TOL_V):
                    sib.append(q)
        if sib:
            res['value_status'] = 'FOUND ONLY IN SIBLING FILE'
            res['config'] = sib[0]
            if lo is not None and hi is not None:
                nums = literal_numbers(sib[0])
                okl = np.any(np.abs(nums - lo) <= TOL_V); okh = np.any(np.abs(nums - hi) <= TOL_V)
                res['ci_status'] = 'CI ends found in sibling' if (okl and okh) else 'CI ends NOT both in sibling (lo %s, hi %s)' % (okl, okh)
    results.append(res)
    print('%4d %-6s %-8s %-45s %-10s %-22s %s' % (res['row'], r['part'], r['task/pod'], r['id'][:45], vv[:10], res['value_status'], res['ci_status']), flush=True)

# =============================================================== 2b. printed-number audit (own rounding)
from decimal import Decimal, ROUND_HALF_UP
printed_rows = []
for i_, r in enumerate(S):
    pc_ = r['printed_check']
    if not pc_:
        continue
    m = re.match(r'printed ([\d.]+)', pc_)
    pr = m.group(1)
    v_ = lead_num(r['value_verified'])
    if pr == '89':
        indep = 'MATCH' if v_ == 89 else 'MISMATCH'; rv = str(int(v_)) if v_ is not None else ''
    else:
        x = Decimal(r['value_verified'].split()[0].lstrip('+'))
        if 'cos' in r['id']: x = abs(x)
        rv = str(x.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)); indep = 'MATCH' if rv == pr else 'MISMATCH'
    builder = pc_.rsplit(': ', 1)[-1]
    printed_rows.append(dict(row=i_ + 2, id=r['id'], role=r['paper_role'], printed=pr, verified=r['value_verified'], check_round=rv, indep=indep, builder=builder, agree=indep == builder))

# =============================================================== 3. sentences
tex = open(TEX).read()
def active_mask(t):
    """True for characters in active text: not after an unescaped % on the line, not inside a comment env."""
    act = np.ones(len(t), bool)
    for m in re.finditer(r'\\begin\{comment\}.*?\\end\{comment\}', t, flags=re.S):
        act[m.start():m.end()] = False
    pos = 0
    for line in t.split('\n'):
        m = re.search(r'(?<!\\)%', line)
        if m:
            act[pos + m.start(): pos + len(line)] = False
        pos += len(line) + 1
    return act
ACT = active_mask(tex)
def norm(s):
    s = re.sub(r'\s*\\cite\{[^}]*\}', '', s)
    return re.sub(r'\s+', ' ', s).strip()
TEXN = norm(tex)

def split_claims(ps):
    """paper_sentence -> list of (raw part, quote to look for or None, skip reason)."""
    out = []
    for part in [x.strip() for x in ps.split(' | ') if x.strip()]:
        if part.startswith('23 Sep addition'):
            out.append((part, None, '23 Sep addition')); continue
        if part.startswith('decision pending'):
            out.append((part, None, 'decision pending')); continue
        q = part
        m = re.match(r'^(?:Table \d row|Figure 1 caption:) "(.*?)"', q)
        if m:
            tail = q[m.end():].strip()
            out.append((part, m.group(1), None))
            if tail:
                out.append((tail, None, 'builder annotation' if not tail.startswith('Omitted-dataset sentence: 23 Sep') else '23 Sep addition'))
            continue
        q = re.sub(r'\s*\(red span\)$', '', q)
        q = re.sub(r';\s*Pitt value: decision pending.*$', '', q)
        out.append((part, q, None))
    return out

sent = {}
for r in S:
    ps = r['paper_sentence'].strip()
    if not ps:
        continue
    for raw, q, why in split_claims(ps):
        if raw in sent:
            continue
        if q is None:
            sent[raw] = ('skipped: ' + why, '')
            continue
        j = tex.find(q)
        if j >= 0:
            st = 'VERBATIM' + ('' if ACT[j] else ' but inside commented-out text')
        elif norm(q) in TEXN:
            st = 'MATCH ONLY AFTER removing \\cite{} / collapsing spaces (not verbatim)'
        else:
            st = 'NOT FOUND'
        sent[raw] = (st, q)

# =============================================================== write
cols = ['row', 'part', 'task', 'id', 'paper_role', 'sampled', 'value_verified', 'lo', 'hi', 'n', 'file', 'resolved', 'method',
        'config', 'check_value', 'check_lo', 'check_hi', 'value_status', 'ci_status', 'note']
with open(OUT + '/check_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for x in results: w.writerow(x)
    f.write('\n')
    w2 = csv.writer(f)
    w2.writerow(['SENTENCE CHECK', 'status', 'quoted text looked for', 'paper_sentence part'])
    for raw, (st, q) in sent.items():
        w2.writerow(['sentence', st, q, raw])
    f.write('\n')
    w2.writerow(['ROW COUNT CHECK'])
    w2.writerow(['summary rows', len(S)]); w2.writerow(['fragment rows', len(frag_rows)])
    w2.writerow(['fragment rows missing from summary', sum(missing.values())] + ['%s | %s | %s' % k_ for k_ in missing])
    w2.writerow(['summary rows not in any fragment', sum(extra.values())] + ['%s | %s | %s' % k_ for k_ in extra])
    w2.writerow(['per-fragment count mismatches', len(frag_count_mismatch)] + ['%s %s' % (p, c) for p, c in frag_count_mismatch.items()])
    w2.writerow(['duplicate summary ids', len(dup_uid)])
    w2.writerow(['ids repeated inside a fragment', len(dup_frag_ids)] + ['%s | %s' % k_ for k_ in dup_frag_ids[:50]])
    for s_ in side:
        w2.writerow(['side check', s_])
    f.write('\n')
    w2.writerow(['PRINTED NUMBER AUDIT', 'row', 'id', 'paper_role', 'printed', 'value_verified', 'check rounding', 'check verdict', 'builder verdict', 'agree'])
    for x in printed_rows:
        w2.writerow(['printed', x['row'], x['id'], x['role'], x['printed'], x['verified'], x['check_round'], x['indep'], x['builder'], x['agree']])

print('\n==== ROW COUNT')
print('summary rows', len(S), 'fragment rows', len(frag_rows), 'fragments', len(frag_files))
print('missing from summary', sum(missing.values()), list(missing)[:10])
print('extra in summary', sum(extra.values()), list(extra)[:10])
print('per-fragment count mismatches', frag_count_mismatch)
print('duplicate summary ids', dup_uid[:10])
print('ids repeated inside a fragment', len(dup_frag_ids), dup_frag_ids)
for s_ in side: print('side:', s_)
print('\n==== VALUES: checked', len(results), '(paper value %d, random %d of %d others, extra %d)' % (sum(1 for x in results if x['sampled'] == 'paper value'), k, len(others), sum(1 for x in results if x['sampled'].startswith('extra'))))
print(Counter(x['value_status'] for x in results))
print(Counter(x['ci_status'].split(' (')[0] for x in results))
print('\n==== PRINTED AUDIT', len(printed_rows), Counter((x['indep'], x['builder']) for x in printed_rows))
for x in printed_rows:
    if x['indep'] == 'MISMATCH' or not x['agree']:
        print('  PRINTED', x['indep'], 'builder', x['builder'], x['row'], x['id'], x['role'], 'printed', x['printed'], 'verified', x['verified'], '->', x['check_round'])
print('\n==== SENTENCES')
print(Counter(st for st, q in sent.values()))
for raw, (st, q) in sent.items():
    if not st.startswith(('VERBATIM', 'skipped')) or 'commented' in st:
        print(st, '||', q[:160])
