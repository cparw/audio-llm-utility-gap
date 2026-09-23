#!/usr/bin/env python3
"""POD4b independent verifier (part 20).
Qwen3-Omni-30B-A3B zero shot on the 468 interviewer-free Pitt clips.

Written from scratch for verification. Imports nothing from the compute job.
Recomputes every number from per-clip csv files:
  AUC   explicit Mann-Whitney over all pos x neg pairs, ties count 0.5,
        cross-checked by a trapezoid ROC built from the distinct thresholds.
  CI    2000 draws, numpy default_rng(0) fresh per cell, speakers resampled with
        replacement (all clips of a drawn speaker come along), percentile 2.5/97.5.
        Speaker order = np.unique over the speaker strings (sorted).
        A draw where a needed quantity loses a class is dropped and counted.
  paired: one speaker draw per replicate, both quantities recomputed inside it.

usage:
  python3 POD4b_verify.py --new NEW.csv --orig ORIG.csv --rerun RERUN.csv --spine SPINE.csv
        [--spk-col speaker] [--p-col p_yes] [--out out.json]
"""
import argparse, json, sys
import numpy as np, pandas as pd

DRAWS = 2000
PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."


def auc_pairs(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def auc_trapz(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    P = (y == 1).sum(); N = (y == 0).sum()
    if P == 0 or N == 0:
        return np.nan
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def groups_of(spk):
    spk = np.asarray(spk).astype(str)
    uniq = np.unique(spk)
    idx = {u: i for i, u in enumerate(uniq)}
    code = np.array([idx[x] for x in spk])
    return uniq, [np.flatnonzero(code == i) for i in range(len(uniq))]


def boot_cells(spk, cell_fns, draws=DRAWS, seed=0):
    """cell_fns: dict name -> f(rows) returning a float (nan if degenerate).
    One speaker draw per replicate shared by every cell (paired)."""
    uniq, groups = groups_of(spk)
    K = len(uniq)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in cell_fns}
    bad = 0
    for _ in range(draws):
        d = rng.integers(0, K, K)
        rows = np.concatenate([groups[i] for i in d])
        v = {k: f(rows) for k, f in cell_fns.items()}
        if any(np.isnan(x) for x in v.values()):
            bad += 1
            continue
        for k, x in v.items():
            vals[k].append(x)
    out = {}
    for k, arr in vals.items():
        arr = np.array(arr)
        out[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out, K, bad


def boot_single(spk, f, seed=0):
    r, K, bad = boot_cells(spk, {'x': f}, seed=seed)
    return r['x'], K, bad


def load_scores(path, pcol, spkcol=None):
    d = pd.read_csv(path)
    d['key'] = d['clip'].astype(str).map(lambda x: x.split('/')[-1])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--orig', required=True)
    ap.add_argument('--rerun', required=True)
    ap.add_argument('--spine', required=True, help='pitt_conflict_manifest.csv (set, grp, spk, label, segment_path)')
    ap.add_argument('--p-col', default='p_yes')
    ap.add_argument('--new-spk-col', default='speaker')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    sp = pd.read_csv(a.spine, dtype={'spk': str})
    sp['key'] = sp.segment_path.map(lambda x: x.split('/')[-1])
    sp['spkkey'] = sp.grp.astype(str) + sp.spk.astype(str)
    sp = sp[['key', 'set', 'spkkey', 'label']]
    assert len(sp) == 468 and sp.key.is_unique

    new = load_scores(a.new, a.p_col)
    orig = load_scores(a.orig, a.p_col)
    rer = load_scores(a.rerun, a.p_col)
    rep = {}
    rep['n_new_rows'] = int(len(new)); rep['new_unique_clips'] = int(new.key.nunique())
    m = sp.merge(new[['key', a.p_col, 'label', a.new_spk_col] + (['prompt'] if 'prompt' in new else [])]
                 .rename(columns={a.p_col: 'p_new', 'label': 'label_new', a.new_spk_col: 'spk_new'}),
                 on='key', how='left')
    m = m.merge(orig[['key', a.p_col]].rename(columns={a.p_col: 'p_orig'}), on='key', how='left')
    m = m.merge(rer[['key', a.p_col]].rename(columns={a.p_col: 'p_rerun'}), on='key', how='left')
    rep['missing_new'] = int(m.p_new.isna().sum())
    rep['missing_orig'] = int(m.p_orig.isna().sum())
    rep['missing_rerun'] = int(m.p_rerun.isna().sum())
    rep['label_mismatch_new_vs_spine'] = int((m.label_new != m.label).sum())
    # speaker partition in the new csv must equal the spine partition
    a2b = m.groupby('spk_new').spkkey.nunique().max(); b2a = m.groupby('spkkey').spk_new.nunique().max()
    rep['speaker_partition_identical'] = bool(a2b == 1 and b2a == 1)
    rep['n'] = int(m.p_new.notna().sum())
    rep['n_speakers_spine'] = int(m.spkkey.nunique())
    rep['n_speakers_newcsv'] = int(m.spk_new.nunique())
    rep['n_pos'] = int((m.label == 1).sum()); rep['n_neg'] = int((m.label == 0).sum())
    rep['arm_counts'] = m.set.value_counts().to_dict()
    if 'prompt' in m:
        rep['prompts_in_csv'] = sorted(m.prompt.dropna().unique().tolist())
        rep['prompt_verbatim_ok'] = rep['prompts_in_csv'] == [PROMPT]
    m = m.dropna(subset=['p_new', 'p_orig'])

    y = m.label.values.astype(int)
    arm = m.set.values.astype(str)
    s_new = m.p_new.values.astype(float); s_orig = m.p_orig.values.astype(float)
    c = arm == 'conflict'; g = arm == 'agreement'

    point = {}
    for nm, s in [('new', s_new), ('orig', s_orig)]:
        for cell, mask in [('overall', np.ones(len(y), bool)), ('conflict', c), ('agreement', g)]:
            a1 = auc_pairs(y[mask], s[mask]); a2 = auc_trapz(y[mask], s[mask])
            point[f'{nm}_{cell}'] = dict(auc=float(a1), auc_trapz=float(a2), trapz_agree=bool(abs(a1 - a2) < 1e-9),
                                          n=int(mask.sum()))
    mr = m.dropna(subset=['p_rerun'])
    point['rerun_overall'] = dict(auc=float(auc_pairs(mr.label.values, mr.p_rerun.values)),
                                  auc_trapz=float(auc_trapz(mr.label.values, mr.p_rerun.values)), n=int(len(mr)))

    results = {}
    for spkcol in ['spkkey', 'spk_new']:
        spk = m[spkcol].values.astype(str)
        fns = {
            'new_overall': lambda r: auc_pairs(y[r], s_new[r]),
            'new_conflict': lambda r: auc_pairs(y[r][arm[r] == 'conflict'], s_new[r][arm[r] == 'conflict']),
            'new_agreement': lambda r: auc_pairs(y[r][arm[r] == 'agreement'], s_new[r][arm[r] == 'agreement']),
            'orig_overall': lambda r: auc_pairs(y[r], s_orig[r]),
            'orig_conflict': lambda r: auc_pairs(y[r][arm[r] == 'conflict'], s_orig[r][arm[r] == 'conflict']),
            'orig_agreement': lambda r: auc_pairs(y[r][arm[r] == 'agreement'], s_orig[r][arm[r] == 'agreement']),
        }
        # paired shared draw across all six plus deltas
        uniq, groups = groups_of(spk); K = len(uniq)
        rng = np.random.default_rng(0)
        store = {k: [] for k in fns}; bad = 0
        for _ in range(DRAWS):
            d = rng.integers(0, K, K); rows = np.concatenate([groups[i] for i in d])
            v = {k: f(rows) for k, f in fns.items()}
            if any(np.isnan(x) for x in v.values()):
                bad += 1; continue
            for k, x in v.items():
                store[k].append(x)
        st = {k: np.array(v) for k, v in store.items()}
        res = {}
        for k in fns:
            res[k] = (float(np.percentile(st[k], 2.5)), float(np.percentile(st[k], 97.5)))
        for cell in ['overall', 'conflict', 'agreement']:
            dd = st[f'orig_{cell}'] - st[f'new_{cell}']
            res[f'delta_orig_minus_new_{cell}'] = (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
        for nm in ['new', 'orig']:
            dd = st[f'{nm}_conflict'] - st[f'{nm}_agreement']
            res[f'{nm}_conflict_minus_agreement'] = (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
        res['_K'] = K; res['_bad'] = bad
        # alternative convention: each arm bootstrapped over its own speakers only
        alt = {}
        for nm, s in [('new', s_new), ('orig', s_orig)]:
            for cell, mask in [('conflict', c), ('agreement', g)]:
                idx = np.flatnonzero(mask)
                (lo, hi), K2, b2 = boot_single(spk[idx], lambda r, idx=idx, s=s: auc_pairs(y[idx][r], s[idx][r]))
                alt[f'{nm}_{cell}_armonly'] = (lo, hi, K2, b2)
            (lo, hi), K2, b2 = boot_single(spk, lambda r, s=s: auc_pairs(y[r], s[r]))
            alt[f'{nm}_overall_single'] = (lo, hi, K2, b2)
        res['_alt'] = alt
        results[spkcol] = res

    deltas = {cell: point[f'orig_{cell}']['auc'] - point[f'new_{cell}']['auc'] for cell in ['overall', 'conflict', 'agreement']}
    out = dict(report=rep, point=point, deltas_orig_minus_new=deltas, boot=results)
    txt = json.dumps(out, indent=1, default=str)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt)


if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] == 'results'):
    main()


# ---------------------------------------------------------------------------
# Mode 2: verify every *.RESULT in a synced copy of /workspace/scores/part20/POD4b
#   python3 POD4b_verify.py results DIR SPINE SHIPPED RERUN FOLDS OUT_TSV
# ---------------------------------------------------------------------------
def _boot_pairs_cells(spk, fns, draws=DRAWS, seed=0):
    uniq, groups = groups_of(spk); K = len(uniq)
    rng = np.random.default_rng(seed)
    st = {k: [] for k in fns}; bad = 0
    for _ in range(draws):
        d = rng.integers(0, K, K); rows = np.concatenate([groups[i] for i in d])
        v = {k: f(rows) for k, f in fns.items()}
        if any(np.isnan(x) for x in v.values()):
            bad += 1; continue
        for k, x in v.items():
            st[k].append(x)
    return {k: np.array(v) for k, v in st.items()}, K, bad


def verify_results(D, spine, shipped, rerun, folds, out_tsv):
    import glob, os, hashlib
    sp = pd.read_csv(spine, dtype={'spk': str})
    sp['key'] = sp.segment_path.map(lambda x: x.split('/')[-1])
    sp['spkkey'] = sp.grp.astype(str) + sp.spk.astype(str)
    SP = sp.set_index('key')
    noinv = pd.read_csv(os.path.join(D, 'q3o_pitt_noinv_zeroshot_scores.csv')).set_index('orig_clip_id')
    SH = pd.read_csv(shipped).set_index('clip'); RR = pd.read_csv(rerun).set_index('clip')
    fo = pd.read_csv(folds).set_index('clip_id')
    SPOD = None
    spf = os.path.join(D, 'q3o_pitt_orig468_samepod_zeroshot_scores.csv')
    if os.path.exists(spf):
        SPOD = pd.read_csv(spf)
        SPOD = SPOD.set_index('orig_clip_id' if 'orig_clip_id' in SPOD else 'clip')
    rows_c = pd.read_csv(os.path.join(D, 'POD4b_rows.tsv'), sep='\t').set_index('id')
    checks_global = {}
    checks_global['noinv_rows'] = len(noinv); checks_global['noinv_unique'] = int(noinv.index.nunique())
    checks_global['noinv_prompts'] = sorted(noinv.prompt.unique().tolist())
    checks_global['noinv_prompt_ok'] = checks_global['noinv_prompts'] == [PROMPT]
    checks_global['noinv_label_vs_spine_mismatch'] = int((noinv.label != SP.loc[noinv.index, 'label'].values).sum())
    checks_global['noinv_set_vs_spine_mismatch'] = int((noinv.set != SP.loc[noinv.index, 'set'].values).sum())
    checks_global['noinv_spk_vs_spine_mismatch'] = int((noinv.speaker != SP.loc[noinv.index, 'spkkey'].values).sum())
    checks_global['noinv_spk_vs_folds_mismatch'] = int((noinv.speaker != fo.loc[noinv.index, 'speaker_id'].values).sum())
    checks_global['noinv_dur_scored_eq_dur'] = bool((noinv.dur_s == noinv.dur_scored_s).all())
    checks_global['noinv_min_mass'] = float(noinv.answer_mass.min())
    checks_global['folds_md5'] = hashlib.md5(open(folds, 'rb').read()).hexdigest()
    checks_global['folds_sha256_first_1MB'] = hashlib.sha256(open(folds, 'rb').read()[:1 << 20]).hexdigest()
    checks_global['shipped_sha256_first_1MB'] = hashlib.sha256(open(shipped, 'rb').read()[:1 << 20]).hexdigest()
    checks_global['spine_sha256_first_1MB'] = hashlib.sha256(open(spine, 'rb').read()[:1 << 20]).hexdigest()
    checks_global['noinv_sha256_first_1MB'] = hashlib.sha256(open(os.path.join(D, 'q3o_pitt_noinv_zeroshot_scores.csv'), 'rb').read()[:1 << 20]).hexdigest()

    out = []
    for rf in sorted(glob.glob(os.path.join(D, '*.RESULT'))):
        rid = os.path.basename(rf)[:-len('.RESULT')]
        sc = json.load(open(os.path.join(D, rid + '.sidecar.json')))
        pc = pd.read_csv(os.path.join(D, rid + '.perclip.csv'))
        rec = dict(id=rid, what=sc['what'], v_c=sc['value'], lo_c=sc['ci95'][0], hi_c=sc['ci95'][1],
                   n_c=sc['n'], nspk_c=sc['n_speakers'], issues=[])
        if sc.get('prompt_verbatim') != PROMPT: rec['issues'].append('sidecar prompt not verbatim')
        rr = rows_c.loc[rid]
        if not (abs(rr['value'] - sc['value']) < 1e-12 and abs(rr['lo'] - sc['ci95'][0]) < 1e-12 and abs(rr['hi'] - sc['ci95'][1]) < 1e-12
                and rr['n'] == sc['n'] and rr['n_spk'] == sc['n_speakers']):
            rec['issues'].append('POD4b_rows.tsv differs from sidecar')
        arm = 'overall' if rid.endswith('_overall') else ('conflict' if rid.endswith('_conflict') else 'agreement')
        key = pc['orig_clip_id'].values
        exp_n = {'overall': 468, 'conflict': 146, 'agreement': 322}[arm]
        if len(pc) != exp_n or pd.Series(key).nunique() != exp_n: rec['issues'].append(f'n {len(pc)} != {exp_n}')
        if arm != 'overall' and not (pc.set == arm).all(): rec['issues'].append('set column not all ' + arm)
        exp_keys = set(sp.key) if arm == 'overall' else set(sp.key[sp.set == arm])
        if set(key) != exp_keys: rec['issues'].append('clip set differs from spine arm')
        if (pc.label.values != SP.loc[key, 'label'].values).any(): rec['issues'].append('label vs spine')
        if (pc.set.values != SP.loc[key, 'set'].values).any(): rec['issues'].append('set vs spine')
        if (pc.speaker.values != SP.loc[key, 'spkkey'].values).any(): rec['issues'].append('speaker vs spine')
        y = pc.label.values.astype(int); spk = pc.speaker.values.astype(str)
        if '_diff_' in rid:
            if '_shipped_' in rid: src = SH
            elif '_rerun_' in rid: src = RR
            elif '_samepod_' in rid: src = SPOD
            else: raise SystemExit('unknown original source for ' + rid)
            if np.abs(pc.p_yes_original.values - src.loc[key, 'p_yes'].values).max() > 0: rec['issues'].append('p_yes_original != source file')
            if np.abs(pc.p_yes_noinv.values - noinv.loc[key, 'p_yes'].values).max() > 0: rec['issues'].append('p_yes_noinv != noinv scores csv')
            so = pc.p_yes_original.values.astype(float); sn = pc.p_yes_noinv.values.astype(float)
            ao, an = auc_pairs(y, so), auc_pairs(y, sn)
            to, tn = auc_trapz(y, so), auc_trapz(y, sn)
            if abs(ao - to) > 1e-9 or abs(an - tn) > 1e-9: rec['issues'].append('trapezoid disagrees')
            val = ao - an
            st, K, bad = _boot_pairs_cells(spk, {'o': lambda r: auc_pairs(y[r], so[r]), 'n': lambda r: auc_pairs(y[r], sn[r])})
            dd = st['o'] - st['n']
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            rec['auc_original_v'] = ao; rec['auc_noinv_v'] = an
            if abs(sc.get('auc_original', ao) - ao) > 5e-7 or abs(sc.get('auc_noinv', an) - an) > 5e-7: rec['issues'].append('sidecar component aucs differ')
        else:
            if '_noinv_' in rid: src = noinv; srcname = 'noinv'
            elif '_shipped_' in rid: src = SH; srcname = 'shipped'
            elif '_rerun_' in rid: src = RR; srcname = 'rerun'
            elif '_samepod_' in rid: src = SPOD; srcname = 'samepod'
            else: raise SystemExit('unknown source for ' + rid)
            if np.abs(pc.p_yes.values - src.loc[key, 'p_yes'].values).max() > 0: rec['issues'].append(f'p_yes != {srcname} source file')
            s = pc.p_yes.values.astype(float)
            val = auc_pairs(y, s)
            if abs(val - auc_trapz(y, s)) > 1e-9: rec['issues'].append('trapezoid disagrees')
            st, K, bad = _boot_pairs_cells(spk, {'a': lambda r: auc_pairs(y[r], s[r])})
            lo, hi = float(np.percentile(st['a'], 2.5)), float(np.percentile(st['a'], 97.5))
        rec.update(v_v=val, lo_v=lo, hi_v=hi, n_v=len(pc), nspk_v=K, bad=bad)
        if bad != 2000 - sc.get('usable_draws', 2000): rec['issues'].append(f'usable draws differ: mine {2000-bad}')
        ag = (round(val, 4) == round(sc['value'], 4) and round(lo, 4) == round(sc['ci95'][0], 4)
              and round(hi, 4) == round(sc['ci95'][1], 4) and len(pc) == sc['n'] and K == sc['n_speakers'])
        rec['agree_4dp'] = bool(ag and not rec['issues'])
        rec['file'] = rr['file']
        out.append(rec)

    def two(v, lo, hi, diff):
        f = (lambda x: f"{x:+.2f}") if diff else (lambda x: f"{x:.2f}")
        if diff:
            tail = 'interval includes zero' if lo <= 0 <= hi else ('interval above zero' if lo > 0 else 'interval below zero')
        else:
            tail = 'above 0.5' if lo > 0.5 else ('below 0.5' if hi < 0.5 else 'interval includes 0.5')
        return f"{f(v)} [{f(lo)}, {f(hi)}], {tail}", tail
    lines = []
    for r in out:
        diff = '_diff_' in r['id']
        td, vs = two(r['v_v'], r['lo_v'], r['hi_v'], diff)
        r['two_dp'] = td; r['vs_half'] = vs
        lines.append(r)
    return checks_global, lines


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'results':
    D, spine, shipped, rerun, folds, out_json = sys.argv[2:8]
    g, lines = verify_results(D, spine, shipped, rerun, folds, None)
    print(json.dumps(g, indent=1))
    for r in lines:
        flag = 'AGREE' if r['agree_4dp'] else 'DISAGREE'
        print(f"{flag} {r['id']}: computed {r['v_c']:.4f} [{r['lo_c']:.4f}, {r['hi_c']:.4f}] n={r['n_c']} spk={r['nspk_c']} | "
              f"verified {r['v_v']:.4f} [{r['lo_v']:.4f}, {r['hi_v']:.4f}] n={r['n_v']} spk={r['nspk_v']} bad={r['bad']} issues={r['issues']}")
    json.dump(dict(global_checks=g, results=lines), open(out_json, 'w'), indent=1, default=str)
