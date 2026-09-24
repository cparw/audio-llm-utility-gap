#!/usr/bin/env python3
"""Independent verifier for PART 25 Track B (Holm over the tex's significant differences).

Written from scratch. Imports nothing from Track B (scripts/B_holm*.py, verify/B_verify.py).
Rebuilds every per-clip frame from the ORIGINAL source files with dict joins,
redraws the 2000 speaker bootstrap draws, recomputes point, interval, p and Holm,
and compares against (a) Track B per-clip csvs, (b) Track B saved draw npz files,
(c) Track B B_holm.tsv, (d) Track A draw npz files, (e) the numbers printed in the tex.
Reads only; writes only under part25/verify/.
"""
import os, re, json, hashlib, datetime, glob, sys, platform
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import sklearn, scipy

P25 = '<local data dir>/release_from_mac/scores/part25'
BD = P25 + '/B'
AD = P25 + '/A'
OUT = P25 + '/verify'
IND = OUT + '/B_indep'
REL = '<local data dir>/release'
OF = REL + '/omni_final'
P14 = REL + '/edaic_rerun/part14'
P16 = REL + '/edaic_rerun/part16'
TEX = '<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex'
TEX_SHA = '6c886f3513e93de7f78a7324f608091e7cdc24f257089bb5bd0a5c40638e4a20'
NB = 2000
TIE = 1e-12

SOURCES = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def src(rid, p):
    SOURCES.setdefault(rid, []).append({'path': p, 'sha256': sha(p)})
    return p


def fnum(v):
    s = str(v).strip()
    m = re.match(r'^np\.float64\((.*)\)$', s)
    return float(m.group(1)) if m else float(s)


def bname(p):
    return os.path.basename(str(p))


def rows_of(path, **kw):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, **kw)
    return df.to_dict('records')


# ---------------------------------------------------------------- frame builders
# Each builder returns (kind, records) where records is a list of dicts with
# clip, speaker (str), label (int), and the value fields named in FIELDS[kind].

def gap_trackA(rid, ds):
    a = rows_of(src(rid, f'{AD}/per_clip/A_{ds}_perclip.csv'))
    z = {r['clip']: r for r in rows_of(src(rid, f'{OF}/omni_{ds}_zeroshot_scores.csv'))}
    out = []
    for r in a:
        zr = z[r['clip']]
        assert int(zr['label']) == int(r['label']) and str(zr['speaker']) == str(r['speaker']), r['clip']
        assert float(r['p_yes_zeroshot_saved']) == float(zr['p_yes'])
        out.append({'clip': r['clip'], 'speaker': str(r['speaker']), 'label': int(r['label']),
                    **{f'r{k}': float(r[f'p_probe_seed{k}']) for k in range(5)},
                    'zs': float(zr['p_yes'])})
    assert len(out) == len(z)
    return 'mean5', out


def gap_pitt(rid):
    zf = np.load(src(rid, f'{REL}/overnight2/part10/pitt_enc_nested5_oof.npz'), allow_pickle=False)
    z = {r['clip']: r for r in rows_of(src(rid, f'{OF}/omni_pitt_zeroshot_scores.csv'))}
    out = []
    for i, nm in enumerate(zf['name']):
        zr = z[str(nm)]
        assert int(zr['label']) == int(zf['label'][i]) and zr['speaker'] == str(zf['spk'][i])
        out.append({'clip': str(nm), 'speaker': str(zf['spk'][i]), 'label': int(zf['label'][i]),
                    **{f'r{k}': float(zf['oof'][k, i]) for k in range(5)}, 'zs': float(zr['p_yes'])})
    assert len(out) == len(z) == 468
    return 'mean5', out


def gap_edaic(rid, which):
    a3 = rows_of(src(rid, f'{P25}/../part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv'))
    a2 = {r['pid']: r for r in rows_of(src(rid, f'{P25}/../part23/A/A2_edaic_whole_zeroshot_perclip.csv'))}
    out = []
    for r in a3:
        zr = a2[r['pid']]
        assert int(zr['label']) == int(r['label'])
        assert fnum(zr['p_yes_whole']) == fnum(r['p_yes_whole'])
        out.append({'clip': r['pid'], 'speaker': str(r['speaker']), 'label': int(r['label']),
                    **{f'r{k}': float(r[f'oof_{which}_r{k}']) for k in range(5)},
                    'zs': fnum(zr['p_yes_whole'])})
    assert len(out) == len(a2) == 275
    return 'mean5', out


def two_by_clip(rid, pa, key_a, val_a, pb, key_b, val_b, spk_a='speaker', lab_a='label', spk_b='speaker', lab_b='label',
                ka=lambda s: s, kb=lambda s: s):
    A = rows_of(src(rid, pa))
    Bm = {}
    for r in rows_of(src(rid, pb)):
        k = kb(r[key_b])
        assert k not in Bm, ('dup', k)
        Bm[k] = r
    out = []
    for r in A:
        k = ka(r[key_a])
        br = Bm[k]
        assert int(br[lab_b]) == int(r[lab_a]), k
        assert str(br[spk_b]) == str(r[spk_a]), k
        out.append({'clip': k, 'speaker': str(r[spk_a]), 'label': int(r[lab_a]),
                    'a': float(r[val_a]), 'b': float(br[val_b])})
    assert len(out) == len(Bm) == len(A)
    return 'two', out


# E-DAIC arm scores rebuilt from the part14 per-model files (not from the T1 file)
P14FILES = {'o25': ('p14_o25_zeroshot_scores.csv', 'clip', 'speaker'),
            'q2a': ('p14_q2a_zeroshot_scores.csv', 'clip', 'speaker'),
            'q3o': ('p14_q3o_zeroshot_scores.csv', 'clip', 'speaker'),
            'af2': ('p14_af2.csv', 'clip_path', 'speaker_id'),
            'af3': ('p14_af3.csv', 'clip_path', 'speaker_id'),
            'kimi': ('p14_kimi_zeroshot_scores.csv', 'clip', 'speaker')}
EDAIC_DUP_INFO = {}


def edaic_arms(rid, m, conflict_only=False):
    fn, ck, sk = P14FILES[m]
    seen = {}
    dup_rows = 0
    dup_maxdiff = 0.0
    for r in rows_of(src(rid, f'{P14}/{fn}')):
        c = bname(r[ck])
        c = c[:-4] if c.endswith('.wav') else c
        if c in seen:
            dup_rows += 1
            dup_maxdiff = max(dup_maxdiff, abs(float(r['p_yes']) - seen[c]['score']))
            assert seen[c]['label'] == int(r['label']) and seen[c]['speaker'] == str(r[sk])
            continue
        arm = c.split('_')[0]
        assert arm in ('conflict', 'agreement')
        seen[c] = {'clip': c, 'speaker': str(r[sk]), 'label': int(r['label']), 'arm': arm, 'score': float(r['p_yes'])}
    EDAIC_DUP_INFO[m] = {'dup_rows_dropped': dup_rows, 'max_score_diff_between_duplicates': dup_maxdiff}
    # cross-check against the T1 file (in_distinct_set rows)
    t1 = rows_of(src(rid, f'{REL}/edaic_rerun/part17/T1_perclip_all966.csv'))
    t1d = {r['clip_id']: r for r in t1 if r['in_distinct_set'] == '1'}
    assert set(t1d) == set(seen), m
    mx = max(abs(float(t1d[c][f'p_{m}']) - seen[c]['score']) for c in seen)
    EDAIC_DUP_INFO[m]['max_abs_diff_vs_T1'] = mx
    assert mx < 1e-12, (m, mx)
    for c in seen:
        assert t1d[c]['set'] == seen[c]['arm'] and int(t1d[c]['label']) == seen[c]['label'] and str(t1d[c]['speaker_id']) == seen[c]['speaker']
    recs = list(seen.values())
    if conflict_only:
        recs = [r for r in recs if r['arm'] == 'conflict']
        return 'one', recs
    return 'arms', recs


PITT_SCORES = {'q2a': ('<local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv', 'clip', 'speaker'),
               'o25': (f'{OF}/omni_pitt_zeroshot_scores.csv', 'clip', 'speaker'),
               'q3o': (f'{REL}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv', 'clip', 'speaker'),
               'af3': (f'{REL}/overnight2/part10/af3_pitt.csv', 'clip_path', 'speaker_id'),
               'kimi': (f'{REL}/overnight2/part3/kimi_pitt_zeroshot_scores.csv', 'clip', 'speaker')}
MANIFEST = '<local data dir>/DementiaBank/pitt_conflict_manifest.csv'


def pitt_arms(rid, m, conflict_only=False):
    man = rows_of(src(rid, MANIFEST))
    if m == 'af2':
        sc = {}
        for f in ('af2_orig30_conflict.csv', 'af2_orig30_agreement.csv'):
            for r in rows_of(src(rid, f'{P25}/../part20/POD4c/{f}')):
                assert r['orig_clip_id'] not in sc
                sc[r['orig_clip_id']] = (str(r['speaker_id']), int(r['label']), float(r['p_yes']), r['arm'])
    else:
        p, ck, sk = PITT_SCORES[m]
        sc = {}
        for r in rows_of(src(rid, p)):
            c = bname(r[ck])
            assert c not in sc
            sc[c] = (str(r[sk]), int(r['label']), float(r['p_yes']), None)
    out = []
    for r in man:
        c = bname(r['segment_path'])
        spk = r['grp'] + r['spk']
        s = sc[c]
        assert s[0] == spk and s[1] == int(r['label']), c
        assert c.startswith(r['set'] + '_')
        if s[3] is not None:
            assert s[3] == r['set']
        out.append({'clip': c, 'speaker': spk, 'label': int(r['label']), 'arm': r['set'], 'score': s[2]})
    assert len(out) == len(sc) == 468
    if conflict_only:
        return 'one', [o for o in out if o['arm'] == 'conflict']
    return 'arms', out


def build(rid):
    if rid in ('G_pcgita', 'G_neurovoz', 'G_adresso', 'G_adress2020'):
        return gap_trackA(rid, rid[2:])
    if rid == 'G_pitt':
        return gap_pitt(rid)
    if rid == 'G_edaic_llm':
        return gap_edaic(rid, 'llm')
    if rid == 'G_edaic_enc':
        return gap_edaic(rid, 'enc')
    if rid == 'Gs_edaic_ansstate':
        return gap_edaic(rid, 'ans')
    if rid == 'Gs_kcl_single':
        return two_by_clip(rid, f'{OF}/omni_kcl_enc_nested_oof.csv', 'clip', 'p_probe',
                           f'{OF}/omni_kcl_zeroshot_scores.csv', 'clip', 'p_yes')
    if rid.startswith('R_edaic_'):
        return edaic_arms(rid, rid.split('_')[-1])
    if rid.startswith('X_edaic_conf_'):
        return edaic_arms(rid, rid.split('_')[-1], conflict_only=True)
    if rid.startswith('R_pitt_') or rid.startswith('Rs_pitt_'):
        return pitt_arms(rid, rid.split('_')[-1])
    if rid.startswith('X_pitt_conf_'):
        return pitt_arms(rid, rid.split('_')[-1], conflict_only=True)
    if rid in ('F_pcgita', 'F_neurovoz', 'F_pitt', 'F_adresso', 'F_adress2020', 'Fs_kcl'):
        ds = rid.split('_')[1]
        return two_by_clip(rid, f'{OF}/omnisft_{ds}_oof.csv', 'path', 'p_yes',
                           f'{OF}/omni_{ds}_zeroshot_scores.csv', 'clip', 'p_yes', ka=bname)
    if rid == 'Fs_edaic_seed0':
        A = rows_of(src(rid, f'{P25}/../part24/FT/FT24_whole_seed0_perclip.csv'))
        Z = {r['pid']: r for r in rows_of(src(rid, f'{P25}/../part23/A/A2_edaic_whole_zeroshot_perclip.csv'))}
        out = []
        for r in A:
            z = Z[r['pid']]
            assert int(z['label']) == int(r['label']) and r['seed'] == '0'
            out.append({'clip': r['pid'], 'speaker': str(r['pid']), 'label': int(r['label']),
                        'a': float(r['p_yes']), 'b': fnum(z['p_yes_whole'])})
        assert len(out) == len(Z) == 275
        return 'two', out
    if rid == 'F_lora_minus_projector':
        return two_by_clip(rid, f'{P16}/POD2/lora_pitt_oof.csv', 'name', 'p_yes',
                           f'{OF}/omnisft_pitt_oof.csv', 'path', 'p_yes', spk_a='spk', kb=bname)
    if rid == 'C_pg_probe_drop':
        return two_by_clip(rid, f'{P16}/POD4/p16_pg_after_enc_nested_oof.csv', 'clip', 'p_probe',
                           f'{P16}/POD4/p16_pg_before_enc_nested_oof.csv', 'clip', 'p_probe')
    if rid == 'C_nv_probe_drop':
        return two_by_clip(rid, f'{P16}/POD4/p16_nv_after_enc_nested_oof.csv', 'clip', 'p_probe',
                           f'{P16}/POD4/p16_nv_before_enc_nested_oof.csv', 'clip', 'p_probe')
    if rid == 'Cs_pg_answer':
        return two_by_clip(rid, f'{P16}/POD4/p16_pg_after_zeroshot_scores.csv', 'clip', 'p_yes',
                           f'{P16}/POD4/p16_pg_before_zeroshot_scores.csv', 'clip', 'p_yes')
    raise KeyError(rid)


# ---------------------------------------------------------------- statistics

def auc(y, s):
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s))


def rank_auc(y, s):
    r = rankdata(s)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def estimate(kind, y, cols, arm, rows):
    """Return (diff, terms) on the given row index set."""
    yy = y[rows]
    if kind == 'mean5':
        per = [auc(yy, cols[f'r{k}'][rows]) for k in range(5)]
        z = auc(yy, cols['zs'][rows])
        m5 = float(np.mean(per)) if not np.any(np.isnan(per)) else np.nan
        return m5 - z, [m5, z] + per
    if kind == 'two':
        a = auc(yy, cols['a'][rows]); b = auc(yy, cols['b'][rows])
        return a - b, [a, b]
    if kind == 'arms':
        cm = arm[rows] == 'conflict'
        am = arm[rows] == 'agreement'
        c = auc(yy[cm], cols['score'][rows][cm]); g = auc(yy[am], cols['score'][rows][am])
        return c - g, [c, g]
    if kind == 'one':
        c = auc(yy, cols['score'][rows])
        return c - 0.5, [c, 0.5]
    raise KeyError(kind)


FIELDS = {'mean5': ['r0', 'r1', 'r2', 'r3', 'r4', 'zs'], 'two': ['a', 'b'], 'arms': ['score'], 'one': ['score']}


def run_boot(kind, recs):
    spk = np.array([r['speaker'] for r in recs], dtype=str)
    y = np.array([r['label'] for r in recs], dtype=int)
    cols = {f: np.array([r[f] for r in recs], dtype=float) for f in FIELDS[kind]}
    arm = np.array([r.get('arm', '') for r in recs], dtype=str)
    uniq = np.unique(spk)
    members = {k: np.flatnonzero(spk == u) for k, u in enumerate(uniq)}
    allrows = np.arange(len(recs))
    pd_, pt = estimate(kind, y, cols, arm, allrows)
    rng = np.random.default_rng(0)
    n = len(uniq)
    diffs = np.full(NB, np.nan)
    terms = np.full((NB, len(pt)), np.nan)
    sidx = np.empty((NB, n), dtype=np.int64)
    for b in range(NB):
        idx = rng.choice(n, size=n, replace=True)
        sidx[b] = idx
        rows = np.concatenate([members[i] for i in idx])
        d, t = estimate(kind, y, cols, arm, rows)
        diffs[b] = d
        terms[b] = t
    # rank-AUC cross check of the point terms
    if kind == 'mean5':
        rk = [np.mean([rank_auc(y, cols[f'r{k}']) for k in range(5)]), rank_auc(y, cols['zs'])]
    elif kind == 'two':
        rk = [rank_auc(y, cols['a']), rank_auc(y, cols['b'])]
    elif kind == 'arms':
        rk = [rank_auc(y[arm == 'conflict'], cols['score'][arm == 'conflict']),
              rank_auc(y[arm == 'agreement'], cols['score'][arm == 'agreement'])]
    else:
        rk = [rank_auc(y, cols['score']), 0.5]
    return {'uniq': uniq, 'point_diff': pd_, 'point_terms': pt, 'diffs': diffs, 'terms': terms, 'sidx': sidx,
            'rank_point_terms': rk, 'n_clips': len(recs), 'n_spk': n, 'n_pos': int(y.sum()),
            'n_conflict': int((arm == 'conflict').sum()), 'n_agreement': int((arm == 'agreement').sum())}


def pval(d, tie=True):
    d = d[np.isfinite(d)]
    B = len(d)
    if tie:
        d = np.where(np.abs(d) < TIE, 0.0, d)
    le = int((d <= 0).sum()); ge = int((d >= 0).sum())
    return min(1.0, 2 * min((1 + le) / (B + 1), (1 + ge) / (B + 1))), le, ge, B


def holm(pmap):
    ids = list(pmap)
    ps = np.array([pmap[i] for i in ids])
    order = np.argsort(ps, kind='stable')
    m = len(ps)
    adj = {}
    run = 0.0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[j]))
        adj[ids[j]] = run
    return adj


# ---------------------------------------------------------------- main
FAM = {
    'GAPS': ['G_pcgita', 'G_neurovoz', 'G_adresso', 'G_adress2020', 'G_pitt', 'G_edaic_llm', 'G_edaic_enc'],
    'ARMS': ['R_edaic_q2a', 'R_edaic_o25', 'R_edaic_q3o', 'R_edaic_af2', 'R_edaic_af3', 'R_edaic_kimi',
             'R_pitt_q2a', 'R_pitt_o25', 'R_pitt_q3o'],
    'FINE TUNE': ['F_pcgita', 'F_neurovoz', 'F_pitt', 'F_adresso', 'F_adress2020', 'F_lora_minus_projector'],
    'CONTROLS': ['C_pg_probe_drop', 'C_nv_probe_drop'],
    'EXTRA CHANCE': ['X_edaic_conf_q2a', 'X_edaic_conf_o25', 'X_edaic_conf_q3o', 'X_edaic_conf_af3'],
}
FAM_SENS = {
    'GAPS': ['Gs_edaic_ansstate', 'Gs_kcl_single'],
    'ARMS': ['Rs_pitt_af2', 'Rs_pitt_af3', 'Rs_pitt_kimi'],
    'FINE TUNE': ['Fs_kcl', 'Fs_edaic_seed0'],
    'CONTROLS': ['Cs_pg_answer'],
    'EXTRA CHANCE': ['X_edaic_conf_af2', 'X_edaic_conf_kimi'],
}
CHECK_ONLY = ['X_pitt_conf_q2a', 'X_pitt_conf_o25', 'X_pitt_conf_q3o']
ALL_IDS = [i for f in FAM for i in FAM[f]] + [i for f in FAM_SENS for i in FAM_SENS[f]] + CHECK_ONLY


def main():
    t0 = datetime.datetime.utcnow()
    tex_sha_now = sha(TEX)
    btsv = pd.read_csv(BD + '/B_holm.tsv', sep='\t', dtype=str, keep_default_na=False)
    btsv = {r['id']: r for r in btsv.to_dict('records')}
    assert set(btsv) == set(ALL_IDS), set(btsv) ^ set(ALL_IDS)
    res = {}
    for rid in ALL_IDS:
        kind, recs = build(rid)
        bo = run_boot(kind, recs)
        d = bo['diffs']
        lo, hi = np.percentile(d[np.isfinite(d)], [2.5, 97.5])
        p, le, ge, B = pval(d)
        p_notie, le_nt, ge_nt, _ = pval(d, tie=False)
        # compare with Track B per-clip csv
        bpc = pd.read_csv(f'{BD}/per_clip/B_{rid}_perclip.csv', dtype={'clip': str, 'speaker': str})
        bmap = {str(r['clip']): r for r in bpc.to_dict('records')}
        assert set(bmap) == set(r['clip'] for r in recs), (rid, 'clip set differs')
        vcols = [c for c in bpc.columns if c not in ('clip', 'speaker', 'label', 'arm', 'is_conflict', 'is_agreement')]
        myf = FIELDS[kind]
        assert len(vcols) == len(myf), (rid, vcols, myf)
        pc_max = 0.0
        pc_meta_ok = True
        for r in recs:
            br = bmap[r['clip']]
            pc_meta_ok &= (str(br['speaker']) == r['speaker']) and (int(br['label']) == r['label'])
            if 'arm' in r and 'arm' in br:
                pc_meta_ok &= (br['arm'] == r['arm'])
            for f, c in zip(myf, vcols):
                pc_max = max(pc_max, abs(float(br[c]) - r[f]))
        # compare with Track B saved draws
        z = np.load(f'{BD}/draws/B_{rid}_draws.npz', allow_pickle=False)
        same_spk = bool(np.array_equal(z['speakers'].astype(str), bo['uniq']))
        same_idx = bool(np.array_equal(z['speaker_idx'].astype(np.int64), bo['sidx']))
        bd = z['diff']
        nanpat = bool(np.array_equal(np.isnan(bd), np.isnan(d)))
        maxdraw = float(np.nanmax(np.abs(bd - d)))
        maxterms = float(np.nanmax(np.abs(z['terms'] - bo['terms']))) if z['terms'].shape == bo['terms'].shape else None
        p_saved, le_s, ge_s, B_s = pval(bd)
        lo_s, hi_s = np.percentile(bd[np.isfinite(bd)], [2.5, 97.5])
        # Track A draws for the gap rows
        a_eq = None
        amap = {'G_pcgita': 'pcgita', 'G_neurovoz': 'neurovoz', 'G_adresso': 'adresso', 'G_adress2020': 'adress2020', 'G_pitt': 'pitt'}
        if rid in amap:
            az = np.load(f'{AD}/draws/A_{amap[rid]}_draws.npz', allow_pickle=False)
            a_eq = {'byte_equal_diff': bool(az['diff'].tobytes() == bd.tobytes()),
                    'same_speaker_idx': bool(np.array_equal(az['speaker_idx'].astype(np.int64), bo['sidx'])),
                    'max_abs_vs_mine': float(np.nanmax(np.abs(az['diff'] - d))),
                    'point_diff_A': float(az['point_diff'])}
        near0 = float(np.nanmin(np.abs(d)))
        res[rid] = {'kind': kind, 'value': bo['point_diff'], 'lo': float(lo), 'hi': float(hi), 'p': p,
                    'n_le0': le, 'n_ge0': ge, 'usable': B, 'p_no_tie_rule': p_notie,
                    'n_clips': bo['n_clips'], 'n_spk': bo['n_spk'], 'n_pos': bo['n_pos'],
                    'n_conflict': bo['n_conflict'], 'n_agreement': bo['n_agreement'],
                    'point_terms': [float(x) for x in bo['point_terms']],
                    'rank_auc_point_terms': [float(x) for x in bo['rank_point_terms']],
                    'min_abs_draw': near0,
                    'vs_B_perclip_max_abs': pc_max, 'vs_B_perclip_meta_ok': bool(pc_meta_ok),
                    'vs_B_draws': {'same_speaker_order': same_spk, 'same_speaker_idx': same_idx, 'same_nan_pattern': nanpat,
                                   'max_abs_draw_diff': maxdraw, 'max_abs_terms_diff': maxterms,
                                   'p_from_saved': p_saved, 'lo_from_saved': float(lo_s), 'hi_from_saved': float(hi_s),
                                   'point_diff_saved': float(z['point_diff'])},
                    'trackA': a_eq, 'sources': SOURCES.get(rid, [])}
        # write my rebuilt per-clip csv, sidecar and draws
        pcp = f'{IND}/per_clip/V_{rid}_perclip.csv'
        pd.DataFrame(recs).to_csv(pcp, index=False)
        np.savez(f'{IND}/draws/V_{rid}_draws.npz', diff=d, terms=bo['terms'], speaker_idx=bo['sidx'],
                 speakers=bo['uniq'], point_diff=np.float64(bo['point_diff']), point_terms=np.array(bo['point_terms']))
        side = dict(res[rid]); side.update({'id': rid, 'per_clip': pcp, 'per_clip_sha256': sha(pcp),
                                           'draws': f'{IND}/draws/V_{rid}_draws.npz',
                                           'bootstrap': {'n_draws': NB, 'rng': 'numpy default_rng(0), fresh per cell',
                                                         'resample': 'np.unique string speaker ids, idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker',
                                                         'percentiles': [2.5, 97.5], 'auc': 'sklearn.metrics.roc_auc_score',
                                                         'p_rule': 'min(1, 2*min((1+#d<=0)/(B+1), (1+#d>=0)/(B+1))), |d|<1e-12 counted as 0'},
                                           'written_utc': datetime.datetime.utcnow().isoformat() + 'Z'})
        with open(f'{IND}/per_clip/V_{rid}_perclip.sidecar.json', 'w') as f:
            json.dump(side, f, indent=1)
        print(f"{rid:24s} {kind:5s} {bo['point_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} B={B} "
              f"maxdraw={maxdraw:.1e} pc={pc_max:.1e} spk={same_spk}/{same_idx} A={a_eq and a_eq['byte_equal_diff']}", flush=True)

    # Holm within families, from my draws and from the saved draws
    hol = {}
    for fam in FAM:
        prim = FAM[fam]; full = prim + FAM_SENS[fam]
        h1 = holm({i: res[i]['p'] for i in prim}); h2 = holm({i: res[i]['p'] for i in full})
        h1s = holm({i: res[i]['vs_B_draws']['p_from_saved'] for i in prim})
        h2s = holm({i: res[i]['vs_B_draws']['p_from_saved'] for i in full})
        for i in full:
            hol[i] = {'family': fam, 'm': len(prim) if i in prim else 0, 'm_full': len(full),
                      'holm': h1.get(i), 'holm_full': h2[i], 'holm_saved': h1s.get(i), 'holm_full_saved': h2s[i]}
    # leave-one-out gaps (m=6)
    loo = {}
    for drop in FAM['GAPS']:
        keep = [i for i in FAM['GAPS'] if i != drop]
        h = holm({i: res[i]['p'] for i in keep})
        loo[drop] = sum(v <= 0.05 for v in h.values())
    # compare with B_holm.tsv
    tsv_cmp = {}
    for rid in ALL_IDS:
        b = btsv[rid]; r = res[rid]; h = hol.get(rid, {})
        chk = {
            'value': f"{r['value']:+.4f}" == b['value'],
            'lo': f"{r['lo']:+.4f}" == b['lo'],
            'hi': f"{r['hi']:+.4f}" == b['hi'],
            'p': f"{r['p']:.4f}" == b['p'],
            'n_le0': str(r['n_le0']) == b['n_le0'], 'n_ge0': str(r['n_ge0']) == b['n_ge0'],
            'usable': str(r['usable']) == b['usable_draws'],
            'n_clips': str(r['n_clips']) == b['n_clips'], 'n_spk': str(r['n_spk']) == b['n_spk'],
        }
        if h.get('holm') is not None:
            chk['holm'] = f"{h['holm']:.4f}" == b['holm_p']
            chk['survives'] = ('yes' if h['holm'] <= 0.05 else 'no') == b['survives']
            chk['m'] = str(h['m']) == b['m_family']
        if h:
            chk['holm_full'] = f"{h['holm_full']:.4f}" == b['holm_p_full']
            chk['survives_full'] = ('yes' if h['holm_full'] <= 0.05 else 'no') == b['survives_full']
            chk['m_full'] = str(h['m_full']) == b['m_family_full']
        tsv_cmp[rid] = {'all_match': all(chk.values()), 'fields': chk}
    out = {'res': res, 'holm': hol, 'gaps_leave_one_out_survivors_of_6': loo, 'tsv_compare': tsv_cmp,
           'edaic_dup_info': EDAIC_DUP_INFO, 'tex_sha256_now': tex_sha_now, 'tex_sha_ok': tex_sha_now == TEX_SHA,
           'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                        'scipy': scipy.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
           'started_utc': t0.isoformat() + 'Z', 'finished_utc': datetime.datetime.utcnow().isoformat() + 'Z'}
    with open(f'{IND}/V_results.json', 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('TSV all rows match:', sum(v['all_match'] for v in tsv_cmp.values()), 'of', len(tsv_cmp))
    for rid, v in tsv_cmp.items():
        if not v['all_match']:
            print('  MISMATCH', rid, {k: x for k, x in v['fields'].items() if not x})
    print('loo', loo)


if __name__ == '__main__':
    main()
