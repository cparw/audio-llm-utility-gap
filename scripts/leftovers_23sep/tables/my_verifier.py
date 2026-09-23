#!/usr/bin/env python3
"""Independent recompute of every new value this task writes (leftovers_23sep/tables).

1. T7b intervals for the E-DAIC 300 s mean-of-five LM (0.7001) and encoder (0.5884) probes, and their paired
   differences from the zero-shot answer, from release/edaic_rerun/part17/T7b/T7b_edaic300_meanof5_perclip.csv.
2. PART 23 values appended to master_lookup.csv, from the PART 23 per-clip files.
3. The facts behind the fixed notes: sftfull n, PART 22 S1 processor values, E-DAIC window counts.

Own code: rank-based Mann-Whitney AUC (average ranks for ties, scipy.stats.rankdata), cross-checked with
sklearn roc_auc_score. Bootstrap: 2000 draws, a fresh numpy default_rng(0) per cell, speakers resampled with
idx = rng.choice(N, size=N, replace=True) over the unique sorted speaker ids, all clips of a drawn speaker
kept, draws with one class skipped, 2.5 and 97.5 percentiles; paired cells use the same idx.
Writes my_verifier_out.json and prints one line per value.
"""
import json, glob, re, sys, os, csv, platform
import numpy as np, pandas as pd
from scipy.stats import rankdata
import sklearn
from sklearn.metrics import roc_auc_score

R = '<local data dir>/release'
P23 = '<local data dir>/release_from_mac/scores/part23'
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
res = {'env': dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, machine=platform.machine())}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    b = roc_auc_score(y, s)
    assert abs(a - b) < 1e-12, (a, b)
    return float(a)


def fnum(x):
    return float(str(x).replace('np.float64(', '').replace(')', ''))


def boot(spk, y, stat, n=2000):
    """stat(ii) -> float. Speakers resampled; all clips of a drawn speaker kept."""
    spk = np.asarray(spk)
    u = np.array(sorted(set(spk.tolist())))
    clips = {s: np.where(spk == s)[0] for s in u}
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(u), size=len(u), replace=True)
        ii = np.concatenate([clips[u[k]] for k in idx])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        vals.append(stat(ii))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def cell(name, value, lohi, extra=None):
    d = dict(value=round(value, 6), value_4dp=round(value, 4), lo=round(lohi[0], 4), hi=round(lohi[1], 4), draws=lohi[2])
    if extra: d.update(extra)
    res[name] = d
    print('%-44s %.4f [%.4f, %.4f] draws %d' % (name, value, lohi[0], lohi[1], lohi[2]))


# ---------------------------------------------------------------- 1. T7b
T = pd.read_csv(R + '/edaic_rerun/part17/T7b/T7b_edaic300_meanof5_perclip.csv')
T['pid'] = T['pid'].astype(int)
T = T.sort_values('pid').reset_index(drop=True)
y = T['label'].values.astype(int); spk = T['pid'].values; pz = T['p_yes_answer'].values.astype(float)
res['T7b_n'] = dict(n=len(T), n_spk=int(len(set(spk))), n_pos=int(y.sum()))
cell('T7b_zeroshot_answer', auc(y, pz), boot(spk, y, lambda ii: auc(y[ii], pz[ii])))
for k in ('llm', 'enc', 'ans', 'proj'):
    V = [T['oof_%s_r%d' % (k, r)].values.astype(float) for r in range(5)]
    per = [auc(y, v) for v in V]
    m5 = float(np.mean(per))
    cell('T7b_%s_meanof5' % k, m5, boot(spk, y, lambda ii: float(np.mean([auc(y[ii], v[ii]) for v in V]))), dict(per_repeat=[round(p, 4) for p in per]))
    cell('T7b_%s_meanof5_minus_answer' % k, m5 - auc(y, pz),
         boot(spk, y, lambda ii: float(np.mean([auc(y[ii], v[ii]) for v in V])) - auc(y[ii], pz[ii])))

# ---------------------------------------------------------------- 2. PART 23
A2 = pd.read_csv(P23 + '/A/A2_edaic_whole_zeroshot_perclip.csv').sort_values('pid').reset_index(drop=True)
A3 = pd.read_csv(P23 + '/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv').sort_values('pid').reset_index(drop=True)
assert (A2['pid'].values == A3['pid'].values).all() and (A2['label'].values == A3['label'].values).all()
ya = A2['label'].values.astype(int); sa = A2['pid'].values
pw = A2['p_yes_whole'].map(fnum).values; p3 = A2['p_yes_300s_T7b'].map(fnum).values
pw3 = A3['p_yes_whole'].map(fnum).values
res['P23_A_n'] = dict(n=len(A2), n_spk=int(len(set(sa))), n_pos=int(ya.sum()), max_abs_pyes_A2_vs_A3=float(np.max(np.abs(pw - pw3))))
# the 300 s column must be the T7b zero-shot scores
t7map = dict(zip(T['pid'].values, pz))
res['P23_A_300s_equals_T7b_maxabs'] = float(np.max(np.abs(p3 - np.array([t7map[p] for p in sa]))))
cell('P23_zs_whole', auc(ya, pw), boot(sa, ya, lambda ii: auc(ya[ii], pw[ii])))
cell('P23_zs_300s', auc(ya, p3), boot(sa, ya, lambda ii: auc(ya[ii], p3[ii])))
cell('P23_zs_whole_minus_300s', auc(ya, pw) - auc(ya, p3), boot(sa, ya, lambda ii: auc(ya[ii], pw[ii]) - auc(ya[ii], p3[ii])))
for k in ('enc', 'proj', 'llm', 'ans'):
    V = [A3['oof_%s_r%d' % (k, r)].values.astype(float) for r in range(5)]
    per = [auc(ya, v) for v in V]; m5 = float(np.mean(per))
    cell('P23_%s_meanof5_whole' % k, m5, boot(sa, ya, lambda ii: float(np.mean([auc(ya[ii], v[ii]) for v in V]))), dict(per_repeat=[round(p, 4) for p in per]))
    cell('P23_%s_meanof5_whole_minus_zs_whole' % k, m5 - auc(ya, pw),
         boot(sa, ya, lambda ii: float(np.mean([auc(ya[ii], v[ii]) for v in V])) - auc(ya[ii], pw[ii])))


def ft(mode):
    fs = sorted(glob.glob(P23 + '/FT/FT_whole_fold*_oof.csv')) if mode == 'whole' else sorted(glob.glob(P23 + '/pull/p23-ft*/out/FT_ctrl300_fold*_oof.csv'))
    F = pd.concat([pd.read_csv(f) for f in fs]).sort_values('pid').reset_index(drop=True)
    return F, fs


Fw, fw = ft('whole'); Fc, fc = ft('ctrl300')
for F, mode in ((Fw, 'whole'), (Fc, 'ctrl300')):
    assert len(F) == 275 and F['pid'].nunique() == 275
    yf = F['label'].values.astype(int); sf = F['pid'].values; pf = F['p_yes'].values.astype(float)
    folds = {int(k): round(auc(g['label'].values.astype(int), g['p_yes'].values), 4) for k, g in F.groupby('fold')}
    cell('P23_ft_%s_pooled' % mode, auc(yf, pf), boot(sf, yf, lambda ii: auc(yf[ii], pf[ii])), dict(per_fold=folds, files=len(fw if mode == 'whole' else fc)))
    M = F.merge(A2[['pid']].assign(pw=pw, p3=p3), on='pid')
    assert len(M) == 275
    ym = M['label'].values.astype(int); sm = M['pid'].values; pm = M['p_yes'].values.astype(float); wm = M['pw'].values; tm = M['p3'].values
    cell('P23_ft_%s_minus_zs_whole' % mode, auc(ym, pm) - auc(ym, wm), boot(sm, ym, lambda ii: auc(ym[ii], pm[ii]) - auc(ym[ii], wm[ii])))
    if mode == 'ctrl300':
        cell('P23_ft_ctrl300_minus_zs_300s', auc(ym, pm) - auc(ym, tm), boot(sm, ym, lambda ii: auc(ym[ii], pm[ii]) - auc(ym[ii], tm[ii])))
Mwc = Fw.merge(Fc[['pid', 'p_yes']], on='pid', suffixes=('', '_c'))
yw = Mwc['label'].values.astype(int); sw = Mwc['pid'].values; a_ = Mwc['p_yes'].values; b_ = Mwc['p_yes_c'].values
cell('P23_ft_whole_minus_ctrl300', auc(yw, a_) - auc(yw, b_), boot(sw, yw, lambda ii: auc(yw[ii], a_[ii]) - auc(yw[ii], b_[ii])))
f3 = Fw[Fw['fold'] == 3]
res['P23_ft_whole_fold3'] = dict(n=len(f3), median_answer_mass=float(f3['answer_mass'].median()), n_mass_below_0_01=int((f3['answer_mass'] < 0.01).sum()),
                                 other_folds_min_median_mass=float(min(g['answer_mass'].median() for k, g in Fw.groupby('fold') if k != 3)))
print('fold 3', res['P23_ft_whole_fold3'])
C = pd.read_csv(P23 + '/C/C_trunc_275_derived.csv')
res['P23_trunc'] = dict(n=len(C), q3o_cut=int(C['q3o_cut'].sum()), af3_cut=int(C['af3_cut'].sum()), windows_over_600s=int((C['win_dur_s'] > 600.0 + 1e-6).sum()),
                        windows_over_900_0001s=int((C['win_dur_s'] > 900.001).sum()), q3o_audio_tokens_max=int(C['q3o_audio_tokens'].max()))
print('trunc', res['P23_trunc'])
A1 = pd.read_csv(P23 + '/A/A1_edaic_whole_audiotok.csv')
res['P23_tokens'] = dict(n=len(A1), cols=list(A1.columns)[:12])
tokcol = [c for c in A1.columns if 'tok' in c.lower()]
res['P23_tokens']['n_at_22500'] = {c: int((A1[c] == 22500).sum()) for c in tokcol if np.issubdtype(A1[c].dtype, np.number)}
print('tokens', res['P23_tokens'])

# ---------------------------------------------------------------- 3. facts behind the fixed notes
sf = pd.read_csv(R + '/edaic_rerun/part16/POD1/p14_o25_sftfull.csv')
res['sftfull_csv'] = dict(rows=len(sf), speakers=int(sf['speaker'].nunique()), unique_ids=int(sf['id'].nunique()))
print('sftfull', res['sftfull_csv'])
tm = json.load(open(R + '/scores/part22/verify/out/tokens_mine.json'))
res['P22_S1'] = dict(chunk_length=tm['chunk_length'], n_samples=tm['n_samples'],
                     tok_300s_default=sorted(set(c['default']['n_audio_tokens'] for c in tm['clips'])),
                     tok_900s_trunc_off=sorted(set(c['trunc_off_audio_kwargs']['n_audio_tokens'] for c in tm['clips'] if abs(c['dur_s'] - 900.0) < 1e-6)))
ad = pd.read_csv(R + '/scores/part22/verify/out/all_default.csv')
res['P22_S1']['all_default_cols'] = list(ad.columns)
dcol = [c for c in ad.columns if 'dur' in c.lower()][0]; tcol = [c for c in ad.columns if 'tok' in c.lower()][0]
res['P22_S1']['all_default_n'] = len(ad)
res['P22_S1']['gt300_n'] = int((ad[dcol] > 300.0).sum())
res['P22_S1']['gt300_at_7500'] = int(((ad[dcol] > 300.0) & (ad[tcol] == 7500)).sum())
res['P22_S1']['formula_match'] = int((ad[tcol] == np.minimum(np.ceil(np.minimum(ad[dcol], 300.0) * 100 / 4), 7500)).sum()) if False else None
wf = pd.read_csv(R + '/edaic_rerun/variants/windows_full.csv')
res['P22_S1']['capped900'] = int(wf['capped'].sum()) if 'capped' in wf.columns else None
res['P22_S1']['windows_full_n'] = len(wf)
cfg = open(R + '/scores/part22/P22_step1_processor_config.md').read()
res['P22_S1']['tf_version_in_config_md'] = sorted(set(re.findall(r'5\.\d+\.\d+', cfg)))
res['P22_S1']['trunc_default_line'] = [l.strip()[:160] for l in cfg.split('\n') if 'truncation' in l and ('default' in l.lower() or 'True' in l)][:3]
print('P22_S1', {k: v for k, v in res['P22_S1'].items() if k != 'all_default_cols'})
json.dump(res, open(os.path.join(OUT, 'my_verifier_out.json'), 'w'), indent=1)
