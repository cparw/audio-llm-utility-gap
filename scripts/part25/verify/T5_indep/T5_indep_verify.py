#!/usr/local/bin/python3
# PART 25 T5 independent verifier. Written from scratch; imports nothing from the T5 track scripts.
# Reads raw answer files, pod OOF npz / per-layer csv files and saved curves; writes only under part25/verify.
import os, re, json, glob, hashlib, time, datetime
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

G = '<local data dir>/release_from_mac/scores'
PULL = G + '/leftovers_23sep/t5/pull'
OF = '<local data dir>/release/omni_final'
REL = '<local data dir>/release'
T5 = G + '/part25/T5'
VOUT = G + '/part25/verify'
WD = VOUT + '/T5_indep'
TEXROOT = '<local data dir>/paper1_submission_23sep'
FINALTEX = TEXROOT + '/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex'
NB = 2000

for p in (WD, VOUT):
    assert 'omni_final' not in p
os.makedirs(WD, exist_ok=True)

LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rank_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    r = rankdata(s); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def fnum(x):
    if isinstance(x, str):
        m = re.fullmatch(r'\s*np\.float64\((.*)\)\s*', x)
        if m:
            x = m.group(1)
    return float(x)

# ---------------- answer files (raw) ----------------
ANS = {
    'neurovoz': (OF + '/omni_neurovoz_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'kcl': (OF + '/omni_kcl_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'adresso': (OF + '/omni_adresso_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'adress2020': (OF + '/omni_adress2020_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'edaic_first30': (OF + '/omni_edaic_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'pitt': (OF + '/omni_pitt_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'edaic_whole_p23': (G + '/part23/A/A2_edaic_whole_zeroshot_perclip.csv', 'pid', 'speaker', 'p_yes_whole'),
    'edaic_lifted_p22': (REL + '/scores/part22/edaic_lifted_perclip.csv', 'pid', 'pid', 'p_yes_lifted'),
    'edaic_full_o25': (REL + '/edaic_rerun/variants/o25_full_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
    'edaic_full_statesrun': (REL + '/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv', 'clip', 'speaker', 'p_yes'),
}
EDAIC = {'edaic_first30', 'edaic_whole_p23', 'edaic_lifted_p22', 'edaic_full_o25', 'edaic_full_statesrun'}

def key_of(ds, v):
    v = str(v)
    if ds in EDAIC or ds.startswith('edaic'):
        return str(int(float(v.replace('.wav', ''))))
    return v

def load_answer(ds):
    f, cc, sc, pc = ANS[ds]
    d = pd.read_csv(f, dtype=str)
    out = pd.DataFrame({
        'key': [key_of(ds, v) for v in d[cc]],
        'speaker': d[sc].astype(str).values,
        'label': d['label'].astype(float).astype(int).values,
        'p': [fnum(v) for v in d[pc]],
    })
    assert out['key'].is_unique, ds
    return out, f, pc

# ---------------- bootstrap ----------------
def speaker_groups(spk, order='str'):
    spk = np.asarray(spk).astype(str)
    if order == 'str':
        uniq = np.unique(spk)
    elif order == 'num':
        uniq = np.array(sorted(set(spk), key=lambda s: float(s)))
    else:  # first appearance
        _, first = np.unique(spk, return_index=True)
        uniq = spk[np.sort(first)]
    pos = {u: i for i, u in enumerate(uniq)}
    code = np.array([pos[s] for s in spk])
    groups = [np.where(code == i)[0] for i in range(len(uniq))]
    return uniq, groups

def boot_matrix(y, S, spk, order='str'):
    """S: (n, k) score matrix. Fresh default_rng(0) per cell; every cell of one dataset shares n_spk,
    so each cell draws the same indices; computed per column with its own fresh rng to follow the rule."""
    uniq, groups = speaker_groups(spk, order)
    n_spk = len(uniq)
    k = S.shape[1]
    out = np.full((NB, k), np.nan)
    idx_all = None
    for j in range(k):
        rng = np.random.default_rng(0)
        idxs = np.empty((NB, n_spk), dtype=np.int64)
        for b in range(NB):
            idx = rng.choice(n_spk, size=n_spk, replace=True)
            idxs[b] = idx
            cl = np.concatenate([groups[i] for i in idx])
            yb = y[cl]
            if yb.min() == yb.max():
                continue
            out[b, j] = roc_auc_score(yb, S[cl, j])
        if idx_all is None:
            idx_all = idxs
        else:
            assert np.array_equal(idx_all, idxs)
    return out, idx_all, uniq

def ci(v):
    v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(v.size)

# ---------------- curves ----------------
def read_curve(p):
    d = pd.read_csv(p)
    return d

def curve_col(p, col='auc_oof'):
    d = pd.read_csv(p)
    return d[d.columns[0]].values.astype(int), d[col].values.astype(float)

def count_above(v, thr):
    return int((np.asarray(v) > thr).sum())

# ---------------- refit files ----------------
REFIT = {
    'neurovoz': [('lo-t5-8', 'neurovoz_encproj')],
    'adress2020': [('lo-t5-7', 'adress2020_encproj')],
    'edaicfull': [('lo-t5-9', 'edaicfull_encproj')],
    'edaic30': [('lo-t5-10', 'edaic30_encproj'), ('lo-t5-9', 'edaic30_encproj'), ('lo-t5-4', 'edaic30_encproj')],
    'pitt': [('lo-t5-10', 'pitt_encproj'), ('lo-t5-1', 'pitt_encproj')],
}
REFIT_DS = {'neurovoz': 'neurovoz', 'adress2020': 'adress2020', 'edaic_first30': 'edaic30', 'pitt': 'pitt',
            'edaic_whole_p23': 'edaicfull', 'edaic_lifted_p22': 'edaicfull', 'edaic_full_o25': 'edaicfull',
            'edaic_full_statesrun': 'edaicfull'}

def blas_of(pod):
    e = json.load(open(f'{PULL}/{pod}/out/env_{pod}.json'))
    archs = sorted({t.get('architecture') for t in e.get('threadpool_info', []) if t.get('internal_api') == 'openblas'})
    return ','.join(a for a in archs if a)

def enc_files(pod, job):
    """All encoder curve files for one job on one pod, with a fold vector keyed by clip."""
    od = f'{PULL}/{pod}/out'
    res = []
    # podfile*/macfile* with OOF npz
    for npz in sorted(glob.glob(f'{od}/{job}__*_oof.npz')):
        tag = os.path.basename(npz)[len(job) + 2:-len('_oof.npz')]
        csv = f'{od}/{job}__{tag}_enc.csv'
        z = np.load(npz, allow_pickle=True)
        names = [str(x) for x in z['name']]
        res.append(dict(pod=pod, job=job, tag=tag, csv=csv, names=names, fold=np.asarray(z['fold']),
                        label=np.asarray(z['label']), oof=np.asarray(z['enc']), meta=f'{od}/{job}__{tag}_meta.json'))
    # vrefit (second code path)
    vj = f'{od}/{job}__vrefit.json'
    if os.path.exists(vj):
        v = json.load(open(vj))
        res.append(dict(pod=pod, job=job, tag='vrefit', csv=f'{od}/{job}__vrefit_enc.csv', names=None, fold=None,
                        label=None, oof=np.load(f'{od}/{job}__vrefit_enc_oof.npy'), fold_file=v['fold_file']))
    # exact curves_from_states.py
    ex = f'{od}/{job}_encoder_perlayer.csv'
    if os.path.exists(ex):
        fn = pd.read_csv(f'{od}/{job}_folds_native.csv', dtype=str)
        res.append(dict(pod=pod, job=job, tag='exact_script', csv=ex, names=list(fn['clip_id']),
                        fold=fn['fold'].astype(int).values, label=None, oof=None))
    return res

def same_partition(names_a, fold_a, names_b, fold_b):
    ma = dict(zip(names_a, fold_a)); mb = dict(zip(names_b, fold_b))
    if set(ma) != set(mb):
        return False, 'clip sets differ'
    exact = all(ma[k] == mb[k] for k in ma)
    # partition equality regardless of fold id labels
    pa = {}
    for k in ma:
        pa.setdefault(ma[k], set()).add(k)
    pb = {}
    for k in mb:
        pb.setdefault(mb[k], set()).add(k)
    part = sorted(map(frozenset, pa.values()), key=lambda s: min(s)) == sorted(map(frozenset, pb.values()), key=lambda s: min(s))
    return part, ('fold ids equal' if exact else 'fold ids differ')

# =====================================================================
R = {}   # recomputed facts
t0 = time.time()
log('verifier start', datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))

# ---------- answer AUC, CI, bootstrap for every dataset ----------
ALL_DS = ['neurovoz', 'kcl', 'adresso', 'adress2020', 'edaic_whole_p23', 'edaic_lifted_p22', 'edaic_full_o25',
          'edaic_full_statesrun', 'edaic_first30', 'pitt']
SAVED = {'neurovoz': OF + '/omni_neurovoz_encoder_perlayer.csv', 'kcl': OF + '/omni_kcl_encoder_perlayer.csv',
         'adresso': OF + '/omni_adresso_encoder_perlayer.csv', 'adress2020': OF + '/omni_adress2020_encoder_perlayer.csv',
         'edaic_first30': OF + '/omni_edaic_encoder_perlayer.csv', 'pitt': OF + '/omni_pitt_encoder_perlayer.csv',
         'edaic_whole_p23': REL + '/edaic_rerun/variants/o25_full_encoder_perlayer.csv'}
for k in ('edaic_lifted_p22', 'edaic_full_o25', 'edaic_full_statesrun'):
    SAVED[k] = SAVED['edaic_whole_p23']

# main refit OOF (podfile of the main pod) per refit job
def main_oof(rj):
    pod, job = REFIT[rj][0]
    fl = [f for f in enc_files(pod, job) if f['tag'] == 'podfile']
    assert len(fl) == 1
    return fl[0]

perclip_written = []
for ds in ALL_DS:
    a, af, pc = load_answer(ds)
    y = a['label'].values; p = a['p'].values
    auc = roc_auc_score(y, p); auc_r = rank_auc(y, p)
    info = dict(n_clips=len(a), n_speakers=int(a['speaker'].nunique()), n_pos=int(y.sum()), answer_auc=auc,
                answer_auc_rank=auc_r, answer_file=af, answer_col=pc, answer_sha256=sha(af))
    # saved curve counts (answer thresholds filled after CI)
    rj = REFIT_DS.get(ds)
    cols = [p]
    colnames = ['p_yes_answer']
    if rj:
        mo = main_oof(rj)
        m = dict(zip(mo['names'], range(len(mo['names']))))
        keys = [key_of(ds, n) for n in mo['names']]
        km = dict(zip(keys, range(len(keys))))
        assert set(km) == set(a['key']), (ds, 'clip set mismatch', len(set(km) ^ set(a['key'])))
        order = np.array([km[k] for k in a['key']])
        lab_oof = mo['label'][order]
        info['label_mismatch_answer_vs_oof'] = int((lab_oof != y).sum())
        L = mo['oof'][:, :][order]
        for j in range(L.shape[1]):
            cols.append(L[:, j]); colnames.append(f'refit_oof_enc_L{j}')
        info['refit_oof_source'] = f"{PULL}/{REFIT[rj][0][0]}/out/{REFIT[rj][0][1]}__podfile_oof.npz"
    S = np.column_stack(cols)
    draws, idx, uniq = boot_matrix(y, S, a['speaker'].values, 'str')
    lo, hi, nuse = ci(draws[:, 0])
    info.update(answer_lo=lo, answer_hi=hi, boot_usable=nuse, n_spk_boot=len(uniq))
    # sensitivity to speaker ordering (answer only)
    alt = {}
    for order in ('num', 'first'):
        try:
            d2, _, _ = boot_matrix(y, S[:, :1], a['speaker'].values, order)
            alt[order] = ci(d2[:, 0])[:2]
        except Exception as e:
            alt[order] = str(e)[:60]
    info['answer_ci_other_speaker_orders'] = alt
    if rj:
        layer_draws = draws[:, 1:]
        delta = layer_draws - draws[:, [0]]
        dlo = np.nanpercentile(delta, 2.5, axis=0); dhi = np.nanpercentile(delta, 97.5, axis=0)
        info['paired_delta_lo_gt0'] = int((dlo > 0).sum())
        info['paired_delta_hi_lt0'] = int((dhi < 0).sum())
        info['paired_delta_lo'] = dlo.tolist(); info['paired_delta_hi'] = dhi.tolist()
        info['refit_oof_auc_from_npz'] = [roc_auc_score(y, S[:, j]) for j in range(1, S.shape[1])]
        np.savez_compressed(f'{WD}/T5v_draws_{ds}.npz', speakers=uniq, speaker_idx=idx.astype(np.int32),
                            answer_draws=draws[:, 0], layer_draws=layer_draws, delta_draws=delta,
                            delta_lo=dlo, delta_hi=dhi)
        # compare with the track's saved draws, draw by draw
        tb = f'{T5}/boot_r2/T5_boot_{ds}.npz'
        if os.path.exists(tb):
            z = np.load(tb, allow_pickle=True)
            cmp = {}
            cmp['speakers_equal'] = bool(np.array_equal(np.asarray(z['speakers']).astype(str), uniq))
            cmp['speaker_idx_equal'] = bool(np.array_equal(np.asarray(z['speaker_idx']), idx))
            cmp['answer_draws_maxabs'] = float(np.nanmax(np.abs(np.asarray(z['answer_draws']) - draws[:, 0])))
            if 'probe_draws' in z.files:
                cmp['probe_draws_maxabs'] = float(np.nanmax(np.abs(np.asarray(z['probe_draws']) - layer_draws)))
                cmp['delta_draws_maxabs'] = float(np.nanmax(np.abs(np.asarray(z['delta_draws']) - delta)))
            info['track_draw_compare'] = cmp
    else:
        np.savez_compressed(f'{WD}/T5v_draws_{ds}.npz', speakers=uniq, speaker_idx=idx.astype(np.int32),
                            answer_draws=draws[:, 0])
        tb = f'{T5}/boot_r2/T5_boot_{ds}.npz'
        if os.path.exists(tb):
            z = np.load(tb, allow_pickle=True)
            info['track_draw_compare'] = {'answer_draws_maxabs': float(np.nanmax(np.abs(np.asarray(z['answer_draws']) - draws[:, 0])))}
    # per-clip csv of exactly what the verifier used
    pcsv = f'{WD}/T5v_perclip_{ds}.csv'
    out = pd.DataFrame({'key': a['key'], 'speaker': a['speaker'], 'label': y})
    for j, c in enumerate(colnames):
        out[c] = S[:, j]
    out.to_csv(pcsv, index=False)
    perclip_written.append(pcsv)
    # saved curve counts
    if ds in SAVED:
        lay, sv = curve_col(SAVED[ds])
        info['saved_curve'] = SAVED[ds]
        info['saved_above_answer'] = count_above(sv, auc); info['saved_above_hi'] = count_above(sv, hi)
        info['saved_min'] = float(sv.min()); info['saved_min_layer'] = int(lay[sv.argmin()])
        info['saved_max'] = float(sv.max()); info['saved_max_layer'] = int(lay[sv.argmax()])
        info['saved_not_above_hi'] = [int(l) for l, v in zip(lay, sv) if not v > hi]
    R[ds] = info
    log(f"{ds:22s} n={info['n_clips']} spk={info['n_speakers']} auc={auc:.4f} (rank {auc_r:.4f}) ci=[{lo:.4f},{hi:.4f}] usable={nuse} "
        f"alt_orders={ {k: (tuple(round(x, 4) for x in v) if isinstance(v, tuple) else v) for k, v in alt.items()} } "
        f"saved {info.get('saved_above_answer')}/{info.get('saved_above_hi')} paired_lo>0={info.get('paired_delta_lo_gt0')} "
        f"cmp={info.get('track_draw_compare')}  t={time.time()-t0:.0f}s")

# ---------- refit curve counts over every file ----------
REF = {}
for rj, pods in REFIT.items():
    files = []
    for pod, job in pods:
        for f in enc_files(pod, job):
            f['blas'] = blas_of(pod)
            files.append(f)
    ref = [f for f in files if f['pod'] == pods[0][0] and f['tag'] == 'podfile'][0]
    rows = []
    for f in files:
        lay, v = curve_col(f['csv'])
        rec = dict(pod=f['pod'], tag=f['tag'], blas=f['blas'], csv=f['csv'], auc_oof=v, layers=lay)
        # fold identity
        if f['tag'] == 'vrefit':
            pm = json.load(open(ref['meta']))
            fm = json.load(open(f"{PULL}/{f['pod']}/out/{f['job']}__podfile_meta.json"))
            rec['same_folds'] = (f['fold_file'] == pm['fold_file'] == fm['fold_file'])
            rec['fold_note'] = 'fold_file ' + f['fold_file']
            # recompute AUC from vrefit OOF using podfile labels of the same pod (same order)
            pf = [g for g in files if g['pod'] == f['pod'] and g['tag'] == 'podfile'][0]
            rc = np.array([roc_auc_score(pf['label'], f['oof'][:, j]) for j in range(f['oof'].shape[1])])
            rec['oof_recompute_maxabs'] = float(np.max(np.abs(rc - v)))
        else:
            same, note = same_partition(ref['names'], ref['fold'], f['names'], f['fold'])
            rec['same_folds'] = same; rec['fold_note'] = note
            if f['oof'] is not None:
                rc = np.array([roc_auc_score(f['label'], f['oof'][:, j]) for j in range(f['oof'].shape[1])])
                rc2 = np.array([rank_auc(f['label'], f['oof'][:, j]) for j in range(f['oof'].shape[1])])
                rec['oof_recompute_maxabs'] = float(np.max(np.abs(rc - v)))
                rec['oof_rank_vs_sklearn_maxabs'] = float(np.max(np.abs(rc - rc2)))
        rows.append(rec)
    REF[rj] = rows

def refit_summary(ds):
    rj = REFIT_DS[ds]
    info = R[ds]
    ans, hi = info['answer_auc'], info['answer_hi']
    rows = REF[rj]
    main = [r for r in rows if r['pod'] == REFIT[rj][0][0] and r['tag'] == 'exact_script'][0]
    lay, v = main['layers'], main['auc_oof']
    s = dict(refit_file=main['csv'], refit_above_answer=count_above(v, ans), refit_above_hi=count_above(v, hi),
             refit_min=float(v.min()), refit_min_layer=int(lay[v.argmin()]), refit_max=float(v.max()),
             refit_max_layer=int(lay[v.argmax()]),
             refit_not_above_answer=[int(l) for l, x in zip(lay, v) if not x > ans],
             refit_not_above_hi=[int(l) for l, x in zip(lay, v) if not x > hi],
             npz_vs_exact_maxabs=float(np.max(np.abs(np.array(info['refit_oof_auc_from_npz']) - v))))
    same = [r for r in rows if r['same_folds']]
    other = [r for r in rows if not r['same_folds']]
    s['same_fold_files'] = len(same)
    s['same_fold_above_answer'] = sorted({count_above(r['auc_oof'], ans) for r in same})
    s['same_fold_above_hi'] = sorted({count_above(r['auc_oof'], hi) for r in same})
    s['other_fold_files'] = len(other)
    s['other_fold_above_answer'] = sorted({count_above(r['auc_oof'], ans) for r in other})
    s['other_fold_above_hi'] = sorted({count_above(r['auc_oof'], hi) for r in other})
    s['other_fold_tags'] = [f"{r['pod']}:{r['tag']}" for r in other]
    s['oof_recompute_maxabs_all'] = max(r.get('oof_recompute_maxabs', 0.0) for r in rows)
    s['per_file'] = [dict(pod=r['pod'], tag=r['tag'], blas=r['blas'], same_folds=bool(r['same_folds']),
                          above_answer=count_above(r['auc_oof'], ans), above_hi=count_above(r['auc_oof'], hi),
                          oof_recompute_maxabs=r.get('oof_recompute_maxabs')) for r in rows]
    return s

for ds in REFIT_DS:
    R[ds]['refit'] = refit_summary(ds)
    s = R[ds]['refit']
    log(f"REFIT {ds:22s} main {s['refit_above_answer']}/32 ans, {s['refit_above_hi']}/32 hi; min {s['refit_min']:.4f}@{s['refit_min_layer']} "
        f"max {s['refit_max']:.4f}@{s['refit_max_layer']}; same-fold files {s['same_fold_files']} ans {s['same_fold_above_answer']} hi {s['same_fold_above_hi']}; "
        f"other-fold {s['other_fold_files']} ans {s['other_fold_above_answer']} hi {s['other_fold_above_hi']} ({s['other_fold_tags']}); "
        f"oof recompute max {s['oof_recompute_maxabs_all']:.1e}; npz vs exact {s['npz_vs_exact_maxabs']:.1e}")

# ---------- P23 lifted curve (Figure 1 E-DAIC), saved only ----------
p23 = G + '/part23/pull/p23-a-probes/p23a/out/o25_whole_encoder_perlayer.csv'
lay, v = curve_col(p23)
R['p23_saved'] = dict(file=p23, sha256=sha(p23), max=float(v.max()), max_layer=int(lay[v.argmax()]),
                      vs={k: (count_above(v, R[k]['answer_auc']), count_above(v, R[k]['answer_hi']))
                          for k in ('edaic_whole_p23', 'edaic_lifted_p22', 'edaic_full_o25', 'edaic_full_statesrun')})
log('P23 saved lifted curve', R['p23_saved'])

# ---------- curve match ----------
CM = [
    ('pitt', 'enc', 'lo-t5-1', 'pitt_encproj', 'encoder', OF + '/omni_pitt_encoder_perlayer.csv'),
    ('pitt', 'proj', 'lo-t5-1', 'pitt_encproj', 'proj', OF + '/omni_pitt_proj_perlayer.csv'),
    ('pitt', 'enc', 'lo-t5-10', 'pitt_encproj', 'encoder', OF + '/omni_pitt_encoder_perlayer.csv'),
    ('pitt', 'proj', 'lo-t5-10', 'pitt_encproj', 'proj', OF + '/omni_pitt_proj_perlayer.csv'),
    ('pitt', 'llm', 'lo-t5-2', 'pitt_llm', 'llm', OF + '/omni_pitt_llm_perlayer.csv'),
    ('pitt', 'ans', 'lo-t5-3', 'pitt_ans', 'ans', OF + '/omni_pitt_ans_perlayer.csv'),
    ('pitt', 'ans', 'lo-t5-13', 'pitt_ans', 'ans', OF + '/omni_pitt_ans_perlayer.csv'),
    ('edaic30', 'enc', 'lo-t5-4', 'edaic30_encproj', 'encoder', OF + '/omni_edaic_encoder_perlayer.csv'),
    ('edaic30', 'proj', 'lo-t5-4', 'edaic30_encproj', 'proj', OF + '/omni_edaic_proj_perlayer.csv'),
    ('edaic30', 'enc', 'lo-t5-9', 'edaic30_encproj', 'encoder', OF + '/omni_edaic_encoder_perlayer.csv'),
    ('edaic30', 'proj', 'lo-t5-9', 'edaic30_encproj', 'proj', OF + '/omni_edaic_proj_perlayer.csv'),
    ('edaic30', 'enc', 'lo-t5-10', 'edaic30_encproj', 'encoder', OF + '/omni_edaic_encoder_perlayer.csv'),
    ('edaic30', 'proj', 'lo-t5-10', 'edaic30_encproj', 'proj', OF + '/omni_edaic_proj_perlayer.csv'),
    ('edaic30', 'llm', 'lo-t5-5', 'edaic30_llm', 'llm', OF + '/omni_edaic_llm_perlayer.csv'),
    ('edaic30', 'ans', 'lo-t5-6', 'edaic30_ans', 'ans', OF + '/omni_edaic_ans_perlayer.csv'),
    ('adress2020', 'enc', 'lo-t5-7', 'adress2020_encproj', 'encoder', OF + '/omni_adress2020_encoder_perlayer.csv'),
    ('adress2020', 'proj', 'lo-t5-7', 'adress2020_encproj', 'proj', OF + '/omni_adress2020_proj_perlayer.csv'),
    ('neurovoz', 'enc', 'lo-t5-8', 'neurovoz_encproj', 'encoder', OF + '/omni_neurovoz_encoder_perlayer.csv'),
    ('neurovoz', 'proj', 'lo-t5-8', 'neurovoz_encproj', 'proj', OF + '/omni_neurovoz_proj_perlayer.csv'),
    ('edaicfull', 'enc', 'lo-t5-9', 'edaicfull_encproj', 'encoder', REL + '/edaic_rerun/variants/o25_full_encoder_perlayer.csv'),
    ('edaicfull', 'proj', 'lo-t5-9', 'edaicfull_encproj', 'proj', REL + '/edaic_rerun/variants/o25_full_proj_perlayer.csv'),
]
CMR = []
for ds, st, pod, job, suf, saved in CM:
    rf = f'{PULL}/{pod}/out/{job}_{suf}_perlayer.csv'
    a = pd.read_csv(rf); b = pd.read_csv(saved)
    assert a.columns[0] == b.columns[0] and list(a[a.columns[0]]) == list(b[b.columns[0]]), (rf, saved)
    x = a['auc_oof'].values.astype(float); yv = b['auc_oof'].values.astype(float)
    d = np.abs(x - yv)
    m4 = int((np.round(x, 4) == np.round(yv, 4)).sum())
    rec = dict(curve=ds, stream=st, pod=pod, blas=blas_of(pod), n_layers=len(x), maxabs=float(d.max()), match4dp=m4,
               refit=rf, saved=saved)
    CMR.append(rec)
    log(f"CURVE {ds:10s} {st:4s} {pod:8s} {rec['blas']:9s} maxabs {rec['maxabs']:.2e} ({m4}/{len(x)} at 4dp)")

# counts on Haswell curves vs the AVX-512 curves
def cnt(ds, path):
    lay, v = curve_col(path)
    return count_above(v, R[ds]['answer_auc']), count_above(v, R[ds]['answer_hi'])
R['haswell_counts'] = {
    'pitt lo-t5-1 enc': cnt('pitt', f'{PULL}/lo-t5-1/out/pitt_encproj_encoder_perlayer.csv'),
    'pitt lo-t5-10 enc': cnt('pitt', f'{PULL}/lo-t5-10/out/pitt_encproj_encoder_perlayer.csv'),
    'edaic30 lo-t5-4 enc': cnt('edaic_first30', f'{PULL}/lo-t5-4/out/edaic30_encproj_encoder_perlayer.csv'),
    'edaic30 lo-t5-10 enc': cnt('edaic_first30', f'{PULL}/lo-t5-10/out/edaic30_encproj_encoder_perlayer.csv'),
}
log('Haswell vs AVX-512 counts', R['haswell_counts'])

# ---------- no refit for kcl / adresso ----------
allout = glob.glob(f'{PULL}/*/out/*') + glob.glob(G + '/leftovers_23sep/t5/inputs/*')
R['kcl_adresso_files_in_t5'] = sorted(os.path.basename(f) for f in allout if re.search(r'kcl|adresso', os.path.basename(f), re.I))
np5 = json.load(open(G + '/leftovers_23sep/verify/t5v/v5_notpossible.json'))
R['v5_notpossible'] = dict(n_npz_scanned=np5.get('n_npz_scanned'),
                           kcl_min_pyes_diff=min(c['max_abs_p_yes_diff_on_overlap'] for c in np5['candidates'] if c['target'] == 'kcl'),
                           adresso_237_min_pyes_diff=min(c['max_abs_p_yes_diff_on_overlap'] for c in np5['candidates'] if c['target'] == 'adresso' and c['n_states'] == 237))
log('kcl/adresso files in T5 pulls+inputs:', R['kcl_adresso_files_in_t5'], R['v5_notpossible'])

# ---------- tex checks ----------
texs = sorted(glob.glob(TEXROOT + '/**/*.tex', recursive=True))
hits = []
for t in texs:
    s = open(t, encoding='utf-8', errors='replace').read()
    if re.search(r'every\s+encoder\s+layer', s, re.I):
        hits.append(t)
fl = open(FINALTEX, encoding='utf-8').read().split('\n')
l226 = fl[225]
R['tex'] = dict(n_tex=len(texs), every_encoder_layer_hits=hits, line226_has_phrase='the signal is present at every depth' in l226,
                line226_commented=l226.lstrip().startswith('%'), line226_parkinson="Parkinson's speech" in l226,
                every_depth_lines=[i + 1 for i, x in enumerate(fl) if 'every depth' in x],
                every_depth_live=[i + 1 for i, x in enumerate(fl) if 'every depth' in x and not x.lstrip().startswith('%')],
                table1_edaic=[x.strip() for x in fl if x.strip().startswith('E-DAIC') and '&' in x])
log('TEX', R['tex'])

# ---------- the track's own independent check ----------
vs = json.load(open(T5 + '/verify/T5_verify_r2.sidecar.json'))
hash_ok = {k: (sha(T5 + '/' + k) == v) for k, v in vs['checked'].items()}
mt = []
for root, dirs, files in os.walk(OF):
    for fn in files:
        mt.append(os.path.getmtime(os.path.join(root, fn)))
mt.append(os.path.getmtime(OF))
R['track_check'] = dict(n_ok=vs['n_ok'], n_bad=vs['n_bad'], result=vs['result'], hashes_match_current=hash_ok,
                        omni_final_latest_mtime_utc=datetime.datetime.utcfromtimestamp(max(mt)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        omni_final_n_files=len(mt) - 1)
log('TRACK CHECK', R['track_check'])

def conv(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
json.dump(dict(R=R, CMR=CMR), open(WD + '/T5v_recomputed.json', 'w'), indent=1, default=conv)
open(WD + '/T5v_recompute.log', 'w').write('\n'.join(LOG) + '\n')
log('done', f'{time.time()-t0:.0f}s')
