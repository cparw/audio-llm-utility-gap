#!/usr/local/bin/python3
"""PART 25 independent verifier for track C.

Own code. It imports nothing from part25/C and reads only raw inputs:
  master_lookup.csv, the zero shot answer csv each master row names, the original nested_repeats json
  (or POD3_SIDECAR.json), the probe OOF files (PART 25 Linux refits, PART 16 POD3 files, PART 26 files),
  and the PART 26 Kimi encoder per-clip files and sidecars.
Track C's own outputs (r2/perclip, r2/draws) are read only at the end, to compare against.

AUC = sklearn.metrics.roc_auc_score (every bootstrap AUC is also recomputed with a rank formula).
Paired speaker bootstrap: 2000 draws, fresh np.random.default_rng(0) per cell, unique speaker ids of the probe file
(read as strings, np.unique order), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each drawn speaker,
mean of the five per-repeat AUCs minus the answer AUC in the same draw, 2.5 and 97.5 percentiles.

usage: /usr/local/bin/python3 vC_recompute.py
"""
import csv, json, os, glob, hashlib, time, sys, platform
import numpy as np
from multiprocessing import Pool

ROOT = '<local data dir>/release_from_mac/scores/part25'
VOUT = ROOT + '/verify/C'
MASTER = '<local data dir>/release/master/master_lookup.csv'
PULL = ROOT + '/C/pull'
TR2 = ROOT + '/C/r2'
POD3 = '<local data dir>/release/edaic_rerun/part16/POD3'
P26 = '<local data dir>/release_from_mac/scores/part26'
NB = 2000

MODELS = {'q2a': 'Qwen2-Audio', 'q3o': 'Qwen3-Omni-30B-A3B', 'kimi': 'Kimi-Audio'}
DSNAME = {'pcgita': 'PC-GITA', 'neurovoz': 'NeuroVoz', 'kcl': 'MDVR-KCL', 'edaic': 'E-DAIC',
          'pitt': 'Pitt', 'adresso': 'ADReSSo', 'adress2020': 'ADReSS-2020'}
DSLIST = ['pcgita', 'neurovoz', 'kcl', 'edaic', 'pitt', 'adresso', 'adress2020']

P26_FILES = {
    'pcgita': ('Kimi-Audio_PC-GITA/Kimi-Audio_PC-GITA_perclip.csv', 'Kimi-Audio_PC-GITA/Kimi-Audio_PC-GITA_perclip.sidecar.json',
               ['probe_per_repeat'], ['gap_ci']),
    'neurovoz': ('Kimi-Audio_NeuroVoz/per_clip/p26_kimi_neurovoz_perclip.csv', 'Kimi-Audio_NeuroVoz/per_clip/p26_kimi_neurovoz_perclip.sidecar.json',
                 ['probe_per_repeat'], ['gap_ci']),
    'kcl': ('Kimi-Audio_MDVR-KCL/per_clip/p26_kimi_kcl_perclip.csv', 'Kimi-Audio_MDVR-KCL/per_clip/p26_kimi_kcl_perclip.sidecar.json',
            ['probe', 'per_repeat_auc'], ['gap', 'ci_2p5_97p5']),
    'edaic': ('Kimi-Audio_E-DAIC/p26_kimi_edaic_enc_perclip.csv', 'Kimi-Audio_E-DAIC/p26_kimi_edaic_enc_perclip.sidecar.json',
              ['per_repeat_auc'], ['gap_ci95']),
    'pitt': ('Kimi-Audio_Pitt/Kimi-Audio_Pitt_perclip.csv', 'Kimi-Audio_Pitt/Kimi-Audio_Pitt.sidecar.json',
             ['probe', 'per_repeat_auc'], ['gap', 'ci95']),
    'adresso': ('Kimi-Audio_ADReSSo/per_clip/Kimi-Audio_ADReSSo_perclip.csv', 'Kimi-Audio_ADReSSo/per_clip/Kimi-Audio_ADReSSo_perclip.sidecar.json',
                ['probe', 'per_repeat_auc'], ['gap', 'ci_2p5_97p5']),
    'adress2020': None,
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def dig(d, keys):
    for k in keys:
        d = d[k]
    return d


def master_rows():
    rows = read_csv(MASTER)
    return rows


def find_row(rows, model, ds, stream):
    hits = [(i + 1, r) for i, r in enumerate(rows) if r['model'] == model and r['dataset'] == ds and r['stream'] == stream]
    return hits


def probe_cols(header):
    out = []
    for pre in ('p_seed', 'p_probe_rep', 'p_probe_seed'):
        c = [pre + str(k) for k in range(5)]
        if all(x in header for x in c):
            out.append(c)
    if len(out) != 1:
        raise RuntimeError('probe columns ambiguous or missing: %r' % header)
    return out[0]


def auc_rank(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y)
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def boot(y, P, a, spk):
    from sklearn.metrics import roc_auc_score
    uniq = np.unique(np.asarray(spk, dtype=str))
    spk = np.asarray(spk, dtype=str)
    groups = [np.flatnonzero(spk == u) for u in uniq]
    ns = len(uniq)
    rng = np.random.default_rng(0)
    diff = np.full(NB, np.nan)
    pm = np.full(NB, np.nan)
    an = np.full(NB, np.nan)
    diff_rank = np.full(NB, np.nan)
    skipped = 0
    for b in range(NB):
        idx = rng.choice(ns, size=ns, replace=True)
        ii = np.concatenate([groups[j] for j in idx])
        yy = y[ii]
        if yy.min() == yy.max():
            skipped += 1
            continue
        per = [roc_auc_score(yy, P[ii, r]) for r in range(P.shape[1])]
        pm[b] = float(np.mean(per))
        an[b] = roc_auc_score(yy, a[ii])
        diff[b] = pm[b] - an[b]
        diff_rank[b] = np.mean([auc_rank(yy, P[ii, r]) for r in range(P.shape[1])]) - auc_rank(yy, a[ii])
    ok = ~np.isnan(diff)
    lo, hi = np.percentile(diff[ok], [2.5, 97.5])
    lo_r, hi_r = np.percentile(diff_rank[ok], [2.5, 97.5])
    return dict(diff=diff, probe_mean5=pm, answer=an, diff_rank=diff_rank, lo=float(lo), hi=float(hi),
                lo_rank=float(lo_r), hi_rank=float(hi_r), usable=int(ok.sum()), skipped=skipped, n_spk=ns,
                speaker_ids=uniq, maxabs_rank_vs_sklearn=float(np.nanmax(np.abs(diff - diff_rank))),
                share_le0=float(np.mean(diff[ok] <= 0)))


def build_cell(tag, ds, rows):
    """Returns the spec of one master cell, resolved from raw files by own logic."""
    model = MODELS[tag]
    pstream = 'LM probe' if tag == 'kimi' else 'encoder probe'
    ph = find_row(rows, model, ds, pstream)
    ah = find_row(rows, model, ds, 'zero shot answer')
    enc_h = find_row(rows, model, ds, 'encoder probe')
    assert len(ph) == 1, (tag, ds, pstream, [h[0] for h in ph])
    assert len(ah) == 1, (tag, ds, 'answer', [h[0] for h in ah])
    prow, pr = ph[0]
    arow, ar = ah[0]
    asrc = ar['source'].split(';')[0].strip()
    ans_json_auc = None
    if asrc.endswith('_zeroshot.json'):
        ans_json_auc = json.load(open(asrc)).get('auc')
        afile = asrc[:-len('.json')] + '_scores.csv'
    else:
        afile = asrc
    psrc = pr['source']
    skey = 'llm' if tag == 'kimi' else 'enc'
    if psrc.endswith('.json'):
        ref = json.load(open(psrc))[skey]
        ref_per = ref['per_repeat']
        ref_src = psrc + ' [%s.per_repeat]' % skey
        ref_mean = ref.get('mean')
    else:
        # POD3 csv named directly by the master row; reference = POD3_SIDECAR nested_per_repeat
        sc = json.load(open(POD3 + '/POD3_SIDECAR.json'))
        key = [k for k in sc if k.startswith('streams_' + ds)]
        assert len(key) == 1, key
        ref_per = sc[key[0]]['enc']['nested_per_repeat']
        ref_mean = sc[key[0]]['enc']['nested_auc_mean']
        ref_src = POD3 + '/POD3_SIDECAR.json [%s.enc.nested_per_repeat]' % key[0]
    # probe per-clip file
    if psrc.endswith('.csv'):
        pfile = psrc
    else:
        g = glob.glob(PULL + '/*/out/%s_%s_%s_rep5_oof.csv' % (tag, ds, skey))
        if len(g) == 1:
            pfile = g[0]
        elif len(g) == 0 and tag == 'q3o' and ds == 'edaic':
            pfile = P26 + '/Qwen3-Omni-30B-A3B_E-DAIC/Q3O_EDAIC_perclip.csv'
        else:
            raise RuntimeError('probe file for %s %s: %r' % (tag, ds, g))
    return dict(tag=tag, ds=ds, model=model, pstream=pstream, prow=prow, pmaster=float(pr['auc']), pn=int(pr['n']),
                psrc=psrc, arow=arow, amaster=float(ar['auc']), an=int(ar['n']), asrc=asrc, afile=afile,
                ans_json_auc=ans_json_auc, pfile=pfile, ref_per=ref_per, ref_mean=ref_mean, ref_src=ref_src,
                n_encoder_rows=len(enc_h), kind='master')


def build_kimi_supp(ds, rows):
    ah = find_row(rows, 'Kimi-Audio', ds, 'zero shot answer')
    assert len(ah) == 1
    arow, ar = ah[0]
    asrc = ar['source'].split(';')[0].strip()
    afile = asrc[:-len('.json')] + '_scores.csv' if asrc.endswith('_zeroshot.json') else asrc
    spec = P26_FILES[ds]
    if spec is None:
        folder = P26 + '/Kimi-Audio_' + DSNAME[ds]
        found = [p for p in glob.glob(folder + '/**/*perclip*.csv', recursive=True)
                 if '/pull/' not in p and '/stage/' not in p]
        found_sc = [p for p in glob.glob(folder + '/**/*sidecar*.json', recursive=True)
                    if '/pull/' not in p and '/stage/' not in p]
        return dict(tag='kimi', ds=ds, kind='supp', status='NOT LANDED', found_perclip=found, found_sidecar=found_sc)
    pfile = P26 + '/' + spec[0]
    scfile = P26 + '/' + spec[1]
    sc = json.load(open(scfile))
    return dict(tag='kimi', ds=ds, kind='supp', status='landed', model='Kimi-Audio', pstream='encoder probe (PART 26)',
                arow=arow, amaster=float(ar['auc']), an=int(ar['n']), asrc=asrc, afile=afile, ans_json_auc=None,
                pfile=pfile, p26_sidecar=scfile, ref_per=dig(sc, spec[2]), ref_src=scfile + ' [' + '.'.join(spec[2]) + ']',
                p26_ci=dig(sc, spec[3]))


def run_cell(c):
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    if c.get('status') == 'NOT LANDED':
        return c
    arows = read_csv(c['afile'])
    prows = read_csv(c['pfile'])
    pcols = probe_cols(list(prows[0].keys()))
    aclips = [r['clip'] for r in arows]
    pclips = [r['clip'] for r in prows]
    c['answer_file_sha256'] = sha(c['afile'])
    c['probe_file_sha256'] = sha(c['pfile'])
    c['answer_rows'] = len(arows)
    c['probe_rows'] = len(prows)
    c['answer_dup'] = len(aclips) - len(set(aclips))
    c['probe_dup'] = len(pclips) - len(set(pclips))
    c['probe_cols'] = pcols
    # answer AUC on its own file
    ya = np.array([int(float(r['label'])) for r in arows])
    sa = np.array([float(r['p_yes']) for r in arows])
    c['answer_auc_own_file'] = float(roc_auc_score(ya, sa))
    c['answer_auc_own_file_rank'] = float(auc_rank(ya, sa))
    # probe AUCs on its own file
    yp = np.array([int(float(r['label'])) for r in prows])
    Pp = np.array([[float(r[k]) for k in pcols] for r in prows])
    per = [float(roc_auc_score(yp, Pp[:, k])) for k in range(5)]
    c['per_repeat'] = per
    c['per_repeat_rank'] = [float(auc_rank(yp, Pp[:, k])) for k in range(5)]
    c['probe_mean5_own_file'] = float(np.mean(per))
    c['per_repeat_equal_ref_4dp'] = all('%.4f' % a == '%.4f' % b for a, b in zip(per, c['ref_per']))
    c['per_repeat_maxabs_vs_ref'] = float(max(abs(a - b) for a, b in zip(per, c['ref_per'])))
    # join on clip
    amap = {r['clip']: r for r in arows}
    joined = [r for r in prows if r['clip'] in amap]
    c['n_joined'] = len(joined)
    c['probe_only'] = len(prows) - len(joined)
    c['answer_only'] = len(set(aclips) - set(pclips))
    y = np.array([int(float(r['label'])) for r in joined])
    ya2 = np.array([int(float(amap[r['clip']]['label'])) for r in joined])
    c['label_disagree'] = int((y != ya2).sum())
    P = np.array([[float(r[k]) for k in pcols] for r in joined])
    a = np.array([float(amap[r['clip']]['p_yes']) for r in joined])
    spk = np.array([str(r['speaker']) for r in joined])
    spk_a = np.array([str(amap[r['clip']]['speaker']) for r in joined])
    c['n_pos'] = int(y.sum())
    c['n_neg'] = int(len(y) - y.sum())
    c['n_spk_probe_file'] = int(len(np.unique(spk)))
    c['n_spk_answer_file'] = int(len(np.unique(spk_a)))
    pairs = set(zip(spk, spk_a))
    c['speaker_map_one_to_one'] = (len(pairs) == c['n_spk_probe_file'] == c['n_spk_answer_file'])
    c['speaker_ids_identical'] = bool((spk == spk_a).all())
    c['probe_joined'] = float(np.mean([roc_auc_score(y, P[:, k]) for k in range(5)]))
    c['answer_joined'] = float(roc_auc_score(y, a))
    c['diff_point'] = c['probe_joined'] - c['answer_joined']
    B = boot(y, P, a, spk)
    for k in ('lo', 'hi', 'lo_rank', 'hi_rank', 'usable', 'skipped', 'n_spk', 'maxabs_rank_vs_sklearn', 'share_le0'):
        c[k] = B[k]
    if c['speaker_map_one_to_one'] and not c['speaker_ids_identical']:
        B2 = boot(y, P, a, spk_a)
        c['alt_answer_ids_lo'] = B2['lo']
        c['alt_answer_ids_hi'] = B2['hi']
    # own outputs
    stem = 'vC_%s_%s_%s' % (c['tag'], c['ds'], 'enc_part26' if c['kind'] == 'supp' else ('llm' if c['tag'] == 'kimi' else 'enc'))
    c['stem'] = stem
    pc = VOUT + '/perclip/' + stem + '_perclip.csv'
    with open(pc, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clip', 'speaker', 'speaker_answer_file', 'label'] + ['p_probe_rep%d' % k for k in range(5)] + ['p_yes_answer'])
        for i, r in enumerate(joined):
            w.writerow([r['clip'], spk[i], spk_a[i], y[i]] + ['%.10g' % v for v in P[i]] + ['%.10g' % a[i]])
    c['perclip_file'] = pc
    dr = VOUT + '/draws/' + stem + '_draws.npz'
    np.savez_compressed(dr, diff=B['diff'], probe_mean5=B['probe_mean5'], answer=B['answer'], diff_rank=B['diff_rank'],
                        speaker_ids=B['speaker_ids'], seed=0, n_boot=NB, point_diff=c['diff_point'])
    c['draws_file'] = dr
    # compare with track C outputs (read only)
    if c['kind'] == 'master':
        tstem = 'C_r2_%s_%s_%s' % (c['tag'], c['ds'], 'llm' if c['tag'] == 'kimi' else 'enc')
        tdr = TR2 + '/draws/' + tstem + '_draws.npz'
        tpc = TR2 + '/perclip/' + tstem + '_perclip.csv'
    else:
        tstem = 'C_r2_kimi_%s_enc_part26' % c['ds']
        tdr = TR2 + '/kimi_enc/draws/' + tstem + '_draws.npz'
        tpc = TR2 + '/kimi_enc/perclip/' + tstem + '_perclip.csv'
    if os.path.exists(tdr):
        z = np.load(tdr, allow_pickle=True)
        td = z['diff']
        c['track_draws_maxabs'] = float(np.nanmax(np.abs(td - B['diff']))) if td.shape == B['diff'].shape else 'shape %r' % (td.shape,)
    else:
        c['track_draws_maxabs'] = 'missing'
    if os.path.exists(tpc):
        tr = read_csv(tpc)
        tmap = {r['clip']: r for r in tr}
        cols = list(tr[0].keys())
        tpcols = [k for k in cols if k.startswith('p_probe') or k.startswith('p_seed') or k.startswith('p_enc_probe')][:5]
        tacol = [k for k in cols if 'answer' in k and k.startswith('p_')] or [k for k in cols if k.startswith('p_yes')]
        mx = 0.0
        miss = 0
        for i, r in enumerate(joined):
            t = tmap.get(r['clip'])
            if t is None:
                miss += 1
                continue
            mx = max(mx, max(abs(float(t[k]) - P[i, j]) for j, k in enumerate(tpcols)), abs(float(t[tacol[0]]) - a[i]))
        c['track_perclip_maxabs'] = mx
        c['track_perclip_missing'] = miss + (len(tr) - len(joined) if len(tr) > len(joined) else 0)
    else:
        c['track_perclip_maxabs'] = 'missing'
    c['seconds'] = round(time.time() - t0, 1)
    sc = VOUT + '/sidecars/' + stem + '.sidecar.json'
    c['sidecar_file'] = sc
    json.dump(c, open(sc, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return c


def main():
    import sklearn, scipy
    rows = master_rows()
    cells = []
    for tag in ('q2a', 'q3o', 'kimi'):
        for ds in DSLIST:
            cells.append(build_cell(tag, ds, rows))
    for ds in DSLIST:
        cells.append(build_kimi_supp(ds, rows))
    kimi_enc_rows = [i + 1 for i, r in enumerate(rows) if r['model'] == 'Kimi-Audio' and 'encoder' in r['stream'].lower()]
    with Pool(12) as pool:
        res = pool.map(run_cell, cells)
    meta = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script=os.path.abspath(__file__),
                env=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
                master_lookup=MASTER, master_lookup_sha256=sha(MASTER), n_master_rows=len(rows),
                kimi_encoder_rows_in_master=kimi_enc_rows)
    json.dump(dict(meta=meta, cells=res), open(VOUT + '/vC_cells.json', 'w'), indent=1,
              default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('done', meta['created_utc'], len(res))


if __name__ == '__main__':
    main()
