#!/usr/bin/env python3
"""Independent verifier for PART 20 POD3c (Qwen3-Omni transcript only on the E-DAIC 483 pair set).

Recomputes every number from per-clip csv files. Does not import or run the compute job's code.
AUC: explicit Mann-Whitney over every positive x negative pair, ties count 0.5.
Cross-check: trapezoid area under the ROC built from sorted unique thresholds.
Intervals: 2000 draws, numpy default_rng(0) created fresh for every cell, speakers resampled with
replacement from the sorted unique speaker list, percentile 2.5 / 97.5. Paired cells use one speaker
draw per replicate for both sides.

usage: POD3c_verify.py TEXT_CSV AUDIO_CSV TRANSCRIPT_INPUT_CSV [O25_TEXT_REF_CSV] [VALIDATION_CSV]
"""
import sys, json, os
import numpy as np
import pandas as pd

B = 2000


def clean_id(x):
    x = str(x).split('/')[-1]
    return x[:-4] if x.endswith('.wav') else x


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f'none of {names} in {list(df.columns)}')


def load_scores(path, pcol=None):
    df = pd.read_csv(path)
    idc = pick(df, ['id', 'clip', 'clip_id', 'path', 'seg'])
    pc = pcol if pcol else pick(df, ['p_yes', 'pyes', 'score'])
    out = pd.DataFrame({'id': df[idc].map(clean_id), 'p': df[pc].astype(float)})
    for c in ['label', 'speaker', 'speaker_id', 'answer_mass', 'mass', 'prompt']:
        if c in df.columns:
            out[c] = df[c].values
    return out


def mw_matrix(sp, sn):
    """pos x neg comparison matrix: 1 if pos > neg, 0.5 if tie, 0 otherwise."""
    sp = np.asarray(sp, float)[:, None]
    sn = np.asarray(sn, float)[None, :]
    return (sp > sn).astype(float) + 0.5 * (sp == sn)


def auc_mw(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    C = mw_matrix(s[y == 1], s[y == 0])
    return C.sum() / C.size


def auc_trap(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    P = (y == 1).sum(); N = (y == 0).sum()
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    return float(np.trapz(tpr, fpr)) if hasattr(np, 'trapz') else float(np.trapezoid(tpr, fpr))


class Cell:
    """One AUC cell with precomputed pos x neg matrix and row -> speaker index."""
    def __init__(self, s, y, spk_idx):
        s = np.asarray(s, float); y = np.asarray(y, int); spk_idx = np.asarray(spk_idx, int)
        self.C = mw_matrix(s[y == 1], s[y == 0])
        self.sp = spk_idx[y == 1]
        self.sn = spk_idx[y == 0]
        self.point = self.C.sum() / self.C.size

    def draw(self, cnt):
        wp = cnt[self.sp].astype(float); wn = cnt[self.sn].astype(float)
        a = wp.sum(); b = wn.sum()
        if a == 0 or b == 0:
            return np.nan
        return float(wp @ self.C @ wn) / (a * b)


def boot(cells, n_spk, combine):
    rng = np.random.default_rng(0)
    vals = np.empty(B)
    for r in range(B):
        idx = rng.integers(0, n_spk, n_spk)
        cnt = np.bincount(idx, minlength=n_spk)
        vals[r] = combine([c.draw(cnt) for c in cells])
    ok = ~np.isnan(vals)
    lo, hi = np.percentile(vals[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum())


def main():
    text_csv, audio_csv, tin_csv = sys.argv[1:4]
    ref_csv = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
    val_csv = sys.argv[5] if len(sys.argv) > 5 else None

    T = load_scores(text_csv, os.environ.get('PCOL'))
    print('text p column:', os.environ.get('PCOL') or 'p_yes')
    A = load_scores(audio_csv)
    tin = pd.read_csv(tin_csv)
    tin['id'] = tin['id'].map(clean_id)
    checks = {}
    checks['text_rows'] = len(T)
    checks['text_distinct_ids'] = int(T['id'].nunique())
    checks['audio_rows'] = len(A)
    checks['input_rows'] = len(tin)

    # truth for label / speaker / arm = transcript input (same as manifest)
    truth = tin[['id', 'label', 'speaker', 'set']].drop_duplicates('id').set_index('id')
    # dedupe scores: every repeat must carry identical p_yes
    def dedupe(df, name):
        g = df.groupby('id')['p'].agg(['min', 'max'])
        checks[f'{name}_max_repeat_diff'] = float((g['max'] - g['min']).max())
        return df.drop_duplicates('id').set_index('id')['p']
    Tp = dedupe(T, 'text'); Ap = dedupe(A, 'audio')
    checks['text_ids_missing_vs_input'] = int(len(set(truth.index) - set(Tp.index)))
    checks['text_ids_extra_vs_input'] = int(len(set(Tp.index) - set(truth.index)))
    checks['audio_ids_missing_vs_input'] = int(len(set(truth.index) - set(Ap.index)))
    if 'label' in T.columns:
        tl = T.drop_duplicates('id').set_index('id')['label']
        checks['text_label_mismatch_vs_input'] = int((tl.reindex(truth.index) != truth['label']).sum())
    spc = 'speaker' if 'speaker' in T.columns else ('speaker_id' if 'speaker_id' in T.columns else None)
    if spc:
        ts = T.drop_duplicates('id').set_index('id')[spc].astype(int)
        checks['text_speaker_mismatch_vs_input'] = int((ts.reindex(truth.index) != truth['speaker'].astype(int)).sum())
    if 'label' in A.columns:
        al = A.drop_duplicates('id').set_index('id')['label']
        checks['audio_label_mismatch_vs_input'] = int((al.reindex(truth.index) != truth['label']).sum())
    if 'prompt' in T.columns:
        checks['text_prompts_in_csv'] = sorted(T['prompt'].astype(str).unique().tolist())
    mc = 'answer_mass' if 'answer_mass' in T.columns else ('mass' if 'mass' in T.columns else None)
    if mc:
        checks['text_median_mass'] = float(np.median(T[mc].astype(float)))
        checks['text_min_mass'] = float(T[mc].astype(float).min())

    # the 966 row frame in transcript-input order (it holds the 483 + 483 pair rows)
    R = tin[['id', 'label', 'speaker', 'set']].copy()
    R['t'] = R['id'].map(Tp); R['a'] = R['id'].map(Ap)
    checks['rows_with_nan_text'] = int(R['t'].isna().sum())
    checks['rows_with_nan_audio'] = int(R['a'].isna().sum())
    spks = np.unique(R['speaker'].astype(int).values)
    n_spk = len(spks)
    R['si'] = np.searchsorted(spks, R['speaker'].astype(int).values)
    checks['n_speakers_966'] = n_spk
    for arm in ['conflict', 'agreement']:
        sub = R[R['set'] == arm]
        checks[f'{arm}_rows'] = len(sub)
        checks[f'{arm}_distinct'] = int(sub['id'].nunique())
        checks[f'{arm}_pos_rows'] = int((sub['label'] == 1).sum())
        checks[f'{arm}_neg_rows'] = int((sub['label'] == 0).sum())
        checks[f'{arm}_n_spk'] = int(sub['speaker'].nunique())
    agd = R[R['set'] == 'agreement'].drop_duplicates('id')
    checks['agreement_distinct_pos'] = int((agd['label'] == 1).sum())
    checks['agreement_distinct_neg'] = int((agd['label'] == 0).sum())
    checks['agreement_distinct_n_spk'] = int(agd['speaker'].nunique())

    subsets = {
        'conflict483': R[R['set'] == 'conflict'],
        'agree483': R[R['set'] == 'agreement'],
        'agree311': agd,
        'pooled966': R,
        'pooled794': R.drop_duplicates('id'),
    }
    rows = []

    def add(key, what, point, lo, hi, n, nsp, usable, trap=None):
        rows.append(dict(key=key, what=what, value=point, lo=lo, hi=hi, n=n, n_spk=nsp,
                         usable=usable, trap=trap))

    cells = {}
    for stream, col in [('text', 't'), ('audio', 'a')]:
        for sk, df in subsets.items():
            c = Cell(df[col].values, df['label'].values, df['si'].values)
            cells[(stream, sk)] = c
            p_mw = auc_mw(df[col].values, df['label'].values)
            p_tr = auc_trap(df[col].values, df['label'].values)
            assert abs(p_mw - c.point) < 1e-12
            lo, hi, us = boot([c], n_spk, lambda v: v[0])
            add(f'{stream}_{sk}', f'Q3O {stream} AUC {sk}', c.point, lo, hi, len(df),
                int(df['speaker'].nunique()), us, p_tr)
    # paired conflict minus agreement, within stream
    for stream in ['text', 'audio']:
        for ak in ['agree483', 'agree311']:
            c1 = cells[(stream, 'conflict483')]; c2 = cells[(stream, ak)]
            lo, hi, us = boot([c1, c2], n_spk, lambda v: v[0] - v[1])
            n = len(subsets['conflict483']) + len(subsets[ak])
            add(f'{stream}_conflict_minus_{ak}', f'Q3O {stream} paired conflict minus {ak}',
                c1.point - c2.point, lo, hi, n, n_spk, us)
    # paired transcript minus audio, per arm
    for sk in ['conflict483', 'agree483', 'agree311', 'pooled966', 'pooled794']:
        c1 = cells[('text', sk)]; c2 = cells[('audio', sk)]
        lo, hi, us = boot([c1, c2], n_spk, lambda v: v[0] - v[1])
        add(f'text_minus_audio_{sk}', f'Q3O paired transcript minus audio {sk}',
            c1.point - c2.point, lo, hi, len(subsets[sk]), int(subsets[sk]['speaker'].nunique()), us)
    # gap difference: (text conflict-agree) minus (audio conflict-agree)
    for ak in ['agree483', 'agree311']:
        cs = [cells[('text', 'conflict483')], cells[('text', ak)], cells[('audio', 'conflict483')], cells[('audio', ak)]]
        f = lambda v: (v[0] - v[1]) - (v[2] - v[3])
        lo, hi, us = boot(cs, n_spk, f)
        add(f'gapdiff_text_minus_audio_{ak}', f'(text conflict-{ak}) minus (audio conflict-{ak})',
            f([c.point for c in cs]), lo, hi, None, n_spk, us)

    # agreement between the two streams
    for sk in ['pooled966', 'pooled794', 'conflict483', 'agree483', 'agree311']:
        df = subsets[sk]
        t = df['t'].values; a = df['a'].values
        r = float(np.corrcoef(t, a)[0, 1])
        same05 = float(np.mean((t >= 0.5) == (a >= 0.5)))
        same05s = float(np.mean((t > 0.5) == (a > 0.5)))
        add(f'pearson_{sk}', f'Pearson r text vs audio p_yes {sk}', r, None, None, len(df), int(df['speaker'].nunique()), None)
        add(f'samedec_ge05_{sk}', f'same decision share (p>=0.5) {sk}', same05, None, None, len(df), int(df['speaker'].nunique()), None)
        add(f'samedec_gt05_{sk}', f'same decision share (p>0.5) {sk}', same05s, None, None, len(df), int(df['speaker'].nunique()), None)
        add(f'yesrate_text_{sk}', f'text yes rate (p>=0.5) {sk}', float(np.mean(t >= 0.5)), None, None, len(df), None, None)
        add(f'yesrate_audio_{sk}', f'audio yes rate (p>=0.5) {sk}', float(np.mean(a >= 0.5)), None, None, len(df), None, None)
        add(f'meanp_text_{sk}', f'text mean p_yes {sk}', float(np.mean(t)), None, None, len(df), None, None)

    # validation: 5 clips of Qwen2.5-Omni text scored on the pod vs the Part 16 reference
    if ref_csv and val_csv:
        ref = load_scores(ref_csv).drop_duplicates('id').set_index('id')['p']
        v = load_scores(val_csv, os.environ.get('VCOL'))
        print('validation p column:', os.environ.get('VCOL') or 'p_yes')
        v['ref'] = v['id'].map(ref)
        v['absdiff'] = (v['p'] - v['ref']).abs()
        checks['validation_n'] = len(v)
        checks['validation_max_absdiff'] = float(v['absdiff'].max())
        checks['validation_rows'] = v[['id', 'p', 'ref', 'absdiff']].to_dict('records')

    print(json.dumps(checks, indent=1, default=str))
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(out.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    return out, checks


if __name__ == '__main__':
    out, checks = main()
    if len(sys.argv) > 6:
        out.to_csv(sys.argv[6], index=False)
        json.dump(checks, open(sys.argv[6] + '.checks.json', 'w'), indent=1, default=str)
