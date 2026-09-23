#!/usr/bin/env python3
"""Decide the twin (independent verifier value) for every PART 16 / PART 17 fragment row.

Inputs: twin_map_auto.csv from twin_extract.py (same row order as summary_all.csv parts 16 and 17),
summary_all.csv, the task verifier files for the special cases, check_summary.csv (the summary
checker's own recompute), and optionally the p16twins output folder.
Output: twin_decisions.csv in OUTDIR.
"""
import csv, json, re, os, sys, glob
from decimal import Decimal, ROUND_HALF_UP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin_extract import dec, eq, R, P16, P17, SUMMARY

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else '.'
P16TWINS = '<local data dir>/leftovers_23sep/p16twins'
CHECK = '<local data dir>/release_from_mac/summary/check_summary.csv'


def rel(p):
    return p.replace(R + '/', 'release/') if p else ''


def fmt(x, like):
    """format the verifier number like the fragment value: 4 dp for decimals, int for integers, sci for sci."""
    if isinstance(x, str):
        return x
    s = str(like).strip()
    if re.search(r'[eE]', s):
        return '%.3e' % x
    d = dec(s)
    if d is not None and d.as_tuple().exponent >= 0 and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    k = max(4, min(6, -d.as_tuple().exponent)) if d is not None and d.as_tuple().exponent < 0 else 4
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP))


def srctype(f):
    if '/verify/' in f or f.endswith('T7b_verify.json'):
        return 'Mac verifier (verify folder)'
    if 'VERIFY' in os.path.basename(f):
        return 'pod-side verifier'
    if 'm7m11' in f:
        return 'm7m11 task'
    if 'check_summary' in f:
        return 'summary checker recompute'
    if 'p16twins' in f:
        return 'p16twins task'
    return 'other'


summ = [r for r in csv.DictReader(open(SUMMARY, newline='')) if r['part'] in ('16', '17')]
auto = list(csv.DictReader(open(os.path.join(OUTDIR, 'twin_map_auto.csv'), newline='')))
assert len(summ) == len(auto) and all(a['id'] == s['id'] for a, s in zip(auto, summ))

# ------------------------------------------------------------------ special twins
SPECIAL = {}   # uid -> dict(verified, lo, hi, file, loc, note)

# M5: verifier 'checked' list, keyed "<cell> <qty>" with mine / theirs
m5 = json.load(open(P16 + '/verify/M5_verify_out.json'))
chk = {c['what']: c for c in m5['checked']}
for s in summ:
    if s['task/pod'] != 'M5':
        continue
    i = s['id']
    m = re.match(r'M5_paired_(auc_probe|auc_d|auc_without_d)_conflict_minus_agreement$', i)
    if m:
        q = m.group(1); keys = ('paired %s diff' % q, 'paired %s lo' % q, 'paired %s hi' % q)
    else:
        m = re.match(r'M5_(all|conflict_refit|agreement_refit|conflict_restricted|agreement_restricted)_(.+)$', i)
        cell, q = m.groups()
        if q == 'randfloor':
            keys = ('%s rand_mean_abs' % cell, '%s rand_lo' % cell, '%s rand_hi' % cell)
        else:
            keys = ('%s %s' % (cell, q), '%s %s_lo' % (cell, q), '%s %s_hi' % (cell, q))
    v = chk[keys[0]]
    lo = chk.get(keys[1]); hi = chk.get(keys[2])
    SPECIAL[i] = dict(verified=float(v['mine']), lo=float(lo['mine']) if lo and s['lo'] else None,
                      hi=float(hi['mine']) if hi and s['hi'] else None,
                      file=P16 + '/verify/M5_verify_out.json', loc='checked: "%s"' % keys[0], note='')

# M7fix 0.7706: T4 verifier recomputes the Pitt encoder mean of five from the part10 npz and its interval
t4 = open(P17 + '/verify/T4_verify.log').read().split('\n')
l5 = [l for l in t4 if 'Q25O pitt encoder mean-of-5 (row 1)' in l][0]
lci = [l for l in t4 if l.startswith('[OK ] CI T4_pitt_o25_enc_mean5')][0]
mw = float(re.search(r'MW ([\d.]+)', l5).group(1))
own = re.search(r'own: \[([\d.]+), ([\d.]+)\]', lci)
for s in summ:
    if s['task/pod'] == 'M7fix' and 'encoder probe' in s['what']:
        SPECIAL[s['id']] = dict(verified=mw, lo=float(own.group(1)), hi=float(own.group(2)), file=P17 + '/verify/T4_verify.log',
                                loc='line "%s" and line "%s"' % (l5[:60], lci[:40]), note='same quantity as T4_pitt_o25_enc_mean5 (Pitt encoder, mean of five, part10 npz)')

# POD3 3b by-arm refit probe: the Mac verifier (partF) recomputes a different AUC
pf = {x['id']: x for x in json.load(open(P16 + '/verify/POD3_partF_out.json'))}
for key in ('3b_conflict_auc_probe', '3b_agreement_auc_probe'):
    x = pf[key]
    SPECIAL[key] = dict(verified=x['mine'], lo=x['mine_ci'][0], hi=x['mine_ci'][1], file=P16 + '/verify/POD3_partF_out.json',
                        loc='id %s' % key, note='Mac refit of the within-arm probe from raw states; the row is already marked SUPERSEDED / NOT USED')

# T5: verifier results per dataset (counts from the saved curve, recomputed by the verifier)
t5 = json.load(open(P17 + '/verify/T5_verify.json'))['results']
t5d = {x['dataset']: x for x in t5}
T5Q = {'layers_above': 'above', 'layers_strict': 'mean_minus_std', 'layers_above_ci': 'above_ci_hi', 'probe_max': 'probe_max'}
for s in summ:
    if s['task/pod'] != 'T5':
        continue
    m = re.match(r'T5_(.+)_(layers_above_ci|layers_above|layers_strict|probe_max|answer)$', s['id'])
    if not m or m.group(1) not in t5d:
        continue
    ds, q = m.groups(); x = t5d[ds]
    if q == 'answer':
        SPECIAL[s['id']] = dict(verified=x['answer_mw'], lo=x['lo'], hi=x['hi'], file=P17 + '/verify/T5_verify.json', loc='results[dataset=%s] answer_mw, lo, hi' % ds, note='')
    else:
        SPECIAL[s['id']] = dict(verified=float(x[T5Q[q]]), lo=None, hi=None, file=P17 + '/verify/T5_verify.json', loc='results[dataset=%s] %s' % (ds, T5Q[q]),
                                note='verifier recount from the saved curve; its own refit from states does not reproduce the curve at 4 dp (T5_verify.log)')

# POD3 3a rows: the Mac verifier's own value and its integer-speaker interval
p3 = {x['id']: x for x in json.load(open(P16 + '/verify/POD3_verify_out.json'))['rows']}
for s in summ:
    if s['task/pod'] == 'POD3' and s['id'] in p3:
        x = p3[s['id']]
        ci = x.get('mine_ci_integers') or []
        SPECIAL[s['id']] = dict(verified=x['mine'], lo=ci[0] if len(ci) == 2 else None, hi=ci[1] if len(ci) == 2 else None,
                                file=P16 + '/verify/POD3_verify_out.json', loc='rows[id=%s] mine, mine_ci_integers' % s['id'], note='')

# SEED rows 15 to 17 (Qwen3-Omni 3b, all clips): T3 others verifier recomputed from the pod perp columns
t3o = open(P17 + '/verify/T3_verify_others.log').read().split('\n')
def t3line(tag):
    """last occurrence = the Qwen3-Omni section (the Qwen2-Audio section comes first in the log)"""
    hit = (None, None)
    for l in t3o:
        m = re.match(r'\s*all\s+%s\s+([\d.]+) \[([\d.]+), ([\d.]+)\]' % tag, l)
        if m:
            hit = (l.strip(), float(m.group(1)))
    return hit
for sid, tag in (('SEED15_3b', 'probe'), ('SEED16_3b', 'along'), ('SEED17_3b', 'perp_trainz')):
    l, x = t3line(tag)
    if l and any(ss['id'] == sid for ss in summ):
        SPECIAL[sid] = dict(verified=x, lo=None, hi=None, file=P17 + '/verify/T3_verify_others.log', loc='line "%s"' % l, note='same cell as Q3Ofix')

# M2 selection counts: the verifier's own DistilBERT rebuild
m2 = json.load(open(P16 + '/verify/M2_verify_out.json'))
for sid, key in (('M2.pairs', 'n_pairs'), ('M2.clips', 'n_rows'), ('M2.speakers', 'n_speakers')):
    SPECIAL[sid] = dict(verified=float(m2['distilbert_arms'][key]), lo=None, hi=None, file=P16 + '/verify/M2_verify_out.json', loc='distilbert_arms.%s (verifier rebuild)' % key, note='')

# E-DAIC 300 s mean-of-five rows without their own twin: T7b verifier (Linux sklearn 1.9.1 rerun of all five repeats)
t7b = json.load(open(P17 + '/verify/T7b_verify_out/T7b_verify.json'))
tsvc = {x['id']: x for x in t7b['tsv_comparison']}
for s in summ:
    m = re.match(r'(T7_published_(llm|enc|ans|proj)_meanof5|T4_edaic_o25_(llm|enc)_mean5_jsononly)$', s['id'])
    if not m:
        continue
    k = m.group(2) or m.group(3)
    st = t7b['stage_' + k]
    mean5 = sum(st['mine_per_repeat_mw']) / 5.0
    ci = tsvc['T7b_%s_meanof5' % k]['verifier']['mine_mw']
    SPECIAL[s['id']] = dict(verified=mean5, lo=None, hi=None, file=P17 + '/verify/T7b_verify_out/T7b_verify.json',
                            loc='stage_%s.mine_per_repeat_mw (mean of five); interval tsv_comparison T7b_%s_meanof5 mine_mw [%s, %s]' % (k, k, ci[1], ci[2]),
                            note='the row carries no interval; T7b verifier interval [%s, %s]' % (ci[1], ci[2]), ci_extra=(ci[1], ci[2]))

# M7 rep5 probe (AUC of the averaged five-repeat OOF probabilities) = T4 "averaged-prob" Pitt encoder cell
la = [l for l in t4 if 'Q25O pitt encoder averaged-prob (not used)' in l][0]
lca = [l for l in t4 if l.startswith('[OK ] CI T4_pitt_o25_enc_avgprob_NOTUSED')][0]
owa = re.search(r'own: \[([\d.]+), ([\d.]+)\]', lca)
SPECIAL['M7_pitt_probe_rep5'] = dict(verified=float(re.search(r'MW ([\d.]+)', la).group(1)), lo=float(owa.group(1)), hi=float(owa.group(2)),
                                     file=P17 + '/verify/T4_verify.log', loc='line "%s" and CI line for T4_pitt_o25_enc_avgprob_NOTUSED' % la[:70], note='same cell as T4_pitt_o25_enc_avgprob_NOTUSED')
# POD4 4a acceptance rows (text value FAILED_x / PASSED_x): the verifier's five-feature AUC after normalisation
p4 = {}
for r in csv.DictReader(open(P16 + '/verify/POD4_VERIFIER_SUMMARY.tsv', newline=''), delimiter='\t'):
    p4[r['id']] = r
for sid, key in (('POD4_4a_nv_accept', 'nv_5f_after'), ('POD4_4a_pg_accept', 'pg_5f_after')):
    x = float(p4[key]['VERIFIER_value'])
    SPECIAL[sid] = dict(verified=('PASSED_' if x < 0.60 else 'FAILED_') + ('%.4f' % x), lo=None, hi=None, file=P16 + '/verify/POD4_VERIFIER_SUMMARY.tsv',
                        loc='id %s VERIFIER_value %s (acceptance rule: AUC below 0.60 passes)' % (key, p4[key]['VERIFIER_value']), note='')

# p16twins task (Linux x86 pods, sklearn 1.9.1, the pods' library versions): raw pulled outputs, read when present
P16P = P16TWINS + '/pull'
dirf = glob.glob(P16P + '/*/out/twin_direction.json')
if dirf:
    dj = json.load(open(dirf[0])); cells = {c['arm']: c for c in dj['cells']}
    env = 'sklearn %s, numpy %s, python %s' % (dj.get('sklearn'), dj.get('numpy'), dj.get('python'))
    for arm in ('all', 'conflict', 'agreement'):
        c = cells.get(arm)
        if not c:
            continue
        SPECIAL['3b_%s_cos' % arm] = dict(verified=c['cosine'], lo=None, hi=None, file=dirf[0], loc='cells[arm=%s].cosine (%s)' % (arm, env), note='p16twins pod twin (lm_head read on the pod)')
        if arm != 'all':
            SPECIAL['3b_%s_auc_probe' % arm] = dict(verified=c['auc_probe'], lo=c['auc_probe_ci'][0], hi=c['auc_probe_ci'][1], file=dirf[0], loc='cells[arm=%s].auc_probe (%s)' % (arm, env),
                                                   note='p16twins pod refit agrees; the Mac refit (verify/POD3_partF_out.json, sklearn 1.7.2) gives %.4f, a Mac library effect' % pf['3b_%s_auc_probe' % arm]['mine'])
            SPECIAL['3b_%s_auc_wo_d' % arm] = dict(verified=c['auc_wo_d_trainz'], lo=c['auc_wo_d_trainz_ci'][0], hi=c['auc_wo_d_trainz_ci'][1], file=dirf[0],
                                                  loc='cells[arm=%s].auc_wo_d_trainz (%s)' % (arm, env), note='p16twins pod twin')
    if 'all' in cells:
        SPECIAL['SEED14_3b'] = dict(verified=cells['all']['cosine'], lo=None, hi=None, file=dirf[0], loc='cells[arm=all].cosine (%s)' % env, note='p16twins pod twin')
for f in glob.glob(P16P + '/*/out/twin_*_*.json'):
    m = re.match(r'twin_(pitt|pcgita)_(enc|llm|ans)\.json$', os.path.basename(f))
    if not m:
        continue
    j = json.load(open(f)); ds, st = m.groups()
    sid = '3a_%s_%s_best_stage' % (ds, st)
    frag = [x for x in summ if x['id'] == sid]
    layer_ok = bool(frag) and re.search(r'layer %d\b' % j['best_stage'], frag[0]['what']) is not None
    SPECIAL[sid] = dict(verified=j['best_stage_auc_mean'], lo=j['best_ci'][0], hi=j['best_ci'][1], file=f,
                        loc='best_stage %d, best_stage_auc_mean, best_ci (sklearn %s)' % (j['best_stage'], j.get('sklearn')),
                        note='p16twins pod twin; best layer %d %s the layer named in the row' % (j['best_stage'], 'matches' if layer_ok else 'DIFFERS FROM'))

NO_TWIN = {}
# M11 aggregates: only the per-clip inputs have a verifier twin (verify/M11_verify_perclip_mine.json); no saved aggregate twin
for s in summ:
    if s['task/pod'] == 'M11':
        NO_TWIN[s['id']] = 'aggregate over the 37 KCL clips; the per-clip inputs match verify/M11_verify_perclip_mine.json, the aggregate itself has no saved twin'
# M11 KCL AUC rows: same quantity and byte-identical input files as M10 (podE_final/out/o25_kcl_text.csv == overnight2/text_new/o25_kcl_text.csv, md5 8d3a4aa9...)
m10 = json.load(open(P16 + '/verify/M10_verify.json'))
for s_ in summ:
    if s_['task/pod'] == 'M11' and s_['what'] in ('auc_text_condition_recomputed', 'auc_audio_condition_recomputed'):
        k = 'transcript' if 'text' in s_['what'] else 'audio'
        NO_TWIN.pop(s_['id'], None)
        SPECIAL[s_['id']] = dict(verified=m10['point'][k], lo=None, hi=None, file=P16 + '/verify/M10_verify.json', loc='point.%s' % k,
                                 note='M10 verifier, same AUC on the same 37 clips; the M11 text file and the M10 text file are byte-identical (md5 8d3a4aa95328743a55ac11966edd2670)')
# T2: only rows keyed to the verifier id count; curve and _UNVERIFIED seed rows have no twin
WEAK = {}
for s in summ:
    if s['task/pod'] == 'MEDAIC' and (s['id'].endswith('_NOCI') or s['id'].endswith('_POINT')):
        WEAK[s['id']] = 'twin is arithmetic on the original-run json per_repeat list only (no per-clip file exists for these); not an independent AUC recompute'

# check_summary: the summary checker's own recompute (value_status MATCH (recomputed))
chk_rows = {}
for c in csv.DictReader(open(CHECK, newline='')):
    if c.get('part') in ('16', '17') and (c.get('value_status') or '').startswith('MATCH') and (c.get('my_value') or '') != '':
        chk_rows[(c['part'], c['task'], c['id'])] = c

# p16twins output (read at the end if present)
p16t = {}
if os.path.isdir(P16TWINS):
    for f in [x for x in glob.glob(P16TWINS + '/**/*.csv', recursive=True) + glob.glob(P16TWINS + '/**/*.tsv', recursive=True) if not x.endswith('twins_p16.tsv')]:
        try:
            rd = csv.DictReader(open(f, newline=''), delimiter='\t' if f.endswith('.tsv') else ',')
            for r in rd:
                k = r.get('uid') or r.get('id')
                if not k:
                    continue
                vv = r.get('value_verified') or r.get('twin_value') or r.get('verified') or r.get('recomputed')
                if vv not in (None, ''):
                    p16t.setdefault(k, (vv, r.get('lo_verified') or r.get('twin_lo') or r.get('lo', ''), r.get('hi_verified') or r.get('twin_hi') or r.get('hi', ''), f, r))
        except Exception as e:
            print('p16twins read fail', f, e)

# ------------------------------------------------------------------ decide
out = []
review = []
for s, a in zip(summ, auto):
    uid, t = s['id'], s['task/pod']
    v, lo, hi = dec(s['value_computed']), dec(s['lo']), dec(s['hi'])
    has_ci = s['lo'] != '' and s['hi'] != ''
    o = dict(part=s['part'], task=t, id=uid, value=s['value_computed'], lo=s['lo'], hi=s['hi'], role=s['paper_role'],
             value_verified='', lo_verified='', hi_verified='', status='', ci_status='', twin_file='', twin_where='', twin_source='', decision_note='')

    def put(x, xlo, xhi, f, where, note='', src=None):
        o['value_verified'] = fmt(x, s['value_computed'])
        o['twin_file'] = rel(f); o['twin_where'] = where; o['twin_source'] = src or srctype(f); o['decision_note'] = note
        if isinstance(x, str):
            ok = x == s['value_computed']
        else:
            ok = eq(float(x), v) if v is not None else False
        o['status'] = 'yes' if ok else 'NO'
        if has_ci:
            if xlo is None or xhi is None:
                o['ci_status'] = 'interval not in twin'
            else:
                o['lo_verified'] = fmt(xlo, s['lo']); o['hi_verified'] = fmt(xhi, s['hi'])
                o['ci_status'] = 'same' if (eq(float(xlo), lo) and eq(float(xhi), hi)) else 'differs'

    if uid in NO_TWIN:
        o['status'] = 'no twin'; o['decision_note'] = NO_TWIN[uid]
        out.append(o); continue
    if uid in SPECIAL:
        d = SPECIAL[uid]
        put(d['verified'], d['lo'], d['hi'], d['file'], d['loc'], d['note'])
        if d.get('ci_extra'):
            o['lo_verified'], o['hi_verified'] = str(d['ci_extra'][0]), str(d['ci_extra'][1]); o['ci_status'] = 'row has no interval; twin interval given'
        out.append(o); continue

    best = a['twin_value'] != ''
    keyed = a['keyed'] == 'True'
    ov = int(a['ctx_overlap'] or 0) if best else 0
    ci_yes = a['ci_twinned'] == 'yes'
    accepted = False
    if best:
        x = float(a['twin_value'])
        where = a['twin_loc'] + ((' (interval: ' + a['ci_how'] + ')') if a['ci_how'] else '')
        if has_ci and ci_yes:
            put(x, float(lo), float(hi), a['twin_file'], where); accepted = True
        elif has_ci and keyed:
            put(x, None, None, a['twin_file'], where, 'value keyed to the row id; interval not found in the twin record'); accepted = True
            review.append((uid, 'keyed value, interval not twinned', a['twin_file'], a['twin_text'][:200]))
        elif has_ci and ov >= 4:
            put(x, None, None, a['twin_file'], where, 'value matched in a record that names the same quantity; interval not found there'); accepted = True
            review.append((uid, 'value only (ov %d)' % ov, a['twin_file'], a['twin_text'][:200]))
        elif not has_ci and t == 'T2' and not keyed:
            pass
        elif not has_ci and (keyed or ov >= 2):
            put(x, None, None, a['twin_file'], where); accepted = True
            if not keyed:
                review.append((uid, 'no-CI value (ov %d)' % ov, a['twin_file'], a['twin_text'][:200]))
    if not accepted and a['dis_value']:
        x = float(a['dis_value'])
        put(x, None, None, a['dis_file'], a['dis_loc'] + ' (' + a['dis_pair'] + ')', 'verifier recomputed a different value')
        review.append((uid, 'DISAGREE', a['dis_file'], a['dis_text'][:200])); accepted = True
    if not accepted:
        c = chk_rows.get((s['part'], t, uid))
        if c:
            put(float(c['my_value']), float(c['my_lo']) if c['my_lo'] else None, float(c['my_hi']) if c['my_hi'] else None,
                CHECK, 'check_summary.csv row %s (%s)' % (c['row'], c['method'][:60]), 'summary checker recompute; no verifier twin found', 'summary checker recompute')
            accepted = True
    if not accepted and uid in p16t:
        vv, l_, h_, f, r = p16t[uid]
        xv = float(vv) if dec(vv) is not None else vv
        put(xv, float(l_) if dec(l_) is not None and has_ci else None, float(h_) if dec(h_) is not None and has_ci else None, f, 'p16twins', '', 'p16twins task')
        accepted = True
    if not accepted:
        o['status'] = 'no twin'
        if best:
            o['decision_note'] = 'rejected coincidental value match in %s (%s)' % (rel(a['twin_file']), a['twin_loc'])
            review.append((uid, 'REJECTED match (ov %d, keyed %s)' % (ov, keyed), a['twin_file'], a['twin_text'][:200]))
    if uid == 'P16.POD2.L2':
        o['decision_note'] = (o['decision_note'] + '; ' if o['decision_note'] else '') + 'verifier stores lo as 0.43485 (rounded from 0.43484998), so the twin lo is 0.4348; a rounding tie, not a disagreement'
    if uid in WEAK and o['status'] == 'yes':
        o['status'] = 'yes (json arithmetic only)'; o['decision_note'] = WEAK[uid]
    out.append(o)

# ------------------------------------------------------------------ p16twins final table (read at the end)
TW = P16TWINS + '/twins_p16.tsv'
if os.path.exists(TW):
    tw = {}
    for r in csv.DictReader(open(TW, newline=''), delimiter='\t'):
        tw[r['id']] = r
    def twrow(uid):
        if uid in tw:
            return tw[uid]
        for k, r in tw.items():
            if uid.startswith('M7fix#') and k.startswith(uid.split(':')[0] + ' '):
                return r
        return None
    for o in out:
        r = twrow(o['id'])
        if not r:
            continue
        kind = r['kind']
        prev = '%s from %s (%s)' % (o['value_verified'] or 'no twin', o['twin_file'] or '-', o['status']) if o['status'] else ''
        ci = re.findall(r'[-\d.]+', r['recomputed_ci'])
        use = False; weak = False
        if kind == 'recompute' and (o['status'] in ('no twin', 'NO') or o['ci_status'] == 'differs' or o['twin_source'] == 'summary checker recompute'):
            use = True
        elif kind.startswith('log twin') and o['status'] in ('no twin', 'yes (json arithmetic only)'):
            use = True; weak = True
        elif o['status'] == 'no twin':
            use = True
        if not use:
            continue
        s_ = [x for x in summ if x['id'] == o['id']][0]
        vv = r['recomputed_full'] if r['recomputed_full'] not in ('', None) else r['recomputed']
        try:
            x = float(vv)
        except ValueError:
            x = vv
        o['value_verified'] = fmt(x, s_['value_computed'])
        o['lo_verified'] = ci[0] if len(ci) == 2 else ''
        o['hi_verified'] = ci[1] if len(ci) == 2 else ''
        has_ci = s_['lo'] != '' and s_['hi'] != ''
        if has_ci:
            o['ci_status'] = ('same' if (len(ci) == 2 and eq(float(ci[0]), dec(s_['lo'])) and eq(float(ci[1]), dec(s_['hi']))) else ('differs' if len(ci) == 2 else 'interval not in twin'))
        okv = (eq(float(x), dec(s_['value_computed'])) if isinstance(x, float) and dec(s_['value_computed']) is not None else str(vv) == s_['value_computed'])
        o['twin_file'] = rel(TW); o['twin_where'] = 'id %s (kind %s; file %s)' % (r['id'], kind, rel(r['file'])[:160]); o['twin_source'] = 'p16twins task'
        note = 'p16twins: ' + r['how'][:300]
        if prev and 'no twin' not in prev:
            note += ' | previous twin: ' + prev
        if weak:
            o['status'] = 'yes (second record only, not a recompute)' if okv else 'NO'
            if WEAK.get(o['id']):
                note += ' | also: ' + WEAK[o['id']]
        else:
            o['status'] = 'yes' if okv else 'NO'
        o['decision_note'] = note

# ------------------------------------------------------------------ m7m11 task table: M11 group-level values with its own verifier columns
M7M11 = 'scores/leftovers_23sep/m7m11/m7m11_values.tsv'
if os.path.exists(M7M11):
    mm = {r['id'][len('M11fix_'):]: r for r in csv.DictReader(open(M7M11, newline=''), delimiter='\t') if r['id'].startswith('M11fix_')}
    for o in out:
        if o['task'] != 'M11' or o['status'] != 'no twin':
            continue
        s_ = [x for x in summ if x['id'] == o['id']][0]
        r = mm.get(s_['what'])
        if not r or r['verifier_value'] in ('', None):
            continue
        x = float(r['verifier_value'])
        o['value_verified'] = fmt(x, s_['value_computed'])
        o['lo_verified'], o['hi_verified'] = r['verifier_lo'], r['verifier_hi']
        if s_['lo'] and s_['hi']:
            o['ci_status'] = 'same' if (eq(float(r['verifier_lo']), dec(s_['lo'])) and eq(float(r['verifier_hi']), dec(s_['hi']))) else 'differs'
        o['status'] = 'yes' if eq(x, dec(s_['value_computed'])) else 'NO'
        o['twin_file'] = rel(M7M11); o['twin_where'] = 'id %s, verifier_value / verifier_lo / verifier_hi (bootstrap rule: %s)' % (r['id'], r['kind'])
        o['twin_source'] = 'm7m11 task'
        o['decision_note'] = 'm7m11 task verifier column' + ('; the fragment interval used the all-37 speaker resampling of M11_robust.py, the rule (resample the cell speakers) gives the interval shown' if o['ci_status'] == 'differs' else '')

with open(os.path.join(OUTDIR, 'twin_decisions.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
with open(os.path.join(OUTDIR, 'twin_review.tsv'), 'w') as f:
    for r in review:
        f.write('\t'.join(str(x) for x in r) + '\n')
from collections import Counter
print(Counter(o['status'] for o in out))
print(Counter((o['status'], o['ci_status']) for o in out))
print(Counter(o['twin_source'] for o in out))
print('p16twins entries read', len(p16t))
