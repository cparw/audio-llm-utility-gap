#!/usr/bin/env python3
"""Build summary_all_v2.csv from summary_all.csv (v1 is never overwritten).

Changes against v1:
  1. PART 16 and PART 17 fragment rows: value_verified now holds the value from each row's real twin file
     (an independent verifier output), not a copy of value_computed (build_summary.py line 33). Where no
     twin exists the cell is empty and agree_4dp says "no twin". The old agree_4dp text is kept in agree_4dp_v1.
  2. The 0.7001 (LM) and 0.5884 (encoder) mean-of-five rows without an interval get the T7b interval
     (recomputed in my_verifier_out.json), with a note.
  3. The 'what' text of the rows whose fragment note was fixed today (T7.tsv, T4.tsv, P22 S1) is refreshed
     from the fixed fragment files.
New columns at the end: agree_4dp_v1, lo_verified, hi_verified, ci_check, twin_file, twin_where, twin_source, twin_note.
"""
import csv, json, os, sys, datetime
from collections import Counter
OUTDIR = sys.argv[1]
SUM = '<local data dir>/release_from_mac/summary/summary_all.csv'
R = '<local data dir>/release'
V = json.load(open(os.path.join(OUTDIR, 'my_verifier_out.json')))
dec = list(csv.DictReader(open(os.path.join(OUTDIR, 'twin_decisions.csv'), newline='')))
rows = list(csv.DictReader(open(SUM, newline='')))
cols = list(rows[0].keys())
new_cols = cols + ['agree_4dp_v1', 'lo_verified', 'hi_verified', 'ci_check', 'twin_file', 'twin_where', 'twin_source', 'twin_note']

# refreshed notes from the fixed fragment files
fixed_what = {}
for p, part in ((R + '/edaic_rerun/part17/rows/T7.tsv', '17'), (R + '/edaic_rerun/part17/rows/T4.tsv', '17'), (R + '/scores/part22/rows/P22_verified.tsv', '22')):
    for r in csv.DictReader(open(p, newline=''), delimiter='\t'):
        fixed_what[(part, r['id'])] = r['what']

di = iter(dec)
out = []
cnt = Counter()
for r in rows:
    o = dict(r)
    o['agree_4dp_v1'] = r['agree_4dp']
    for c in new_cols[len(cols) + 1:]:
        o[c] = ''
    if r['part'] in ('16', '17'):
        d = next(di)
        assert d['id'] == r['id'] and d['task'] == r['task/pod']
        o['value_verified'] = d['value_verified']
        o['lo_verified'], o['hi_verified'] = d['lo_verified'], d['hi_verified']
        o['ci_check'] = d['ci_status']
        o['twin_file'], o['twin_where'], o['twin_source'], o['twin_note'] = d['twin_file'], d['twin_where'], d['twin_source'], d['decision_note']
        st = d['status']
        if st == 'yes':
            o['agree_4dp'] = 'yes (twin file)' + ('' if d['ci_status'] in ('', 'same', 'row has no interval; twin interval given') else '; interval: ' + d['ci_status'])
        elif st.startswith('yes ('):
            o['agree_4dp'] = st
        elif st == 'NO':
            o['agree_4dp'] = 'NO (twin gives %s)' % d['value_verified']
        else:
            o['agree_4dp'] = 'no twin'
        cat = 'yes' if o['agree_4dp'] == 'yes (twin file)' else ('yes, interval differs' if o['agree_4dp'].startswith('yes (twin file); interval') else
               ('yes, json arithmetic only' if 'json arithmetic' in o['agree_4dp'] else ('yes, second record only' if 'second record' in o['agree_4dp'] else o['agree_4dp'].split(' (')[0])))
        o['_cat'] = cat
        cnt[(r['part'], cat)] += 1
    else:
        o['twin_source'] = 'fragment value_verified column (PART %s verifier), unchanged from v1' % r['part']
    k = (r['part'], r['id'])
    if k in fixed_what and fixed_what[k] != r['what']:
        o['what'] = fixed_what[k]
        o['note'] = (r['note'] + ' | ' if r['note'] else '') + 'note text refreshed 23 Sep from the fixed fragment file'
    # T7b intervals on the 0.7001 / 0.5884 rows that had none
    if r['part'] == '17' and r['lo'] == '' and r['value_computed'].strip() in ('0.7001', '0.5884') and r['id'] in (
            'T4_edaic_o25_llm_mean5_jsononly', 'T4_edaic_o25_enc_mean5_jsononly', 'T7_published_llm_meanof5', 'T7_published_enc_meanof5'):
        kk = 'T7b_llm_meanof5' if r['value_computed'].strip() == '0.7001' else 'T7b_enc_meanof5'
        o['lo'], o['hi'] = '%.4f' % V[kk]['lo'], '%.4f' % V[kk]['hi']
        o['note'] = (o['note'] + ' | ' if o['note'] else '') + ('interval added 23 Sep from T7b (T7b.tsv %s; Linux sklearn 1.9.1 rerun; recomputed in leftovers_23sep/tables/my_verifier_out.json %s)'
                                                               % (kk, kk))
        cnt['t7b_interval_added'] += 1
    out.append(o)
assert next(di, None) is None
path = os.path.join(OUTDIR, 'summary_all_v2.csv')
with open(path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=new_cols, extrasaction='ignore'); w.writeheader(); w.writerows(out)
C = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), source_v1=SUM, rows=len(out),
         agree_4dp_by_part={'%s %s' % k: v for k, v in sorted((k, v) for k, v in cnt.items() if isinstance(k, tuple))}, t7b_intervals_added=cnt['t7b_interval_added'],
         twin_source_counts=dict(Counter(o['twin_source'] for o in out if o['part'] in ('16', '17'))),
         paper_value_rows_16_17_by_status=dict(Counter(o['_cat'] for o in out if o['part'] in ('16', '17') and o['paper_role'] == 'PAPER VALUE')),
         paper_value_rows_without_twin=[o['id'] for o in out if o['part'] in ('16', '17') and o['paper_role'] == 'PAPER VALUE' and o['_cat'] == 'no twin'])
json.dump(C, open(os.path.join(OUTDIR, 'summary_counts_v2.json'), 'w'), indent=1)
print(json.dumps(C, indent=1))
