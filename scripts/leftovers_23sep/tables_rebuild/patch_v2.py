#!/usr/bin/env python3
"""Targeted patch of summary_all_v2.csv (tables task, built 15:30 PDT) for the rows whose twin
files were rewritten after the build (twins_p16.tsv 15:33:54, m7m11_values.tsv 15:33:08).
Only the rows listed in KEEP take the full-rebuild values; every other row stays byte for byte.
A full rebuild is NOT used: it switches 12 other rows to p16twins/verify_twins.tsv, which drops
their intervals and relabels two log twins as recomputes."""
import csv, json, collections, hashlib, datetime
B = '<local data dir>/leftovers_23sep/'
OLD = B + 'tables/summary_all_v2.csv'
NEW = B + 'tables_rebuild/summary_all_v2_fullrebuild_NOT_USED.csv'
OUT = B + 'tables_rebuild/summary_all_v2_patched.csv'
KEEP = {'3a_pcgita_llm_best_stage', '3a_pcgita_ans_best_stage',
        'M11#04:B_xsys_disagree_diff_PD_minus_control', 'M11#10:C_mean_token_logprob_diff_PD_minus_control',
        'M11#16:C_compression_ratio_diff_PD_minus_control', 'M11#22:D_n_words_30s_diff_PD_minus_control',
        'M11#33:B_xsys_disagree_MEDIAN_diff_PD_minus_control', 'M11#36:B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control'}
a = list(csv.DictReader(open(OLD, newline='')))
b = list(csv.DictReader(open(NEW, newline='')))
assert [r['id'] for r in a] == [r['id'] for r in b]
cols = list(a[0].keys())
out, took = [], []
for x, y in zip(a, b):
    if x['part'] == '16' and x['id'] in KEEP:
        out.append(y); took.append(x['id'])
    else:
        out.append(x)
assert sorted(took) == sorted(KEEP), took
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(out)
cnt = collections.Counter((r['part'], r['agree_4dp'].split(';')[0].replace(' (twin file)', '').replace(' (second record only, not a recompute)', ', second record only').strip())
                          for r in out if r['part'] in ('16', '17'))
res = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
       'rows': len(out), 'rows_patched': sorted(took),
       'agree_4dp_by_part_first_clause': {'%s %s' % k: v for k, v in sorted(cnt.items())},
       'agree_4dp_full_text_counts': dict(collections.Counter((r['part'] + ' ' + r['agree_4dp']) for r in out if r['part'] in ('16', '17'))),
       'md5_old': hashlib.md5(open(OLD, 'rb').read()).hexdigest(), 'md5_out': hashlib.md5(open(OUT, 'rb').read()).hexdigest()}
print(json.dumps(res, indent=1))
json.dump(res, open(B + 'tables_rebuild/summary_all_v2_patched.counts.json', 'w'), indent=1)
