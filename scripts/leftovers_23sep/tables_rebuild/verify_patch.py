#!/usr/bin/env python3
"""Independent check of summary_all_v2_patched.csv: values of the 3 changed cells against the
independent verifier outputs (verify/p16twins_VERIFIED.tsv, verify/m7m11_VERIFIED.tsv), which were
written by separate verifier jobs, and every other row equal to the old v2 row. No import of patch_v2.py."""
import csv, re, json, hashlib
B = '<local data dir>/leftovers_23sep/'
def rows(p, d=','):
    with open(p, newline='') as f: return list(csv.DictReader(f, delimiter=d))
def nums(s): return [float(x) for x in re.findall(r'[-+]?\d+\.\d+', s)]
old = rows(B + 'tables/summary_all_v2.csv'); new = rows(B + 'tables_rebuild/summary_all_v2_patched.csv')
same = sum(1 for x, y in zip(old, new) if x == y); diff = [y['id'] for x, y in zip(old, new) if x != y]
pv = {r['id']: r for r in rows(B + 'verify/p16twins_VERIFIED.tsv', '\t')}
mv = {r['id']: r for r in rows(B + 'verify/m7m11_VERIFIED.tsv', '\t')}
N = {r['id']: r for r in new if r['part'] == '16'}
checks = [('3a_pcgita_llm_best_stage', pv['3a_pcgita_llm_best_stage']['recomputed'], 'verify/p16twins_VERIFIED.tsv'),
          ('3a_pcgita_ans_best_stage', pv['3a_pcgita_ans_best_stage']['recomputed'], 'verify/p16twins_VERIFIED.tsv'),
          ('M11#36:B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control', mv['M11fix_B_xsys_disagree_ge20w_MEAN_diff_PD_minus_control']['recomputed'], 'verify/m7m11_VERIFIED.tsv')]
out = []
for rid, ref, src in checks:
    r = N[rid]; rv = nums(ref)
    mine = [float(r['value_verified']), float(r['lo_verified']), float(r['hi_verified'])]
    frag = [float(r['value_computed'])] + ([float(r['lo']), float(r['hi'])] if r['lo'] else [])
    ok = all(round(a, 4) == round(b, 4) for a, b in zip(mine, rv))
    ok_frag = round(frag[0], 4) == round(rv[0], 4)
    out.append(dict(id=rid, value=r['value_verified'], lo=r['lo_verified'], hi=r['hi_verified'], agree_4dp_col=r['agree_4dp'], ci_check=r['ci_check'],
                    independent_ref=ref, independent_src=src, fragment_value=r['value_computed'], match_4dp='yes' if ok and ok_frag else 'NO',
                    file=B + 'tables_rebuild/summary_all_v2_patched.csv'))
with open(B + 'tables_rebuild/summary_v2_patch_values.tsv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter='\t'); w.writeheader(); w.writerows(out)
res = dict(rows_total=len(new), rows_identical_to_old_v2=same, rows_changed=diff, checks=[(o['id'], o['match_4dp']) for o in out],
           no_NO_rows=sum(1 for r in new if r['part'] in ('16', '17') and r['agree_4dp'].startswith('NO')),
           sha256_patched=hashlib.sha256(open(B + 'tables_rebuild/summary_all_v2_patched.csv', 'rb').read()).hexdigest())
json.dump(res, open(B + 'tables_rebuild/summary_v2_patch_values.sidecar.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
for o in out: print(o['id'], o['value'], o['lo'], o['hi'], '|', o['independent_ref'], '|', o['match_4dp'], '|', o['agree_4dp_col'], o['ci_check'])
