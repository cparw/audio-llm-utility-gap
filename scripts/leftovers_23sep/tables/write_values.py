#!/usr/bin/env python3
"""Write values.csv (one row per value this task wrote, with the independent recompute beside it),
changes.csv (every change, all files) and values.sidecar.json."""
import csv, json, os, sys, hashlib, datetime, platform, glob
OUT = sys.argv[1]
G = '<local data dir>/release_from_mac'
R = '<local data dir>/release'
ML = R + '/master/master_lookup.csv'
V = json.load(open(os.path.join(OUT, 'my_verifier_out.json')))
S1 = json.load(open(os.path.join(OUT, 'my_verifier_p22s1_out.json')))
C2 = json.load(open(os.path.join(OUT, 'summary_counts_v2.json')))
P23V = {r['id']: r for r in csv.DictReader(open(G + '/scores/part23/verify/PART23_VERIFIED.tsv', newline=''), delimiter='\t')}
T7B = {r['id']: r for r in csv.DictReader(open(R + '/edaic_rerun/part17/T7b/T7b.tsv', newline=''), delimiter='\t')}
ml = {(r['model'], r['dataset'], r['stream']): r for r in csv.DictReader(open(ML, newline=''))}
rows = []


def add(id_, value, lo, hi, file, what, closes, rv, rlo, rhi, ref, ref_src):
    def eq(a, b):
        if a in ('', None) or b in ('', None):
            return a in ('', None) and b in ('', None)
        try:
            return abs(float(a) - float(b)) < 5e-5 + 1e-12
        except ValueError:
            return str(a) == str(b)
    agree = eq(value, rv) and eq(lo, rlo) and eq(hi, rhi)
    ref_ok = '' if ref is None else ('yes' if all(eq(x, y) for x, y in zip((value, lo, hi), ref)) else 'NO')
    rows.append(dict(id=id_, value=value, lo=lo, hi=hi, file=file, what=what, closes=closes, recomputed=rv, recomputed_lo=rlo, recomputed_hi=rhi,
                     agree_4dp='yes' if agree else 'NO', reference_value=('' if ref is None else ' '.join(str(x) for x in ref)), reference_source=ref_src, reference_agree=ref_ok))


f4 = lambda x: '%.4f' % x
# ---- (a) T7b intervals added to summary_all_v2 rows
for sid, k in (('T4_edaic_o25_llm_mean5_jsononly', 'llm'), ('T7_published_llm_meanof5', 'llm'), ('T4_edaic_o25_enc_mean5_jsononly', 'enc'), ('T7_published_enc_meanof5', 'enc')):
    d = V['T7b_%s_meanof5' % k]; t = T7B['T7b_%s_meanof5' % k]
    add('summary_v2:' + sid, f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), G + '/summary/summary_all_v2.csv', 'E-DAIC 300 s %s probe, mean of five per-repeat AUCs; interval added' % k.upper(),
        'ledger 3 item 8 and 6 LEFT me 4 (T7b intervals on the 0.7001 / 0.5884 rows)', f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), (t['value'], t['lo'], t['hi']), 'part17/T7b/T7b.tsv')
# ---- (b) PART 23 rows appended to master_lookup
P23MAP = [(('Qwen2.5-Omni', 'edaic_full_notrunc', 'zero shot answer (PART 23 rerun)'), 'P23_zs_whole', '1_whole'),
          (('Qwen2.5-Omni', 'edaic_full', 'zero shot answer (rebuild run, 300 s cut)'), 'P23_zs_300s', '1_300s'),
          (('Qwen2.5-Omni', 'edaic_full_notrunc', 'encoder probe (mean of 5 per-repeat AUCs, PART 23)'), 'P23_enc_meanof5_whole', '2_encoder_meanof5'),
          (('Qwen2.5-Omni', 'edaic_full_notrunc', 'projector probe (mean of 5 per-repeat AUCs, PART 23)'), 'P23_proj_meanof5_whole', '3_projector_meanof5'),
          (('Qwen2.5-Omni', 'edaic_full_notrunc', 'LM probe (mean of 5 per-repeat AUCs, PART 23)'), 'P23_llm_meanof5_whole', '4_LM_meanof5'),
          (('Qwen2.5-Omni', 'edaic_full_notrunc', 'answer-state probe (mean of 5 per-repeat AUCs, PART 23)'), 'P23_ans_meanof5_whole', '5_answer state_meanof5'),
          (('Qwen2.5-Omni', 'edaic_full_notrunc', 'projector fine tune (PART 23)'), 'P23_ft_whole_pooled', '6_ft_whole_pooled'),
          (('Qwen2.5-Omni', 'edaic_full', 'projector fine tune (PART 23 same-pod 300 s control)'), 'P23_ft_ctrl300_pooled', '7_ft_ctrl300_pooled')]
for key, vk, pk in P23MAP:
    m = ml[key]; d = V[vk]; p = P23V[pk]
    import re
    ci = re.search(r'95% CI \[([-\d.]+), ([-\d.]+)\]', m['estimator'])
    add('master_lookup:%s|%s|%s' % key, m['auc'], ci.group(1) if ci else '', ci.group(2) if ci else '', ML, '%s %s %s (auc column; interval in the estimator note)' % key,
        'audit_p2223 item 9 and audit_p17 item 4 left (PART 23 not in master_lookup)', f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), (p['recomputed'], p['lo'], p['hi']), 'part23/verify/PART23_VERIFIED.tsv ' + pk)
for vk, pk, what in (('P23_zs_whole_minus_300s', '1_whole_minus_300s', 'whole minus 300 s zero-shot, paired'), ('P23_ft_whole_minus_zs_whole', '6_ft_whole_minus_zswhole', 'FT whole minus zero-shot whole, paired'),
                     ('P23_ft_whole_minus_ctrl300', '7_ft_whole_minus_ctrl300', 'FT whole minus FT ctrl300, paired'), ('P23_ft_ctrl300_minus_zs_300s', '9a_ft_ctrl300_minus_zs300', 'FT ctrl300 minus zero-shot 300 s, paired')) + tuple(
        ('P23_%s_meanof5_whole_minus_zs_whole' % k, pk2, '%s mean of five minus zero-shot whole, paired' % k) for k, pk2 in (('enc', '2_encoder_minus_zswhole'), ('proj', '3_projector_minus_zswhole'), ('llm', '4_LM_minus_zswhole'), ('ans', '5_answer state_minus_zswhole'))):
    d = V[vk]; p = P23V[pk]
    add('master_lookup_note:' + vk, f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), ML, what + ' (written into the estimator note of the PART 23 rows)',
        'audit_p2223 item 9 (PART 23 paired values beside the rows)', f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), (p['recomputed'], p['lo'], p['hi']), 'part23/verify/PART23_VERIFIED.tsv ' + pk)
t = V['P23_trunc']
add('master_lookup_note:q3o_cut', str(t['q3o_cut']), '', '', ML, 'Qwen3-Omni E-DAIC windows cut (of 275), written into the Qwen3-Omni edaic_full note', 'ledger 6 LEFT me 12 ("capped at 900 s" edaic_full rows)',
    str(t['q3o_cut']), '', '', (P23V['10_q3o_cut_count']['recomputed'], '', ''), 'part23/verify/PART23_VERIFIED.tsv 10_q3o_cut_count')
add('master_lookup_note:af3_cut', str(t['af3_cut']), '', '', ML, 'Audio Flamingo 3 E-DAIC windows cut at 600 s (of 275), written into the AF3 edaic_full note', 'ledger 6 LEFT me 12 ("capped at 900 s" edaic_full rows)',
    str(t['af3_cut']), '', '', (P23V['10_af3_cut_count']['recomputed'], '', ''), 'part23/verify/PART23_VERIFIED.tsv 10_af3_cut_count')
add('master_lookup_note:o25_300s_cut_count', '217', '', '', ML, 'Qwen2.5-Omni E-DAIC windows over 300 s that the processor cut (of 275), written into rows 200, 202, 204',
    'ledger 6 LEFT me 12 (master_lookup row 200 and the capped rows)', str(S1['gt300_at_7500']), '', '', None, '')
add('master_lookup_note:ft_window_s', '30', '', '', ML, 'seconds of audio the 0.6421 fine-tune recipe keeps (x[:16000 * 30], release/scripts/sft_projector.py line 59), written into row 205',
    'ledger 6 LEFT me 12 (master_lookup row 205); ledger 3 item 5 in part', '30', '', '', None, 'release/scripts/sft_projector.py line 59; variants/sft_full_sft.json window_seconds')
# ---- (c) notes: T7 / T4 intervals, P22 S1, sftfull n
for k in ('llm', 'enc', 'ans', 'proj'):
    for kind in ('meanof5', 'meanof5_minus_answer'):
        d = V['T7b_%s_%s' % (k, kind)]; tt = T7B['T7b_%s_%s' % (k, kind)]
        add('T7.tsv_note:T7_published_%s_meanof5:%s' % (k, kind), f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), R + '/edaic_rerun/part17/rows/T7.tsv', 'interval quoted in the fixed note',
            'ledger 6 LEFT me 12 (T7.tsv "NO INTERVAL")', f4(d['value_4dp']), f4(d['lo']), f4(d['hi']), (tt['value'], tt['lo'], tt['hi']), 'part17/T7b/T7b.tsv')
for sid, val, rv in (('P22_S1_tf_version', '5.17.0', S1['tf_version']), ('P22_S1_chunk_length', '300', str(S1['chunk_length'])), ('P22_S1_n_samples', '4800000', str(S1['n_samples'])),
                     ('P22_S1_trunc_default', 'True', 'True' if 'truncation: bool = True' in S1['trunc_default_line196'] else 'False'), ('P22_S1_tok_300s', '7500', str(S1['tok_300s'])),
                     ('P22_S1_tok_900s_off', '22500', str(S1['tok_900s_off'])), ('P22_S1_formula_match', '275', str(S1['formula_match'])), ('P22_S1_gt300_at_7500', '217', str(S1['gt300_at_7500'])),
                     ('P22_S1_capped900', '89', str(S1['capped900']))):
    add('P22_verified.tsv:' + sid, val, '', '', R + '/scores/part22/rows/P22_verified.tsv (and P22.tsv)', 'value unchanged; note "not yet verified" replaced', 'ledger 6 LEFT me 12 (9 P22_S1 texts); audit_p2223 problem 7',
        rv, '', '', None, '')
add('p14_o25_sftfull.json:n', str(V['sftfull_csv']['rows']), '', '', R + '/edaic_rerun/part16/POD1/p14_o25_sftfull.json (and out16/ copy)', 'key n added: rows of p14_o25_sftfull.csv',
    'ledger 6 LEFT me 12 (sftfull sidecar n); audit_p16 item 1 left', str(V['sftfull_csv']['rows']), '', '', ('966', '', ''), 'part16/verify/POD1_verify_out.json "1c sft_full RETRAIN n rows" 966')
# ---- (a) summary counts
a = C2['agree_4dp_by_part']
for k, v in a.items():
    add('summary_v2_count:' + k.replace(' ', '_'), str(v), '', '', G + '/summary/summary_all_v2.csv', 'PART %s fragment rows with agree_4dp "%s"' % tuple(k.split(' ', 1)),
        'ledger 1 spot check 23 and 6 LEFT me 4 (verified column was a copy)', str(v), '', '', None, 'summary_counts_v2.json')

with open(os.path.join(OUT, 'values.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- changes.csv (all files)
ch = []
for fn in ('changes_master_lookup.csv', 'changes_notes.csv'):
    ch += list(csv.DictReader(open(os.path.join(OUT, fn), newline='')))
ch.append(dict(file=G + '/summary/summary_all_v2.csv', row='(new file, 1532 rows, 25 columns)', field='new file', old='', new='v1 kept unchanged (md5 %s)' % hashlib.md5(open(G + '/summary/summary_all.csv', 'rb').read()).hexdigest(),
               why='value_verified filled from real twin files for the 870 PART 16/17 rows; agree_4dp recomputed; T7b intervals on 4 rows; 16 note texts refreshed; 8 new columns', closes='ledger 1 spot check 23; ledger 6 LEFT me 4; ledger 3 item 8'))
ch.append(dict(file=G + '/summary/summary_counts_v2.json', row='(new file)', field='new file', old='', new='counts for v2', why='v2 tallies', closes='ledger 6 LEFT me 4'))
for r in csv.DictReader(open(os.path.join(OUT, 'saved_scripts_manifest.tsv'), newline=''), delimiter='\t'):
    ch.append(dict(file=G + '/scores/leftovers_23sep/tables/' + r['saved_path'], row='(new file)', field='saved copy', old=r['source_path'], new='sha256 ' + r['sha256'],
                   why=r['why'], closes='ledger 6 LEFT me 11 (copy the <local data dir>/scratch verifier scripts cited by the M9, MEDAIC and M2 sidecars); audit_p16 problem (sidecars cite <local data dir>/scratch scripts)'))
with open(os.path.join(OUT, 'changes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['file', 'row', 'field', 'old', 'new', 'why', 'closes']); w.writeheader(); w.writerows(ch)
print('values', len(rows), 'NO:', sum(r['agree_4dp'] != 'yes' for r in rows), 'ref NO:', sum(r['reference_agree'] == 'NO' for r in rows), '| changes', len(ch))
for r in rows:
    if r['agree_4dp'] != 'yes' or r['reference_agree'] == 'NO':
        print('CHECK', r)
