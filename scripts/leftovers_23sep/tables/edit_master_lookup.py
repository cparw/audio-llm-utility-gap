#!/usr/bin/env python3
"""(b) append the PART 23 rows to release/master/master_lookup.csv, (c) fix the stale E-DAIC window notes.

Backups: master_lookup.csv.bak_pre_part23 (untouched original, before any change of this task) and
master_lookup.csv.bak_pre_notefix_23sep (after the PART 23 append, before the note fixes).
Every number written here comes from my_verifier_out.json (own recompute) and agrees with
part23/verify/PART23_VERIFIED.tsv and part17 T7b. Rows are appended; no row is reordered.
Writes changes rows to changes_master_lookup.csv in OUTDIR.
"""
import csv, io, json, os, shutil, sys, hashlib

ML = '<local data dir>/release/master/master_lookup.csv'
OUTDIR = sys.argv[1]
V = json.load(open(os.path.join(OUTDIR, 'my_verifier_out.json')))
P23 = '<local data dir>/release_from_mac/scores/part23'
R = '<local data dir>/release'


def rd():
    t = open(ML, newline='').read()
    return list(csv.reader(io.StringIO(t)))


def wr(rows):
    with open(ML, 'w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def c(k):
    d = V[k]
    return d['value_4dp'], '[%.4f, %.4f]' % (d['lo'], d['hi'])


def pv(k):
    d = V[k]
    return '%+.4f [%.4f, %.4f]' % (d['value_4dp'], d['lo'], d['hi'])


changes = []
orig_md5 = md5(ML)
if os.path.exists(ML + '.bak_pre_part23'):
    sys.exit('bak_pre_part23 already exists; refusing to overwrite it')
shutil.copy2(ML, ML + '.bak_pre_part23')
changes.append(dict(file=ML, row='(whole file)', field='backup', old='', new=ML + '.bak_pre_part23 (md5 %s)' % orig_md5, why='copy before the PART 23 append', closes='task (b)'))

rows = rd()
hdr = rows[0]
assert hdr == ['model', 'dataset', 'stream', 'auc', 'n', 'estimator', 'source']
n0 = len(rows) - 1
keys = set((r[0], r[1], r[2]) for r in rows[1:])

# ------------------------------------------------------------------ (b) PART 23 rows
zw, zwci = c('P23_zs_whole'); z3, z3ci = c('P23_zs_300s')
ft = V['P23_ft_whole_pooled']; ftc = V['P23_ft_ctrl300_pooled']
folds_w = [ft['per_fold'][k] for k in sorted(ft['per_fold'])]
folds_c = [ftc['per_fold'][k] for k in sorted(ftc['per_fold'])]
f3 = V['P23_ft_whole_fold3']
WHOLE = ('whole E-DAIC window with truncation lifted (truncation=False; windows up to the 900 s file cap, the 89 capped windows get 22,500 audio tokens), '
         'PART 23 run on an H100 NVL pod')
new = []
new.append(['Qwen2.5-Omni', 'edaic_full_notrunc', 'zero shot answer (PART 23 rerun)', str(zw), '275',
            'zero shot, %s; 95%% CI %s; minus the 300 s cut (0.8278, same clips): %s, not significant; the PART 22 run of the same setup on H200 gave 0.8379 (row above), '
            'per-clip p_yes max diff 0.0617 between the two runs (GPU numerics, part23 A2d gate), the paper must pick one; PART 23, independently recomputed '
            '(part23/verify/PART23_VERIFIED.tsv 1_whole yes; leftovers_23sep/tables/my_verifier_out.json P23_zs_whole)' % (WHOLE, zwci, pv('P23_zs_whole_minus_300s')),
            P23 + '/A/A2_edaic_whole_zeroshot_perclip.csv'])
new.append(['Qwen2.5-Omni', 'edaic_full', 'zero shot answer (rebuild run, 300 s cut)', str(z3), '275',
            'zero shot, rebuild run: the first 300 s of each E-DAIC window (Qwen2_5OmniProcessor default cut, 217 of 275 windows cut); the 300 s reference used by T7b, PART 22 and PART 23; '
            '95%% CI %s; the original run on the same window is 0.8285 (row 200); independently recomputed (T7b verifier 17/17; PART23_VERIFIED.tsv 1_300s yes; '
            'leftovers_23sep/tables/my_verifier_out.json T7b_zeroshot_answer)' % z3ci,
            R + '/edaic_rerun/part17/T7b/T7b_edaic300_meanof5_perclip.csv'])
for k, stream in (('enc', 'encoder probe'), ('proj', 'projector probe'), ('llm', 'LM probe'), ('ans', 'answer-state probe')):
    d = V['P23_%s_meanof5_whole' % k]
    new.append(['Qwen2.5-Omni', 'edaic_full_notrunc', stream + ' (mean of 5 per-repeat AUCs, PART 23)', str(d['value_4dp']), '275',
                'mean of five per-repeat AUCs %s, nested GroupKFold(5) x 5 repeats on states extracted from the %s; 95%% CI [%.4f, %.4f]; paired minus the whole-window zero-shot %.4f: %s; '
                'AUC recomputed from the pod OOF scores (the probes were not refit off the pod); PART 23 (PART23_VERIFIED.tsv yes; my_verifier_out.json P23_%s_meanof5_whole)'
                % ('[' + ', '.join('%.4f' % x for x in d['per_repeat']) + ']', WHOLE, d['lo'], d['hi'], zw, pv('P23_%s_meanof5_whole_minus_zs_whole' % k), k),
                P23 + '/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv'])
new.append(['Qwen2.5-Omni', 'edaic_full_notrunc', 'projector fine tune (PART 23)', str(ft['value_4dp']), '275',
            'projector only fine tune, recipe p15/sft_projector.py with its 30 s slice removed and truncation=False, 5 outer folds, pooled out-of-fold AUC of the two-logit p_yes, %s; '
            'per fold %s; fold 3 collapsed (%d of %d test clips with answer mass under 0.01, median mass %.4f; other folds median mass at least %.4f); 95%% CI [%.4f, %.4f]; '
            'paired minus the whole-window zero-shot: %s; minus the same-pod 300 s control (0.815): %s; PART 23 (PART23_VERIFIED.tsv yes; my_verifier_out.json P23_ft_whole_pooled)'
            % (WHOLE, '[' + ', '.join('%.4f' % x for x in folds_w) + ']', f3['n_mass_below_0_01'], f3['n'], f3['median_answer_mass'], f3['other_folds_min_median_mass'],
               ft['lo'], ft['hi'], pv('P23_ft_whole_minus_zs_whole'), pv('P23_ft_whole_minus_ctrl300')),
            P23 + '/FT'])
new.append(['Qwen2.5-Omni', 'edaic_full', 'projector fine tune (PART 23 same-pod 300 s control)', str(ftc['value_4dp']), '275',
            'projector only fine tune, same pod, recipe and folds as the PART 23 whole-window run, but the processor keeps its default 300 s cut (first 300 s of each window; the recipe 30 s slice removed); '
            'pooled out-of-fold AUC over 5 folds, per fold %s; 95%% CI [%.4f, %.4f]; paired minus the 300 s zero-shot 0.8278: %s; compare row 205 (0.6421, the recipe that heard only the first 30 s); '
            'PART 23 (PART23_VERIFIED.tsv 7_ft_ctrl300_pooled yes; my_verifier_out.json P23_ft_ctrl300_pooled)'
            % ('[' + ', '.join('%.4f' % x for x in folds_c) + ']', ftc['lo'], ftc['hi'], pv('P23_ft_ctrl300_minus_zs_300s')),
            P23 + '/pull'])
for r in new:
    k = (r[0], r[1], r[2])
    assert k not in keys, k
    assert os.path.exists(r[6]), r[6]
    keys.add(k)
rows += new
wr(rows)
for i, r in enumerate(new):
    changes.append(dict(file=ML, row='line %d (appended)' % (n0 + 2 + i), field='new row', old='', new='%s | %s | %s | %s | n %s' % tuple(r[:5]),
                        why='PART 23 value added to master_lookup', closes='audit_p17 item 4 left / audit_p2223 item 9 (PART 23 not in master_lookup); ledger 6 LEFT me 2'))

# ------------------------------------------------------------------ (c) stale notes
shutil.copy2(ML, ML + '.bak_pre_notefix_23sep')
changes.append(dict(file=ML, row='(whole file)', field='backup', old='', new=ML + '.bak_pre_notefix_23sep (md5 %s)' % md5(ML), why='copy after the PART 23 append, before the note fixes', closes='task (c)'))
rows = rd()
CUT300 = ('the first 300 s of each E-DAIC stitched participant window: Qwen2_5OmniProcessor cuts audio at 300 s by default (PART 22 proof, scores/part22/p22_step2_truncation_proof.json), '
          'so 217 of 275 windows were cut and the 900 s file cap (89 of 275 files capped, median window 591.8 s) never came into play')
fix = {}
fix[('Qwen2.5-Omni', 'edaic_full', 'zero shot answer')] = (
    'zero shot, original run (podA, transformers 5.17.0). Window heard: %s. Same window, rebuild run: 0.8278 [0.7703, 0.8822] (row "zero shot answer (rebuild run, 300 s cut)"); '
    'truncation lifted: 0.8379 (PART 22) and 0.8366 (PART 23). E-DAIC 275 speakers' % CUT300)
fix[('Qwen2.5-Omni', 'edaic_full', 'projector probe')] = None  # replace phrase only
fix[('Qwen2.5-Omni', 'edaic_full', 'answer-state probe')] = None
fix[('Qwen2.5-Omni', 'edaic_full', 'projector fine tune')] = (
    'projector only fine tune, 5 fold out of fold AUC, 3 epochs, lr 1e-4, clip list mf_full.csv (E-DAIC windows up to 900 s). Window heard: the first 30 s of each window. '
    'The recipe slices every clip to its first 30 s before the processor (release/scripts/sft_projector.py line 59, x = x[:16000 * 30]; file dated 18 Sep, before this run) and writes '
    'window_seconds 30 into the sidecar (line 102); the run log edaic_rerun/logs/podA/sft_full.log ends with that script\'s RESULT line, and PART 23 records "the recipe heard only the first 30 s" '
    '(part23/FT/FT_whole_fold0_oof.json). So the sidecar field window_seconds 30 is correct, not stale. Without the slice: same-pod 300 s control 0.815, whole window 0.7931 (PART 23 rows). '
    'E-DAIC 275 speakers')
fix[('Qwen2-Audio', 'edaic_full', 'zero shot answer')] = (
    'zero shot, full stitched participant audio capped at 900 s (89 of 275 files capped, median window 591.8 s); the 900 s cap describes the file given to the model, and how much of it '
    'Qwen2-Audio\'s processor keeps was not measured (PART 22/23 measured Qwen2.5-Omni 300 s, Qwen3-Omni 900 s, Audio Flamingo 3 600 s). E-DAIC 275 speakers')
fix[('Qwen3-Omni-30B-A3B', 'edaic_full', 'zero shot answer')] = (
    'zero shot, full stitched participant audio capped at 900 s (89 of 275 capped, median window 591.8 s); Qwen3-Omni hears the whole capped window: 0 of 275 windows cut '
    '(PART 23 item 6, part23/C/C_trunc_275_derived.csv q3o_cut; real forward on pid 716 gives 11,700 audio tokens for 900 s). E-DAIC 275 speakers')
fix[('Audio Flamingo 3', 'edaic_full', 'zero shot answer')] = (
    'zero shot, full stitched participant audio capped at 900 s (89 of 275 capped, median window 591.8 s), but Audio Flamingo 3 hears at most the first 600 s: 136 of 275 windows cut at 600 s '
    '(PART 23 item 6, part23/C/C_trunc_275_derived.csv af3_cut; 136 truncation warnings in the run log; real forward on pid 716 gives 15,000 audio tokens = 600 s). E-DAIC 275 speakers')
fix[('Kimi-Audio', 'edaic_full', 'zero shot answer')] = None  # phrase only, keep the sidecar note
OLDPH = ('zero shot, full stitched participant audio capped at 900 s, 89 of 275 capped, median window 591.8 s',
         'full window capped at 900 s')
done = 0
for i, r in enumerate(rows[1:], start=2):
    k = (r[0], r[1], r[2])
    if k not in fix:
        continue
    old = r[5]
    if fix[k] is not None:
        newtxt = fix[k]
    elif k[0] == 'Kimi-Audio':
        newtxt = old.replace(OLDPH[0], 'zero shot, full stitched participant audio capped at 900 s (89 of 275 files capped, median window 591.8 s); how much of it Kimi-Audio keeps was not measured '
                                       '(PART 22/23 measured Qwen2.5-Omni 300 s, Qwen3-Omni 900 s, Audio Flamingo 3 600 s)')
    else:
        newtxt = old.replace(OLDPH[1], CUT300.replace('the first 300 s', 'window heard: the first 300 s', 1))
    assert newtxt != old, k
    r[5] = newtxt
    done += 1
    changes.append(dict(file=ML, row='line %d' % i, field='estimator (note)', old=old, new=newtxt, why='stale "capped at 900 s" window note',
                        closes={('Qwen2.5-Omni', 'edaic_full', 'zero shot answer'): 'ledger 6 LEFT me 12 (master_lookup row 200); audit_p2223 problem 2; DISCREPANCIES item 129',
                                ('Qwen2.5-Omni', 'edaic_full', 'projector fine tune'): 'ledger 6 LEFT me 12 (master_lookup row 205); audit_p2223 problem 3; DISCREPANCIES item 129; ledger 3 item 5 (FT swing, partly)'}
                        .get(k, 'ledger 6 LEFT me 12 ("capped at 900 s" edaic_full rows); DISCREPANCIES item 129')))
assert done == 8, done
wr(rows)
# integrity: same header, same keys in the same order, only the estimator field changed in the 8 rows
a = list(csv.reader(open(ML + '.bak_pre_part23', newline='')))
b = list(csv.reader(open(ML, newline='')))
assert b[0] == a[0] and len(b) == len(a) + len(new)
diffs = [(i, j) for i in range(len(a)) for j in range(7) if a[i][j] != b[i][j]]
assert all(j == 5 for i, j in diffs) and len(diffs) == 8, diffs
with open(os.path.join(OUTDIR, 'changes_master_lookup.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['file', 'row', 'field', 'old', 'new', 'why', 'closes']); w.writeheader(); w.writerows(changes)
print('appended', len(new), 'fixed', done, 'rows now', len(b) - 1, 'md5', md5(ML))
