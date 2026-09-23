#!/usr/bin/env python3
"""(c) Fix the stale notes named by the 23 Sep audit (outside master_lookup):
  - edaic_rerun/part17/rows/T7.tsv: 4 'NO INTERVAL ... not reproducible on Mac' notes (T7b now has the intervals)
  - edaic_rerun/part17/rows/T4.tsv: 2 'per-clip repeats not on Mac' notes on 0.7001 / 0.5884
  - scores/part22/rows/P22_verified.tsv and P22.tsv: 9 P22_S1 rows that still say 'not yet verified'
  - edaic_rerun/part16/POD1/p14_o25_sftfull.json and POD1/out16/p14_o25_sftfull.json: add the missing key n
Only the note text (the 'what' field) changes in the tsv files; values, intervals and every other byte stay.
Each file is copied to <file>.bak_pre_notefix_23sep first. Numbers come from my_verifier_out.json and
my_verifier_p22s1_out.json (own recompute). Writes changes_notes.csv in OUTDIR.
"""
import csv, json, os, shutil, sys, hashlib
R = '<local data dir>/release'
OUTDIR = sys.argv[1]
V = json.load(open(os.path.join(OUTDIR, 'my_verifier_out.json')))
S1 = json.load(open(os.path.join(OUTDIR, 'my_verifier_p22s1_out.json')))
changes = []


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def backup(p):
    b = p + '.bak_pre_notefix_23sep'
    if os.path.exists(b):
        sys.exit('backup exists already: ' + b)
    shutil.copy2(p, b)
    changes.append(dict(file=p, row='(whole file)', field='backup', old='', new='%s (md5 %s)' % (b, md5(b)), why='copy before the note fix', closes=''))


def edit_tsv(p, repl, closes, why):
    """repl: {row_id: (old_substring, new_substring)} applied to the 'what' field (column 2) only."""
    backup(p)
    lines = open(p, newline='').read().split('\n')
    hit = set()
    for i, l in enumerate(lines):
        c = l.split('\t')
        if c and c[0] in repl:
            o, n = repl[c[0]]
            assert o in c[1], (p, c[0])
            old = c[1]; c[1] = c[1].replace(o, n)
            lines[i] = '\t'.join(c)
            hit.add(c[0])
            changes.append(dict(file=p, row='line %d (%s)' % (i + 1, c[0]), field='what (note)', old=old, new=c[1], why=why, closes=closes))
    assert hit == set(repl), set(repl) - hit
    open(p, 'w', newline='').write('\n'.join(lines))
    # integrity: every other column identical
    a = [l.split('\t') for l in open(p + '.bak_pre_notefix_23sep', newline='').read().split('\n')]
    b = [l.split('\t') for l in open(p, newline='').read().split('\n')]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert len(x) == len(y) and all(x[j] == y[j] for j in range(len(x)) if j != 1)


def ci(k):
    d = V[k]
    return '[%.4f, %.4f]' % (d['lo'], d['hi'])


def pv(k):
    d = V[k]
    return '%+.4f [%.4f, %.4f]' % (d['value_4dp'], d['lo'], d['hi'])


# ---------------------------------------------------------------- T7.tsv
OLD = 'NO INTERVAL: per-repeat vectors not saved and not reproducible on Mac'
rep = {}
for k in ('llm', 'enc', 'ans', 'proj'):
    rep['T7_published_%s_meanof5' % k] = (OLD, 'interval now in T7b.tsv (the Linux sklearn 1.9.1 pod rerun reproduced all five per-repeat vectors, T7b/T7b_edaic300_meanof5_perclip.csv): '
                                               '95%% CI %s, paired minus the zero-shot answer %s; the old Mac note "no interval" is superseded'
                                               % (ci('T7b_%s_meanof5' % k), pv('T7b_%s_meanof5_minus_answer' % k)))
edit_tsv(R + '/edaic_rerun/part17/rows/T7.tsv', rep, 'ledger 6 LEFT me 12 (T7.tsv "NO INTERVAL"); audit_p17 problem 6',
         'stale "no interval" note: T7b has the interval')
# ---------------------------------------------------------------- T4.tsv
OLD4 = '(json per_repeat list, per-clip repeats not on Mac)'
rep = {}
for k, tid in (('llm', 'T4_edaic_o25_llm_mean5_jsononly'), ('enc', 'T4_edaic_o25_enc_mean5_jsononly')):
    rep[tid] = (OLD4, '(json per_repeat list; the per-clip repeats were later reproduced on Linux sklearn 1.9.1 in T7b: 95%% CI %s, paired minus the zero-shot answer %s)'
                % (ci('T7b_%s_meanof5' % k), pv('T7b_%s_meanof5_minus_answer' % k)))
edit_tsv(R + '/edaic_rerun/part17/rows/T4.tsv', rep, 'ledger 6 LEFT me 12 (T4.tsv "no interval"); audit_p17 problem 6',
         'stale "per-clip repeats not on Mac" note: T7b has them')
# ---------------------------------------------------------------- P22 S1
OLD22 = '[value re-derived by the pod job from the cited source after it blanked this row by mistake; not yet verified]'
chk = {'P22_S1_tf_version': 'tf_version %s' % S1['tf_version'], 'P22_S1_chunk_length': 'chunk_length %s' % S1['chunk_length'],
       'P22_S1_n_samples': 'n_samples %s' % S1['n_samples'], 'P22_S1_trunc_default': 'feature_extraction_whisper.py:196 "%s"' % S1['trunc_default_line196'],
       'P22_S1_tok_300s': 'formula %s' % S1['tok_300s'], 'P22_S1_tok_900s_off': 'formula %s' % S1['tok_900s_off'],
       'P22_S1_formula_match': 'formula matches %s of %s' % (S1['formula_match'], S1['n_windows']), 'P22_S1_gt300_at_7500': '%s of %s windows over 300 s at 7500' % (S1['gt300_at_7500'], S1['gt300']),
       'P22_S1_capped900': 'capped %s' % S1['capped900']}
for p in (R + '/scores/part22/rows/P22_verified.tsv', R + '/scores/part22/rows/P22.tsv'):
    rep = {k: (OLD22, '[value re-derived by the pod job from the cited source after it blanked this row by mistake; verified afterwards: the PART 22 verifier agrees (agree_4dp True, '
                      'verify/out/rows_S1_check.json), and a recheck from the cited sources on 23 Sep gives %s (leftovers_23sep/tables/my_verifier_p22s1_out.json)]' % v)
           for k, v in chk.items()}
    edit_tsv(p, rep, 'ledger 6 LEFT me 12 (9 P22_S1 "not yet verified" texts); audit_p2223 item 1 left and problem 7',
             'stale "not yet verified" note on rows that are verified')
# ---------------------------------------------------------------- sftfull sidecar n
n = V['sftfull_csv']['rows']
for p in (R + '/edaic_rerun/part16/POD1/p14_o25_sftfull.json', R + '/edaic_rerun/part16/POD1/out16/p14_o25_sftfull.json'):
    backup(p)
    raw = open(p).read()
    j = json.loads(raw)
    assert 'n' not in j
    out = {}
    for k, v in j.items():
        out[k] = v
        if k == 'prompt':
            out['n'] = n
    indent = 1 if raw.startswith('{\n "') else 2
    assert json.dumps(j, indent=indent) + ('\n' if raw.endswith('\n') else '') == raw, 'format would change: ' + p
    open(p, 'w').write(json.dumps(out, indent=indent) + ('\n' if raw.endswith('\n') else ''))
    j2 = json.load(open(p))
    assert all(j2[k] == j[k] for k in j) and j2['n'] == n
    changes.append(dict(file=p, row='key n', field='sidecar key', old='(missing)', new='n: %d (rows of p14_o25_sftfull.csv, recounted; 138 speakers)' % n,
                        why='POD1 verifier required key n was missing (verify/POD1_verify_out.json "missing:[\'n\']")', closes='ledger 6 LEFT me 12 (p14_o25_sftfull.json n); audit_p16 item 1 left'))
with open(os.path.join(OUTDIR, 'changes_notes.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['file', 'row', 'field', 'old', 'new', 'why', 'closes']); w.writeheader(); w.writerows(changes)
print(len(changes), 'change records')
