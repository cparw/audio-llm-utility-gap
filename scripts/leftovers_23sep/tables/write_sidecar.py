#!/usr/bin/env python3
import json, hashlib, os, datetime, platform, csv, sklearn, numpy, scipy, sys
OUT = sys.argv[1]
R='<local data dir>/release'; G='<local data dir>/release_from_mac'
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
C2=json.load(open(os.path.join(OUT,'summary_counts_v2.json')))
nv=sum(1 for _ in csv.DictReader(open(os.path.join(OUT,'values.csv'))))
files_changed = {R+'/master/master_lookup.csv': R+'/master/master_lookup.csv.bak_pre_part23'}
for f in ('/edaic_rerun/part17/rows/T7.tsv','/edaic_rerun/part17/rows/T4.tsv','/scores/part22/rows/P22_verified.tsv','/scores/part22/rows/P22.tsv','/edaic_rerun/part16/POD1/p14_o25_sftfull.json','/edaic_rerun/part16/POD1/out16/p14_o25_sftfull.json'):
    files_changed[R+f] = R+f+'.bak_pre_notefix_23sep'
sc = dict(
 what='leftovers_23sep task "tables": (a) summary_all_v2.csv with value_verified filled from real twin files, (b) PART 23 rows appended to master_lookup.csv, (c) stale notes fixed, (d) <local data dir>/scratch verifier scripts saved',
 generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
 machine='Mac only, no pod (no refit needed); /usr/local/bin/python3 %s, numpy %s, scipy %s, sklearn %s' % (platform.python_version(), numpy.__version__, scipy.__version__, sklearn.__version__),
 per_value_file='values.csv: %d values, each recomputed by scripts/my_verifier.py or scripts/my_verifier_p22s1.py and compared with its reference (PART23_VERIFIED.tsv, T7b.tsv, POD1 verifier); 0 disagreements' % nv,
 verifier_rule='AUC by average-rank Mann-Whitney (scipy rankdata), asserted equal to sklearn roc_auc_score to 1e-12; bootstrap 2000 draws, fresh numpy default_rng(0) per cell, idx = rng.choice(N, size=N, replace=True) over the unique sorted speaker ids (E-DAIC pids 300 to 718, one clip per speaker), draws with one class skipped (none occurred), 2.5 and 97.5 percentiles, paired cells share idx',
 twin_rule='A twin is a number written by an independent verifier: the verify/ folders of PART 16 and 17, pod-side VERIFY files, or today\'s sibling tasks p16twins (twins_p16.tsv and its pod pulls, Linux sklearn 1.9.1) and m7m11 (m7m11_values.tsv verifier columns). Copies of the claimed value (keys claimed, theirs, original, orig, task, primary, POD4_value, master, json_auc, figure_answer) are never used. Match at the fragment precision (4 dp at most). Interval twinned when the verifier lo and hi sit in the same record or in records keyed to the same quantity. Decided by scripts/twin_extract.py + scripts/twin_decide.py; each accepted twin value re-found in its twin file by scripts/check_twins.py (all but one: P16.POD2.L2 lo is stored as 0.43485 = 0.43484998, a rounding tie).',
 summary_v2_counts=C2,
 master_lookup=dict(rows_before=234, rows_after=242, appended=8, estimator_notes_fixed=8, md5_before=hashlib.md5(open(R+'/master/master_lookup.csv.bak_pre_part23','rb').read()).hexdigest(), md5_after=hashlib.md5(open(R+'/master/master_lookup.csv','rb').read()).hexdigest()),
 files_changed_in_place={k: dict(backup=v, sha256_before=sha(v), sha256_after=sha(k)) for k, v in files_changed.items()},
 new_files=[G+'/summary/summary_all_v2.csv', G+'/summary/summary_counts_v2.json', R+'/master/master_lookup.csv.bak_pre_part23', R+'/master/master_lookup.csv.bak_pre_notefix_23sep'],
 not_touched='summary_all.csv v1 (md5 %s), any omni_final folder, the Overleaf tex, git; no pod created' % hashlib.md5(open(G+'/summary/summary_all.csv','rb').read()).hexdigest(),
 scripts=sorted('scripts/'+f for f in os.listdir(os.path.join(OUT,'scripts'))),
 n=dict(summary_rows=1532, fragment_rows_16_17=870, t7b_rows=275, part23_rows=275, sftfull_rows=966))
json.dump(sc, open(os.path.join(OUT,'values.sidecar.json'),'w'), indent=1)
print('sidecar ok', nv)
