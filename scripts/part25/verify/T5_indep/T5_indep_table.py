#!/usr/local/bin/python3
# Builds T5_VERIFIED.tsv from T5v_recomputed.json (written by T5_indep_verify.py). No track code imported.
import json, hashlib, datetime, os
import numpy as np

G = '<local data dir>/release_from_mac/scores'
WD = G + '/part25/verify/T5_indep'
OUT = G + '/part25/verify/T5_VERIFIED.tsv'
J = json.load(open(WD + '/T5v_recomputed.json'))
R, CMR = J['R'], J['CMR']

def r4(x): return round(float(x), 4)
def e2(x): return float('%.2e' % float(x))
def lay(xs):
    xs = list(xs)
    if not xs: return 'none'
    if xs == list(range(xs[0], xs[-1] + 1)) and len(xs) > 2: return f'{xs[0]} to {xs[-1]}'
    return ','.join(map(str, xs))

rows = []
def add(i, claimed, recomputed, checks):
    ok = all(bool(c) for c in checks)
    rows.append((i, claimed, recomputed, 'yes' if ok else 'no', checks))

def cm(ds, st, pod):
    return [c for c in CMR if c['curve'] == ds and c['stream'] == st and c['pod'] == pod][0]

# ---------- NeuroVoz ----------
n = R['neurovoz']; s = n['refit']
add('T5_neurovoz_refit_above_answer',
    '32/32; ans 0.6951 [0.6356, 0.7519]; 1270 clips, 107 spk, 2000/2000 draws; lowest layer 0 at 0.8361',
    f"{s['refit_above_answer']}/32; ans {r4(n['answer_auc'])} [{r4(n['answer_lo'])}, {r4(n['answer_hi'])}]; {n['n_clips']} clips, {n['n_speakers']} spk, {n['boot_usable']}/2000 draws; lowest layer {s['refit_min_layer']} at {r4(s['refit_min'])}",
    [s['refit_above_answer'] == 32, r4(n['answer_auc']) == 0.6951, r4(n['answer_lo']) == 0.6356, r4(n['answer_hi']) == 0.7519,
     n['n_clips'] == 1270, n['n_speakers'] == 107, n['boot_usable'] == 2000, s['refit_min_layer'] == 0, r4(s['refit_min']) == 0.8361,
     n['track_draw_compare']['answer_draws_maxabs'] == 0.0])
add('T5_neurovoz_refit_above_hi',
    '32/32 above 0.7519; 3 same-fold refit files 32/32; paired delta lo>0 on 32/32',
    f"{s['refit_above_hi']}/32; same-fold files {s['same_fold_files']} give {s['same_fold_above_hi']}; paired delta lo>0 {n['paired_delta_lo_gt0']}/32 (min lo {min(n['paired_delta_lo']):.4f}); draws equal to track draw by draw (max abs {n['track_draw_compare']['delta_draws_maxabs']})",
    [s['refit_above_hi'] == 32, s['same_fold_files'] == 3, s['same_fold_above_hi'] == [32], s['same_fold_above_answer'] == [32],
     n['paired_delta_lo_gt0'] == 32])

# ---------- KCL / ADReSSo ----------
nf = R['kcl_adresso_files_in_t5']; v5 = R['v5_notpossible']
k = R['kcl']
add('T5_kcl_refit',
    'NOT LANDED (no refit exists); ans 0.7143 [0.5210, 0.8810]; 37 clips, 37 spk; saved 32/32 above ans, 0/32 above hi; saved max 0.8482',
    f"no kcl file in any T5 pull or input ({len(nf)} kcl/adresso files); {v5['n_npz_scanned']} npz scanned in t5v; ans {r4(k['answer_auc'])} [{r4(k['answer_lo'])}, {r4(k['answer_hi'])}]; {k['n_clips']} clips, {k['n_speakers']} spk; saved {k['saved_above_answer']}/32 above ans, {k['saved_above_hi']}/32 above hi; saved max {r4(k['saved_max'])}",
    [len(nf) == 0, r4(k['answer_auc']) == 0.7143, r4(k['answer_lo']) == 0.5210, r4(k['answer_hi']) == 0.8810, k['n_clips'] == 37,
     k['n_speakers'] == 37, k['saved_above_answer'] == 32, k['saved_above_hi'] == 0, r4(k['saved_max']) == 0.8482])
k = R['adresso']
add('T5_adresso_refit',
    'NOT LANDED (no refit exists); ans 0.6150 [0.5432, 0.6890]; 237 clips, 237 spk; saved 32/32 above ans, 32/32 above hi; saved min 0.7413',
    f"no adresso file in any T5 pull or input; ans {r4(k['answer_auc'])} [{r4(k['answer_lo'])}, {r4(k['answer_hi'])}]; {k['n_clips']} clips, {k['n_speakers']} spk; saved {k['saved_above_answer']}/32 above ans, {k['saved_above_hi']}/32 above hi; saved min {r4(k['saved_min'])}",
    [len(nf) == 0, r4(k['answer_auc']) == 0.6150, r4(k['answer_lo']) == 0.5432, r4(k['answer_hi']) == 0.6890, k['n_clips'] == 237,
     k['n_speakers'] == 237, k['saved_above_answer'] == 32, k['saved_above_hi'] == 32, r4(k['saved_min']) == 0.7413])

# ---------- ADReSS-2020 ----------
n = R['adress2020']; s = n['refit']
add('T5_adress2020_refit_above_answer',
    '32/32; ans 0.6241 [0.5348, 0.7148]; 156 clips, 156 spk; lowest layer 2 at 0.6491, margin 0.025',
    f"{s['refit_above_answer']}/32; ans {r4(n['answer_auc'])} [{r4(n['answer_lo'])}, {r4(n['answer_hi'])}]; {n['n_clips']} clips, {n['n_speakers']} spk; lowest layer {s['refit_min_layer']} at {r4(s['refit_min'])}, margin {round(s['refit_min'] - n['answer_auc'], 3)}",
    [s['refit_above_answer'] == 32, r4(n['answer_auc']) == 0.6241, r4(n['answer_lo']) == 0.5348, r4(n['answer_hi']) == 0.7148,
     n['n_clips'] == 156, n['n_speakers'] == 156, s['refit_min_layer'] == 2, r4(s['refit_min']) == 0.6491,
     round(s['refit_min'] - n['answer_auc'], 3) == 0.025])
add('T5_adress2020_refit_above_hi',
    '24/32; layers 0 to 7 not above 0.7148; saved curve 24/32; paired delta lo>0 on 22/32',
    f"{s['refit_above_hi']}/32; not above hi: {lay(s['refit_not_above_hi'])}; saved {n['saved_above_hi']}/32; paired delta lo>0 {n['paired_delta_lo_gt0']}/32; same-fold files {s['same_fold_files']} give {s['same_fold_above_hi']}",
    [s['refit_above_hi'] == 24, s['refit_not_above_hi'] == list(range(8)), n['saved_above_hi'] == 24, n['paired_delta_lo_gt0'] == 22])

# ---------- E-DAIC whole ----------
w = R['edaic_whole_p23']; s = w['refit']; p23 = R['p23_saved']
others = {kk: (R[kk]['refit']['refit_above_answer'], r4(R[kk]['answer_auc'])) for kk in ('edaic_lifted_p22', 'edaic_full_o25', 'edaic_full_statesrun')}
tab = R['tex']['table1_edaic']
add('T5_edaic_whole_refit_above_answer',
    '0/32; ans 0.8366 [0.7786, 0.8866] (Table 1 0.84); 275 clips, 275 spk; 0/32 vs 0.8379, 0.8285, 0.8278; refit max 0.6922 at layer 21; P23 saved curve 0/32, max 0.6875',
    f"{s['refit_above_answer']}/32; ans {r4(w['answer_auc'])} [{r4(w['answer_lo'])}, {r4(w['answer_hi'])}] (Table 1 row: {tab[0] if tab else 'n/a'}); {w['n_clips']} clips, {w['n_speakers']} spk; others {'; '.join(f'{a}/32 vs {b}' for a, b in others.values())}; refit max {r4(s['refit_max'])} at layer {s['refit_max_layer']}; P23 saved {p23['vs']['edaic_whole_p23'][0]}/32, max {r4(p23['max'])}",
    [s['refit_above_answer'] == 0, r4(w['answer_auc']) == 0.8366, r4(w['answer_lo']) == 0.7786, r4(w['answer_hi']) == 0.8866,
     w['n_clips'] == 275, w['n_speakers'] == 275, others['edaic_lifted_p22'] == (0, 0.8379), others['edaic_full_o25'] == (0, 0.8285),
     others['edaic_full_statesrun'] == (0, 0.8278), r4(s['refit_max']) == 0.6922, s['refit_max_layer'] == 21,
     p23['vs']['edaic_whole_p23'][0] == 0, r4(p23['max']) == 0.6875, bool(tab) and '0.84' in tab[0]])
hi4 = {kk: R[kk]['refit']['refit_above_hi'] for kk in ('edaic_whole_p23', 'edaic_lifted_p22', 'edaic_full_o25', 'edaic_full_statesrun')}
add('T5_edaic_whole_refit_above_hi',
    '0/32 for all four whole-interview answer files',
    '; '.join(f"{kk} {v}/32 (hi {r4(R[kk]['answer_hi'])})" for kk, v in hi4.items()),
    [all(v == 0 for v in hi4.values())])

# ---------- E-DAIC first 30 s ----------
f = R['edaic_first30']; s = f['refit']
add('T5_edaic_first30_refit_above_answer',
    '2/32; ans 0.6620 [0.5831, 0.7315]; 275 clips, 275 spk; only layers 0 and 1; 2/32 over all 32 same-fold files',
    f"{s['refit_above_answer']}/32; ans {r4(f['answer_auc'])} [{r4(f['answer_lo'])}, {r4(f['answer_hi'])}]; {f['n_clips']} clips, {f['n_speakers']} spk; above: {lay([l for l in range(32) if l not in s['refit_not_above_answer']])}; {s['same_fold_files']} same-fold files give {s['same_fold_above_answer']}",
    [s['refit_above_answer'] == 2, r4(f['answer_auc']) == 0.6620, r4(f['answer_lo']) == 0.5831, r4(f['answer_hi']) == 0.7315,
     f['n_clips'] == 275, f['n_speakers'] == 275, [l for l in range(32) if l not in s['refit_not_above_answer']] == [0, 1],
     s['same_fold_files'] == 32, s['same_fold_above_answer'] == [2]])
add('T5_edaic_first30_refit_above_hi', '0/32',
    f"{s['refit_above_hi']}/32; same-fold files give {s['same_fold_above_hi']}",
    [s['refit_above_hi'] == 0, s['same_fold_above_hi'] == [0]])

# ---------- Pitt ----------
p = R['pitt']; s = p['refit']
add('T5_pitt_refit_extra',
    '31/32 above ans, 25/32 above hi; ans 0.6578 [0.6033, 0.7125]; 468 clips, 228 speaker ids; layer 2 at 0.6517; layers 0 to 6 not above hi; paired delta lo>0 18/32',
    f"{s['refit_above_answer']}/32 above ans, {s['refit_above_hi']}/32 above hi; ans {r4(p['answer_auc'])} [{r4(p['answer_lo'])}, {r4(p['answer_hi'])}]; {p['n_clips']} clips, {p['n_speakers']} speaker ids; not above ans: {lay(s['refit_not_above_answer'])} at {r4(s['refit_min'])}; not above hi: {lay(s['refit_not_above_hi'])}; paired delta lo>0 {p['paired_delta_lo_gt0']}/32",
    [s['refit_above_answer'] == 31, s['refit_above_hi'] == 25, r4(p['answer_auc']) == 0.6578, r4(p['answer_lo']) == 0.6033,
     r4(p['answer_hi']) == 0.7125, p['n_clips'] == 468, p['n_speakers'] == 228, s['refit_not_above_answer'] == [2],
     r4(s['refit_min']) == 0.6517, s['refit_not_above_hi'] == list(range(7)), p['paired_delta_lo_gt0'] == 18])

# ---------- verdict ----------
cnt = {kk: (R[kk]['refit']['refit_above_answer'], R[kk]['refit']['refit_above_hi']) for kk in
       ('neurovoz', 'adress2020', 'edaic_whole_p23', 'edaic_first30', 'pitt')}
holds = [kk for kk, v in cnt.items() if v[0] == 32]
holds_hi = [kk for kk, v in cnt.items() if v[1] == 32]
fails = [kk for kk, v in cnt.items() if v[0] < 32]
tx = R['tex']
add('T5_every_encoder_layer_verdict',
    'holds on NeuroVoz and ADReSS-2020 only; fails on E-DAIC (and Pitt); KCL and ADReSSo cannot be judged on the refit; above hi only NeuroVoz; phrase not in any tex; line 226 has "the signal is present at every depth" (Parkinson\'s)',
    f"holds (32/32 above ans): {', '.join(holds)}; above hi 32/32: {', '.join(holds_hi)}; fails: {', '.join(f'{kk} {cnt[kk][0]}/32' for kk in fails)}; no refit: kcl, adresso; 'every encoder layer' hits in {tx['n_tex']} tex files: {len(tx['every_encoder_layer_hits'])}; final tex line 226 has phrase: {tx['line226_has_phrase']}, commented: {tx['line226_commented']}, Parkinson's: {tx['line226_parkinson']}; live 'every depth' lines: {tx['every_depth_live']}",
    [sorted(holds) == ['adress2020', 'neurovoz'], holds_hi == ['neurovoz'], sorted(fails) == ['edaic_first30', 'edaic_whole_p23', 'pitt'],
     len(tx['every_encoder_layer_hits']) == 0, tx['line226_has_phrase'], not tx['line226_commented'], tx['line226_parkinson'],
     tx['every_depth_live'] == [226]])

# ---------- curve match ----------
same = [('pitt', 'enc', 'lo-t5-10'), ('pitt', 'proj', 'lo-t5-10'), ('pitt', 'llm', 'lo-t5-2'), ('pitt', 'ans', 'lo-t5-13'),
        ('edaic30', 'enc', 'lo-t5-9'), ('edaic30', 'proj', 'lo-t5-9'), ('edaic30', 'enc', 'lo-t5-10'), ('edaic30', 'proj', 'lo-t5-10'),
        ('edaic30', 'llm', 'lo-t5-5'), ('edaic30', 'ans', 'lo-t5-6')]
sr = [cm(*t) for t in same]
add('T5_curve_match_same_run_avx512', 'max abs diff 0 on every curve; all layers match at 4 dp',
    '; '.join(f"{c['curve']} {c['stream']} {c['pod']} ({c['blas']}) {c['maxabs']:.1e} {c['match4dp']}/{c['n_layers']}" for c in sr),
    [c['maxabs'] < 1e-12 and c['match4dp'] == c['n_layers'] and c['blas'] == 'SkylakeX' for c in sr])
oe = {('neurovoz', 'enc'): (1.09e-03, 5), ('adress2020', 'enc'): (2.88e-03, 1), ('edaicfull', 'enc'): (2.97e-03, 2),
      ('neurovoz', 'proj'): (4.96e-06, None), ('adress2020', 'proj'): (1.31e-03, None), ('edaicfull', 'proj'): (7.25e-04, None)}
pods = {'neurovoz': 'lo-t5-8', 'adress2020': 'lo-t5-7', 'edaicfull': 'lo-t5-9'}
chk = []; txt = []
for (ds, st), (d, m) in oe.items():
    c = cm(ds, st, pods[ds])
    chk.append(e2(c['maxabs']) == d and (m is None or c['match4dp'] == m))
    txt.append(f"{ds} {st} {c['maxabs']:.2e} ({c['match4dp']}/{c['n_layers']})")
add('T5_curve_match_other_extraction',
    'close, not exact; enc NeuroVoz 1.09e-03 (5/32), ADReSS-2020 2.88e-03 (1/32), E-DAIC whole o25_full 2.97e-03 (2/32); proj 4.96e-06, 1.31e-03, 7.25e-04',
    'close, not exact; ' + '; '.join(txt), chk)
hw = {('pitt', 'enc', 'lo-t5-1'): (6.78e-04, 8), ('pitt', 'proj', 'lo-t5-1'): (5.81e-05, None), ('pitt', 'ans', 'lo-t5-3'): (7.94e-04, 4),
      ('edaic30', 'enc', 'lo-t5-4'): (5.07e-04, 7), ('edaic30', 'proj', 'lo-t5-4'): (3.62e-04, None)}
chk = []; txt = []
for (ds, st, pod), (d, m) in hw.items():
    c = cm(ds, st, pod)
    chk.append(e2(c['maxabs']) == d and (m is None or c['match4dp'] == m) and c['blas'] == 'Haswell')
    txt.append(f"{ds} {st} {pod} ({c['blas']}) {c['maxabs']:.2e} ({c['match4dp']}/{c['n_layers']})")
hc = R['haswell_counts']
chk.append(hc['pitt lo-t5-1 enc'] == hc['pitt lo-t5-10 enc'] and hc['edaic30 lo-t5-4 enc'] == hc['edaic30 lo-t5-10 enc'])
txt.append(f"counts Haswell vs AVX-512: pitt {hc['pitt lo-t5-1 enc']} vs {hc['pitt lo-t5-10 enc']}, edaic30 {hc['edaic30 lo-t5-4 enc']} vs {hc['edaic30 lo-t5-10 enc']}")
add('T5_curve_match_haswell_pods',
    'close, not exact; Pitt enc 6.78e-04 (8/32), proj 5.81e-05, ans 7.94e-04 (4/29), E-DAIC first30 enc 5.07e-04 (7/32), proj 3.62e-04; counts unchanged',
    'close, not exact; ' + '; '.join(txt), chk)

# ---------- track's own check ----------
tc = R['track_check']
add('T5_independent_check', '544 checks, 0 bad; omni_final last modified 2026-09-18',
    f"track sidecar says {tc['n_ok'] + tc['n_bad']} checks, {tc['n_bad']} bad; sidecar hashes match current T5 files: {all(tc['hashes_match_current'].values())}; omni_final latest mtime {tc['omni_final_latest_mtime_utc']} over {tc['omni_final_n_files']} files; this verifier also reproduced every T5 draw array draw by draw",
    [tc['n_ok'] == 544, tc['n_bad'] == 0, all(tc['hashes_match_current'].values()), tc['omni_final_latest_mtime_utc'].startswith('2026-09-18')])

with open(OUT, 'w') as fh:
    fh.write('id\tclaimed\trecomputed\tverified\n')
    for i, c, r, v, _ in rows:
        fh.write(f"{i}\t{c.replace(chr(9), ' ')}\t{r.replace(chr(9), ' ')}\t{v}\n")
h = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
side = dict(task='PART 25 T5 independent verifier', date_utc=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            tsv=OUT, tsv_sha256=h, scripts=[WD + '/T5_indep_verify.py', WD + '/T5_indep_table.py'],
            recomputed_json=WD + '/T5v_recomputed.json', draws=sorted(x for x in os.listdir(WD) if x.startswith('T5v_draws_')),
            perclip=sorted(x for x in os.listdir(WD) if x.startswith('T5v_perclip_')),
            method='sklearn roc_auc_score (rank formula cross check); speaker bootstrap 2000 draws, fresh default_rng(0) per cell, '
                   'np.unique(str) speaker order, idx = rng.choice(n_spk, n_spk, replace=True), all clips of each drawn speaker, '
                   'usable = both classes present, np.percentile 2.5/97.5; paired delta = layer AUC minus answer AUC in the same draw; '
                   'strict > for all counts; 4 dp match = equal after round(x, 4)',
            notes=['Answer CI depends slightly on speaker order: NeuroVoz numeric order gives [0.6357, 0.7485], first-appearance [0.6357, 0.7515], '
                   'string order (used by the track and here) [0.6356, 0.7519]. No count changes (refit min 0.8361).',
                   'Pitt has 228 speaker ids but Dementia172 and Control172 share a number; Table 1 prints 227.',
                   'Mac env for this check: sklearn 1.7.2, numpy 2.2.6, scipy 1.15.3; the track draws were reproduced exactly.'],
            rows=[dict(id=i, claimed=c, recomputed=r, verified=v, n_subchecks=len(ch), n_subchecks_failed=sum(1 for x in ch if not x))
                  for i, c, r, v, ch in rows])
json.dump(side, open(OUT.replace('.tsv', '.sidecar.json'), 'w'), indent=1)
for i, c, r, v, ch in rows:
    print(f"{i}\t{v}\t({len(ch)} subchecks)\t{r}")
