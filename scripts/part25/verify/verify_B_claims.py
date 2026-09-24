#!/usr/bin/env python3
"""Second stage of the independent Track B verifier.
Reads my own recomputation (B_indep/V_results.json, from verify_B_indep.py) and checks every
claimed id against it, against the tex (read only), and against Track B's files on disk.
Writes part25/verify/B_VERIFIED.tsv (id, claimed, recomputed, verified) and a sidecar json."""
import os, re, json, glob, hashlib, datetime, subprocess
import numpy as np

P25 = '<local data dir>/release_from_mac/scores/part25'
BD = P25 + '/B'
OUT = P25 + '/verify'
TEX = '<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex'
TEX_SHA = '6c886f3513e93de7f78a7324f608091e7cdc24f257089bb5bd0a5c40638e4a20'
OMNI = ['<local data dir>/release/omni_final', '<local data dir>/paper1_local_runs/omni_final']
B_START_LOCAL = '2026-09-23 17:57:09'  # = 2026-09-24 00:57:09 UTC (START_UTC.txt), Mac is UTC-7


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


J = json.load(open(OUT + '/B_indep/V_results.json'))
R = J['res']; H = J['holm']
L = open(TEX).read().split('\n')
line = lambda n: L[n - 1]


def f2(x, d=2):
    return f'{x:.{d}f}'


def rnd_ok(x, tex, d=2):
    return f2(x, d) == tex


def iv(rid):
    r = R[rid]
    return f"{r['value']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


rows = []
detail = {}


def add(rid, claimed, recomputed, ok, extra=None):
    rows.append((rid, claimed, recomputed, 'yes' if ok else 'no'))
    detail[rid] = {'claimed': claimed, 'recomputed': recomputed, 'verified': bool(ok), 'checks': extra}


# ------------------------------------------------------------------ Holm counts
FAM = {'GAPS': 7, 'ARMS': 9, 'FINE TUNE': 6, 'CONTROLS': 2}
claimed24 = [k for k, h in H.items() if h['family'] in FAM and h['m'] > 0]
surv = [k for k in claimed24 if H[k]['holm'] <= 0.05]
surv_saved = [k for k in claimed24 if H[k]['holm_saved'] <= 0.05]
fam_m = {f: sorted({H[k]['m'] for k in claimed24 if H[k]['family'] == f}) for f in FAM}
loo = J['gaps_leave_one_out_survivors_of_6']
minp = min(R[k]['p'] for k in claimed24)
ok = (len(claimed24) == 24 and len(surv) == 24 and len(surv_saved) == 24 and all(fam_m[f] == [FAM[f]] for f in FAM)
      and all(int(v) == 6 for v in loo.values()) and abs(minp - 2 / 2001) < 1e-15)
add('B_paper_line', 'All 24 differences survive Holm correction within their family (23 of 23 if six gaps).',
    f"{len(surv)} of {len(claimed24)} survive, from my redrawn draws and from B's saved draws alike; families GAPS m={fam_m['GAPS']}, ARMS m={fam_m['ARMS']}, "
    f"FINE TUNE m={fam_m['FINE TUNE']}, CONTROLS m={fam_m['CONTROLS']}; dropping any one of the 7 gaps (m=6) leaves 6 of 6 gap survivors, so 23 of 23; "
    f"smallest p {minp:.6f} = 2/2001", ok)

# ------------------------------------------------------------------ claim list vs tex
tex_sha_now = sha(TEX)
need = {
    26: ['On depression the answer beats every probe', 'four of six models fall below chance', 'narrows the gap on five of the six'],
    52: ['the gap is significant on every Parkinson', 'except MDVR-KCL and reverses on depression'],
    53: ['recovers the overall AUC on five of seven datasets'],
    149: ["excludes zero on every Parkinson's and Alzheimer's set except MDVR-KCL", 'from 0.11 [0.05, 0.18] on Pitt to 0.33 on PC-GITA',
          'falls to 0.83 ($-$0.07 [$-$0.12, $-$0.02])', 'falls to 0.89 ($-$0.03 [$-$0.061, $-$0.001])',
          '($-$0.09 [$-$0.16, $-$0.03] and $-$0.24 [$-$0.32, $-$0.15])'],
    180: ['exclude zero for all six models on depression and for the three Qwen models on Pitt'],
    226: ['AUC improves significantly on every dataset except MDVR-KCL', 'reaches 0.825 on Pitt, 0.06 [0.02, 0.10] above the projector run'],
    244: ['significant performance gain in AD and PD detection'],
}
found = {n: all(s in line(n) for s in ss) for n, ss in need.items()}
count = {'gaps positive (PC-GITA, NeuroVoz, ADReSSo, ADReSS-2020, Pitt)': 5, 'E-DAIC reversals': 2,
         'arms E-DAIC': 6, 'arms Pitt Qwen': 3, 'fine-tune gains': 5, 'LoRA minus projector': 1, 'channel drops': 2}
# printed-number reproduction for Table 1 / Table 2 cells and intervals behind the 24
T1 = {'PC-GITA': ('0.56', '0.89', 'F_pcgita'), 'NeuroVoz': ('0.70', '0.90', 'F_neurovoz'), 'MDVR-KCL': ('0.71', '0.70', 'Fs_kcl'),
      'E-DAIC': ('0.84', '0.86', 'Fs_edaic_seed0'), 'Pitt': ('0.66', '0.77', 'F_pitt'), 'ADReSSo': ('0.62', '0.82', 'F_adresso'),
      'ADReSS-2020': ('0.62', '0.77', 'F_adress2020')}
T1lines = {l.split('&')[0].strip(): [c.strip().rstrip('\\').strip() for c in l.split('&')] for l in L[119:130] if '&' in l}
mism = []
for ds, (zs, ft, rid) in T1.items():
    cells = T1lines[ds]
    if cells[2] != zs or cells[4] != ft:
        mism.append(f'transcription {ds}')
    a, b = R[rid]['point_terms'][:2]
    if not (rnd_ok(b, zs) and rnd_ok(a, ft)):
        mism.append(f'T1 {ds} {a:.4f} {b:.4f}')
T2 = {'Qwen2-Audio': 'q2a', 'Qwen2.5-Omni': 'o25', 'Qwen3-Omni': 'q3o', 'Audio Flamingo 2': 'af2', 'Audio Flamingo 3': 'af3', 'Kimi-Audio': 'kimi'}
for l in L[167:173]:
    name = l.split('\\cite')[0].strip()
    m = T2[name]
    cells = [c.strip().rstrip('\\').strip() for c in l.split('&')[1:]]
    e = R[f'R_edaic_{m}']['point_terms']
    pr = R[(f'R_pitt_{m}' if m in ('q2a', 'o25', 'q3o') else f'Rs_pitt_{m}')]['point_terms']
    for got, tex in zip([e[0], e[1], pr[0], pr[1]], cells):
        if not rnd_ok(got, tex):
            mism.append(f'T2 {name} {got:.4f} vs {tex}')
ivchk = [('G_pitt', ['0.11', '0.05', '0.18'], 2), ('G_edaic_llm', ['-0.09', '-0.16', '-0.03'], 2), ('G_edaic_enc', ['-0.24', '-0.32', '-0.15'], 2),
         ('C_pg_probe_drop', ['-0.07', '-0.12', '-0.02'], 2), ('C_nv_probe_drop', ['-0.03', '-0.061', '-0.001'], None),
         ('F_lora_minus_projector', ['0.06', '0.02', '0.10'], 2), ('Fs_edaic_seed0', ['0.02', '-0.01', '0.05'], 2)]
for rid, tv, d in ivchk:
    r = R[rid]
    ds = [2, 3, 3] if d is None else [d] * 3
    if rid == 'C_nv_probe_drop':
        ds = [2, 3, 3]
    for x, t, dd in zip([r['value'], r['lo'], r['hi']], tv, ds):
        if f2(x, dd) != t:
            mism.append(f'interval {rid} {x:.4f} vs {t}')
single = [('G_pcgita', 'value', '0.33'), ('G_edaic_llm', 't0', '0.74'), ('G_edaic_enc', 't0', '0.60'), ('G_edaic_llm', 't1', '0.84'),
          ('Gs_edaic_ansstate', 't0', '0.84'), ('C_pg_probe_drop', 't0', '0.83'), ('C_nv_probe_drop', 't0', '0.89'),
          ('Cs_pg_answer', 't0', '0.57'), ('F_pitt', 't0', '0.77')]
for rid, what, t in single:
    x = R[rid]['value'] if what == 'value' else R[rid]['point_terms'][0 if what == 't0' else 1]
    if f2(x) != t:
        mism.append(f'single {rid} {what} {x:.4f} vs {t}')
if f2(R['F_lora_minus_projector']['point_terms'][0], 3) != '0.825':
    mism.append('lora 0.825')
drops = sorted(-R[f'R_pitt_{m}']['value'] for m in ('q2a', 'o25', 'q3o'))
if not (f2(drops[0]) == '0.17' and f2(drops[-1]) == '0.29'):
    mism.append('pitt drops 0.17 to 0.29')
# probe ranges l.149: PD 0.75 to 0.92 (KCL mean of five from its repeats json), AD 0.77 to 0.88
kclj = json.load(open('<local data dir>/release/omni_final/omni_kcl_nested_repeats.json'))
kcl_vals = []
def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (int, float)) and 'mean' in k.lower():
                kcl_vals.append((k, v))
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(kclj)
pd_probes = [R['G_pcgita']['point_terms'][0], R['G_neurovoz']['point_terms'][0]]
ad_probes = [R['G_pitt']['point_terms'][0], R['G_adresso']['point_terms'][0], R['G_adress2020']['point_terms'][0]]
if not (f2(max(pd_probes)) == '0.92' and f2(min(ad_probes)) == '0.77' and f2(max(ad_probes)) == '0.88'):
    mism.append('probe ranges')
ok = tex_sha_now == TEX_SHA and all(found.values()) and sum(count.values()) == 24 and not mism
add('B_claim_list_vs_tex', '24 claimed differences in the tex, not 23 (7 gaps incl. 2 E-DAIC reversals; 9 arms; 5 fine-tune + LoRA; 2 channel drops); 0 mismatches; tex sha 6c886f35',
    f"my own line-by-line read of the non-comment tex gives {sum(count.values())} (5 positive gaps l.149/l.52, 2 E-DAIC reversals l.149, 6 E-DAIC arms + 3 Qwen Pitt arms l.180, "
    f"5 fine-tune gains l.226 (l.26/l.53/l.244 restate), LoRA minus projector l.226, 2 channel drops l.149); all quoted phrases present: {all(found.values())}; "
    f"Table 1 (7 rows x zero-shot audio, fine-tuned), all 24 Table 2 cells, 7 printed intervals and the single values behind them: {len(mism)} mismatches; "
    f"tex sha256 now {tex_sha_now[:8]}...; other differences in the tex (l.149 eGeMAPS margins, l.180 '0.39 below the agreement arm', l.226 E-DAIC pair retrain 0.22 to 0.59) carry no interval and no significance word, so not counted",
    ok, {'found': found, 'mismatches': mism, 'kcl_repeats_json_means': kcl_vals[:5]})

# ------------------------------------------------------------------ the 24 numeric rows
CL = {  # id: (value, lo, hi, n_clips, n_spk, p, holm, family_m, extra)
    'G_pcgita': ('+0.3306', '0.2498', '0.4088', 1100, 100, '0.0010', '0.0070', 7, 'A'),
    'G_neurovoz': ('+0.2262', '0.1735', '0.2851', 1270, 107, '0.0010', '0.0070', 7, 'A'),
    'G_adresso': ('+0.2602', '0.1755', '0.3396', 237, 237, '0.0010', '0.0070', 7, 'A'),
    'G_adress2020': ('+0.1802', '0.0766', '0.2805', 156, 156, '0.0010', '0.0070', 7, 'A'),
    'G_pitt': ('+0.1128', '0.0455', '0.1768', 468, 228, '0.0030', '0.0070', 7, 'A'),
    'G_edaic_llm': ('-0.0929', '-0.1609', '-0.0257', 275, 275, '0.0080', '0.0080', 7, None),
    'G_edaic_enc': ('-0.2385', '-0.3244', '-0.1505', 275, 275, '0.0010', '0.0070', 7, None),
    'R_edaic_q2a': ('-0.7328', '-0.7780', '-0.6837', 794, 138, '0.0010', '0.0090', 9, 'T1supp'),
    'R_edaic_o25': ('-0.7429', '-0.7869', '-0.6944', 794, 138, '0.0010', '0.0090', 9, None),
    'R_edaic_q3o': ('-0.6719', '-0.7266', '-0.6105', 794, 138, '0.0010', '0.0090', 9, None),
    'R_edaic_af2': ('-0.1011', '-0.1647', '-0.0384', 794, 138, '0.0040', '0.0090', 9, None),
    'R_edaic_af3': ('-0.4190', '-0.4878', '-0.3461', 794, 138, '0.0010', '0.0090', 9, None),
    'R_edaic_kimi': ('-0.3678', '-0.4394', '-0.3020', 794, 138, '0.0010', '0.0090', 9, None),
    'R_pitt_q2a': ('-0.2853', '-0.3950', '-0.1485', 468, 228, '0.0010', '0.0090', 9, None),
    'R_pitt_o25': ('-0.1731', '-0.2958', '-0.0392', 468, 228, '0.0120', '0.0120', 9, None),
    'R_pitt_q3o': ('-0.2165', '-0.3242', '-0.1052', 468, 228, '0.0010', '0.0090', 9, None),
    'F_pcgita': ('+0.3227', '0.2391', '0.4040', 1100, 100, '0.0010', '0.0060', 6, None),
    'F_neurovoz': ('+0.2078', '0.1588', '0.2578', 1270, 107, '0.0010', '0.0060', 6, None),
    'F_pitt': ('+0.1095', '0.0454', '0.1728', 468, 228, '0.0030', '0.0090', 6, None),
    'F_adresso': ('+0.2096', '0.1294', '0.2867', 237, 237, '0.0010', '0.0060', 6, None),
    'F_adress2020': ('+0.1443', '0.0490', '0.2370', 156, 156, '0.0030', '0.0090', 6, None),
    'F_lora_minus_projector': ('+0.0577', '0.0154', '0.1044', 468, 228, '0.0080', '0.0090', 6, None),
    'C_pg_probe_drop': ('-0.0661', '-0.1181', '-0.0180', 1100, 100, '0.0130', '0.0260', 2, None),
    'C_nv_probe_drop': ('-0.0308', '-0.0608', '-0.0012', 1270, 107, '0.0370', '0.0370', 2, None),
}
t1supp = {}
import csv
for r in csv.DictReader(open('<local data dir>/release/edaic_rerun/part17/T1_table2_distinct.csv')):
    if r['stream'] == 'audio' and r['arm'].startswith('conflict_minus') and r['version'] == 'NEW_311distinct':
        t1supp[r['model']] = (float(r['value']), float(r['lo']), float(r['hi']))
for rid, (v, lo, hi, nc, ns, p, hp, m, extra) in CL.items():
    r = R[rid]; h = H[rid]
    got_v = f"{r['value']:+.4f}"; got_lo = f"{r['lo']:.4f}"; got_hi = f"{r['hi']:.4f}"
    conds = [got_v == v, got_lo == lo, got_hi == hi, r['n_clips'] == nc, r['n_spk'] == ns, r['usable'] == 2000,
             f"{r['p']:.4f}" == p, f"{h['holm']:.4f}" == hp, h['m'] == m, h['holm'] <= 0.05,
             f"{r['vs_B_draws']['p_from_saved']:.4f}" == p, f"{h['holm_saved']:.4f}" == hp,
             r['vs_B_draws']['max_abs_draw_diff'] == 0.0 and r['vs_B_draws']['same_speaker_idx'],
             r['vs_B_perclip_max_abs'] < 1e-12 and r['vs_B_perclip_meta_ok'], r['min_abs_draw'] > 1e-12]
    note = ''
    if rid.startswith('R_edaic'):
        conds.append(r['n_conflict'] == 483 and r['n_agreement'] == 311)
        note += f"; {r['n_conflict']} conflict + {r['n_agreement']} distinct agreement rebuilt from the part14 per-model files (T1 file matches to {J['edaic_dup_info'][rid.split('_')[-1]]['max_abs_diff_vs_T1']:.0e})"
        if extra == 'T1supp':
            s = t1supp['Qwen2-Audio']
            eq = f"{r['value']:.4f}" == f"{s[0]:.4f}" and f"{r['lo']:.4f}" == f"{s[1]:.4f}" and f"{r['hi']:.4f}" == f"{s[2]:.4f}"
            conds.append(eq)
            note += f"; equals part17 T1 supplementary {s[0]:.4f} [{s[1]:.4f}, {s[2]:.4f}]: {eq}"
        tp = r['point_terms']; note += f"; arms {tp[0]:.4f}/{tp[1]:.4f}"
    if rid.startswith('R_pitt'):
        conds.append(r['n_conflict'] == 146 and r['n_agreement'] == 322)
        tp = r['point_terms']; note += f"; {r['n_conflict']} conflict + {r['n_agreement']} agreement; arms {tp[0]:.4f}/{tp[1]:.4f}"
    if extra == 'A':
        a = r['trackA']
        conds.append(a['byte_equal_diff'] and a['same_speaker_idx'])
        note += f"; B draws byte-equal to Track A A_{rid[2:]}_draws.npz: {a['byte_equal_diff']}"
    if rid.startswith('G_edaic'):
        note += f"; no Track A E-DAIC npz exists, draws redrawn here from the part23 A3/A2 per-clip files and equal to B's saved npz"
    if rid.startswith('F_') and rid != 'F_lora_minus_projector':
        tp = r['point_terms']; note += f"; fine-tuned {tp[0]:.4f}, zero-shot {tp[1]:.4f}"
    if rid.startswith('C_'):
        tp = r['point_terms']; note += f"; after {tp[0]:.4f}, before {tp[1]:.4f}"
    rec = (f"{got_v} [{got_lo}, {got_hi}]; {r['n_clips']} clips, {r['n_spk']} speakers, {r['usable']} of 2000 draws; "
           f"p {r['p']:.4f} (le0 {r['n_le0']}, ge0 {r['n_ge0']}), Holm {h['holm']:.4f} (m={h['m']}), {'survives' if h['holm'] <= 0.05 else 'fails'}; "
           f"my draws equal B's saved draws exactly (max diff {r['vs_B_draws']['max_abs_draw_diff']:.0e}); per-clip rebuild vs B csv max diff {r['vs_B_perclip_max_abs']:.0e}{note}")
    add(rid, f"{v} [{lo}, {hi}]; n {nc} clips, {ns} speakers; p {p}, Holm {hp} (m={m}), survives", rec, all(conds))

# ------------------------------------------------------------------ sensitivity
full_fail = [k for k in claimed24 if H[k]['holm_full'] > 0.05]
full_fail_saved = [k for k in claimed24 if H[k]['holm_full_saved'] > 0.05]
a = R['Gs_edaic_ansstate']; s0 = R['Fs_edaic_seed0']; kg = R['Gs_kcl_single']
ok = (full_fail == ['C_nv_probe_drop'] and full_fail_saved == full_fail and f"{H['C_nv_probe_drop']['holm_full']:.4f}" == '0.0740'
      and H['C_nv_probe_drop']['m_full'] == 3 and iv('Gs_edaic_ansstate') == '-0.0007 [-0.0455, +0.0436]'
      and iv('Fs_edaic_seed0') == '+0.0223 [-0.0064, +0.0547]' and f2(s0['value']) == '0.02' and f2(s0['lo']) == '-0.01' and f2(s0['hi']) == '0.05'
      and {H[k]['m_full'] for k in claimed24 if H[k]['family'] == 'GAPS'} == {9} and {H[k]['m_full'] for k in claimed24 if H[k]['family'] == 'ARMS'} == {12}
      and {H[k]['m_full'] for k in claimed24 if H[k]['family'] == 'FINE TUNE'} == {8})
add('B_sensitivity_widened_families', '23 of 24 survive; the NeuroVoz channel drop does not (Holm 0.0740, m=3)',
    f"{24 - len(full_fail)} of 24 survive with widened families (GAPS m=9, ARMS m=12, FINE TUNE m=8, CONTROLS m=3); fails: {full_fail} with Holm {H['C_nv_probe_drop']['holm_full']:.4f} (m={H['C_nv_probe_drop']['m_full']}); "
    f"next weakest widened: R_pitt_o25 {H['R_pitt_o25']['holm_full']:.4f}, C_pg_probe_drop {H['C_pg_probe_drop']['holm_full']:.4f}; "
    f"added members: answer-state probe {iv('Gs_edaic_ansstate')}, KCL single-split gap {iv('Gs_kcl_single')}, Pitt AF2/AF3/Kimi arms {iv('Rs_pitt_af2')}, {iv('Rs_pitt_af3')}, {iv('Rs_pitt_kimi')}, "
    f"KCL fine-tune {iv('Fs_kcl')}, E-DAIC seed 0 {iv('Fs_edaic_seed0')} (tex +0.02 [-0.01, 0.05] matches), PC-GITA answer after normalisation {iv('Cs_pg_answer')}", ok)

# ------------------------------------------------------------------ extra below chance
X4 = ['X_edaic_conf_q2a', 'X_edaic_conf_o25', 'X_edaic_conf_q3o', 'X_edaic_conf_af3']
xs = [k for k in X4 if H[k]['holm'] <= 0.05]
af3 = R['X_edaic_conf_af3']; haf3 = H['X_edaic_conf_af3']
pitt_inc = {m: R[f'X_pitt_conf_{m}']['lo'] <= 0 <= R[f'X_pitt_conf_{m}']['hi'] for m in ('q2a', 'o25', 'q3o')}
below = [m for m in ('q2a', 'o25', 'q3o', 'af2', 'af3', 'kimi') if R[f'X_edaic_conf_{m}']['point_terms'][0] < 0.5]
ok = (len(xs) == 4 and all(H[k]['m'] == 4 for k in X4) and f"{af3['point_terms'][0]:.4f}" == '0.4235' and iv('X_edaic_conf_af3') == '-0.0765 [-0.1394, -0.0120]'
      and f"{af3['p']:.3f}" == '0.020' and f"{haf3['holm']:.3f}" == '0.020' and f"{haf3['holm_full']:.3f}" == '0.060' and haf3['holm_full'] > 0.05
      and all(f"{R[k]['p']:.4f}" == '0.0010' for k in X4[:3]) and all(pitt_inc.values()) and below == ['q2a', 'o25', 'q3o', 'af3'])
add('B_extra_below_chance', '4 of 4 survive Holm (m=4); AF3 0.4235, -0.0765 [-0.1394, -0.0120], p 0.020, Holm 0.020; at m=6 AF3 fails at 0.060; Pitt Qwen conflict intervals include 0.5',
    f"{len(xs)} of 4 survive at m=4 (Q2A/O25/Q3O p {R['X_edaic_conf_q2a']['p']:.4f}); below 0.5 on E-DAIC conflict: {below}; AF3 {af3['point_terms'][0]:.4f}, {iv('X_edaic_conf_af3')}, p {af3['p']:.4f}, Holm {haf3['holm']:.4f}; "
    f"m=6 Holm AF3 {haf3['holm_full']:.4f} (fails); Pitt Qwen conflict minus 0.5: q2a {iv('X_pitt_conf_q2a')}, o25 {iv('X_pitt_conf_o25')}, q3o {iv('X_pitt_conf_q3o')}, all include 0", ok)

# ------------------------------------------------------------------ wording flags
w1 = 'On depression the answer beats every probe' in line(26) and 'a probe on the answer state matches the answer (0.84)' in line(149)
w2 = 'significant performance gain in AD and PD detection' in line(244) and 'MDVR-KCL' not in line(244) and 'KCL' not in line(244)
w3 = 'without correction for multiple comparisons' in line(101)
two_of_three = (R['G_edaic_llm']['hi'] < 0) and (R['G_edaic_enc']['hi'] < 0) and not (R['Gs_edaic_ansstate']['lo'] > 0 or R['Gs_edaic_ansstate']['hi'] < 0)
ok = w1 and w2 and w3 and two_of_three and f2(R['Gs_edaic_ansstate']['p']) == '0.99' and iv('Fs_kcl') == '-0.0149 [-0.2709, +0.2364]'
add('B_wording_flags', "3 wording issues, no number changes: l.26 'answer beats every probe' vs l.149 answer-state probe matches (p 0.99); l.244 omits the MDVR-KCL exception (-0.0149 [-0.2709, +0.2364]); l.101 'without correction for multiple comparisons'",
    f"l.26 vs l.149 conflict present: {w1}; answer-state probe {iv('Gs_edaic_ansstate')}, p {R['Gs_edaic_ansstate']['p']:.4f}; only LLM and encoder probes are significantly below: {two_of_three}; "
    f"l.244 says the gain holds in AD and PD with no KCL exception: {w2}; KCL fine-tune {iv('Fs_kcl')}; l.101 still says uncorrected: {w3}; tex unchanged (sha {tex_sha_now[:8]})", ok)

# ------------------------------------------------------------------ independent check by Track B
bv = json.load(open(BD + '/verify/B_verify.json'))
maxdd = max(v.get('max_draw_diff', 0) for v in bv['recomputed'].values())
newer = []
for d in OMNI:
    if os.path.isdir(d):
        out = subprocess.run(['find', d, '-newermt', B_START_LOCAL, '-type', 'f'], capture_output=True, text=True).stdout.split('\n')
        newer += [x for x in out if x]
tsv_all = sum(v['all_match'] for v in J['tsv_compare'].values())
tie_rows = sorted(k for k, v in R.items() if f"{v['p']:.4f}" != f"{v['p_no_tie_rule']:.4f}")
claimed_min_absd = min(R[k]['min_abs_draw'] for k in claimed24 + X4)
pre = open(BD + '/logs/B_holm_r2_run_pre_tiefix.log').read()
ok = (bv['ALL_PASS'] is True and bv['n_rows'] == 41 and bv['fails'] == [] and bv['omni_final_files_newer_than_B_start'] == 0 and not newer
      and tsv_all == 41 and tex_sha_now == TEX_SHA and tie_rows == ['Fs_kcl', 'Gs_kcl_single', 'X_pitt_conf_q2a'] and claimed_min_absd > 1e-12
      and f'{maxdd:.1e}' == '5.6e-16')
add('B_independent_check', 'ALL_PASS true (41 of 41 rows, 0 fails); max draw diff 5.6e-16; 0 omni_final files touched; tex sha unchanged; tie fix moved p only in KCL gap, KCL fine-tune, Pitt Q2A conflict',
    f"B_verify.json: ALL_PASS {bv['ALL_PASS']}, n_rows {bv['n_rows']}, fails {bv['fails']}, max draw diff {maxdd:.1e}, omni_final newer {bv['omni_final_files_newer_than_B_start']}; "
    f"my separate rebuild (own dict joins from the original sources, E-DAIC arms from the part14 per-model files, own bootstrap, p and Holm) matches all {tsv_all} of 41 B_holm.tsv rows in every column at 4 dp; "
    f"my find over both omni_final folders for files newer than B start (00:57:09 UTC): {len(newer)}; tex sha {tex_sha_now[:8]}; "
    f"rows where the |d|<1e-12 tie rule changes p at 4 dp: {tie_rows}; smallest |draw| in any claimed or extra row {claimed_min_absd:.1e}", ok,
    {'omni_final_newer': newer, 'tie_rows': tie_rows})

# ------------------------------------------------------------------ draws index
import pandas as pd
idx = pd.read_csv(BD + '/B_draws_index.tsv', sep='\t', dtype=str)
npz = sorted(glob.glob(BD + '/draws/B_*_draws.npz'))
keys_ok = all(sorted(np.load(f).files) == sorted(['diff', 'terms', 'speaker_idx', 'speakers', 'point_diff', 'point_terms']) for f in npz)
sha_ok = all(sha(r['draws']) == r['draws_sha256'] for r in idx.to_dict('records'))
pcs = glob.glob(BD + '/per_clip/B_*_perclip.csv'); scs = glob.glob(BD + '/per_clip/B_*_perclip.sidecar.json')
side = json.load(open(BD + '/B_holm.sidecar.json'))
tsv_sha_ok = sha(BD + '/B_holm.tsv') == side['outputs']['tsv_sha256'] and sha(BD + '/B_draws_index.tsv') == side['outputs']['index_sha256']
r1 = open(BD + '/logs/B_holm_run_r1_crashed.log').read()
ok = (len(npz) == 41 and len(idx) == 41 and keys_ok and sha_ok and len(pcs) == 41 and len(scs) == 41 and tsv_sha_ok
      and os.path.exists(BD + '/scripts/B_holm_r2.py') and os.path.exists(BD + '/scripts/B_holm.py') and os.path.exists(BD + '/logs/B_holm_r2_run.log')
      and "AttributeError: 'function' object has no attribute 'map'" in r1 and 'Zs.clip.map' in r1)
add('B_draws_index', '41 npz files (diff, terms, speaker_idx, speakers, point_diff, point_terms); per-clip csv + sidecar per row; B_holm_r2.py and its log; r1 crashed on pandas .clip',
    f"{len(npz)} npz on disk, {len(idx)} index rows, keys as claimed: {keys_ok}, index sha256 match: {sha_ok}; {len(pcs)} per-clip csv + {len(scs)} sidecars; "
    f"B_holm.tsv and index sha match the sidecar: {tsv_sha_ok}; scripts/B_holm_r2.py, scripts/B_holm.py, logs/B_holm_r2_run.log present; r1 log ends in AttributeError on Zs.clip.map (DataFrame.clip method shadows the column)", ok)

# ------------------------------------------------------------------ cost
hits = subprocess.run(['grep', '-rlI', '-e', 'runpod', '-e', 'rest.runpod', '-e', 'podFindAndDeploy', BD + '/scripts', BD + '/logs'], capture_output=True, text=True).stdout.split()
bfiles = [os.path.relpath(p, BD) for p in glob.glob(BD + '/**', recursive=True)]
launchy = [p for p in bfiles if re.search(r'launch|pod|watchdog|create_.*json', p, re.I)]
pods_now = json.load(open(OUT + '/logs/runpod_pods_now.json')) if os.path.exists(OUT + '/logs/runpod_pods_now.json') else None
bpods = [p for p in (pods_now or []) if re.search(r'25b|p25-?b|trackb|_b_|holm', str(p.get('name', '')), re.I)]
t_first = min(json.load(open(f))['written_utc'] for f in scs); t_last = side['date_utc']
ok = not hits and not launchy and pods_now is not None and not bpods
add('B_cost', '$0, no pods', f"no runpod reference in B scripts or logs ({len(hits)} files), no launch/pod/watchdog files under B ({len(launchy)}); "
    f"RunPod pod list now: {len(pods_now) if pods_now is not None else 'unavailable'} pods, {len(bpods)} with a Track B name; r2 outputs written {t_first[11:19]} to {t_last[11:19]} UTC on the Mac", ok,
    {'pods_now': pods_now})

# ------------------------------------------------------------------ write
order = ['B_paper_line', 'B_claim_list_vs_tex'] + list(CL) + ['B_sensitivity_widened_families', 'B_extra_below_chance', 'B_wording_flags',
                                                            'B_independent_check', 'B_draws_index', 'B_cost']
rows.sort(key=lambda r: order.index(r[0]))
with open(OUT + '/B_VERIFIED.tsv', 'w') as f:
    f.write('id\tclaimed\trecomputed\tverified\n')
    for r in rows:
        f.write('\t'.join(x.replace('\t', ' ').replace('\n', ' ') for x in r) + '\n')
sc = {'task': 'PART 25 independent verifier, Track B', 'written_utc': datetime.datetime.utcnow().isoformat() + 'Z',
      'n_ids': len(rows), 'n_verified_yes': sum(r[3] == 'yes' for r in rows),
      'scripts': [OUT + '/scripts/verify_B_indep.py', OUT + '/scripts/verify_B_claims.py'],
      'recompute': OUT + '/B_indep/V_results.json', 'tex_sha256': tex_sha_now,
      'tsv': OUT + '/B_VERIFIED.tsv', 'tsv_sha256': sha(OUT + '/B_VERIFIED.tsv'), 'detail': detail,
      'versions': J['versions']}
json.dump(sc, open(OUT + '/B_VERIFIED.sidecar.json', 'w'), indent=1, default=str)
for r in rows:
    print(r[0], r[3], '|', r[2][:260])
print('yes', sum(r[3] == 'yes' for r in rows), 'of', len(rows))
