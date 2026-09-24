#!/usr/local/bin/python3
"""Compare track C claims with the verifier's own recomputation (vC_cells.json) and write C_VERIFIED.tsv.
Claims are typed in below from the track C return value (not read from track C files).
usage: /usr/local/bin/python3 vC_compare.py
"""
import json, csv, time, os
ROOT = '<local data dir>/release_from_mac/scores/part25'
VOUT = ROOT + '/verify/C'
TSV = ROOT + '/verify/C_VERIFIED.tsv'

# id: (probe, answer, diff, lo, hi, n_clips, n_spk, probe_row, answer_row)
CLAIM = {
    'C_q2a_pcgita': ('0.9097', '0.5369', '+0.3728', '+0.3104', '+0.4271', 1100, 100, 72, 73),
    'C_q2a_neurovoz': ('0.9324', '0.5519', '+0.3805', '+0.3278', '+0.4325', 1270, 107, 68, 69),
    'C_q2a_kcl': ('0.7714', '0.5923', '+0.1791', '-0.0503', '+0.4305', 37, None, 64, 65),
    'C_q2a_edaic': ('0.5673', '0.6464', '-0.0791', '-0.1664', '+0.0120', 275, None, 52, 54),
    'C_q2a_pitt': ('0.7846', '0.6173', '+0.1673', '+0.0944', '+0.2432', 468, 228, 76, 78),
    'C_q2a_adresso': ('0.8537', '0.6649', '+0.1888', '+0.1154', '+0.2682', 237, None, 46, 48),
    'C_q2a_adress2020': ('0.8315', '0.6824', '+0.1491', '+0.0640', '+0.2368', 156, None, 41, 43),
    'C_q3o_pcgita': ('0.8969', '0.5967', '+0.3002', '+0.2298', '+0.3695', 1100, 100, 169, 172),
    'C_q3o_neurovoz': ('0.9372', '0.6393', '+0.2979', '+0.2360', '+0.3578', 1270, 107, 163, 166),
    'C_q3o_kcl': ('0.7661', '0.7738', '-0.0077', '-0.1618', '+0.1385', 37, None, 157, 160),
    'C_q3o_edaic': ('0.5828', '0.6290', '-0.0462', '-0.1412', '+0.0527', 275, None, 144, 147),
    'C_q3o_pitt': ('0.8122', '0.7619', '+0.0503', '-0.0087', '+0.1095', 468, 228, 175, 178),
    'C_q3o_adresso': ('0.8750', '0.7774', '+0.0976', '+0.0362', '+0.1616', 237, None, 137, 140),
    'C_q3o_adress2020': ('0.8445', '0.7097', '+0.1348', '+0.0417', '+0.2335', 156, None, 131, 134),
    'C_kimi_pcgita': ('0.8715', '0.6147', '+0.2568', '+0.1839', '+0.3284', 1100, 100, 33, 35),
    'C_kimi_neurovoz': ('0.9334', '0.6204', '+0.3130', '+0.2556', '+0.3681', 1270, 107, 30, 32),
    'C_kimi_kcl': ('0.7310', '0.7113', '+0.0197', '-0.1643', '+0.2141', 37, None, 27, 29),
    'C_kimi_edaic': ('0.4840', '0.6293', '-0.1453', '-0.2334', '-0.0454', 275, None, 23, 25),
    'C_kimi_pitt': ('0.8374', '0.7180', '+0.1194', '+0.0587', '+0.1788', 468, 228, 36, 38),
    'C_kimi_adresso': ('0.8993', '0.7531', '+0.1462', '+0.0821', '+0.2096', 237, None, 20, 22),
    'C_kimi_adress2020': ('0.9200', '0.7053', '+0.2147', '+0.1331', '+0.3032', 156, None, 17, 19),
}
PITT_ALT = ('-0.0058', '+0.1108')
# supplement: ds -> (probe mean5, diff, lo, hi)
SUPP = {'pcgita': ('0.8768', '+0.2621', '+0.1810', '+0.3364'), 'neurovoz': ('0.9319', '+0.3115', '+0.2515', '+0.3674'),
        'kcl': ('0.7631', '+0.0518', '-0.1637', '+0.2901'), 'edaic': ('0.5823', '-0.0469', '-0.1393', '+0.0533'),
        'pitt': ('0.8112', '+0.0932', '+0.0331', '+0.1547'), 'adresso': ('0.8682', '+0.1151', '+0.0481', '+0.1831')}
PDAD = ['pcgita', 'neurovoz', 'kcl', 'pitt', 'adresso', 'adress2020']


def f4(x):
    return '%.4f' % x


def s4(x):
    return '%+.4f' % x


def main():
    D = json.load(open(VOUT + '/vC_cells.json'))
    cells = D['cells']
    M = {('C_%s_%s' % (c['tag'], c['ds'])): c for c in cells if c['kind'] == 'master'}
    S = {c['ds']: c for c in cells if c['kind'] == 'supp'}
    out = []
    for cid, cl in CLAIM.items():
        c = M[cid]
        cp, ca, cd, clo, chi, cn, cs, cpr, car = cl
        checks = []
        checks.append(('probe_row', c['prow'] == cpr))
        checks.append(('answer_row', c['arow'] == car))
        checks.append(('master_probe', f4(c['pmaster']) == cp))
        checks.append(('master_answer', f4(c['amaster']) == ca))
        checks.append(('probe_recomputed', f4(c['probe_joined']) == cp))
        checks.append(('probe_own_file', f4(c['probe_mean5_own_file']) == cp))
        checks.append(('per_repeat_eq_original_4dp', c['per_repeat_equal_ref_4dp']))
        checks.append(('answer_recomputed', f4(c['answer_joined']) == ca))
        checks.append(('answer_own_file', f4(c['answer_auc_own_file']) == ca))
        if c.get('ans_json_auc') is not None:
            checks.append(('answer_json_auc', f4(float(c['ans_json_auc'])) == ca))
        checks.append(('diff_master', s4(round(c['pmaster'] - c['amaster'], 4)) == cd))
        # claimed diff = difference of the two 4dp master values; the unrounded diff may differ by one unit in the 4th dp
        checks.append(('diff_point_within_rounding', abs(c['diff_point'] - float(cd)) <= 0.0001 + 1e-9))
        checks.append(('lo', s4(c['lo']) == clo))
        checks.append(('hi', s4(c['hi']) == chi))
        checks.append(('lo_hi_rankAUC', s4(c['lo_rank']) == clo and s4(c['hi_rank']) == chi))
        checks.append(('n_clips', c['n_joined'] == cn and c['pn'] == cn and c['an'] == cn))
        checks.append(('no_lost_clips', c['probe_only'] == 0 and c['answer_only'] == 0 and c['label_disagree'] == 0
                       and c['probe_dup'] == 0 and c['answer_dup'] == 0))
        checks.append(('n_spk', (cs is None) or c['n_spk'] == cs))
        checks.append(('usable_2000', c['usable'] == 2000))
        if cid == 'C_q3o_pitt':
            checks.append(('alt_padded_ids', s4(c.get('alt_answer_ids_lo', 9)) == PITT_ALT[0] and s4(c.get('alt_answer_ids_hi', 9)) == PITT_ALT[1]))
        if cid.startswith('C_kimi'):
            checks.append(('no_kimi_encoder_row', c['n_encoder_rows'] == 0))
        fails = [k for k, v in checks if not v]
        rec = ('probe %s vs answer %s, diff %s [%s, %s]; n %d clips, %d speakers; per-repeat %s (orig %s); '
               'draws vs track maxabs %s; track perclip maxabs %s; master rows %d/%d') % (
            f4(c['probe_joined']), f4(c['answer_joined']), s4(c['diff_point']), s4(c['lo']), s4(c['hi']), c['n_joined'],
            c['n_spk'], [round(x, 4) for x in c['per_repeat']], c['ref_per'], c['track_draws_maxabs'],
            c['track_perclip_maxabs'], c['prow'], c['arow'])
        if cid == 'C_q3o_pitt':
            rec += '; padded answer-file ids [%s, %s]' % (s4(c['alt_answer_ids_lo']), s4(c['alt_answer_ids_hi']))
        if s4(c['diff_point']) != cd:
            rec += '; note: claimed diff %s is master 4dp probe minus master 4dp answer; unrounded diff %.6f rounds to %s' % (cd, c['diff_point'], s4(c['diff_point']))
        if fails:
            rec += '; FAILED: ' + ','.join(fails)
        claimed = 'probe %s vs answer %s, diff %s [%s, %s], n %s%s, rows %d/%d' % (cp, ca, cd, clo, chi, cn,
                  (' clips, %d speakers' % cs) if cs else '', cpr, car)
        out.append((cid, claimed, rec, 'yes' if not fails else 'no'))
    # counts
    def cnt(keys):
        above = [k for k in keys if M[k]['diff_point'] > 0]
        excl = [k for k in above if M[k]['lo'] > 0 or M[k]['hi'] < 0]
        return len(keys), len(above), len(excl), [k for k in keys if k not in above], [k for k in keys if not (M[k]['lo'] > 0 or M[k]['hi'] < 0)]
    q = ['C_%s_%s' % (t, d) for t in ('q2a', 'q3o') for d in PDAD]
    k = ['C_kimi_%s' % d for d in PDAD]
    nq, aq, eq, belowq, inclq = cnt(q)
    nk, ak, ek, belowk, inclk = cnt(k)
    na, aa, ea, belowa, incla = cnt(q + k)
    ok = (nq, aq, eq) == (12, 11, 9) and (nk, ak, ek) == (6, 6, 5) and (na, aa, ea) == (18, 17, 14) \
        and belowa == ['C_q3o_kcl'] and sorted(incla) == sorted(['C_q2a_kcl', 'C_q3o_kcl', 'C_q3o_pitt', 'C_kimi_kcl'])
    # the claim names the 3 intervals that include 0 among the 17 above; q3o_kcl (below) also includes 0
    incl_above = [x for x in incla if x not in belowa]
    ok = ok and sorted(incl_above) == sorted(['C_q2a_kcl', 'C_q3o_pitt', 'C_kimi_kcl'])
    all_master_ok = all(r[3] == 'yes' for r in out)
    out.append(('C_count_18_PDAD',
                'Qwen 12: 11 above, 9 of 11 exclude 0; Kimi LM 6: 6 above, 5 exclude 0; all 18: 17 above, 14 of 17 exclude 0; only q3o KCL below; above-but-includes-0: q2a KCL, q3o Pitt, Kimi KCL',
                'Qwen %d: %d above, %d exclude 0; Kimi LM %d: %d above, %d exclude 0; all %d: %d above, %d exclude 0; below: %s; above but include 0: %s' % (
                    nq, aq, eq, nk, ak, ek, na, aa, ea, belowa, incl_above),
                'yes' if (ok and all_master_ok) else 'no'))
    e = {t: M['C_%s_edaic' % t] for t in ('q2a', 'q3o', 'kimi')}
    se = S['edaic']
    ed_ok = all(e[t]['diff_point'] < 0 for t in e) and [t for t in e if e[t]['hi'] < 0 or e[t]['lo'] > 0] == ['kimi'] \
        and s4(se['diff_point']) == '-0.0469' and s4(se['lo']) == '-0.1393' and s4(se['hi']) == '+0.0533' \
        and all(r[3] == 'yes' for r in out if r[0].endswith('_edaic'))
    out.append(('C_edaic_summary',
                'answer beats probe for all three; q2a -0.0791 [-0.1664, +0.0120]; q3o -0.0462 [-0.1412, +0.0527]; Kimi LM -0.1453 [-0.2334, -0.0454] only one excluding 0; Kimi PART 26 enc -0.0469 [-0.1393, +0.0533]',
                'q2a %s [%s, %s]; q3o %s [%s, %s]; Kimi LM %s [%s, %s]; Kimi PART 26 enc' % tuple(
                    x for t in ('q2a', 'q3o', 'kimi') for x in (s4(e[t]['diff_point']), s4(e[t]['lo']), s4(e[t]['hi'])))
                + ' %s [%s, %s]' % (s4(se['diff_point']), s4(se['lo']), s4(se['hi'])),
                'yes' if ed_ok else 'no'))
    # supplement
    parts = []
    sok = True
    for ds, (cp, cd, clo, chi) in SUPP.items():
        c = S[ds]
        good = (f4(c['probe_joined']) == cp and s4(c['diff_point']) == cd and s4(c['lo']) == clo and s4(c['hi']) == chi
                and c['per_repeat_equal_ref_4dp'] and s4(c['p26_ci'][0]) == clo and s4(c['p26_ci'][1]) == chi
                and c['usable'] == 2000 and c['probe_only'] == 0 and c['answer_only'] == 0 and c['label_disagree'] == 0
                and f4(c['answer_joined']) == f4(c['amaster']))
        sok = sok and good
        parts.append('%s %s, %s [%s, %s]%s' % (ds, f4(c['probe_joined']), s4(c['diff_point']), s4(c['lo']), s4(c['hi']),
                                               '' if good else ' MISMATCH'))
    a20 = S['adress2020']
    parts.append('adress2020 %s (perclip found %d, sidecar found %d at %s)' % (
        a20.get('status'), len(a20.get('found_perclip', [])), len(a20.get('found_sidecar', [])), D['meta']['created_utc']))
    above5 = [ds for ds in SUPP if ds != 'edaic' and S[ds]['diff_point'] > 0]
    excl5 = [ds for ds in above5 if S[ds]['lo'] > 0 or S[ds]['hi'] < 0]
    parts.append('PD/AD landed: %d of 5 above, %d exclude 0' % (len(above5), len(excl5)))
    sok = sok and len(above5) == 5 and len(excl5) == 4 and a20.get('status') == 'NOT LANDED'
    out.append(('C_kimi_encoder_part26_supplement',
                'PC-GITA 0.8768 +0.2621 [+0.1810, +0.3364]; NeuroVoz 0.9319 +0.3115 [+0.2515, +0.3674]; KCL 0.7631 +0.0518 [-0.1637, +0.2901]; E-DAIC 0.5823 -0.0469 [-0.1393, +0.0533]; Pitt 0.8112 +0.0932 [+0.0331, +0.1547]; ADReSSo 0.8682 +0.1151 [+0.0481, +0.1831]; ADReSS-2020 NOT LANDED; 5 of 5 above, 4 exclude 0',
                '; '.join(parts), 'yes' if sok else 'no'))
    with open(TSV, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'claimed', 'recomputed', 'verified'])
        for r in out:
            w.writerow(r)
    side = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), tsv=TSV, cells_json=VOUT + '/vC_cells.json',
                recompute_meta=D['meta'], script=os.path.abspath(__file__), n_ids=len(out), n_yes=sum(r[3] == 'yes' for r in out))
    json.dump(side, open(ROOT + '/verify/C_VERIFIED.sidecar.json', 'w'), indent=1)
    for r in out:
        print(r[0], r[3], '|', r[2])


if __name__ == '__main__':
    main()
