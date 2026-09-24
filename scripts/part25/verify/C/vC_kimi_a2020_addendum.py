#!/usr/local/bin/python3
"""Addendum: Kimi-Audio ADReSS-2020 PART 26 encoder cell, which landed at 03:42Z after track C's NOT LANDED note.
Uses the verifier's own vC_recompute.py functions (not track C code).
usage: /usr/local/bin/python3 vC_kimi_a2020_addendum.py"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vC_recompute as V
V.P26_FILES['adress2020'] = ('Kimi-Audio_ADReSS-2020/per_clip/p26_kimi_adress2020_perclip.csv',
                             'Kimi-Audio_ADReSS-2020/per_clip/p26_kimi_adress2020_perclip.sidecar.json',
                             ['probe', 'per_repeat_auc'], ['gap', 'ci_2p5_97p5'])
rows = V.master_rows()
c = V.build_kimi_supp('adress2020', rows)
c = V.run_cell(c)
keys = ['arow', 'amaster', 'afile', 'pfile', 'probe_file_sha256', 'n_joined', 'probe_only', 'answer_only', 'label_disagree',
        'n_spk', 'per_repeat', 'ref_per', 'per_repeat_equal_ref_4dp', 'probe_joined', 'answer_joined', 'diff_point', 'lo', 'hi',
        'lo_rank', 'hi_rank', 'p26_ci', 'usable', 'sidecar_file', 'draws_file', 'perclip_file']
print(json.dumps({k: c.get(k) for k in keys}, indent=1, default=str))
