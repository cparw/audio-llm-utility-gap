#!/usr/bin/env python3
"""Independent check of the 9 PART 22 S1 rows whose note still says 'not yet verified'.
Reads the cited sources directly: podA SETUP3.log, the transformers 5.17.0 source unzipped under scores/part22/src,
the Hub preprocessor_config.json, windows_full.csv and the rebuild per-clip file full_zeroshot_scores.csv."""
import json, re, glob, sys, os
import numpy as np, pandas as pd
R = '<local data dir>/release'
out = {}
setup = open(R + '/edaic_rerun/logs/podA/SETUP3.log').read().split('\n')
out['tf_version'] = re.search(r'tf (\S+)', setup[14]).group(1) if len(setup) > 14 and 'tf ' in setup[14] else [l for l in setup if 'OK tf' in l]
fe = glob.glob(R + '/scores/part22/src/tf5170/**/feature_extraction_whisper.py', recursive=True)[0]
fl = open(fe).read().split('\n')
out['trunc_default_line196'] = fl[195].strip()
out['n_samples_line91'] = fl[90].strip()
pc = glob.glob(R + '/scores/part22/src/hf/**/preprocessor_config.json', recursive=True)[0]
cfg = json.load(open(pc)); out['chunk_length'] = cfg.get('chunk_length'); out['n_samples'] = cfg.get('chunk_length') * cfg.get('sampling_rate')
def tok(nsamp):
    frames = -(-nsamp // 160)  # ceil(L/160), the documented frame rule (config md lines 332-340)
    a = (frames - 1) // 2 + 1
    return (a - 2) // 2 + 1
out['tok_300s'] = int(tok(300 * 16000)); out['tok_900s_off'] = int(tok(900 * 16000))
wf = pd.read_csv(R + '/edaic_rerun/variants/windows_full.csv')
fz = pd.read_csv(R + '/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv')
wf['pid'] = wf[[c for c in wf.columns if c in ('pid', 'participant', 'speaker')][0]].astype(int)
fz['pid'] = fz['speaker'].astype(int)
m = fz.merge(wf, on='pid', suffixes=('', '_w'))
dcol = 'win_dur_s' if 'win_dur_s' in m.columns else [c for c in m.columns if 'dur' in c][0]
L = np.minimum(np.round(m[dcol].values * 16000).astype(np.int64), 4800000)
m['tok_formula'] = [tok(int(x)) for x in L]
out['n_windows'] = len(m); out['dur_col'] = dcol
out['formula_match'] = int((m['tok_formula'] == m['n_audio_tok']).sum())
gt = m[m[dcol] > 300.0]
out['gt300'] = len(gt); out['gt300_at_7500'] = int((gt['n_audio_tok'] == 7500).sum())
out['capped900'] = int(wf['capped'].sum())
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'my_verifier_p22s1_out.json'), 'w'), indent=1)
