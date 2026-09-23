#!/usr/bin/env python3
"""Build summary_all.csv, summary_paper.md and summary_counts.json from the verified
row fragments of Parts 16, 17, 20 and 22. Reads only; writes only to OUT."""
import csv, glob, json, os, re, sys, datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

R = '<local data dir>/release'
OUT = 'summary'
TEX = R + '/edaic_rerun/part17/part12_prep/pasted_tex_dryrun.tex'

# ---------------------------------------------------------------- load rows
def rd(path):
    with open(path, newline='') as f:
        lines = [l.rstrip('\r') for l in f.read().split('\n') if l.strip()]
    hdr = lines[0].split('\t')
    if hdr[0] != 'id':  # part16 POD1.tsv has no header row
        hdr = ['id', 'what', 'value', 'lo', 'hi', 'n', 'n_spk', 'file']; body = lines
    else:
        body = lines[1:]
    out = []
    for l in body:
        c = l.split('\t'); c += [''] * (len(hdr) - len(c))
        out.append(dict(zip(hdr, c)))
    return hdr, out

rows = []
def add(part, task, path, r, hdr):
    d = dict(part=part, task=task, id=r['id'], what=r['what'], lo=r.get('lo', ''), hi=r.get('hi', ''),
             n=r.get('n', ''), n_spk=r.get('n_spk', ''), file=r.get('file', ''), src=path)
    if 'value_verified' in hdr and 'value_computed' in hdr:
        d.update(vc=r['value_computed'], vv=r['value_verified'], agree_file=r.get('agree_4dp', ''), single=False)
    else:
        d.update(vc=r['value'], vv=r['value'], agree_file='', single=True)
    for k in ('lo', 'hi'):
        if d[k] in ('NA', '\u2014'): d[k] = ''
    rows.append(d)

for p in sorted(glob.glob(R + '/edaic_rerun/part16/rows/*.tsv')):
    h, rs = rd(p); [add('16', os.path.basename(p)[:-4], p, r, h) for r in rs]
md16 = open(R + '/edaic_rerun/part16/PART16_RESULTS.md').read().split('\n')
for i, l in enumerate(md16[10:27]):
    c = [x.strip() for x in l.strip().strip('|').split('|')]
    lo = hi = ''; extra = ''
    m = re.match(r'\[(-?[\d.]+),\s*(-?[\d.]+)\]$', c[3])
    if m: lo, hi = m.groups()
    elif c[3] not in ('\u2014', ''): extra = ' (' + c[3] + ')'
    rows.append(dict(part='16', task='SEED', id='SEED%02d_%s' % (i + 1, c[0]), what=c[1] + extra, lo=lo, hi=hi,
                     n=c[4], n_spk=c[5], file=c[6], src=R + '/edaic_rerun/part16/PART16_RESULTS.md (seed table, lines 11-27)',
                     vc=c[2], vv=c[2], agree_file='', single=True))
for p in sorted(glob.glob(R + '/edaic_rerun/part17/rows/*.tsv')):
    h, rs = rd(p); [add('17', os.path.basename(p)[:-4], p, r, h) for r in rs]
POD1B = R + '/scores/part20/rows/POD1b.tsv'
pod1b_present = os.path.exists(POD1B)
for p in sorted(glob.glob(R + '/scores/part20/rows/*.tsv')):
    if 'provisional' in p: continue
    h, rs = rd(p); [add('20', os.path.basename(p)[:-4], p, r, h) for r in rs]
p = R + '/scores/part22/rows/P22_verified.tsv'; h, rs = rd(p); [add('22', 'P22', p, r, h) for r in rs]

# unique ids where the fragment reuses one id for several values
from collections import Counter, defaultdict
cnt = Counter((r['part'], r['task'], r['id']) for r in rows); seen = Counter()
for r in rows:
    k = (r['part'], r['task'], r['id'])
    if cnt[k] > 1:
        seen[k] += 1
        slug = re.sub(r'[^A-Za-z0-9.]+', '_', r['what']).strip('_')[:70]
        r['uid'] = '%s#%02d:%s' % (r['id'], seen[k], slug)
    else:
        r['uid'] = r['id']

# ---------------------------------------------------------------- numbers
def num(s):
    try: return Decimal(str(s).strip().lstrip('+'))
    except (InvalidOperation, ValueError): return None
def r2(s):
    d = num(s)
    return None if d is None else d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

def lead(s):
    m = re.match(r'^\s*([+-]?\d[\d.]*(?:[eE][+-]?\d+)?)\s*(\[.*\])?\s*$', str(s))
    return num(m.group(1)) if m else None
def agree(r):
    if r['single']:
        return 'verified in fragment'
    flag = r['agree_file'].strip()
    fl = ' (file flag: %s)' % (flag if flag else 'blank')
    vv = r['vv'].strip()
    if vv == '' or vv.lower() in ('nan', 'none') or vv.startswith('not measured'):
        return 'MISSING verified value' + fl
    a, b = lead(r['vc']), lead(r['vv'])
    if a is not None and b is not None:
        return ('yes' if abs(a - b) < Decimal('0.00005') else 'NO') + fl
    if r['vc'].strip() == vv:
        return 'yes (text value)' + fl
    if flag.lower().startswith(('true', 'yes')):
        return 'yes (verifier worded the value differently)' + fl
    if r['vc'].strip() == '' and b is not None and flag.lower().startswith(('true', 'yes')):
        return 'yes (verified value only)' + fl
    return 'NO' + fl

# ---------------------------------------------------------------- paper claims
Q = {  # verbatim spans of the pasted tex (05:58Z), active text only
 'ABS_PROBE': ('abstract', 'Probes detect the disorders from the representations with AUC from 0.75 to 0.92, while the zero-shot answer sits well below on Parkinson\'s and Alzheimer\'s'),
 'ABS_BELOW': ('abstract', 'Where the words contradict the diagnosis, four of six models fall below chance.'),
 'ABS_FT': ('abstract', 'Retraining only the projector narrows the gap on Parkinson\'s and Alzheimer\'s.'),
 'T1_PCGITA': ('Table 1', 'Table 1 row "PC-GITA & 1100 (100) & 0.56 & 0.52 & 0.89"'),
 'T1_NV': ('Table 1', 'Table 1 row "NeuroVoz & 1270 (107) & 0.70 & 0.54 & 0.90"'),
 'T1_KCL': ('Table 1', 'Table 1 row "MDVR-KCL & 37 (37) & 0.71 & 0.76 & 0.70"'),
 'T1_EDAIC': ('Table 1', 'Table 1 row "E-DAIC & 275 (275) & 0.83 & 0.82 & 0.64"'),
 'T1_PITT': ('Table 1', 'Table 1 row "Pitt & 468 (228) & 0.66 & 0.66 & 0.77"'),
 'T1_ADRESSO': ('Table 1', 'Table 1 row "ADReSSo & 237 (237) & 0.62 & 0.55 & 0.82"'),
 'T1_ADR20': ('Table 1', 'Table 1 row "ADReSS-2020 & 156 (156) & 0.62 & 0.75 & 0.77"'),
 'T2_Q2A': ('Table 2', 'Table 2 row "Qwen2-Audio & 0.23 & 0.95 & 0.41 & 0.70"'),
 'T2_O25': ('Table 2', 'Table 2 row "Qwen2.5-Omni & 0.22 & 0.95 & 0.53 & 0.71"'),
 'T2_Q3O': ('Table 2', 'Table 2 row "Qwen3-Omni & 0.30 & 0.96 & 0.61 & 0.82"'),
 'T2_AF2': ('Table 2', 'Table 2 row "Audio Flamingo 2 & 0.53 & 0.61 & 0.63 & 0.55"'),
 'T2_AF3': ('Table 2', 'Table 2 row "Audio Flamingo 3 & 0.42 & 0.81 & 0.61 & 0.59", Pitt cells: decision pending (AF3 Pitt copy)'),
 'T2_KIMI': ('Table 2', 'Table 2 row "Kimi-Audio & 0.53 & 0.87 & 0.72 & 0.72"'),
 'R_ZS': ('results text', 'Zero-shot, Qwen2.5-Omni stays between 0.56 and 0.71 on the Parkinson\'s and Alzheimer\'s sets, and only E-DAIC is higher'),
 'R_PITT_TEXT': ('results text', 'For AD the transcript alone matches the audio on the Pitt dataset'),
 'R_PD_TEXT': ('results text', 'On the Parkinson\'s sets the transcript sits near chance, since every speaker reads the same sentences.'),
 'R_EDAIC_TEXT': ('results text', 'On E-DAIC the audio answer on the whole interview matches the transcript alone.'),
 'R_BESTPROBE': ('results text', 'The best probe achieves an AUC of 0.75 to 0.92 on the Parkinson\'s sets and 0.77 on the Pitt dataset; Pitt value: decision pending (Pitt encoder 0.7706 vs 0.7761)'),
 'R_Q2A_RANGE': ('results text', 'The same probes on Qwen2-Audio achieve 0.77 to 0.93 against answers of 0.54 to 0.68 on the Parkinson\'s and Alzheimer\'s sets.'),
 'R_Q3O_RANGE': ('results text', 'On Qwen3-Omni they obtain AUC scores of 0.77 to 0.94 against 0.60 to 0.78.'),
 'R_CHANNEL': ('results text', 'Recording features from the quietest frames separate the NeuroVoz groups at 0.87 by spectral centroid alone and PC-GITA at 0.69'),
 'R_LM080': ('results text', 'probing the language model\'s representations at the audio-token positions on Alzheimer\'s rises to 0.80, and the answer state itself achieves 0.77.'),
 'R_COS': ('results text', 'The cosine between the probe direction and the Yes minus No readout direction is 0.01 in magnitude, no larger than between two random directions.'),
 'R_REMOVE': ('results text', 'Removing the readout direction from the state leaves the probe at 0.77.'),
 'R_EDAIC_PROBE': ('results text', 'On E-DAIC the best probe is 0.69, at the language model, below the answer of 0.83.'),
 'R_BELOW': ('results text', 'On the depression segments four of the six models fall below chance when the words point away from the label, while every model except Audio Flamingo 2 scores above 0.80 when the words and the label agree.'),
 'R_EXCL0': ('results text', 'Speaker-level bootstrap intervals for the difference between the two arms exclude zero for all six models.'),
 'R_PITTDROP': ('results text', 'On the Pitt conflict segments the three Qwen models drop by 0.18 to 0.29 against their agreement segments, Kimi-Audio holds, and both Audio Flamingo models sit near chance on both sets'),
 'R_FT_DROP': ('results text', 'After fine-tuning the projector, AUC improves on every dataset except MDVR-KCL, which has 37 speakers, and E-DAIC, where it drops'),
 'R_FT_ARMS': ('results text', 'On the Pitt conflict segments the fine-tuned answer stays at 0.52, while the agreement segments rise to 0.86.'),
 'METHOD_PY': ('method text', 'so $p(\\text{Y})$ is the probability of \\emph{Yes} conditioned on the answer being one of the two.'),
 'INTRO_COND': ('method text', 'We perform these comparisons under three different conditions so that the diagnostic content of the spoken words differs between them'),
 'IMPL_EDAIC': ('implementation text', 'The model hears the participant\'s whole interview, capped at 15 minutes for the 89 longest.'),
 'FIG1': ('Figure 1 caption', 'Figure 1 caption: "The 32 audio encoder stages come first, then the projector, then the 29 language model stages read at the audio-token positions." Omitted-dataset sentence: 23 Sep addition, wording to be matched to the final tex (topic: whether NeuroVoz, MDVR-KCL, ADReSSo and ADReSS-2020 follow their condition\'s pattern)'),
 'CONC': ('conclusion', 'resulting in a significant performance gain in AD and MDD detection. (red span)'),
}
A23 = {  # 23 Sep additions named in the PART 21 brief from the authors (12:02Z) and 11:29Z decisions
 'A23_GAP': 'the paired gap intervals',
 'A23_Q3O': 'the Qwen3-Omni cells',
 'A23_DIR': 'the three backbone direction tests',
 'A23_CHAN': 'the channel normalisation',
 'A23_TXTPAIRS': 'the transcript on the pairs',
 'A23_PROMPT': 'the prompt wording and threshold checks',
 'A23_LABEL': 'the label cutoff check',
 'A23_YES': 'the yes rates',
 'A23_LORA': 'LoRA',
 'A23_GREEDY': 'the greedy decoding check',
 'A23_KCL': 'the KCL intervals',
 'A23_PITTARMS': 'Pitt transcript by arm on the Sep 20 file, so the sentence matches Table 1',
 'A23_EDAICPROBE': 'E-DAIC probe values 0.7001 / 0.5884 (mean of five) with paired intervals against 0.8278',
}
for k, v in A23.items():
    Q[k] = ('results text: 23 Sep additions', '23 Sep addition, wording to be matched to the final tex (topic: %s)' % v)

PENDING_ENC = 'decision pending (Pitt encoder 0.7706 vs 0.7761)'
PENDING_AF3 = 'decision pending (AF3 Pitt copy: part10 copy gives the printed 0.61 / 0.59, part3 copy cited by master_lookup gives 0.62 / 0.58)'

PV, SUP, REL, SUPS = 'PAPER VALUE', 'SUPPORT', 'RELEASED ONLY', 'SUPERSEDED / NOT USED'

def A(role, claims=(), printed=None, note='', pending=None):
    return dict(role=role, claims=list(claims), printed=printed, note=note, pending=pending)

T2P = {  # printed Table 2 cells: model -> (dep conflict, dep agree, pitt conflict, pitt agree)
 'Qwen2-Audio': ('0.23', '0.95', '0.41', '0.70', 'T2_Q2A'), 'Qwen2.5-Omni': ('0.22', '0.95', '0.53', '0.71', 'T2_O25'),
 'Qwen3-Omni-30B-A3B': ('0.30', '0.96', '0.61', '0.82', 'T2_Q3O'), 'Audio Flamingo 2': ('0.53', '0.61', '0.63', '0.55', 'T2_AF2'),
 'Audio Flamingo 3': ('0.42', '0.81', '0.61', '0.59', 'T2_AF3'), 'Kimi-Audio': ('0.53', '0.87', '0.72', '0.72', 'T2_KIMI')}
T1A = {'pitt': ('T1_PITT', '0.66'), 'adresso': ('T1_ADRESSO', '0.62'), 'adress2020': ('T1_ADR20', '0.62'),
       'pcgita': ('T1_PCGITA', '0.56'), 'neurovoz': ('T1_NV', '0.70'), 'kcl': ('T1_KCL', '0.71')}
SUP311 = 'Old 483-row agreement value (repeated segments); PART 21 brief from the authors: Table 2 E-DAIC agreement on 311 distinct segments, old 483-row values listed as superseded'
NOTE311 = 'decided 311 distinct agreement segments (PART 21 brief from the authors); the pasted tex still prints the old 483-row value'
SEP15 = 'Sep 15 transcript run; Table 1 prints the Sep 20 run and the authors (23 Sep 11:29Z) moved the by-arm sentence to the Sep 20 file'
AVGP = 'AUC of probabilities averaged over five repeats; decision by the authors 23 Sep: not used (paper uses mean of five per-repeat AUCs)'
IFREE = 'interviewer-free window result from PART 20; PART 21 brief from the authors lists it as released, not in the paper'
EW = 'E-DAIC window rerun (first/middle 30 s or middle 300 s); PART 21 brief from the authors lists it as released, not in the paper'

def classify(r):
    t, i, w, part = r['task'], r['id'], r['what'], r['part']
    # ------------------------------------------------ Part 16
    if part == '16':
        if t == 'EDAICFULL':
            if i == 'edaic_full_zeroshot_answer':
                return A(PV, ['T1_EDAIC', 'R_EDAIC_PROBE', 'A23_EDAICPROBE'], '0.83', 'rebuild run, 300 s processor cut; the original run 0.8285 also prints 0.83')
            if i.startswith('edaic_full_'): return A(SUPS, note=AVGP)
            return A(REL, note=EW)
        if t == 'M1':
            if i == 'M1_audio_all468': return A(PV, ['T1_PITT'], '0.66')
            if i == 'M1_audio_conflict': return A(PV, ['T2_O25'], '0.53')
            if i == 'M1_audio_agreement': return A(PV, ['T2_O25'], '0.71')
            if i == 'M1_audio_conflict_minus_agreement': return A(PV, ['R_PITTDROP', 'A23_GAP'], None, 'Qwen2.5-Omni Pitt drop; printed range starts at 0.18, this value rounds to 0.17')
            if i == 'M1_ALT_text_all468': return A(PV, ['T1_PITT'], '0.66', 'Sep 20 run, the run Table 1 prints')
            if i in ('M1_ALT_text_conflict', 'M1_ALT_text_agreement'): return A(PV, ['A23_PITTARMS'])
            return A(SUPS, note=SEP15)
        if t == 'M10':
            if 'transcript-only answer' in w: return A(PV, ['T1_KCL', 'A23_KCL', 'R_PD_TEXT'], '0.76', 'interval excludes 0.5 while the text says the Parkinson\'s transcript sits near chance')
            if 'audio zero-shot' in w: return A(PV, ['T1_KCL', 'A23_KCL'], '0.71')
            if 'encoder probe' in w: return A(PV, ['A23_KCL'], None, 'single saved OOF repeat; master_lookup Qwen2.5-Omni KCL encoder probe (mean of 5) is 0.7506')
            return A(PV, ['A23_KCL'])
        if t == 'M11':
            if w == 'auc_text_condition_recomputed': return A(PV, ['T1_KCL'], '0.76')
            if w == 'auc_audio_condition_recomputed': return A(PV, ['T1_KCL'], '0.71')
            if w == 'E_auc_from_passage_identity_alone': return A(REL, note='bears on "every speaker reads the same sentences": KCL speakers read two different passages (DISCREPANCIES M11)')
            return A(REL, note='MDVR-KCL ASR / WER analysis; not named for the paper')
        if t == 'M2': return A(REL, note='second scorer set (DistilBERT SST-2 rebuild); PART 21 brief from the authors: released, not in the paper')
        if t == 'M3':
            m = re.match(r'M3_(\w+?)_clip$', i)
            if m: c, p_ = T1A[m.group(1)]; return A(PV, [c, 'R_ZS'], p_)
            return A(REL, note='speaker-level AUC check')
        if t == 'M4':
            if w.startswith('yes_rate'): return A(PV, ['A23_YES'])
            return A(SUP, ['METHOD_PY'], note='answer mass P(Yes)+P(No)')
        if t == 'M5':
            if 'refit' in i: return A(SUPS, note='probe refit within the arm; not used (T3 marks the refit estimator not used)')
            if 'without_d' in i: return A(SUPS, note='first pass standardised with the training fold; replaced by T3 perp_z (DISCREPANCIES HIGH)')
            if 'randfloor' in i: return A(SUPS, note='first pass floor used an advancing generator; replaced by T3 reseeded floor')
            return A(SUP, ['A23_DIR'], note='same value as the T3 decided row (Qwen2-Audio direction test)')
        if t == 'M7':
            if i in ('M7_pitt_probe', 'M7_pitt_gap'): return A(SUPS, note='single-split Pitt probe file superseded; corrected in M7fix (0.7706, +0.1128)')
            if 'rep5' in i: return A(SUPS, note=AVGP)
            if i == 'M7_mean_gap': return A(SUPS, note='mean includes the superseded Pitt gap 0.1391 and the 30 s E-DAIC gap')
            ds = i.split('_')[1]
            if ds == 'edaic': return A(REL, note='first 30 s E-DAIC window; the paper E-DAIC row is the full window (authors 23 Sep 07:59Z)')
            if i.endswith('_zeroshot'): c, p_ = T1A[ds]; return A(PV, [c, 'R_ZS'], p_)
            mast = dict(adresso='0.8752', adress2020='0.8043', pcgita='0.8941', neurovoz='0.9213', kcl='0.7506')[ds]
            if i.endswith('_gap'): return A(PV, ['A23_GAP'], None, 'probe side is the single saved OOF split, not the mean-of-five value %s in master_lookup' % mast)
            return A(SUP, ['A23_GAP'], None, 'single saved OOF split; master_lookup mean-of-five encoder probe is %s' % mast)
        if t == 'M7fix':
            if 'paired' in w: return A(PV, ['A23_GAP'], None, 'Pitt probe minus zero-shot with the mean-of-five encoder 0.7706', PENDING_ENC)
            return A(PV, ['R_BESTPROBE', 'ABS_PROBE'], '0.77', 'mean of five per-repeat AUCs', PENDING_ENC)
        if t == 'M8':
            parts = [x.strip() for x in w.split('|')]
            model, sset, arm, src = parts[0], parts[1], parts[2], parts[3].replace('src=', '')
            key = 'Qwen3-Omni-30B-A3B' if model.startswith('Qwen3') else model
            pr = T2P[key]
            if sset.startswith('A_edaic'):
                if arm == 'conflict': return A(PV, [pr[4], 'R_BELOW', 'ABS_BELOW'], pr[0])
                return A(SUPS, note=SUP311)
            if src in ('af2_pitt_audio', 'alt_overnight_copy'): return A(SUPS, note='non-canonical Pitt copy; decision by the authors 23 Sep (SUPERSEDED.csv)')
            if key == 'Audio Flamingo 3':
                if arm == 'conflict_minus_agreement': return A(PV, ['R_PITTDROP'], None, 'part3 copy', PENDING_AF3)
                return A(PV, [pr[4]], pr[2] if arm == 'conflict' else pr[3], 'part3 copy', PENDING_AF3)
            if arm == 'conflict': return A(PV, [pr[4]], pr[2])
            if arm == 'agreement': return A(PV, [pr[4]], pr[3])
            cl = ['R_PITTDROP'] + (['A23_GAP'] if key.startswith('Qwen') else [])
            pnote = {'Qwen2.5-Omni': 'rounds to 0.17; printed range starts at 0.18', 'Qwen2-Audio': 'top of the printed 0.18 to 0.29 range',
                     'Qwen3-Omni-30B-A3B': 'inside the printed range', 'Kimi-Audio': '"Kimi-Audio holds"',
                     'Audio Flamingo 2': 'AF2 Pitt conflict interval lies above 0.5, which "near chance" does not describe'}[key]
            return A(PV, cl, None, pnote)
        if t == 'M9':
            if i in ('M9.CELLS', 'M9.FILES'): return A(REL, note='inventory count')
            return A(SUP, ['METHOD_PY'], note='answer mass P(Yes)+P(No)')
        if t == 'MEDAIC':
            if i == 'MEDAIC_full_answer': return A(PV, ['T1_EDAIC', 'R_ZS'], '0.83', 'original full-window run; the rebuild 0.8278 also prints 0.83')
            if i == 'MEDAIC_full_sft': return A(PV, ['T1_EDAIC', 'R_FT_DROP'], '0.64')
            if i == 'MEDAIC_full_sft_minus_answer': return A(PV, ['R_FT_DROP', 'A23_GAP', 'CONC'], None, 'E-DAIC fine-tune drop excludes zero; contradicts the conclusion red span "gain in ... MDD detection"')
            if i == 'MEDAIC_full_text': return A(PV, ['T1_EDAIC'], '0.82')
            if i == 'MEDAIC_full_text_minus_answer': return A(SUP, ['R_EDAIC_TEXT'])
            if i in ('MEDAIC_full_llm_nested_NOCI', 'MEDAIC_full_llm_minus_answer_POINT', 'MEDAIC_full_enc_nested_NOCI', 'MEDAIC_full_enc_minus_answer_POINT'):
                return A(SUPS, note='original-run json 0.6853 / 0.5930 superseded by 0.7001 / 0.5884 (SUPERSEDED.csv, decision by the authors 23 Sep)')
            if i == 'MEDAIC_answer_full_minus_30s': return A(REL, note=EW)
            if 'full_' in i: return A(REL, note='original-run answer-state / projector json aggregate, not printed')
            return A(REL, note=EW)
        if t == 'POD1':
            if i.startswith('1a_'):
                if 'agreement' in i: return A(SUPS, note=SUP311 + ' (transcript cell)')
                return A(PV, ['A23_TXTPAIRS'])
            if i.startswith('1f_'): return A(PV, ['A23_GREEDY'])
            if i == 'VALIDATION_conflict_auc': return A(PV, ['T2_O25'], '0.22')
            if i == 'VALIDATION_agreement_auc': return A(SUPS, note=SUP311)
            if i.startswith('1d_'): return A(PV, ['A23_LABEL'])
            if i.startswith('1e') or i.startswith('1b_'): return A(PV, ['A23_PROMPT'])
            if i.startswith('1c_'): return A(REL, note='sft_full projector retrain; does not reproduce 0.6421 because the original training audio is unrecoverable (DISCREPANCIES POD1 1c)')
        if t == 'POD2':
            k = i.split('.')[-1]
            if k == 'V1': return A(PV, ['T1_PITT', 'ABS_FT'], '0.77')
            if k == 'V2': return A(PV, ['T1_PITT'], '0.66')
            if k == 'V3': return A(PV, ['R_FT_ARMS'], '0.52')
            if k == 'V4': return A(PV, ['R_FT_ARMS'], '0.86')
            if k == 'V5': return A(SUP, ['R_FT_ARMS'])
            if k.startswith('X'): return A(SUPS, note='pre-existing LoRA file; master_lookup records the POD2 fresh run 0.8250 (PART 17, authors)')
            if k in ('P1', 'P2', 'L1', 'L2', 'L3', 'L4', 'D1', 'D2'): return A(PV, ['A23_LORA'])
            if k in ('P3', 'P4', 'L7', 'L8'): return A(SUP, ['A23_LORA'])
            if k == 'D3': return A(PV, ['A23_GAP', 'ABS_FT', 'CONC'], None, 'Pitt projector gain; supports the AD half of the conclusion')
            return A(REL, note='run cost / resource record')
        if t == 'POD3':
            if i == '3b_all_cos': return A(PV, ['A23_DIR'])
            if i in ('3b_all_auc_probe', '3b_all_auc_d'): return A(PV, ['A23_DIR'])
            if i == '3b_all_auc_wo_d': return A(SUPS, note='training-fold standardisation; canonical value is Q3Ofix 0.8276')
            if i.startswith('3b_'):
                if i.endswith('auc_d'): return A(SUP, ['A23_DIR'], note='same value as Q3Ofix')
                if 'wo_d' in i: return A(SUPS, note='training-fold standardisation; canonical value in Q3Ofix')
                return A(SUPS, note='probe refit within the arm; decision by the authors 23 Sep: not used (0.6394 / 0.9032 full fit restricted to the arm is used)')
            if i == '3a_pitt_zeroshot_shipped': return A(SUP, ['R_Q3O_RANGE'], note='shipped file; Table 2 Qwen3-Omni Pitt arms come from it')
            if i == '3a_pcgita_zeroshot_shipped': return A(PV, ['R_Q3O_RANGE'], '0.60', 'lower end of the printed answer range')
            if i in ('3a_pitt_enc_nested', '3a_pcgita_enc_nested'): return A(PV, ['A23_Q3O', 'R_Q3O_RANGE'])
            if 'best_stage' in i: return A(REL, note='oracle-selected best stage, descriptive only')
            if 'zeroshot_pod' in i or 'attn' in i: return A(REL, note='pod rerun of the Qwen3-Omni zero-shot; drift versus the shipped file (DISCREPANCIES POD3 3a)')
            return A(REL, note='POD3 re-extraction; master_lookup keeps the original json as the source for this stream')
        if t == 'POD4':
            if i.startswith('POD4_4a'):
                if i == 'POD4_4a_nv_cen_before': return A(PV, ['R_CHANNEL', 'A23_CHAN'], '0.87', 'tonight quiet-frame centroid alone (direction-free) is 0.7782; the source of the printed 0.87 is not in tonight fragments')
                if re.search(r'_(nf|snr|cen|ro|tilt)_', i): return A(SUP, ['A23_CHAN'], note='single channel feature')
                if i == 'POD4_4a_pg_5f_before': return A(PV, ['A23_CHAN'], None, 'five quiet-frame features together; the printed PC-GITA 0.69 in the results text is not reproduced by any tonight row')
                return A(PV, ['A23_CHAN'])
            if i == 'POD4_4b_pub_zs': return A(PV, ['T1_PITT'], '0.66')
            if i == 'POD4_4b_pub_enc': return A(SUPS, note='single-split Pitt encoder baseline 0.7969, superseded by 0.7706')
            return A(REL, note='interviewer removal on Pitt with the duration control; PART 21 brief from the authors: released, not in the paper')
        if t == 'POD4B':
            if i == '4b_adress2020_published_zs': return A(PV, ['T1_ADR20'], '0.62')
            return A(REL, note='interviewer removal on ADReSSo / ADReSS-2020 with the duration control; PART 21 brief from the authors: released, not in the paper')
        if t == 'Q3Ofix':
            if 'conflict: probe' in w or 'agreement: probe' in w and 'minus' not in w:
                if 'minus' not in w: return A(PV, ['A23_DIR', 'A23_Q3O'], None, 'Qwen3-Omni by arm, full fit restricted to the arm (decision by the authors 23 Sep)')
            return A(PV, ['A23_DIR'])
        if t == 'SEED':
            n_ = int(i[4:6])
            m = {1: A(SUP, ['A23_TXTPAIRS'], note='pooled over both arms'), 2: A(PV, ['A23_TXTPAIRS']), 3: A(SUPS, note=SUP311 + ' (transcript cell)'),
                 4: A(SUP, ['A23_TXTPAIRS'], note='pooled over both arms'), 5: A(PV, ['A23_TXTPAIRS']), 6: A(SUPS, note=SUP311 + ' (transcript cell)'),
                 7: A(PV, ['A23_TXTPAIRS']), 8: A(PV, ['A23_TXTPAIRS']), 9: A(PV, ['A23_Q3O', 'R_Q3O_RANGE']),
                 10: A(REL, note='one repeat; master_lookup keeps the original json for this stream'), 11: A(REL, note='POD3 re-extraction; master_lookup keeps the original json for this stream'),
                 12: A(REL, note='POD3 re-extraction; master_lookup keeps the original json for this stream'), 13: A(REL, note='pod rerun; drift versus the shipped 0.7619'),
                 14: A(PV, ['A23_DIR']), 15: A(PV, ['A23_DIR']), 16: A(PV, ['A23_DIR']),
                 17: A(SUPS, note='training-fold standardisation; canonical value is Q3Ofix 0.8276')}[n_]
            return m
    # ------------------------------------------------ Part 17
    if part == '17':
        if t == 'T1':
            m = re.match(r'T1_(o25t|q2at|o25|q2a|q3o|af2|af3|kimi)_(.*)', i); mk, rest = m.groups()
            model = {'o25': 'Qwen2.5-Omni', 'q2a': 'Qwen2-Audio', 'q3o': 'Qwen3-Omni-30B-A3B', 'af2': 'Audio Flamingo 2', 'af3': 'Audio Flamingo 3', 'kimi': 'Kimi-Audio'}.get(mk)
            if mk in ('o25t', 'q2at'):
                if rest == 'conflict': return A(PV, ['A23_TXTPAIRS'])
                if 'OLD' in rest: return A(SUPS, note=SUP311 + ' (transcript cell)')
                if 'NEW' in rest and 'paired' not in rest: return A(PV, ['A23_TXTPAIRS'], None, 'transcript agreement on 311 distinct segments (PART 17 task 1 from the authors)')
                return A(SUP, ['A23_TXTPAIRS'])
            pr = T2P[model]
            if rest == 'conflict': return A(PV, [pr[4], 'R_BELOW', 'ABS_BELOW'], pr[0])
            if 'OLD' in rest: return A(SUPS, note=SUP311)
            if 'paired' in rest: return A(PV, ['R_EXCL0'], None, 'paired conflict minus agreement on the 311 distinct agreement segments')
            return A(PV, [pr[4], 'R_BELOW'], pr[1], NOTE311)
        if t == 'T2': return A(REL, note='fold file reproduction check (folds released; PART 17 task 2 from the authors)')
        if t == 'T3':
            u = re.search(r'used=([a-z]+)', w).group(1)
            if u == 'yes': return A(PV, ['A23_DIR'])
            if u == 'secondary': return A(SUP, ['A23_DIR'])
            if u == 'no': return A(SUPS, note='refit within the arm, marked not used in the fragment')
            return A(REL, note='nested-probe supplement, marked supplement in the fragment')
        if t == 'T4':
            if i == 'T4_pitt_o25_enc_mean5': return A(PV, ['R_BESTPROBE', 'ABS_PROBE'], '0.77', 'mean of five per-repeat AUCs', PENDING_ENC)
            if 'NOTUSED' in i or 'avgprob' in i: return A(SUPS, note=AVGP)
            if i == 'T4_edaic_o25_llm_mean5_jsononly': return A(PV, ['R_EDAIC_PROBE', 'A23_EDAICPROBE'], '0.69', 'decided E-DAIC LM probe 0.7001 (authors 23 Sep 11:29Z); the pasted tex still prints 0.69; json only here, interval in T7b')
            if i == 'T4_edaic_o25_enc_mean5_jsononly': return A(PV, ['A23_EDAICPROBE'], None, 'json only here, interval in T7b')
            if i in ('T4_pcgita_q3o_enc_mean5', 'T4_pitt_q3o_enc_mean5'): return A(PV, ['A23_Q3O', 'R_Q3O_RANGE'])
            if 'SUPERSEDED' in i: return A(SUPS, note='non-canonical Pitt copy; decision by the authors 23 Sep (SUPERSEDED.csv)')
            if i == 'T4_pitt_kimi_pooled': return A(SUP, ['T2_KIMI'], note='canonical file, pooled over both arms')
            if i == 'T4_pitt_af2_pooled': return A(SUP, ['T2_AF2'], note='canonical file, pooled over both arms')
            if i == 'T4_pitt_kimi_conflict': return A(PV, ['T2_KIMI', 'R_PITTDROP'], '0.72')
            if i == 'T4_pitt_kimi_agreement': return A(PV, ['T2_KIMI', 'R_PITTDROP'], '0.72')
            if i == 'T4_pitt_af2_conflict': return A(PV, ['T2_AF2'], '0.63')
            if i == 'T4_pitt_af2_agreement': return A(PV, ['T2_AF2'], '0.55', 'decision text from the authors typed 0.5536; the file gives 0.5544 (same 2 dp)')
            if 'q3o' in i and 'fullfit' in i: return A(PV, ['A23_Q3O', 'A23_DIR'], None, 'full fit restricted to the arm (decision by the authors 23 Sep)')
            if 'lora' in i: return A(PV, ['A23_LORA'])
            return A(REL, note='master_lookup bookkeeping count')
        if t == 'T5':
            ds = i.split('_')[1]
            if i.endswith('_answer'):
                if ds in T1A: c, p_ = T1A[ds]; return A(PV, [c, 'R_ZS'], p_)
                if i == 'T5_edaic_full_answer': return A(PV, ['T1_EDAIC'], '0.83', 'original full-window run')
                return A(REL, note=EW)
            if ds in ('neurovoz', 'kcl', 'adresso', 'adress2020'): return A(PV, ['FIG1'], None, 'omitted-dataset curve (PART 17 task 5 from the authors, for the Figure 1 caption)')
            if ds in ('pitt', 'pcgita') or i.startswith('T5_edaic_full'): return A(SUP, ['FIG1'], note='dataset shown in Figure 1')
            return A(REL, note=EW)
        if t == 'T6':
            if i == 'T6_s20_text_all468': return A(PV, ['T1_PITT'], '0.66', 'Sep 20 run, the run Table 1 prints')
            if i in ('T6_s20_text_conflict', 'T6_s20_text_agreement', 'T6_s20_text_conflict_minus_agreement'): return A(PV, ['A23_PITTARMS'])
            if i == 'T6_s20_text_minus_audio_all468': return A(SUP, ['R_PITT_TEXT'])
            if i.startswith('T6_s20_text_minus_audio') or i in ('T6_s20_gap_text_minus_gap_audio', 'T6_armgap_ratio_s20'): return A(SUP, ['A23_PITTARMS'])
            if i == 'T6_audio_all468': return A(PV, ['T1_PITT'], '0.66')
            if i == 'T6_audio_conflict': return A(PV, ['T2_O25'], '0.53')
            if i == 'T6_audio_agreement': return A(PV, ['T2_O25'], '0.71')
            if i == 'T6_audio_conflict_minus_agreement': return A(PV, ['R_PITTDROP', 'A23_GAP'], None, 'rounds to 0.17; printed range starts at 0.18')
            if 's15' in i and 's20_minus_s15' not in i and 'perclip' not in i: return A(SUPS, note=SEP15)
            return A(REL, note='comparison of the Sep 15 and Sep 20 transcript runs')
        if t == 'T7':
            if i == 'T7_zeroshot_answer': return A(PV, ['T1_EDAIC', 'R_EDAIC_PROBE', 'A23_EDAICPROBE'], '0.83', 'rebuild run, 300 s processor cut')
            if i == 'T7_published_llm_meanof5': return A(PV, ['R_EDAIC_PROBE', 'A23_EDAICPROBE'], '0.69', 'decided E-DAIC LM probe 0.7001 (authors 23 Sep 11:29Z); the pasted tex still prints 0.69')
            if i == 'T7_published_enc_meanof5': return A(PV, ['A23_EDAICPROBE'])
            if i.startswith('T7_published_'): return A(SUP, ['R_EDAIC_PROBE'], note='other stages, below the LM probe')
            if 'notused' in i: return A(SUPS, note=AVGP)
            return A(SUPS, note='proxy Mac rerun, labelled not the paper value in the fragment')
        if t == 'T7b':
            if i == 'T7b_zeroshot_answer': return A(PV, ['T1_EDAIC', 'R_EDAIC_PROBE', 'A23_EDAICPROBE'], '0.83', 'rebuild run, 300 s processor cut')
            if i == 'T7b_llm_meanof5': return A(PV, ['R_EDAIC_PROBE', 'A23_EDAICPROBE'], '0.69', 'decided E-DAIC LM probe 0.7001 (authors 23 Sep 11:29Z); the pasted tex still prints 0.69')
            if i == 'T7b_enc_meanof5': return A(PV, ['A23_EDAICPROBE'])
            if i in ('T7b_llm_meanof5_minus_answer', 'T7b_enc_meanof5_minus_answer'): return A(PV, ['A23_EDAICPROBE', 'A23_GAP'])
            if 'notused' in i: return A(SUPS, note=AVGP)
            return A(SUP, ['R_EDAIC_PROBE'], note='answer-state / projector, same rebuild run')
    # ------------------------------------------------ Part 20
    if part == '20':
        if t == 'POD1':
            if '(paper file)' in w and 'minus' not in w.lower() and 'gap' not in i:
                arm = 'all' if i.endswith('_all') else i.rsplit('_', 1)[1]
                return A(PV, ['T1_PITT'] if arm == 'all' else ['R_FT_ARMS'], {'all': '0.77', 'conflict': '0.52', 'agreement': '0.86'}[arm], 'paper projector fine-tune file, recomputed beside POD1')
            if i.startswith('POD1_std_'):
                arm = i.rsplit('_', 1)[1]
                return A(PV, ['T1_PITT'] if arm == 'all' else ['R_FT_ARMS'], {'all': '0.77', 'conflict': '0.52', 'agreement': '0.86'}[arm], 'paper projector fine-tune file, recomputed beside POD1')
            if i in ('POD1_gap_std',) or (i.endswith('_gap_ref') and '(paper file)' in w): return A(SUP, ['R_FT_ARMS'])
            return A(REL, note='balanced fine tuning / same-pod control; PART 21 brief from the authors: released, not in the paper')
        if t == 'POD1b': return A(REL, note='seed spread of the standard projector fine-tune; not named for the paper')
        if t == 'POD2':
            if i.startswith('POD2.earlier.') or i == 'POD2.params.lora': return A(SUP, ['A23_LORA'], note='the paper LoRA run recomputed beside the balanced LoRA')
            return A(REL, note='balanced LoRA / control; PART 21 brief from the authors: balanced fine tuning released, not in the paper')
        if t == 'POD3': return A(REL, note='projector fine-tune on the 966 E-DAIC pair clips; not named for the paper')
        if t == 'POD3b':
            if 'envcheck' in i: return A(SUP, ['T2_O25'], note='Part 14 set re-scored in a new environment, matches Table 2 at 2 dp')
            return A(REL, note='second scorer set (DistilBERT SST-2 pairs); PART 21 brief from the authors: released, not in the paper')
        if t == 'POD3c':
            if i == 'POD3c_q3o_audio_conflict483': return A(PV, ['T2_Q3O', 'R_BELOW', 'ABS_BELOW'], '0.30')
            if i == 'POD3c_q3o_audio_agreement483': return A(SUPS, note=SUP311)
            if i == 'POD3c_q3o_audio_agreement311': return A(PV, ['T2_Q3O', 'R_BELOW'], '0.96', NOTE311)
            if i == 'POD3c_q3o_audio_conf_minus_agree483': return A(SUPS, note=SUP311)
            if i == 'POD3c_q3o_audio_conf_minus_agree311': return A(PV, ['R_EXCL0'])
            if 'tokenset' in i or 'mass' in i: return A(REL, note='token set / answer mass check')
            return A(SUP, ['A23_TXTPAIRS'], note='Qwen3-Omni transcript on the pairs (Part 20 extension; the 23 Sep list names the transcript on the pairs)')
        if t == 'POD4':
            if i == 'paper_o25_orig_recording_overall': return A(PV, ['T1_PITT'], '0.66')
            if i == 'paper_o25_orig_recording_conflict': return A(PV, ['T2_O25'], '0.53')
            if i == 'paper_o25_orig_recording_agreement': return A(PV, ['T2_O25'], '0.71')
            if i == 'paper_q2a_orig_recording_conflict': return A(PV, ['T2_Q2A'], '0.41')
            if i == 'paper_q2a_orig_recording_agreement': return A(PV, ['T2_Q2A'], '0.70')
            if i == 'paper_q2a_orig_recording_overall': return A(SUP, ['R_Q2A_RANGE'], note='Qwen2-Audio Pitt answer, inside the printed 0.54 to 0.68')
            return A(REL, note=IFREE + ' (or same-pod rerun / wording / dtype check)')
        if t == 'POD4b':
            if i == 'P20_4b_q3o_orig_shipped_conflict': return A(PV, ['T2_Q3O'], '0.61')
            if i == 'P20_4b_q3o_orig_shipped_agreement': return A(PV, ['T2_Q3O'], '0.82')
            if i == 'P20_4b_q3o_orig_shipped_overall': return A(SUP, ['R_Q3O_RANGE'], note='shipped Qwen3-Omni Pitt answer')
            return A(REL, note=IFREE)
        if t == 'POD4c':
            if i in ('af3_orig_part10_conflict', 'af3_orig_part10_agreement', 'af3_orig_part3_conflict', 'af3_orig_part3_agreement'):
                cp = 'part10' if 'part10' in i else 'part3'
                return A(PV, ['T2_AF3'], '0.61' if i.endswith('conflict') else '0.59', cp + ' copy', PENDING_AF3)
            if i in ('af3_orig_part10_all', 'af3_orig_part3_all'): return A(SUP, ['T2_AF3'], note='pooled over both arms', pending=PENDING_AF3)
            if i == 'af2_orig_af2pitt468_conflict': return A(PV, ['T2_AF2'], '0.63')
            if i == 'af2_orig_af2pitt468_agreement': return A(PV, ['T2_AF2'], '0.55')
            if i == 'af2_orig_af2pitt468_all': return A(SUP, ['T2_AF2'], note='canonical file, pooled over both arms')
            return A(REL, note=IFREE)
        if t == 'POD4d':
            if i == 'kimi_pitt_orig_canonical_conflict': return A(PV, ['T2_KIMI'], '0.72')
            if i == 'kimi_pitt_orig_canonical_agreement': return A(PV, ['T2_KIMI'], '0.72')
            if i == 'kimi_pitt_orig_canonical_overall': return A(SUP, ['T2_KIMI'], note='canonical file, pooled over both arms')
            return A(REL, note=IFREE)
        if t == 'POD5':
            if i == 'POD5_zs_o25_paper_overall': return A(PV, ['T1_PITT'], '0.66')
            if i == 'POD5_zs_o25_paper_conflict': return A(PV, ['T2_O25'], '0.53')
            if i == 'POD5_zs_o25_paper_agreement': return A(PV, ['T2_O25'], '0.71')
            if i == 'POD5_enc_mac_seed_paper_overall': return A(PV, ['R_BESTPROBE'], '0.77', 'Mac splits, mean of five (0.7706)', PENDING_ENC)
            if i == 'POD5_enc_pod_seed_original_rerun_overall': return A(PV, ['R_BESTPROBE'], '0.77', 'pod splits, mean of five (0.7761, the pod json value)', PENDING_ENC)
            if i == 'POD5_llm_pod_seed_original_rerun_overall': return A(PV, ['R_LM080'], '0.80')
            if i == 'POD5_dir_orig_probe': return A(PV, ['R_LM080', 'A23_DIR'], '0.77', 'answer-state final-layer probe (README estimator rule 0.7709)')
            if i == 'POD5_dir_orig_after_removal': return A(PV, ['R_REMOVE', 'A23_DIR'], '0.77')
            if i == 'POD5_dir_orig_cos': return A(PV, ['R_COS', 'A23_DIR'], '0.01', 'magnitude')
            if i == 'POD5_dir_orig_kept_minus_removed': return A(SUP, ['R_REMOVE'])
            if i == 'POD5_sft_paper_overall': return A(PV, ['T1_PITT'], '0.77')
            if i == 'POD5_sft_paper_conflict': return A(PV, ['R_FT_ARMS'], '0.52')
            if i == 'POD5_sft_paper_agreement': return A(PV, ['R_FT_ARMS'], '0.86')
            if 'original_rerun' in i or '_paper_' in i: return A(REL, note='original-window rerun of a stream or arm the paper does not print')
            return A(REL, note=IFREE)
        if t == 'POD6':
            if i.startswith('A_tfidf'):
                if 'rawCHAT' in i: return A(REL, note='sensitivity on raw CHAT text')
                return A(SUP, ['INTRO_COND'], note='TF-IDF text classifier per dataset; ordering holds only with the full E-DAIC transcript and the Parkinson\'s signal comes from ASR output (DISCREPANCIES POD6 A)')
            if i == 'B_zeroshot_nvfull_ref': return A(PV, ['T1_NV'], '0.70')
            if i == 'B_encprobe_mean5_nvfull_samepod': return A(SUP, ['ABS_PROBE', 'R_BESTPROBE'], note='same-pod re-extraction check against the NeuroVoz encoder probe (master 0.9213)')
            if i.startswith('B_'): return A(REL, note='NeuroVoz matched subset; PART 21 brief from the authors: released, not in the paper')
            if i.startswith('C_af3_part10'):
                if 'minus' in i: return A(PV, ['R_PITTDROP'], None, 'part10 copy', PENDING_AF3)
                return A(PV, ['T2_AF3'], '0.61' if i.endswith('conflict') else '0.59', 'part10 copy', PENDING_AF3)
            if i.startswith('C_M8_'):
                m = re.match(r'C_M8_(.+?)_(primary|af2_pitt468|af2_pitt_audio|alt_overnight_copy)_(conflict_minus_agreement|conflict|agreement)$', i)
                mdl, src, arm = m.groups()
                key = {'Qwen2.5-Omni': 'Qwen2.5-Omni', 'Qwen2-Audio': 'Qwen2-Audio', 'Qwen3-Omni-30B-A3B': 'Qwen3-Omni-30B-A3B', 'AudioFlamingo2': 'Audio Flamingo 2', 'AudioFlamingo3': 'Audio Flamingo 3', 'Kimi-Audio': 'Kimi-Audio'}[mdl]
                fake = dict(r); fake['part'] = '16'; fake['task'] = 'M8'; fake['what'] = '%s | B_pitt | %s | src=%s' % (key, arm, src)
                return classify(fake)
            if i.startswith('C_T6_'):
                fake = dict(r); fake['part'] = '17'; fake['task'] = 'T6'; fake['id'] = 'T6_' + i[5:]
                return classify(fake)
    # ------------------------------------------------ Part 22
    if part == '22':
        if i == 'P22_S1_capped900': return A(PV, ['IMPL_EDAIC'], '89', 'the 89 windows capped at 900 s; the processor still cut every window to its first 300 s')
        if i in ('P22_all275_trunc', 'P22_auc_default_all275'): return A(PV, ['T1_EDAIC'], '0.83', 'rebuild run, 300 s processor cut')
        return A(SUP, ['IMPL_EDAIC'], note='PART 22: the scored run heard only the first 300 s (217 of 275 windows cut); the whole-window rerun gives 0.8379, paired +0.0101 [-0.0309, 0.0518]; the sentence needs rewording (23 Sep)')
    raise KeyError((part, t, i))

for r in rows:
    a = classify(r)
    if a is None: raise KeyError((r['part'], r['task'], r['id']))
    r.update(a)
    r['agree_4dp'] = agree(r)
    # paper sentence text
    if r['role'] in (PV, SUP):
        s = [Q[c][1] for c in r['claims']]
        if r['pending']: s = [r['pending']] + s
        r['paper_sentence'] = ' | '.join(s)
    else:
        r['paper_sentence'] = ''
    # printed check
    r['printed_check'] = ''
    if r['printed'] is not None:
        v = num(r['vv'])
        if v is not None:
            if r['printed'] == '89':
                ok = v == Decimal('89')
                r['printed_check'] = 'printed 89: ' + ('MATCH' if ok else 'MISMATCH')
            else:
                rv = r2(abs(v)) if r['id'] == 'POD5_dir_orig_cos' else r2(v)
                ok = str(rv) == r['printed']
                r['printed_check'] = 'printed %s, verified rounds to %s: %s' % (r['printed'], rv, 'MATCH' if ok else 'MISMATCH')

# ---------------------------------------------------------------- write CSV
os.makedirs(OUT, exist_ok=True)
cols = ['part', 'task/pod', 'id', 'what', 'value_computed', 'value_verified', 'lo', 'hi', 'n', 'n_spk', 'file', 'agree_4dp', 'paper_role', 'paper_sentence', 'printed_check', 'note', 'fragment']
with open(OUT + '/summary_all.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(cols)
    for r in rows:
        wr.writerow([r['part'], r['task'], r['uid'], r['what'], r['vc'], r['vv'], r['lo'], r['hi'], r['n'], r['n_spk'], r['file'],
                     r['agree_4dp'], r['role'], r['paper_sentence'], r['printed_check'], r['note'], r['src']])

# ---------------------------------------------------------------- counts
def is_verified(r):
    return r['agree_4dp'] == 'verified in fragment' or r['agree_4dp'].startswith('yes')
C = dict(
    generated_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    total_rows=len(rows),
    per_part=dict(Counter(r['part'] for r in rows)),
    per_part_task={k: v for k, v in sorted(Counter('%s/%s' % (r['part'], r['task']) for r in rows).items())},
    per_role=dict(Counter(r['role'] for r in rows)),
    per_part_role={p: dict(Counter(r['role'] for r in rows if r['part'] == p)) for p in sorted(set(r['part'] for r in rows))},
    verified=dict(total=sum(is_verified(r) for r in rows),
                  verified_in_fragment_single_value=sum(r['agree_4dp'] == 'verified in fragment' for r in rows),
                  computed_equals_verified_4dp=sum(r['agree_4dp'].startswith('yes') for r in rows),
                  disagree_or_missing=sum(not is_verified(r) for r in rows)),
    paper_value_rows=sum(r['role'] == PV for r in rows),
    paper_value_decision_pending=sum(r['role'] == PV and bool(r['pending']) for r in rows),
    paper_value_printed_checks=dict(Counter(r['printed_check'].split(': ')[-1] for r in rows if r['role'] == PV and r['printed_check'])),
    pod1b='included (verified file present)' if pod1b_present else 'pending verification',
    sources=dict(part16=R + '/edaic_rerun/part16/rows/*.tsv + PART16_RESULTS.md seed table', part17=R + '/edaic_rerun/part17/rows/*.tsv',
                 part20=R + '/scores/part20/rows/<POD>.tsv (verified files only)', part22=R + '/scores/part22/rows/P22_verified.tsv',
                 tex=TEX, master=R + '/master/master_lookup.csv'),
)
json.dump(C, open(OUT + '/summary_counts.json', 'w'), indent=1)

# ---------------------------------------------------------------- markdown
def fmt_ci(r):
    return '[%s, %s]' % (r['lo'], r['hi']) if r['lo'] and r['hi'] else 'no interval'
SECTIONS = ['abstract', 'Table 1', 'Table 2', 'results text', 'results text: 23 Sep additions', 'Figure 1 caption', 'conclusion', 'method text', 'implementation text']
by_claim = defaultdict(list)
for r in rows:
    if r['role'] == PV:
        for c in r['claims']: by_claim[c].append(r)
L = []
L.append('# Paper values verified tonight, ICASSP 2027 "The Representation Utility Gap in Audio LLMs for Clinical Speech"')
L.append('')
L.append('Built %s from the verified row fragments of Parts 16, 17, 20 and 22. Paper text is the pasted tex of 05:58Z (%s). Sentences the authors added later on 23 Sep are not in that paste, so rows that feed them say "23 Sep addition, wording to be matched to the final tex". The 23 Sep topics are the ones the authors listed in the PART 21 brief (12:02Z) and the 11:29Z decisions.' % (C['generated_utc'], TEX))
L.append('')
L.append('## Counts')
L.append('')
L.append('- Rows (one per value): %d' % C['total_rows'])
L.append('- Per part: ' + ', '.join('Part %s %d' % (k, v) for k, v in sorted(C['per_part'].items())))
L.append('- Per role: ' + ', '.join('%s %d' % (k, v) for k, v in sorted(C['per_role'].items())))
for p, d in C['per_part_role'].items():
    L.append('  - Part %s: ' % p + ', '.join('%s %d' % (k, v) for k, v in sorted(d.items())))
v = C['verified']
L.append('- Verified: %d of %d (%d single values verified in their fragment, Parts 16 and 17; %d with computed equal to verified at 4 dp, Parts 20 and 22); disagree or missing: %d' % (v['total'], C['total_rows'], v['verified_in_fragment_single_value'], v['computed_equals_verified_4dp'], v['disagree_or_missing']))
L.append('- PAPER VALUE rows: %d, of which decision pending: %d' % (C['paper_value_rows'], C['paper_value_decision_pending']))
L.append('- PAPER VALUE rows checked against a printed number: ' + ', '.join('%s %d' % (k, n_) for k, n_ in sorted(C['paper_value_printed_checks'].items())))
L.append('- POD1b: ' + C['pod1b'])
L.append('')
L.append('Each line: verified value, 95% speaker interval, n (speakers), file, and the sentence it feeds. Rows with the same value, interval, n and file are merged onto one line and their ids listed. MISMATCH means the verified value does not round to the number printed in the 05:58Z paste.')
for sec in SECTIONS:
    claims = [c for c in Q if Q[c][0] == sec and by_claim.get(c)]
    if not claims: continue
    L.append('')
    L.append('## ' + sec[0].upper() + sec[1:])
    for c in claims:
        L.append('')
        L.append('### ' + Q[c][1])
        grp = defaultdict(list)
        for r in by_claim[c]:
            grp[(r['vv'], r['lo'], r['hi'], r['n'], r['n_spk'], r['file'], r['printed_check'] if c == r['claims'][0] else '', r['pending'] or '', r['note'])].append(r)
        for k, rs in grp.items():
            r = rs[0]
            ids = ', '.join('%s/%s' % (x['task'], x['uid']) for x in rs)
            extra = []
            if r['printed_check'] and c == r['claims'][0]: extra.append(r['printed_check'])
            if r['pending']: extra.append(r['pending'])
            if r['note']: extra.append(r['note'])
            L.append('- %s %s, n %s (%s spk), file `%s`, what: %s. Sentence: "%s". %sRows: %s' % (
                r['vv'], fmt_ci(r), r['n'], r['n_spk'] or '?', r['file'], r['what'], Q[c][1],
                ('Check: ' + '; '.join(extra) + '. ') if extra else '', ids))
open(OUT + '/summary_paper.md', 'w').write('\n'.join(L) + '\n')
print(json.dumps(C, indent=1))
