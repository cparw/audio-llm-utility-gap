import sys, os, json, subprocess, datetime
sys.path.insert(0,'<local data dir>/scratch/p16m')
import pandas as pd, numpy as np
from auclib import auc_rank, boot_paired, boot_one, sha1mb

OUT='<local data dir>/Desktop/release/edaic_rerun/part16/'
V='<local data dir>/Desktop/release/edaic_rerun/variants/'
O='<local data dir>/Desktop/release/omni_final/'
TX='<local data dir>/Desktop/release/overnight2/text_new/o25_edaicfull_text.csv'
NJ=V+'o25_full_nested_repeats.json'

SRC={
 'full_answer'   : V+'o25_full_zeroshot_scores.csv',
 'full_sft_oof'  : V+'sft_full_oof.csv',
 'full_text'     : TX,
 'full_nested_json': NJ,
 'w30_answer'    : O+'omni_edaic_zeroshot_scores.csv',
 'w30_enc_oof'   : O+'omni_edaic_enc_nested_oof.csv',
 'w30_llm_oof'   : O+'omni_edaic_llm_nested_oof.csv',
 'w30_ans_oof'   : O+'omni_edaic_ans_nested_oof.csv',
 'w30_proj_oof'  : O+'omni_edaic_proj_nested_oof.csv',
 'windows_full'  : V+'windows_full.csv',
}

# ---------- load ----------
zsf = pd.read_csv(SRC['full_answer']);  zsf['spk']=zsf.speaker.astype(str)
zs30= pd.read_csv(SRC['w30_answer']);   zs30['spk']=zs30.speaker.astype(str)
sft = pd.read_csv(SRC['full_sft_oof']); sft['spk']=sft.speaker.astype(str)
txt = pd.read_csv(SRC['full_text']);    txt['spk']=txt.speaker_id.astype(str)
win = pd.read_csv(SRC['windows_full']); win['spk']=win.pid.astype(str)
nested=json.load(open(NJ))

rows=[]   # id, what, value, lo, hi, n, n_spk, file
def emit(i,what,v,lo,hi,n,ns,f):
    rows.append((i,what,
        ('' if v is None else f'{v:.4f}'),
        ('' if lo is None else f'{lo:.4f}'),
        ('' if hi is None else f'{hi:.4f}'), n, ns, f))
    print(f'{i:<26} {what:<52} {("NA" if v is None else f"{v:.4f}"):>8} '
          f'[{("NA" if lo is None else f"{lo:.4f}")}, {("NA" if hi is None else f"{hi:.4f}")}]  n={n} spk={ns}', flush=True)

res={}
# ---------- STEP 6 overlap ----------
a=set(zsf.spk); b=set(zs30.spk)
same = (a==b)
lab_ok = bool((zsf.set_index('spk').label.sort_index()==zs30.set_index('spk').label.sort_index()).all())
res['speaker_overlap']={'n_full':len(a),'n_30s':len(b),'intersection':len(a&b),
    'identical_sets':bool(same),'labels_identical':lab_ok,
    'full_only':sorted(a-b),'w30_only':sorted(b-a)}
print(f'\nSTEP6 speaker sets identical={same} intersection={len(a&b)} labels_identical={lab_ok}\n')

# ---------- FULL window ----------
print('--- FULL WINDOW (median 591.8 s, cap 900 s, 89/275 capped) ---')
ans_full = auc_rank(zsf.label, zsf.p_yes)
lo,hi,u = boot_one(zsf.label, zsf.p_yes, zsf.spk, 2000, 0)
emit('MEDAIC_full_answer','Omni zero-shot answer AUC, full window',ans_full,lo,hi,len(zsf),zsf.spk.nunique(),'edaic_rerun/variants/o25_full_zeroshot_scores.csv')
res['full_answer']={'auc':ans_full,'lo':lo,'hi':hi,'usable':u,'n':len(zsf)}

PAIRS=[]
m=zsf[['spk','label','p_yes','mass']].merge(sft[['spk','p_yes']].rename(columns={'p_yes':'p_sft'}),on='spk',how='inner')
PAIRS.append(('full_sft','projector fine-tune OOF',m,'p_sft','edaic_rerun/variants/sft_full_oof.csv'))
m2=zsf[['spk','label','p_yes']].merge(txt[['spk','p_yes','answer_mass']].rename(columns={'p_yes':'p_text'}),on='spk',how='inner')
PAIRS.append(('full_text','transcript-only readout',m2,'p_text','overnight2/text_new/o25_edaicfull_text.csv'))

for key,name,mm,col,f in PAIRS:
    ap=auc_rank(mm.label,mm[col]); aa=auc_rank(mm.label,mm.p_yes)
    r=boot_paired(mm.label,mm[col],mm.p_yes,mm.spk,2000,0)
    excl = not (r['d_lo']<0<r['d_hi'])
    emit(f'MEDAIC_{key}',f'{name} AUC, full window',ap,r['a_lo'],r['a_hi'],len(mm),mm.spk.nunique(),f)
    emit(f'MEDAIC_{key}_minus_answer',f'PAIRED {name} minus answer, full window (excl0={excl})',
         ap-aa,r['d_lo'],r['d_hi'],len(mm),mm.spk.nunique(),f)
    res[key]={'probe_auc':ap,'probe_lo':r['a_lo'],'probe_hi':r['a_hi'],
              'answer_auc':aa,'answer_lo':r['b_lo'],'answer_hi':r['b_hi'],
              'paired_diff':ap-aa,'diff_lo':r['d_lo'],'diff_hi':r['d_hi'],
              'excludes_zero':excl,'usable_draws':r['usable'],'n':len(mm),'join_losses':len(zsf)-len(mm)}

# nested probe: aggregate only, NO interval
print('\n--- FULL WINDOW nested probe: AGGREGATE ONLY, no per-clip file, NO interval computable ---')
for st,lbl in [('llm','LM probe'),('enc','encoder probe'),('ans','answer-state probe'),('proj','projector probe')]:
    v=nested[st]['mean']
    emit(f'MEDAIC_full_{st}_nested_NOCI',f'{lbl} nested mean, full window (json aggregate, NOT recomputed, no per-clip file)',
         v,None,None,nested[st]['n'],nested[st]['n'],'edaic_rerun/variants/o25_full_nested_repeats.json')
    emit(f'MEDAIC_full_{st}_minus_answer_POINT',f'POINT-ONLY {lbl} minus answer, full window (no interval possible)',
         v-ans_full,None,None,nested[st]['n'],nested[st]['n'],'edaic_rerun/variants/o25_full_nested_repeats.json')
res['full_nested_aggregate']={k:{'mean':nested[k]['mean'],'repeats':nested[k]['repeats'],
    'per_repeat':nested[k]['per_repeat'],'minus_answer_point':nested[k]['mean']-ans_full,
    'interval':'NOT COMPUTABLE - no per-clip OOF file exists for the full window'} for k in nested}

# ---------- 30 s window ----------
print('\n--- 30 s WINDOW (first 30 s, M7 row) ---')
res['w30']={}
for st,lbl in [('enc','encoder probe'),('llm','LM probe'),('ans','answer-state probe'),('proj','projector probe')]:
    pr=pd.read_csv(SRC[f'w30_{st}_oof']); pr['spk']=pr.speaker.astype(str)
    mm=zs30[['spk','label','p_yes','mass']].merge(pr[['spk','p_probe']],on='spk',how='inner')
    ap=auc_rank(mm.label,mm.p_probe); aa=auc_rank(mm.label,mm.p_yes)
    r=boot_paired(mm.label,mm.p_probe,mm.p_yes,mm.spk,2000,0)
    excl = not (r['d_lo']<0<r['d_hi'])
    emit(f'MEDAIC_w30_{st}',f'{lbl} nested OOF AUC, 30 s window',ap,r['a_lo'],r['a_hi'],len(mm),mm.spk.nunique(),f'omni_final/omni_edaic_{st}_nested_oof.csv')
    emit(f'MEDAIC_w30_{st}_minus_answer',f'PAIRED {lbl} minus answer, 30 s window (excl0={excl})',
         ap-aa,r['d_lo'],r['d_hi'],len(mm),mm.spk.nunique(),f'omni_final/omni_edaic_{st}_nested_oof.csv')
    res['w30'][st]={'probe_auc':ap,'probe_lo':r['a_lo'],'probe_hi':r['a_hi'],
        'answer_auc':aa,'answer_lo':r['b_lo'],'answer_hi':r['b_hi'],
        'paired_diff':ap-aa,'diff_lo':r['d_lo'],'diff_hi':r['d_hi'],
        'excludes_zero':excl,'usable_draws':r['usable'],'n':len(mm),'join_losses':len(zs30)-len(mm)}
ans30=auc_rank(zs30.label,zs30.p_yes); lo3,hi3,u3=boot_one(zs30.label,zs30.p_yes,zs30.spk,2000,0)
emit('MEDAIC_w30_answer','Omni zero-shot answer AUC, 30 s window',ans30,lo3,hi3,len(zs30),zs30.spk.nunique(),'omni_final/omni_edaic_zeroshot_scores.csv')
res['w30_answer']={'auc':ans30,'lo':lo3,'hi':hi3,'usable':u3}

# ---------- answer shift across windows (paired, same speakers) ----------
mw=zsf[['spk','label','p_yes']].rename(columns={'p_yes':'p_full'}).merge(
   zs30[['spk','p_yes']].rename(columns={'p_yes':'p_30'}),on='spk',how='inner')
r=boot_paired(mw.label,mw.p_full,mw.p_30,mw.spk,2000,0)
d=auc_rank(mw.label,mw.p_full)-auc_rank(mw.label,mw.p_30)
excl=not(r['d_lo']<0<r['d_hi'])
emit('MEDAIC_answer_full_minus_30s','PAIRED answer AUC full minus 30 s, same 275 speakers (excl0=%s)'%excl,
     d,r['d_lo'],r['d_hi'],len(mw),mw.spk.nunique(),'o25_full_zeroshot_scores.csv + omni_edaic_zeroshot_scores.csv')
res['answer_full_minus_30s']={'diff':d,'lo':r['d_lo'],'hi':r['d_hi'],'excludes_zero':excl,'usable':r['usable']}

# ---------- per-clip csv ----------
pc = zsf[['spk','label','p_yes','mass']].rename(columns={'p_yes':'p_yes_answer','mass':'answer_mass'})
pc.insert(0,'id',zsf['clip'].values)
pc = pc.merge(sft[['spk','p_yes']].rename(columns={'p_yes':'probe_oof_projector_sft'}),on='spk',how='left')
pc = pc.merge(txt[['spk','p_yes','answer_mass']].rename(columns={'p_yes':'text_p_yes','answer_mass':'text_answer_mass'}),on='spk',how='left')
pc = pc.merge(zs30[['spk','p_yes','mass']].rename(columns={'p_yes':'p_yes_answer_30s','mass':'answer_mass_30s'}),on='spk',how='left')
for st in ['enc','llm','ans']:
    pr=pd.read_csv(SRC[f'w30_{st}_oof']); pr['spk']=pr.speaker.astype(str)
    pc=pc.merge(pr[['spk','p_probe']].rename(columns={'p_probe':f'probe_oof_30s_{st}'}),on='spk',how='left')
pc = pc.merge(win[['spk','win_dur_s','capped']],on='spk',how='left')
pc['probe_oof_full_nested']=np.nan   # explicitly absent
pc['window']='full (cap 900 s)'
pc=pc.rename(columns={'spk':'spk'})
pc.to_csv(OUT+'MEDAIC_full_window.csv',index=False)
print('\nwrote',OUT+'MEDAIC_full_window.csv',pc.shape)

# ---------- sidecar ----------
side={
 'cell':'MEDAIC, E-DAIC representation-utility gap on the FULL window vs the 30 s window',
 'date':datetime.date.today().isoformat(),
 'model_id':'Qwen/Qwen2.5-Omni-7B',
 'prompt_audio':"Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.",
 'prompt_text':"Based only on this transcript of what the person said, does this speaker show signs of depression? Answer with one word, Yes or No.",
 'condition':'mdd',
 'n':int(len(zsf)),'n_speakers':int(zsf.spk.nunique()),'n_positive':int(zsf.label.sum()),
 'bootstrap':{'draws':2000,'rng':'numpy.random.default_rng(0) reseeded per cell',
   'unit':'speaker, resampled with replacement','percentiles':[2.5,97.5],
   'paired':'one speaker list drawn per replicate, BOTH quantities recomputed inside that draw'},
 'auc':'recomputed with the tie-averaged rank (Mann-Whitney) formula from per-clip scores; never copied from a json',
 'folds':'no fold assignment was created by this cell. Every probe number here comes from an OOF file that already carried its own speaker-disjoint nested GroupKFold(5) assignment (nested_repeats_all.py). No new GroupKFold was needed.',
 'p_yes':'P(Yes)/(P(Yes)+P(No)) at the first answer position, bare and leading-space variants of Yes/yes/YES and No/no/NO, single-token encodings only; answer_mass = the denominator, carried per clip',
 'full_window_def':'stitched participant audio, capped at 900 s; 89 of 275 capped; median window 591.8 s (edaic_rerun/variants/windows_full.json)',
 'w30_window_def':'first 30 s of the raw session audio (omni_final/omni_edaic_zeroshot.json, window_seconds 30)',
 'MISSING':'The full-window nested probe has NO per-clip OOF file and NO states npz anywhere on this Mac or on <local data dir>/paper1_local_runs. Its AUCs therefore could NOT be recomputed from per-clip scores and NO interval was attached to them. See DISCREPANCIES.md entry "MEDAIC / Part 16" (HIGH).',
 'sources':{k:{'path':v,'sha256_first_1MB':(sha1mb(v) if os.path.exists(v) else 'MISSING'),
               'n_rows':(int(sum(1 for _ in open(v))-1) if os.path.exists(v) and v.endswith('.csv') else None)}
            for k,v in SRC.items()},
 'command':'/usr/local/bin/python3 '+os.path.abspath(__file__),
 'results':res,
}
json.dump(side,open(OUT+'MEDAIC_full_window.sidecar.json','w'),indent=1,default=str)
print('wrote',OUT+'MEDAIC_full_window.sidecar.json')

with open(OUT+'rows/MEDAIC.tsv','w') as fh:
    fh.write('id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n')
    for r_ in rows: fh.write('\t'.join(str(x) for x in r_)+'\n')
print('wrote',OUT+'rows/MEDAIC.tsv',len(rows),'value lines')
