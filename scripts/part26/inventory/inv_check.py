# PART 26 inventory check: states vs zero-shot per-clip file vs saved five-repeat fold files.
import numpy as np, pandas as pd, json, os, sys
from sklearn.metrics import roc_auc_score
R='<local data dir>/release'
P2='<local data dir>/paper1_local_runs/probe2'
FO=R+'/folds'
C=pd.read_csv('<local data dir>/release_from_mac/scores/part25/C/C_other_models.tsv',sep='\t')
dsmap={'PC-GITA':'pcgita','NeuroVoz':'neurovoz','MDVR-KCL':'kcl','E-DAIC':'edaic','Pitt':'pitt','ADReSSo':'adresso','ADReSS-2020':'adress2020'}
states={
 ('Qwen2-Audio',d):f'{P2}/{d}_states.npz' for d in dsmap.values()}
for d in ['pcgita','neurovoz','kcl','adresso','adress2020']:
  states[('Qwen3-Omni-30B-A3B',d)]=f'{R}/overnight2/q3o/q3o_{d}_states.npz'
states[('Qwen3-Omni-30B-A3B','pitt')]=f'{R}/edaic_rerun/part16/pod_sync/q3o_pitt_states.npz'
# POD3 pcgita re-extraction copy on G-Drive, checked separately
alt={('Qwen3-Omni-30B-A3B','pcgita'):'<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pcgita_states.npz',
     ('Qwen3-Omni-30B-A3B','pitt'):'<local data dir>/paper1_local_runs/part16_POD3_states/q3o_pitt_states.npz'}
out=[]
for i,r in C.iterrows():
  m=r.model; d=dsmap[r.dataset]
  zf=r.zeroshot_perclip_source
  zs=pd.read_csv(zf)
  zs['clip']=zs['clip'].astype(str)
  auc_zs=roc_auc_score(zs.label,zs.p_yes)
  rec=dict(model=m,dataset=r.dataset,zs_file=zf,zs_n=len(zs),zs_auc=round(auc_zs,4),zs_master=r.zeroshot_auc_master)
  # fold files
  pitt_ids = 'Control15ids_' if (m.startswith('Qwen3') and d=='pitt') else ''
  ffs=[]
  for s in range(5):
    cand=[f'{FO}/{d}_groupkfold5_pod_{pitt_ids}seed{s}.csv',f'{FO}/{d}_groupkfold5_pod_{pitt_ids}seed{s}_UNVERIFIED.csv']
    ff=[c for c in cand if os.path.exists(c)]
    ffs.append(ff[0] if ff else None)
  rec['fold_files']=ffs
  fsets=[]; fspk=None
  for ff in ffs:
    f=pd.read_csv(ff); fsets.append(set(f.clip_id.astype(str)))
    fspk=f
  rec['folds_n']=len(fspk); rec['folds_nspk']=fspk.speaker_id.nunique()
  rec['folds_same_clips_as_zs']=all(s==set(zs['clip']) for s in fsets)
  sf=states.get((m,d))
  rec['states_file']=sf
  if sf and os.path.exists(sf):
    z=np.load(sf,allow_pickle=True)
    nm=[str(x) for x in z['name']]; spk=[str(x) for x in z['spk']]
    rec['states_keys']={k:list(z[k].shape) for k in z.files if k in('enc','proj','llm','ans')}
    rec['states_n']=len(nm); rec['states_uniq']=len(set(nm))
    rec['states_same_set_zs']=set(nm)==set(zs['clip'])
    rec['states_same_order_zs']=nm==list(zs['clip'])
    rec['states_same_set_folds']=all(s==set(nm) for s in fsets)
    sm=dict(zip(nm,spk)); fm=dict(zip(fspk.clip_id.astype(str),fspk.speaker_id.astype(str)))
    rec['states_spk_eq_fold_spk']=all(sm[c]==fm[c] for c in nm)
    rec['states_nspk']=len(set(spk))
    lab=dict(zip(nm,z['label'])); zl=dict(zip(zs['clip'],zs.label))
    rec['labels_agree_zs']=all(int(lab[c])==int(zl[c]) for c in nm)
    pz=dict(zip(nm,z['p_yes'])); zp=dict(zip(zs['clip'],zs.p_yes))
    rec['states_pyes_maxdiff_vs_zs']=float(max(abs(pz[c]-zp[c]) for c in nm))
    e=z['enc'] if 'enc' in z.files else None
    rec['enc_layers']=int(e.shape[1]) if e is not None else 0
    rec['enc_finite']=bool(np.isfinite(e).all()) if e is not None else None
  else:
    rec['states_n']=0
  out.append(rec)
  print(json.dumps(rec,default=str)); sys.stdout.flush()
json.dump(out,open('scores/part26/inventory/inv_check.json','w'),indent=1,default=str)
# alt POD3 copies
for k,v in alt.items():
  if os.path.exists(v):
    z=np.load(v,allow_pickle=True); w=np.load(states.get(k) or v,allow_pickle=True)
    print('ALT',k,v,{kk:z[kk].shape for kk in z.files if kk in('enc',)}, 'names equal to primary', list(z['name'])==list(w['name']) if k in states else None, 'enc maxdiff', float(np.abs(z['enc']-w['enc']).max()) if k in states and z['enc'].shape==w['enc'].shape else None)
