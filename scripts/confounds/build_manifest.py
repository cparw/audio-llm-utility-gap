import pandas as pd, os
OUT='/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun/confounds/clip_manifest.csv'
OLD='/Users/chaitanyaparwatkar/Desktop/Hard drive data for paper'
NEW='/Volumes/G-Drive Pro/paper work/Hard drive data for paper'
rows=[]
def add(ds,src,paths,spk,lab,zipsrc=''):
    for p,s,l in zip(paths,spk,lab):
        rows.append(dict(dataset=ds,path=p,speaker=str(s),label=int(l),zip=zipsrc,manifest=src))
for ds,f in [('pitt','/Volumes/G-Drive Pro/paper1_local_runs/pitt_manifest.csv'),
             ('adresso','/Volumes/G-Drive Pro/paper1_local_runs/adresso_manifest.csv'),
             ('adress2020','/Volumes/G-Drive Pro/paper1_local_runs/adress2020_manifest.csv'),
             ('kcl','/Volumes/G-Drive Pro/paper1_local_runs/kcl_manifest.csv')]:
    d=pd.read_csv(f); d['path']=d.path.str.replace(OLD,NEW,regex=False)
    add(ds,f,d.path,d.speaker,d.label)
# pcgita from zip
ZP='/Volumes/G-Drive Pro/paper1_local_runs/drive_data/PC-GITA/OneDrive_1_12-11-2023.zip'
f='/Volumes/G-Drive Pro/paper1_local_runs/drive_data/pcgita_read.csv'
d=pd.read_csv(f); d=d[~d.filepath.str.contains('las que sobraron')].copy()
d['mem']=d.filepath.str.replace('/project2/msoleyma_946/speech_health/spanish_dataset/','',regex=False)
add('pcgita',f,d.mem,d.speaker_id,d.label,zipsrc=ZP)
# neurovoz read
f='/Volumes/G-Drive Pro/paper1_local_runs/drive_data/neurovoz.csv'
d=pd.read_csv(f); d=d[d.task_type=='read'].copy()
d['path']=d.filepath.str.replace('/project2/msoleyma_946/speech_health/spanish_neurovoz/','/Volumes/G-Drive Pro/paper1_local_runs/drive_data/spanish_neurovoz/',regex=False)
add('neurovoz',f,d.path,d.group.astype(str)+'_'+d.speaker_id.astype(str),d.label)
# edaic
f='/Volumes/G-Drive Pro/paper1_local_runs/drive_data/dcaps.csv'
d=pd.read_csv(f)
d['path']='/Volumes/G-Drive Pro/paper1_local_runs/drive_data/dcaps_proc/'+d.speaker_id.astype(str)+'.wav'
add('edaic',f,d.path,d.speaker_id,d.label)
m=pd.DataFrame(rows)
# subsample >600 with seed 0
keep=[]
for ds,g in m.groupby('dataset'):
    if len(g)>600:
        g=g.sample(600,random_state=0)
    keep.append(g)
m2=pd.concat(keep).sort_values(['dataset','path'])
m2['exists']=[True if r.zip else os.path.exists(r.path) for r in m2.itertuples()]
m2.to_csv(OUT,index=False)
print(m2.groupby('dataset').agg(n=('path','size'),spk=('speaker','nunique'),pos=('label','sum'),missing=('exists',lambda s:(~s).sum())))
