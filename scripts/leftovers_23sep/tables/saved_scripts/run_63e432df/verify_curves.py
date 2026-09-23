import csv, json
R='<local data dir>/release'
P='<local data dir>/release_from_mac/scores/part23/pull/p23-a-probes/p23a/out'
ex=json.load(open('<local data dir>/paper1_submission_23sep/project_page_build/explorer_data.json'))
S={s['id']:s for s in ex['layers']['series']}
def rd(path,col):
    rows=list(csv.DictReader(open(path)))
    k='layer' if 'layer' in rows[0] else ('stage' if 'stage' in rows[0] else None)
    if k: rows.sort(key=lambda r:int(r[k]))
    return [round(float(r[col]),4) for r in rows], list(rows[0].keys())
files={('pcgita','encoder'):(R+'/overnight/curves/omni/pcgita_encoder.csv','auc'),
('pcgita','lm'):(R+'/overnight/curves/omni/pcgita_llm.csv','auc'),
('pitt','encoder'):(R+'/overnight/curves/omni/pitt_encoder.csv','auc'),
('pitt','lm'):(R+'/overnight/curves/omni/pitt_llm.csv','auc'),
('edaic_300s','encoder'):(R+'/edaic_rerun/variants/o25_full_encoder_perlayer.csv','auc_oof'),
('edaic_300s','lm'):(R+'/edaic_rerun/variants/o25_full_llm_perlayer.csv','auc_oof'),
('edaic_whole','encoder'):(P+'/o25_whole_encoder_perlayer.csv','auc_oof'),
('edaic_whole','lm'):(P+'/o25_whole_llm_perlayer.csv','auc_oof')}
for (sid,part),(f,c) in files.items():
    v,cols=rd(f,c)
    j=S[sid][part]
    print(sid,part,len(v),len(j),'MATCH' if v==j else 'DIFF', cols[:4])
