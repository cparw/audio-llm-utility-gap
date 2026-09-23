"""PART 14 step 2: re-select conflict/agreement using the citable sentiment scorer.
Keeps px_build.py's other rules exactly."""
import pandas as pd, numpy as np, json, os
P="/Users/chaitanyaparwatkar/Desktop/af2_run/speech-health/results/health/paradox"
R="/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun"
S=pd.read_csv(f"{R}/part14_segments_sentiment.csv")
OLD=pd.read_csv(f"{P}/manifest.csv")
S["span"]=S.end-S.start
# px_build eligibility gate, line 154, unchanged
E=S[(S.speech_s>=5)&(S.nw>=25)&(S.span>=8)].copy()
print("eligible (speech_s>=5, nw>=25, span>=8): %d from %d speakers"%(len(E),E.pid.nunique()))
POS=E.clearly_positive==1
NEG=E.clearly_negative==1
# px_build conflict rule, line 159-160, lexicon swapped for sentiment
CONF=((E.y==1)&(E.sev>=15)&POS)|((E.y==0)&(E.sev<=4)&NEG)
# agreement: same speaker's closest-length segment whose sentiment MATCHES the label
AGRpool=((E.y==1)&NEG)|((E.y==0)&POS)
conf=E[CONF].copy()
conf["direction"]=np.where(conf.y==1,"depressed_positive_words","control_negative_words")
print("\n=== NEW CONFLICT SET (sentiment >=0.70) ===")
print("  conflict segments %d, speakers %d"%(len(conf),conf.pid.nunique()))
print("  depressed-positive %d, control-negative %d"%((conf.y==1).sum(),(conf.y==0).sum()))
pool=E[AGRpool]
pairs=[]
for i,r in conf.iterrows():
    cand=pool[(pool.pid==r.pid)&(~pool.index.isin([i]))]
    if len(cand):
        j=(cand.span-r.span).abs().idxmin(); pairs.append((i,j))
    else: pairs.append((i,None))
Pp=pd.DataFrame(pairs,columns=["conf_idx","ctrl_idx"])
unmatched=Pp.ctrl_idx.isna().sum()
Pp=Pp[Pp.ctrl_idx.notna()].copy(); Pp["ctrl_idx"]=Pp.ctrl_idx.astype(int)
print("  matched pairs %d, unmatched %d"%(len(Pp),unmatched))
cm=E.loc[Pp.conf_idx].copy(); am=E.loc[Pp.ctrl_idx].copy()
keep=pd.concat([cm.assign(set="conflict",pair_id=range(len(cm)),seg_uid=["c%06d"%k for k in cm.index]),
                am.assign(set="agreement",pair_id=range(len(am)),seg_uid=["a%06d"%k for k in am.index])])
keep["speaker_id"]=keep.pid; keep["label"]=keep.y
print("  total rows %d, unique segments %d, speakers %d"%(
    len(keep),keep.groupby(['set','seg_uid']).ngroups,keep.speaker_id.nunique()))
# overlap with the old 390
oldkeys=set(zip(OLD.set,OLD.seg_uid)); newkeys=set(zip(keep.set,keep.seg_uid))
oldseg=set(OLD.seg_uid); newseg=set(keep.seg_uid)
print("\n=== OVERLAP WITH THE OLD 390 ===")
print("  old rows %d (unique seg %d), new rows %d (unique seg %d)"%(
    len(OLD),OLD.groupby(['set','seg_uid']).ngroups,len(keep),keep.groupby(['set','seg_uid']).ngroups))
print("  same (set,seg_uid): %d"%len(oldkeys&newkeys))
print("  seg_uid in both regardless of arm: %d"%len(oldseg&newseg))
print("  new-only seg_uid: %d   old-only seg_uid: %d"%(len(newseg-oldseg),len(oldseg-newseg)))
print("  speakers: old %d, new %d, shared %d"%(
    OLD.speaker_id.nunique(),keep.speaker_id.nunique(),
    len(set(OLD.speaker_id)&set(keep.speaker_id))))
cols=["seg_uid","set","pair_id","speaker_id","label","sev","start","end","span","speech_s","nw",
      "n_turns","p_pos","p_neg","p_neu","clearly_positive","clearly_negative","val","npos","nneg","oof","text"]
cols=[c for c in cols if c in keep.columns]
keep[cols].to_csv(f"{R}/part14_manifest_new.csv",index=False)
json.dump({"part":"14","scorer":"cardiffnlp/twitter-roberta-base-sentiment-latest","threshold":0.70,
 "rules_kept_from_px_build":"speech_s>=5, nw>=25, span>=8; conflict = (y==1 & sev>=15 & clearly positive) or (y==0 & sev<=4 & clearly negative); agreement = same speaker's closest-span segment whose sentiment matches the label",
 "n_conflict":int((keep.set=="conflict").sum()),"n_agreement":int((keep.set=="agreement").sum()),
 "n_pairs":int(len(Pp)),"unmatched":int(unmatched),"speakers":int(keep.speaker_id.nunique()),
 "depressed_positive":int((cm.y==1).sum()),"control_negative":int((cm.y==0).sum()),
 "overlap_same_arm_and_seg":len(oldkeys&newkeys),"overlap_seg_any_arm":len(oldseg&newseg),
 "new_only_segments":len(newseg-oldseg),"old_only_segments":len(oldseg-newseg),
 "sentiment_vs_old_lexicon_agreement_on_390":0.9359,
 "sentiment_vs_old_lexicon_kappa_on_390":0.8718,
 "AFFECTED":"D13 dq_lex unavailable; this is the citable replacement"},
 open(f"{R}/part14_manifest_new.json","w"),indent=1)
print("\nwrote part14_manifest_new.csv / .json")
