import sys, os, json, numpy as np, pandas as pd, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT="<local data dir>/Desktop/release/edaic_rerun/part16/POD1"
F=f"{OUT}/p14_o25_greedy.csv"
d=pd.read_csv(F)
print("rows",len(d),"speakers",d.spk.nunique(),"arms",d.arm.value_counts().to_dict())

# cross-check the logit p_yes against the existing Part 14 audio run
ref=pd.read_csv("scores/edaic_conflict_v2/p14_o25_zeroshot_scores.csv")
man=pd.read_csv("manifests/part14_manifest_new.csv")
man['key']=man['set']+'_'+man.speaker_id.astype(str)+'_'+man.seg_uid
assert (d.id.values==man.key.values).all() and (ref['clip'].str.replace('.wav','',regex=False).values==man.key.values).all()
md=float(np.abs(d.p_yes.values-ref.p_yes.values).max())
print(f"max |p_yes greedy - p_yes part14 audio| = {md:.3e}   (same forward pass definition)")

d['fw_norm']=d.first_word.fillna("").astype(str).str.replace(r"[^A-Za-z]","",regex=True).str.lower()
d['is_yn']=d.fw_norm.isin(["yes","no"])
share=d.is_yn.mean()
sub=d[d.is_yn]
agree=(sub.fw_norm==sub.logit_decision.str.lower()).mean()
print(f"\nSHARE first word is Yes/No : {share:.4f}  ({d.is_yn.sum()}/{len(d)})")
print(f"AGREEMENT gen vs logit>0.5 : {agree:.4f}  (over the {len(sub)} Yes/No clips)")
agree_all=((d.fw_norm==d.logit_decision.str.lower())).mean()
print(f"AGREEMENT over ALL {len(d)} clips (non-Yes/No counted as disagree) = {agree_all:.4f}")

rows=[]
print("\nBY ARM:")
for arm in ("conflict","agreement"):
    a=d[d.arm==arm]; s=a[a.is_yn]
    sh=a.is_yn.mean(); ag=(s.fw_norm==s.logit_decision.str.lower()).mean()
    print(f"  {arm:10s} n={len(a)} share_yes_no={sh:.4f} ({a.is_yn.sum()}/{len(a)})  agreement={ag:.4f} (n={len(s)})")
    rows.append((f"1f_share_yesno_{arm}",f"share first generated word is Yes/No, {arm} arm",sh,"","",len(a),a.spk.nunique(),F))
    rows.append((f"1f_agree_gen_logit_{arm}",f"agreement generated word vs logit decision, {arm} arm",ag,"","",len(s),s.spk.nunique(),F))

print("\nYES/NO BREAKDOWN of the generated first word:")
print(d.fw_norm.value_counts().head(10).to_string())
other=d[~d.is_yn]
print(f"\nNON-Yes/No clips: {len(other)}")
if len(other):
    print("10 most common first tokens when it is neither Yes nor No:")
    for w,c in collections.Counter(other.first_word.fillna("<empty>").astype(str)).most_common(10):
        print(f"   {w!r:20s} {c}")
else:
    print("  none - the model emitted Yes or No as the first word on every clip.")

json.dump({"n":len(d),"share_yes_no":round(float(share),6),
  "agreement_gen_vs_logit_over_yesno":round(float(agree),6),
  "agreement_over_all_clips":round(float(agree_all),6),
  "by_arm":{a:{"n":int((d.arm==a).sum()),
               "share_yes_no":round(float(d[d.arm==a].is_yn.mean()),6),
               "agreement":round(float((lambda s:(s.fw_norm==s.logit_decision.str.lower()).mean())(d[(d.arm==a)&d.is_yn])),6)}
            for a in ("conflict","agreement")},
  "first_word_counts":{str(k):int(v) for k,v in d.fw_norm.value_counts().head(15).items()},
  "non_yesno_first_tokens":{str(w):int(c) for w,c in collections.Counter(other.first_word.fillna("<empty>").astype(str)).most_common(10)},
  "max_abs_diff_vs_part14_audio_p_yes":md},
  open(f"{OUT}/p14_o25_greedy_analysis.json","w"),indent=1)

rows=[("1f_share_yesno_all","share first generated word is Yes/No, all clips",share,"","",len(d),d.spk.nunique(),F),
      ("1f_agree_gen_logit_all","agreement generated word vs logit decision, Yes/No clips",agree,"","",len(sub),sub.spk.nunique(),F)]+rows
with open("reports/part16/rows/POD1.tsv","a") as fh:
    for r in rows:
        fh.write("\t".join([r[0],r[1]]+[f"{v:.4f}" if isinstance(v,float) else str(v) for v in r[2:5]]+[str(r[5]),str(r[6]),r[7]])+"\n")
print("\nrows appended:",len(rows))
