import pandas as pd, re, numpy as np
P="/project2/msoleyma_946/speech_health/results_chaitanya/committed/paradox"
M=pd.read_csv(f"{P}/manifest.csv")
C=M[M.set=="conflict"]; Ag=M[M.set=="agreement"]
neg=C[C.label==0]; pos=C[C.label==1]
G=re.compile(r"guilt",re.I); A=re.compile(r"(mad|angry|anger|argument|arguing)",re.I)
H=re.compile(r"(enjoy|fun|happy|love|like to|hobb)",re.I)
print("control-negative segs (n=%d): guilt %d, anger %d, either %d"%(len(neg),neg.text.str.contains(G).sum(),neg.text.str.contains(A).sum(),(neg.text.str.contains(G)|neg.text.str.contains(A)).sum()))
print("depressed-positive segs (n=%d): hobby/enjoy language %d"%(len(pos),pos.text.str.contains(H).sum()))
Qmid=re.compile(r"(what do you|how would you|tell me about|do you feel|have you ever|what are you|how have you|when did you|why did you)",re.I)
print("\nresidual mid-segment interviewer stems: conflict %d/%d  agreement %d/%d"%(C.text.str.contains(Qmid).sum(),len(C),Ag.text.str.contains(Qmid).sum(),len(Ag)))
S=pd.read_csv(f"{P}/daic_segments_scored.csv")
print("all %d segments with residual stems: %d (%.1f%%)"%(len(S),S.text.fillna("").str.contains(Qmid).sum(),100*S.text.fillna("").str.contains(Qmid).mean()))
print("\nconflict dur: median %.1f IQR %.1f-%.1f min %.1f max %.1f"%(C.clip_dur_s.median(),C.clip_dur_s.quantile(.25),C.clip_dur_s.quantile(.75),C.clip_dur_s.min(),C.clip_dur_s.max()))
print("agreement dur: median %.1f IQR %.1f-%.1f"%(Ag.clip_dur_s.median(),Ag.clip_dur_s.quantile(.25),Ag.clip_dur_s.quantile(.75)))
d=np.abs(C.sort_values("pair_id").clip_dur_s.values-Ag.sort_values("pair_id").clip_dur_s.values)
print("paired |dur delta|: median %.1fs p75 %.1fs"%(np.median(d),np.percentile(d,75)))
print("\nclips on disk:", len(set(M.filepath)))
