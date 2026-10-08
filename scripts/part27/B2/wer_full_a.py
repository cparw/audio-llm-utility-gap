# WER of the 400 Hz low-passed transcripts against the unfiltered transcripts (reference), computation A (jiwer if present, else rapidfuzz-free DP).
import json, glob, re, sys, numpy as np
rows=[r for f in sorted(glob.glob(sys.argv[1]+'/b2f_transcripts_*.json')) for r in json.load(open(f))['rows']]
def norm(t): t=t.lower(); t=re.sub(r"[^a-z0-9' ]+"," ",t); return t.split()
def ed(a,b):
    d=list(range(len(b)+1))
    for i in range(1,len(a)+1):
        p=d[:]; d[0]=i
        for j in range(1,len(b)+1): d[j]=min(p[j]+1,d[j-1]+1,p[j-1]+(a[i-1]!=b[j-1]))
    return d[-1]
E=np.array([ed(norm(r['unfiltered']),norm(r['lp400'])) for r in rows],float); N=np.array([len(norm(r['unfiltered'])) for r in rows],float)
pooled=E.sum()/N.sum(); per=E/np.maximum(N,1)
rng=np.random.default_rng(0); bs=[]
for _ in range(2000):
    i=rng.choice(len(rows),len(rows),replace=True); bs.append(E[i].sum()/N[i].sum())
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"n_windows {len(rows)} ref_words {int(N.sum())} hyp_words {sum(len(norm(r['lp400'])) for r in rows)}")
print(f"pooled WER {pooled:.4f} [{lo:.4f}, {hi:.4f}]  median per-window {np.median(per):.4f}  min {per.min():.4f} max {per.max():.4f}")
json.dump({"pooled_wer":pooled,"ci":[lo,hi],"per_window":dict(zip([r['pid'] for r in rows],per.tolist())),"ref_words":int(N.sum())},open(sys.argv[1]+'/wer_full_a.json','w'),indent=1)
