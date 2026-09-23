import sys,os,json,glob,csv,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_fp import decode,mfcc
from multiprocessing import Pool
S=os.path.dirname(os.path.abspath(__file__))
NFR=2900
def seq(p):
    try:
        x=decode(p,dur=30); m=mfcc(x)[:,1:]
        if len(m)<NFR: m=np.pad(m,((0,NFR-len(m)),(0,0)))
        m=m[:NFR]; m=m-m.mean(0); s=m.std(0); s[s<1e-6]=1e-6; m=m/s
        v=m.ravel(); v=v-v.mean(); n=np.linalg.norm(v)
        return (p,(v/n).astype(np.float32) if n>0 else None)
    except Exception: return (p,None)
if __name__=="__main__":
    HD="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"
    extra=sorted(glob.glob(f"{HD}/DementiaBank/Pitt/0extra/*/*.wav"))
    ao=pd.read_csv(f"{S}/adresso_xcorr.csv")
    un=ao[ao.margin<0.5030]["query"].tolist()
    qs=[f"{HD}/DementiaBank/challenges/ADReSS-M/ADReSS-M-train_x/train/{q}" for q in un]
    print("extra refs:",len(extra),"unmatched queries:",len(qs),flush=True)
    with Pool(8) as pool: res=dict(pool.map(seq,extra+qs,chunksize=4))
    P=[p for p in extra if res.get(p) is not None]
    M=np.stack([res[p] for p in P])
    rows=[]
    for q in qs:
        if res.get(q) is None: continue
        c=M@res[q]; o=np.argsort(-c)
        rows.append(dict(query=os.path.basename(q),best=os.path.basename(P[o[0]]),
                         grp=P[o[0]].split("/")[-2],r1=round(float(c[o[0]]),4),
                         second=os.path.basename(P[o[1]]),r2=round(float(c[o[1]]),4),
                         margin=round(float(c[o[0]]-c[o[1]]),4)))
    pd.DataFrame(rows).to_csv(f"{S}/adresso_unmatched_vs_0extra.csv",index=False)
    df=pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("accepted at margin>=0.503:",int((df.margin>=0.503).sum()),"of",len(df))
