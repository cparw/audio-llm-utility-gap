import sys,os,json,csv,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_fp import decode,mfcc
from multiprocessing import Pool
S=os.path.dirname(os.path.abspath(__file__))
sets=json.load(open(f"{S}/sets.json"))
NFR=2900   # ~29 s of frames

def seq(p):
    try:
        x=decode(p,dur=30)
        m=mfcc(x)[:,1:]           # drop c0 -> gain invariant
        if len(m)<NFR: m=np.pad(m,((0,NFR-len(m)),(0,0)))
        m=m[:NFR]
        m=m-m.mean(0); s=m.std(0); s[s<1e-6]=1e-6; m=m/s
        v=m.ravel(); v=v-v.mean(); n=np.linalg.norm(v)
        return (p, (v/n).astype(np.float32) if n>0 else v.astype(np.float32))
    except Exception as e:
        return (p,None)

if __name__=="__main__":
    pitt=sets["pitt_cookie_source_mp3"]; ao=sets["adresso_source_mp3"]; ad=sets["adress2020_fullwave"]
    allp=pitt+ao+ad
    with Pool(8) as pool: res=dict(pool.map(seq,allp,chunksize=4))
    P=[p for p in pitt if res.get(p) is not None]
    M=np.stack([res[p] for p in P])
    def best(q_list,name):
        rows=[]
        Q=np.stack([res[p] for p in q_list if res.get(p) is not None])
        names=[p for p in q_list if res.get(p) is not None]
        C=Q@M.T
        for i,q in enumerate(names):
            o=np.argsort(-C[i])
            rows.append(dict(query=os.path.basename(q), set=name,
                             best_pitt=P[o[0]].split("/")[-3]+":"+os.path.basename(P[o[0]])[:-4],
                             r1=round(float(C[i,o[0]]),4),
                             second=P[o[1]].split("/")[-3]+":"+os.path.basename(P[o[1]])[:-4],
                             r2=round(float(C[i,o[1]]),4)))
        return rows
    rows=best(ao,"adresso_source_mp3")+best(ad,"adress2020_fullwave")
    out=f"{S}/xcorr_to_pitt.csv"
    with open(out,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    print("wrote",out,len(rows))
    for nm in ["adresso_source_mp3","adress2020_fullwave"]:
        rr=[r for r in rows if r["set"]==nm]
        r1=np.array([r["r1"] for r in rr]); r2=np.array([r["r2"] for r in rr])
        print(f"{nm}: n={len(rr)} r1 median={np.median(r1):.4f} min={r1.min():.4f} ; r1>0.9:{int((r1>0.9).sum())} r1>0.7:{int((r1>0.7).sum())} r1>0.5:{int((r1>0.5).sum())}; median(r1-r2)={np.median(r1-r2):.4f}")
