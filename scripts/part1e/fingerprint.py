import sys,os,json,csv,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_fp import decode,mfcc,frame_db,md5file
from multiprocessing import Pool
S=os.path.dirname(os.path.abspath(__file__))
sets=json.load(open(f"{S}/sets.json"))

def one(a):
    ds,p=a
    try:
        m5=md5file(p)
        x=decode(p)
        n=len(x); dur=n/16000.0
        db=frame_db(x)
        p5=float(np.percentile(db,5))
        frac6=float(np.mean(db<(p5+6.0)))
        frac50=float(np.mean(db< -50.0))
        head=x[:160000]                       # first 10 s
        mv=mfcc(head).mean(axis=0)
        return dict(dataset=ds,path=p,md5=m5,dur_s=round(dur,4),
                    db_p5=round(p5,4),db_med=round(float(np.median(db)),4),
                    db_max=round(float(db.max()),4),
                    frac_below_floor6=round(frac6,4),frac_below_m50=round(frac50,4),
                    n_frames=len(db),
                    mfcc=";".join(f"{v:.6f}" for v in mv))
    except Exception as e:
        return dict(dataset=ds,path=p,md5="",dur_s=-1,db_p5=0,db_med=0,db_max=0,
                    frac_below_floor6=-1,frac_below_m50=-1,n_frames=0,mfcc="",err=str(e)[:150])

if __name__=="__main__":
    todo=[(ds,p) for ds,ps in sets.items() for p in ps]
    print("files:",len(todo),flush=True)
    out=f"{S}/fp.csv"
    fields=["dataset","path","md5","dur_s","db_p5","db_med","db_max","frac_below_floor6","frac_below_m50","n_frames","mfcc","err"]
    with Pool(8) as pool, open(out,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for i,r in enumerate(pool.imap_unordered(one,todo,chunksize=8)):
            w.writerow(r)
            if (i+1)%250==0: print(i+1,flush=True)
    print("done ->",out,flush=True)
