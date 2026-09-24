import os, csv, glob, tarfile, argparse
import numpy as np, soundfile as sf
BASE="/project2/msoleyma_946/speech_health/DCAPS_challenge"
PROC="/scratch1/parwatka/dcaps_proc"
OUTMAN="/project2/msoleyma_946/speech_health/results_chaitanya/manifests/dcaps.csv"

def load_labels():
    lab={}
    for s in ("train","dev","test"):
        p=f"{BASE}/labels/{s}_split.csv"
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)):
            pid=(r.get("Participant_ID") or "").strip()
            b=r.get("PHQ_Binary") or r.get("PHQ8_Binary") or ""
            if pid and str(b).strip() not in ("",): lab[pid]=int(float(b))
    return lab

def process(tgz, labels, tmp):
    pid=os.path.basename(tgz).split("_")[0]
    if pid not in labels: return None
    with tarfile.open(tgz,"r:gz") as t:
        names=t.getnames()
        aw=[n for n in names if n.endswith("_AUDIO.wav")]; tr=[n for n in names if n.endswith("_Transcript.csv")]
        if not aw or not tr: return None
        t.extract(aw[0],tmp); t.extract(tr[0],tmp)
        wavp=os.path.join(tmp,aw[0]); trp=os.path.join(tmp,tr[0])
    y,sr=sf.read(wavp)
    if getattr(y,"ndim",1)>1: y=y.mean(1)
    segs=[]
    for r in csv.DictReader(open(trp)):
        try: s=float(r["Start_Time"]); e=float(r["End_Time"])
        except Exception: continue
        if e>s: segs.append(y[int(s*sr):int(e*sr)])
    if not segs: return None
    os.makedirs(PROC,exist_ok=True); outp=f"{PROC}/{pid}.wav"
    sf.write(outp,np.concatenate(segs),sr)
    for f in (wavp,trp):
        try: os.remove(f)
        except Exception: pass
    return dict(filepath=outp,dataset="dcaps",speaker_id=pid,label=labels[pid],
        task_type="interview",language="english",age="",sex="",group="",transcript="")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    labels=load_labels(); print("labels loaded:",len(labels),flush=True)
    tmp="/scratch1/parwatka/dcaps_tmp2"; os.makedirs(tmp,exist_ok=True)
    tgzs=sorted(glob.glob(f"{BASE}/data/*_P.tar.gz"));  tgzs=tgzs[:a.limit] if a.limit else tgzs
    rows=[]
    for i,tgz in enumerate(tgzs):
        try:
            r=process(tgz,labels,tmp)
            if r: rows.append(r)
        except Exception as e: print("FAIL",tgz,repr(e),flush=True)
        if (i+1)%25==0: print(f"  {i+1}/{len(tgzs)} done",flush=True)
    if not a.limit or a.limit>=len(glob.glob(f"{BASE}/data/*_P.tar.gz")):
        os.makedirs(os.path.dirname(OUTMAN),exist_ok=True)
        with open(OUTMAN,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["filepath","dataset","speaker_id","label","task_type","language","age","sex","group","transcript"]); w.writeheader(); w.writerows(rows)
        print("WROTE MANIFEST",OUTMAN,flush=True)
    print(f"dcaps: {len(rows)} participants, depressed={sum(r['label'] for r in rows)} nondep={sum(1-r['label'] for r in rows)}",flush=True)

if __name__=="__main__": main()
