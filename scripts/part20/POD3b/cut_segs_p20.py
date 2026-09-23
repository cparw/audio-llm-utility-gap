"""Cut E-DAIC segments from the raw per-participant tarball *_AUDIO.wav.
Adapted from podE_final/edaic_conflict/cut_from_tarballs.py, same direct slice, no remap
(transcript timestamps index the tarball audio). Output name is seg_<index>.wav where index
is the row index in part14_segments_sentiment.csv."""
import os, io, csv, sys, wave, time, tarfile, subprocess, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
BASE="https://dcapswoz.ict.usc.edu/wwwedaic/data"
OUT="/workspace/p20/clips"; TMP="/workspace/p20/tmp"
os.makedirs(OUT,exist_ok=True); os.makedirs(TMP,exist_ok=True)
rows=list(csv.DictReader(open("/workspace/p20/cut_list.csv")))
need=collections.defaultdict(list)
for r in rows: need[int(r["pid"])].append(r)
todo=[p for p in need if not all(os.path.exists(f"{OUT}/seg_{int(r['idx']):06d}.wav") for r in need[p])]
print(f"segments {len(rows)}, speakers {len(need)}, speakers still to do {len(todo)}",flush=True)
t0=time.time(); done=[]; fail=[]
def one(pid):
    tgz=f"{TMP}/{pid}_P.tar.gz"
    try:
        rc=subprocess.run(["curl","-sS","--fail","--max-time","1800","-o",tgz,
                           f"{BASE}/{pid}_P.tar.gz"],capture_output=True)
        if rc.returncode!=0:
            fail.append((pid,"download",rc.stderr.decode()[:120])); return
        raw=None
        with tarfile.open(tgz,"r:gz") as t:
            for m in t:
                if m.name.lower().endswith("audio.wav"):
                    raw=t.extractfile(m).read(); break
        os.remove(tgz)
        if raw is None: fail.append((pid,"no_audio_member","")); return
        w=wave.open(io.BytesIO(raw))
        sr,sw,nch=w.getframerate(),w.getsampwidth(),w.getnchannels()
        data=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
        if nch>1: data=data.reshape(-1,nch).mean(axis=1).astype(np.int16)
        for r in need[pid]:
            s,e=float(r["start"]),float(r["end"])
            a,b=int(s*sr),int(min(e,len(data)/sr)*sr)
            seg=data[a:b]
            if len(seg)<sr:
                fail.append((int(r["idx"]),"too_short",f"{len(seg)/sr:.2f}s")); continue
            o=wave.open(f"{OUT}/seg_{int(r['idx']):06d}.wav","w")
            o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr)
            o.writeframes(seg.tobytes()); o.close()
        done.append(pid)
        if len(done)%20==0:
            print(f"{len(done)}/{len(todo)} speakers  {time.time()-t0:.0f}s  fails {len(fail)}",flush=True)
    except Exception as ex:
        fail.append((pid,"exc",str(ex)[:150]))
        if os.path.exists(tgz):
            try: os.remove(tgz)
            except Exception: pass
with ThreadPoolExecutor(max_workers=int(os.environ.get("NW","24"))) as ex: list(ex.map(one,todo))
n=len([f for f in os.listdir(OUT) if f.endswith(".wav")])
print(f"DONE speakers ok {len(done)} fails {len(fail)} wavs on disk {n}  {time.time()-t0:.0f}s",flush=True)
for f in fail[:40]: print("FAIL",f,flush=True)
