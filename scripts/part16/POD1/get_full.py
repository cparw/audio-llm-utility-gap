"""Fetch the E-DAIC full-session participant audio for the 275 sft speakers.
Saves the FIRST 60 s of each tarball's *_AUDIO.wav (sft_projector.py only ever reads x[:16000*30])
to /workspace/data/full/<pid>.wav, matching the path layout of the original sft_full run."""
import os, io, csv, wave, time, tarfile, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
BASE="https://dcapswoz.ict.usc.edu/wwwedaic/data"
OUT="/workspace/data/full"; TMP="/workspace/full_tmp"; KEEP=60
os.makedirs(OUT,exist_ok=True); os.makedirs(TMP,exist_ok=True)
pids=[int(r["speaker"]) for r in csv.DictReader(open("/workspace/mf_full_rebuilt.csv"))]
todo=[p for p in pids if not os.path.exists(f"{OUT}/{p}.wav")]
print(f"{len(pids)} speakers, {len(todo)} to fetch",flush=True)
t0=time.time(); done=[]; fail=[]
def one(pid):
    tgz=f"{TMP}/{pid}_P.tar.gz"
    for attempt in range(3):
        try:
            rc=subprocess.run(["curl","-sS","--fail","--max-time","1800","-o",tgz,
                               f"{BASE}/{pid}_P.tar.gz"],capture_output=True)
            if rc.returncode!=0:
                if attempt==2: fail.append((pid,"download",rc.stderr.decode()[:100]))
                time.sleep(2); continue
            raw=None
            with tarfile.open(tgz,"r:gz") as t:
                for m in t:
                    if m.name.lower().endswith("audio.wav"):
                        raw=t.extractfile(m).read(); break
            os.remove(tgz)
            if raw is None: fail.append((pid,"no_audio","")); return
            w=wave.open(io.BytesIO(raw)); sr,nch=w.getframerate(),w.getnchannels()
            d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
            if nch>1: d=d.reshape(-1,nch).mean(axis=1).astype(np.int16)
            d=d[:sr*KEEP]
            o=wave.open(f"{OUT}/{pid}.wav","w"); o.setnchannels(1); o.setsampwidth(2)
            o.setframerate(sr); o.writeframes(d.tobytes()); o.close()
            done.append(pid)
            if len(done)%25==0: print(f"{len(done)}/{len(todo)} {time.time()-t0:.0f}s fails {len(fail)}",flush=True)
            return
        except Exception as ex:
            if os.path.exists(tgz):
                try: os.remove(tgz)
                except Exception: pass
            if attempt==2: fail.append((pid,"exc",str(ex)[:120]))
            time.sleep(2)
with ThreadPoolExecutor(max_workers=6) as ex: list(ex.map(one,todo))
print(f"DONE ok {len(done)} fails {len(fail)} on disk {len(os.listdir(OUT))} {time.time()-t0:.0f}s",flush=True)
for f in fail[:30]: print("FAIL",f,flush=True)
