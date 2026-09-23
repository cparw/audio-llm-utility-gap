"""Fetch 20 complete (untruncated) E-DAIC _AUDIO.wav and report their durations,
so we can tell whether /workspace/data/full/<pid>.wav in the original run was the raw
tarball session audio or the participant-only stitched audio."""
import os, io, csv, wave, tarfile, subprocess, numpy as np
BASE="https://dcapswoz.ict.usc.edu/wwwedaic/data"
OUT="/workspace/data/fullraw"; os.makedirs(OUT, exist_ok=True)
pids=[int(r["speaker"]) for r in csv.DictReader(open("/workspace/mf_full_rebuilt.csv"))][:20]
for pid in pids:
    dst=f"{OUT}/{pid}.wav"
    if os.path.exists(dst): continue
    tgz=f"/tmp/{pid}_P.tar.gz"
    rc=subprocess.run(["curl","-sS","--fail","--max-time","900","-o",tgz,f"{BASE}/{pid}_P.tar.gz"],capture_output=True)
    if rc.returncode!=0: print("dlfail",pid,flush=True); continue
    raw=None
    with tarfile.open(tgz,"r:gz") as t:
        for m in t:
            if m.name.lower().endswith("audio.wav"): raw=t.extractfile(m).read(); break
    os.remove(tgz)
    w=wave.open(io.BytesIO(raw)); sr,nch=w.getframerate(),w.getnchannels()
    d=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16)
    if nch>1: d=d.reshape(-1,nch).mean(axis=1).astype(np.int16)
    o=wave.open(dst,"w"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr)
    o.writeframes(d.tobytes()); o.close()
    print(f"{pid} sr={sr} dur={len(d)/sr:.1f}s", flush=True)
w=csv.writer(open("/workspace/mf_fullraw.csv","w",newline="")); w.writerow(["path","label","speaker","arm","id"])
lab={int(r["speaker"]):r["label"] for r in csv.DictReader(open("/workspace/mf_full_rebuilt.csv"))}
for pid in pids:
    if os.path.exists(f"{OUT}/{pid}.wav"): w.writerow([f"{OUT}/{pid}.wav",lab[pid],pid,"full",pid])
print("manifest written",flush=True)
