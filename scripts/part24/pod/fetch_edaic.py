"""Fetch E-DAIC <pid>_P.tar.gz (open URL) for the 275 pids in windows_full.csv.
Keeps only <pid>_AUDIO.wav -> /workspace/edaicfull/wav and <pid>_Transcript.csv -> /workspace/edaicfull/tr
(the layout build_windows.py reads). Records the sha256 and byte size of every tarball. Resumable."""
import os, csv, sys, time, tarfile, hashlib, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor
BASE = "https://dcapswoz.ict.usc.edu/wwwedaic/data"
D = "/workspace/edaicfull"; TMP = os.environ.get("TGZ_TMP", "/root/tgz")
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for d in (f"{D}/wav", f"{D}/tr", TMP): os.makedirs(d, exist_ok=True)
pids = [r["pid"] for r in csv.DictReader(open(f"{D}/windows_full.csv"))]
SHA = f"{D}/tarball_sha256.tsv"
have = set(l.split("\t")[0] for l in open(SHA)) if os.path.exists(SHA) else set()
todo = [p for p in pids if not (p in have and os.path.exists(f"{D}/wav/{p}_AUDIO.wav") and os.path.exists(f"{D}/tr/{p}_Transcript.csv"))]
print(f"{len(pids)} pids, {len(todo)} to fetch, workers {NW}", flush=True)
t0 = time.time(); done = []; fail = []; nbytes = [0]
def one(pid):
    tgz = f"{TMP}/{pid}_P.tar.gz"
    for attempt in range(5):
        try:
            rc = subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "--max-time", "3600", "-o", tgz, f"{BASE}/{pid}_P.tar.gz"], capture_output=True)
            if rc.returncode != 0: raise RuntimeError("curl " + rc.stderr.decode()[:120])
            h = hashlib.sha256(); sz = 0
            with open(tgz, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""): h.update(b); sz += len(b)
            got = {}
            with tarfile.open(tgz, "r:gz") as t:
                for m in t:
                    n = m.name.lower()
                    if n.endswith("_audio.wav") and "wav" not in got:
                        with open(f"{D}/wav/{pid}_AUDIO.wav.part", "wb") as o: shutil.copyfileobj(t.extractfile(m), o)
                        got["wav"] = m.name
                    elif n.endswith("_transcript.csv") and "tr" not in got:
                        with open(f"{D}/tr/{pid}_Transcript.csv.part", "wb") as o: shutil.copyfileobj(t.extractfile(m), o)
                        got["tr"] = m.name
            if set(got) != {"wav", "tr"}: raise RuntimeError(f"members missing {got}")
            os.replace(f"{D}/wav/{pid}_AUDIO.wav.part", f"{D}/wav/{pid}_AUDIO.wav")
            os.replace(f"{D}/tr/{pid}_Transcript.csv.part", f"{D}/tr/{pid}_Transcript.csv")
            os.remove(tgz)
            with open(SHA, "a") as f: f.write(f"{pid}\t{h.hexdigest()}\t{sz}\n")
            done.append(pid); nbytes[0] += sz
            if len(done) % 10 == 0:
                el = time.time() - t0
                print(f"{len(done)}/{len(todo)} {el:.0f}s {nbytes[0]/1e9:.2f} GB {nbytes[0]/1e6/el:.1f} MB/s fails {len(fail)}", flush=True)
            return
        except Exception as ex:
            if os.path.exists(tgz): os.remove(tgz)
            if attempt == 4: fail.append((pid, str(ex)[:160])); print("FAIL", pid, ex, flush=True)
            time.sleep(3 + 5 * attempt)
with ThreadPoolExecutor(max_workers=NW) as ex: list(ex.map(one, todo))
nw = len([f for f in os.listdir(f"{D}/wav") if f.endswith("_AUDIO.wav")])
print(f"FETCH_DONE ok {len(done)} fails {len(fail)} wav_on_disk {nw} {time.time()-t0:.0f}s {nbytes[0]/1e9:.2f} GB", flush=True)
for f in fail: print("FAIL", f, flush=True)
