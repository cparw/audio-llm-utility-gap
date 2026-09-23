"""Re-cut the 794 distinct Part 14 clips on the pod from the raw E-DAIC tarball *_AUDIO.wav, same direct slice
as part16/POD1/cut_segs.py (which re-cut Part 14 clips bit-identically). Each file is checked against the sha256
of the Mac clip in release/edaic_rerun/part14_clips; a mismatch is written to FAIL, never used silently."""
import os, io, csv, wave, time, tarfile, subprocess, hashlib, collections, numpy as np
from concurrent.futures import ThreadPoolExecutor
BASE = "https://dcapswoz.ict.usc.edu/wwwedaic/data"
OUT = "/workspace/p20/part14_clips"; TMP = "/workspace/p20/tmp"
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
rows = list(csv.DictReader(open("/workspace/p20/p20_cut_list.csv")))
def ok(r):
    f = f"{OUT}/{r['name']}"
    return os.path.exists(f) and hashlib.sha256(open(f, "rb").read()).hexdigest() == r["sha256"]
need = collections.defaultdict(list)
for r in rows: need[int(r["pid"])].append(r)
todo = [p for p in need if not all(ok(r) for r in need[p])]
print(f"segments {len(rows)} speakers {len(need)} to do {len(todo)}", flush=True)
t0 = time.time(); done = []; fail = []
def one(pid):
    tgz = f"{TMP}/{pid}_P.tar.gz"
    for attempt in range(3):
        try:
            rc = subprocess.run(["curl", "-sS", "--fail", "--max-time", "1800", "-o", tgz, f"{BASE}/{pid}_P.tar.gz"], capture_output=True)
            if rc.returncode != 0:
                if attempt == 2: fail.append((pid, "download", rc.stderr.decode()[:120]))
                time.sleep(2); continue
            raw = None
            with tarfile.open(tgz, "r:gz") as t:
                for m in t:
                    if m.name.lower().endswith("audio.wav"): raw = t.extractfile(m).read(); break
            os.remove(tgz)
            if raw is None: fail.append((pid, "no_audio", "")); return
            w = wave.open(io.BytesIO(raw)); sr, nch = w.getframerate(), w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if nch > 1: data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
            for r in need[pid]:
                s, e = float(r["start"]), float(r["end"])
                a, b = int(s * sr), int(min(e, len(data) / sr) * sr)
                seg = data[a:b]
                f = f"{OUT}/{r['name']}"
                o = wave.open(f, "w"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes(seg.tobytes()); o.close()
                if hashlib.sha256(open(f, "rb").read()).hexdigest() != r["sha256"]:
                    fail.append((r["name"], "sha_mismatch", "")); os.rename(f, f + ".MISMATCH")
            done.append(pid)
            if len(done) % 10 == 0: print(f"{len(done)}/{len(todo)} speakers {time.time()-t0:.0f}s fails {len(fail)}", flush=True)
            return
        except Exception as ex:
            if attempt == 2: fail.append((pid, "exc", str(ex)[:150]))
            if os.path.exists(tgz):
                try: os.remove(tgz)
                except Exception: pass
with ThreadPoolExecutor(max_workers=int(os.environ.get("NW", "32"))) as ex: list(ex.map(one, todo))
good = sum(ok(r) for r in rows)
print(f"DONE speakers ok {len(done)} fails {len(fail)} sha-verified clips {good}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
for f in fail[:50]: print("FAIL", f, flush=True)
if good == len(rows): open("/workspace/p20/CLIPS_VERIFIED", "w").write(f"{good}\n")
