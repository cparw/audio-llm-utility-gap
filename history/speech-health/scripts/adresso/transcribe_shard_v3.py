import csv, os, sys
from faster_whisper import WhisperModel
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
shard, nshards = int(sys.argv[1]), int(sys.argv[2])
rows = [r for i, r in enumerate(csv.DictReader(open(f"{D}/adresso/adresso_manifest_patient.csv"))) if i % nshards == shard]
out = f"{D}/adresso/transcripts_v3_shard{shard}.csv"
done = set()
if os.path.exists(out):
    done = {r["spk"] for r in csv.DictReader(open(out))}
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=3)
fh = open(out, "a" if done else "w", newline="")
w = csv.writer(fh)
if not done: w.writerow(["spk","label","text"])
for i, r in enumerate(rows):
    if r["spk"] in done: continue
    segs, _ = m.transcribe(f"{D}/{r['segment_path']}", language="en", vad_filter=True)
    w.writerow([r["spk"], r["label"], " ".join(s.text.strip() for s in segs)]); fh.flush()
    if (i+1)%20==0: print(f"shard {shard}: {i+1}/{len(rows)}", flush=True)
print(f"shard {shard} done", flush=True)
