"""Transcripts for the 237 ADReSSo clips with faster-whisper on CPU."""
import csv, os
from faster_whisper import WhisperModel
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
rows = list(csv.DictReader(open(f"{D}/adresso/adresso_manifest.csv")))
m = WhisperModel("small.en", device="cpu", compute_type="int8")
out = f"{D}/adresso/adresso_transcripts.csv"
done = set()
if os.path.exists(out):
    done = {r["spk"] for r in csv.DictReader(open(out))}
fh = open(out, "a" if done else "w", newline="")
w = csv.writer(fh)
if not done: w.writerow(["spk","label","text"])
for i, r in enumerate(rows):
    if r["spk"] in done: continue
    segs, _ = m.transcribe(f"{D}/{r['segment_path']}", language="en", vad_filter=True)
    text = " ".join(s.text.strip() for s in segs)
    w.writerow([r["spk"], r["label"], text]); fh.flush()
    if (i+1)%25==0: print(f"transcribed {i+1}/{len(rows)}", flush=True)
print("transcripts done", flush=True)
