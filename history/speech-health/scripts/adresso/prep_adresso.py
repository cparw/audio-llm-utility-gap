"""Cut the ADReSSo recordings (ADReSS-M english train) into 30 s wavs + manifest."""
import csv, os, subprocess
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
A = f"{D}/DementiaBank/challenges/ADReSS-M"
SRC = f"{A}/ADReSS-M-train_x/train"
OUT = f"{D}/adresso/segments"
os.makedirs(OUT, exist_ok=True)
gt = {r["adressfname"]: r for r in csv.DictReader(open(f"{A}/ADReSS-M-train_x/training-groundtruth.csv"))}
rows = []
for f in sorted(os.listdir(SRC)):
    if not f.endswith(".mp3"): continue
    fid = f[:-4]
    if fid not in gt: print("no label for", fid); continue
    r = gt[fid]
    wav = f"{OUT}/{fid}.wav"
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i",f"{SRC}/{f}",
                        "-ac","1","-ar","16000","-t","30",wav], check=True)
    rows.append(dict(spk=fid, label=1 if r["dx"]=="ProbableAD" else 0,
                     mmse=r["mmse"], age=r["age"],
                     segment_path=f"adresso/segments/{fid}.wav"))
with open(f"{D}/adresso/adresso_manifest.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
print(f"manifest {len(rows)} rows, {sum(r['label'] for r in rows)} alzheimers", flush=True)
