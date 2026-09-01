import csv, os, subprocess, glob
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
A = f"{D}/DementiaBank/challenges/ADReSS-2020"
OUT = f"{D}/adress2020/segments"; os.makedirs(OUT, exist_ok=True)
rows = []
def add(src, sid, split, label):
    wav = f"{OUT}/{sid}.wav"
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i",src,"-ac","1","-ar","16000","-t","30",wav], check=True)
    rows.append(dict(spk=sid, split=split, label=label, segment_path=f"adress2020/segments/{sid}.wav"))
for lab, sub in ((0,"cc"),(1,"cd")):
    for p in sorted(glob.glob(f"{A}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/{sub}/*.wav")):
        add(p, os.path.basename(p)[:-4], "train", lab)
labels = {r["id"]: int(r["label"]) for r in csv.DictReader(open(f"{A}/test_labels_recovered.csv")) if r["label"] not in ("", "None")}
for p in sorted(glob.glob(f"{A}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/Full_wave_enhanced_audio/*.wav")):
    sid = os.path.basename(p)[:-4]
    if sid in labels: add(p, sid, "test", labels[sid])
with open(f"{D}/adress2020/adress2020_manifest.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print(f"{len(rows)} clips: train {sum(r['split']=='train' for r in rows)}, test {sum(r['split']=='test' for r in rows)}")
