import csv,hashlib,os,re
from collections import defaultdict
R="/project2/msoleyma_946/speech_health/results_chaitanya"
rows=list(csv.DictReader(open(f"{R}/manifests/pcgita.csv")))
print("manifest clips:",len(rows),"speakers:",len(set(r["speaker_id"] for r in rows)))
h=defaultdict(list)
for r in rows:
    p=r["filepath"]
    try:
        with open(p,"rb") as f: d=hashlib.md5(f.read()).hexdigest()
    except Exception: continue
    h[d].append((r["speaker_id"],r["label"],os.path.basename(p)))
dupes={k:v for k,v in h.items() if len(v)>1}
print("\nfiles with identical content:",len(dupes))
cross=0
for k,v in dupes.items():
    spk={s for s,_,_ in v}
    if len(spk)>1:
        cross+=1
        if cross<=12:
            print("  SAME AUDIO, DIFFERENT SPEAKER IDS:")
            for s,l,n in v: print("      %-18s label=%s  %s"%(s,l,n))
print("\nduplicate groups spanning more than one speaker id:",cross)
