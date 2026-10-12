import csv,os,re,glob,hashlib,warnings
import pandas as pd, numpy as np
from collections import defaultdict,Counter
warnings.filterwarnings("ignore")
R="/project2/msoleyma_946/speech_health/results_chaitanya"
B="/project2/msoleyma_946/speech_health"

print("="*70); print("1. PC-GITA ROSTER")
md=pd.read_excel(f"{B}/spanish_dataset/Copia de PCGITA_metadata.xlsx","PD+HC")
md.columns=[str(c).strip() for c in md.columns]
ids=[str(x).strip().upper() for x in md["RECODING ORIGINAL NAME"]]
pat=[i for i in ids if "AC" not in i]; ctl=[i for i in ids if "AC" in i]
print(f"  sheet rows {len(ids)}: {len(pat)} patient ids, {len(ctl)} control ids")
rows=list(csv.DictReader(open(f"{R}/manifests/pcgita.csv")))
spk=sorted(set(r["speaker_id"].strip().upper() for r in rows))
extra=[s for s in spk if s not in set(ids)]
print(f"  manifest speakers {len(spk)}, extras not on roster: {len(extra)}")
ex_tasks=Counter(r["task_type"] for r in rows if r["speaker_id"].strip().upper() in set(extra))
print(f"  those extras appear in tasks: {dict(ex_tasks)}")
nopd=sorted(set(r["speaker_id"].strip().upper() for r in rows if "NO PD" in r["filepath"].upper()))
noupdrs=sorted(set(r["speaker_id"].strip().upper() for r in rows if "NO UPDRS" in r["filepath"].upper()))
print(f"  NO PD speakers: {nopd}   all off-roster? {all(s in extra for s in nopd)}")
print(f"  NO UPDRS speakers: {noupdrs}  all off-roster? {all(s in extra for s in noupdrs)}")
lab={r["speaker_id"].strip().upper():r["label"] for r in rows}
print(f"  labels given to NO PD speakers: {[lab[s] for s in nopd]}")

print("="*70); print("2. PC-GITA DUPLICATES (md5 over read task)")
h=defaultdict(list)
for r in rows:
    if r["task_type"]!="read": continue
    try:
        with open(r["filepath"],"rb") as f: h[hashlib.md5(f.read()).hexdigest()].append(r["speaker_id"].strip().upper())
    except Exception: pass
cross=[(k,sorted(set(v))) for k,v in h.items() if len(set(v))>1]
print(f"  duplicate groups spanning >1 speaker id: {len(cross)}")
for k,v in cross: print(f"    {v}")

print("="*70); print("3. NEUROVOZ ESPONTANEA / DDK")
nv=list(csv.DictReader(open(f"{R}/manifests/neurovoz.csv")))
esp=[r for r in nv if "ESPONTANEA" in r["filepath"].upper()]
print(f"  ESPONTANEA rows in manifest: {len(esp)}, task_type assigned: {dict(Counter(r['task_type'] for r in esp))}")
print(f"  their labels: {dict(Counter(r['label'] for r in esp))}")
d1=len(glob.glob(f"{B}/spanish_neurovoz/zenodo_upload/audios/*ESPONTANEA*.wav"))
d2=len(glob.glob(f"{B}/spanish_neurovoz/zenodo_upload_silencemode_1/audios/*ESPONTANEA*.wav"))
print(f"  wavs on disk: zenodo_upload {d1}, silencemode_1 {d2}  -> the 148 was these two dirs")
ddk=[r for r in nv if r["task_type"]=="ddk"]
item=Counter(re.sub(r"^(HC|PD)_","",os.path.basename(r["filepath"]).rsplit(".",1)[0]).rsplit("_",1)[0].upper() for r in ddk)
print(f"  ddk bucket = {len(ddk)} clips, composed of: {dict(item)}")

print("="*70); print("4. ITALIAN SPEAKERS")
it=list(csv.DictReader(open(f"{R}/manifests/italian.csv")))
its=sorted(set(r["speaker_id"] for r in it))
pdspk=sorted(set(r["speaker_id"] for r in it if r["label"]=="1"))
print(f"  manifest speakers {len(its)}, patients {len(pdspk)}, controls {len(its)-len(pdspk)}")
folders=Counter(r["filepath"].split("/")[6] if len(r["filepath"].split("/"))>6 else "?" for r in it)
print(f"  top-level folders: {dict(folders)}")
