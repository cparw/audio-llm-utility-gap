import csv, os, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
B="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"
DB=B+"/DementiaBank"
P="/Volumes/G-Drive Pro/paper1_local_runs"

paths=set()
# PITT
pcm=list(csv.DictReader(open(DB+"/pitt_conflict_manifest.csv")))
for r in pcm:
    paths.add(DB+"/segments/"+os.path.basename(r["segment_path"]))
    w=f'{DB}/Pitt/{r["grp"]}/cookie/0wav/{r["session"]}.wav'
    m=f'{DB}/Pitt/{r["grp"]}/cookie/{r["session"]}.mp3'
    paths.add(w if os.path.exists(w) else m)
# ADRESSO
am=list(csv.DictReader(open(B+"/adresso/adresso_manifest_patient.csv")))
for r in am:
    paths.add(B+"/adresso/segments_patient/"+r["spk"]+".wav")
    paths.add(DB+"/challenges/ADReSS-M/ADReSS-M-train_x/train/"+r["spk"]+".mp3")
# ADRESS2020
a2=list(csv.DictReader(open(B+"/adress2020/adress2020_manifest_patient.csv")))
A20=DB+"/challenges/ADReSS-2020"
for r in a2:
    paths.add(B+"/adress2020/segments_patient/"+r["spk"]+".wav")
    if r["split"]=="train":
        sub="cd" if r["label"]=="1" else "cc"
        paths.add(f"{A20}/ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/{sub}/{r['spk']}.wav")
    else:
        paths.add(f"{A20}/ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/Full_wave_enhanced_audio/{r['spk']}.wav")
# KCL scored clips + kcl_read30b + source
for r in csv.DictReader(open(P+"/kcl_manifest.csv")):
    paths.add(r["path"].replace("/Users/chaitanyaparwatkar/Desktop/Hard drive data for paper",B))
for r in csv.DictReader(open(P+"/kcl30b_manifest.csv")):
    paths.add(r["path"])
# PCGITA scored (pcgita_read.csv -> clips_local/spanish_dataset)
pg=list(csv.DictReader(open(P+"/drive_data/pcgita_read.csv")))
for r in pg:
    rel=r["filepath"].split("/spanish_dataset/",1)[-1]
    paths.add(P+"/clips_local/spanish_dataset/"+rel)
nv=list(csv.DictReader(open(P+"/drive_data/neurovoz.csv")))
for r in nv:
    rel=r["filepath"].split("/spanish_neurovoz/",1)[-1]
    paths.add(P+"/drive_data/spanish_neurovoz/"+rel)

paths=sorted(paths)
print("n paths",len(paths),file=sys.stderr)
def dur(p):
    if not os.path.exists(p): return (p,None)
    try:
        o=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",p],
                         capture_output=True,text=True,timeout=60).stdout.strip()
        return (p,float(o)) if o else (p,None)
    except Exception: return (p,None)
with ThreadPoolExecutor(max_workers=12) as ex:
    res=dict(ex.map(dur,paths))
json.dump(res,open("durations.json","w"))  # output path was a temporary working directory on the authors' machine
print("done, missing:",sum(1 for v in res.values() if v is None),file=sys.stderr)
