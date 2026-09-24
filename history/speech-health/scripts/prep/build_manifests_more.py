import os, csv, re, glob
BASE="/project2/msoleyma_946/speech_health"
OUT="/project2/msoleyma_946/speech_health/results_chaitanya/manifests"
os.makedirs(OUT, exist_ok=True)
VOWELS=set("AEIOU")

def vowel_or_read(tok):
    t=tok.upper()
    if re.fullmatch(r"[AEIOU][0-9]?", t): return "vowel"
    if any(k in t for k in ("PATAKA","PAKATA","PETAKA","KAKA","PAPA","TATA","KA","PA","TA")): return "ddk"
    return "read"

# ---------- NEUROVOZ ----------
nv_base=f"{BASE}/spanish_neurovoz/zenodo_upload"
rows=[]
for grp_file,lab in (("data_hc.csv",0),("data_pd.csv",1)):
    p=f"{nv_base}/metadata/{grp_file}"
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        r={k.strip():v for k,v in r.items()}
        apath=r.get("Audio","").strip()
        if not apath: continue
        fp=f"{nv_base}/audios/"+os.path.basename(apath)
        if not os.path.exists(fp): continue
        fn=os.path.basename(fp)[:-4]; parts=fn.split("_")
        tok=parts[1] if len(parts)>=3 else "read"
        rows.append(dict(filepath=fp,dataset="neurovoz",speaker_id=str(r.get("ID","")).strip(),
            label=lab,task_type=vowel_or_read(tok),language="spanish",
            age=r.get("Age","").strip(),sex=r.get("Sex","").strip(),
            group=("pd" if lab else "hc"),transcript=""))
with open(f"{OUT}/neurovoz.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["filepath","dataset","speaker_id","label","task_type","language","age","sex","group","transcript"]); w.writeheader(); w.writerows(rows)
print(f"neurovoz: {len(rows)} clips, speakers={len(set(r['speaker_id'] for r in rows))}, PD={sum(r['label'] for r in rows)} HC={sum(1-r['label'] for r in rows)}")

# ---------- PC-GITA (spanish_dataset) ----------
pg=f"{BASE}/spanish_dataset"
TASKMAP={"ddk analysis":"ddk","vowels":"vowel","modulated vowels":"vowel","words":"words","monologue":"monologue","read text":"read","sentences":"read"}
rows=[]
for fp in glob.glob(f"{pg}/**/*.wav",recursive=True)+glob.glob(f"{pg}/**/*.WAV",recursive=True):
    low=fp.lower()
    if "/pd/" in low or "patolog" in low: lab=1; g="pd"
    elif "/hc/" in low or "/control/" in low: lab=0; g="hc"
    else: continue
    rel=os.path.relpath(fp,pg); top=rel.split(os.sep)[0].lower()
    task=TASKMAP.get(top,"read")
    m=re.match(r"^([A-Za-z]+\d+)", os.path.basename(fp)); spk=m.group(1) if m else os.path.basename(fp)
    rows.append(dict(filepath=fp,dataset="pcgita",speaker_id=spk,label=lab,task_type=task,
        language="spanish",age="",sex="",group=g,transcript=""))
with open(f"{OUT}/pcgita.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["filepath","dataset","speaker_id","label","task_type","language","age","sex","group","transcript"]); w.writeheader(); w.writerows(rows)
from collections import Counter
tc=Counter(r["task_type"] for r in rows)
print(f"pcgita: {len(rows)} clips, speakers={len(set(r['speaker_id'] for r in rows))}, PD={sum(r['label'] for r in rows)} HC={sum(1-r['label'] for r in rows)}, tasks={dict(tc)}")
