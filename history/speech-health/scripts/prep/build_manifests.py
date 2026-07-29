import os, glob, csv, wave, collections
BASE="/project2/msoleyma_946/speech_health"
OUT=BASE+"/results_chaitanya/manifests"
os.makedirs(OUT, exist_ok=True)

def dur(f):
    try:
        w=wave.open(f,"rb"); d=w.getnframes()/float(w.getframerate()); sr=w.getframerate(); w.close()
        return round(d,2), sr
    except Exception:
        return "", ""

def it_task(fn):
    u=fn.upper()
    if u[:2] in ("VA","VE","VI","VO","VU"): return "vowel"
    if u[:1]=="D" and len(u)>1 and u[1].isdigit(): return "ddk"
    if u[:1]=="B" or u[:2]=="PR" or u[:2]=="FB": return "read"
    return "unknown"

hdr=["filepath","dataset","speaker_id","label","task_type","language","age","sex","transcript","group","duration_sec","sample_rate"]
rows=[]
it=BASE+"/italian"
for f in glob.glob(it+"/**/*.wav", recursive=True):
    rel=os.path.relpath(f,it); parts=rel.split(os.sep); spk_dir="/".join(parts[:-1])
    low=rel.lower()
    label=1 if "parkinson" in low else 0
    grp="pd" if label else ("young_hc" if "young healthy" in low else "elderly_hc")
    d,sr=dur(f)
    rows.append([f,"italian","IT::"+spk_dir.replace(" ","_"),label,it_task(os.path.basename(f)),"Italian","","","",grp,d,sr])
kcl=BASE+"/26-29_09_2017_KCL"
for f in glob.glob(kcl+"/**/*.wav", recursive=True):
    rel=os.path.relpath(f,kcl); parts=rel.split(os.sep)
    label=1 if parts[1].upper()=="PD" else 0
    tt="read" if parts[0].lower().startswith("read") else "spontaneous"
    spk=os.path.basename(f).split("_")[0]
    d,sr=dur(f)
    rows.append([f,"kcl","KCL::"+spk,label,tt,"English","","","",parts[1],d,sr])

for ds in ("italian","kcl"):
    sub=[r for r in rows if r[1]==ds]
    with open(OUT+"/"+ds+".csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(hdr); w.writerows(sub)
    spk={r[2] for r in sub}; pdspk={r[2] for r in sub if r[3]==1}; hcspk={r[2] for r in sub if r[3]==0}
    tt=collections.Counter(r[4] for r in sub)
    bad=sum(1 for r in sub if r[11]=="")
    print("[%s] files=%d speakers=%d (PD=%d HC=%d) tasks=%s unreadable=%d"%(ds,len(sub),len(spk),len(pdspk),len(hcspk),dict(tt),bad))
print("written to", OUT)
