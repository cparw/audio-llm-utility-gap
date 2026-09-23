import os,re,glob,json,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib_cha import par_text, meta
R="/Volumes/G-Drive Pro/paper work/Hard drive data for paper"
AD=f"{R}/DementiaBank/challenges/ADReSS-2020"
PT=f"{R}/DementiaBank/transcripts"
ad={}
for sub,sp in [("ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/transcription/cc","train_cc"),
               ("ADReSS-IS2020-train_x/ADReSS-IS2020-data/train/transcription/cd","train_cd"),
               ("ADReSS-IS2020-test_x/ADReSS-IS2020-data/test/transcription","test")]:
    for f in sorted(glob.glob(f"{AD}/{sub}/*.cha")):
        sid=os.path.basename(f)[:-4]
        ad[sid]=dict(file=f,split=sp,text=par_text(f),**meta(f))
pitt={}
for f in sorted(glob.glob(f"{PT}/*/cookie/*.cha")):
    pid=os.path.basename(f)[:-4]; grp=f.split("/")[-3]
    pitt[f"{grp}:{pid}"]=dict(file=f,text=par_text(f),**meta(f))
def big(t):
    w=t.split(); return set(zip(w,w[1:])) | set(w)
pb={k:big(v["text"]) for k,v in pitt.items()}
rows=[]
for sid,v in sorted(ad.items()):
    a=big(v["text"])
    sc=[]
    for k,b in pb.items():
        if not a or not b: continue
        j=len(a&b)/len(a|b)
        sc.append((j,k))
    sc.sort(reverse=True)
    top=sc[0]; second=sc[1] if len(sc)>1 else (0.0,"")
    rows.append(dict(sid=sid,split=v["split"],age=v["age"],sex=v["sex"],grp=v["grp"],mmse=v["mmse"],
                     best=top[1],j1=round(top[0],4),second=second[1],j2=round(second[0],4),
                     margin=round(top[0]-second[0],4),nwords=len(v["text"].split())))
import csv
out=f"{os.path.dirname(os.path.abspath(__file__))}/ad2020_to_pitt.csv"
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
import statistics
j1=[r["j1"] for r in rows]
print("n=",len(rows),"j1 min",min(j1),"median",statistics.median(j1))
print("j1>=0.9:",sum(1 for x in j1 if x>=0.9),"0.6-0.9:",sum(1 for x in j1 if 0.6<=x<0.9),"<0.6:",sum(1 for x in j1 if x<0.6))
print("ambiguous (margin<0.15):",sum(1 for r in rows if r["margin"]<0.15))
for r in rows[:5]: print(r)
print("worst:")
for r in sorted(rows,key=lambda r:r["j1"])[:8]: print(r)
print("wrote",out)
