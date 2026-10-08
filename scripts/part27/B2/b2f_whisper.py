# B2: whisper-large-v3 on 20 E-DAIC whole-interview windows, 400 Hz low-passed and unfiltered.
import csv, json, time, numpy as np, torch, librosa
from scipy.signal import butter, sosfiltfilt
from transformers import pipeline
D="/workspace/p23a"; CUT=f"{D}/edaicfull/cut"
WIN=[r["pid"] for r in csv.DictReader(open(f"{D}/edaicfull/windows_full.csv"))]
rng=np.random.default_rng(0); pick=sorted(rng.choice(len(WIN),size=20,replace=False).tolist()); pids=[WIN[i] for i in pick]
import os
SH=int(os.environ.get("SHARD","0")); NS=int(os.environ.get("NSHARD","1")); mine=pids[SH::NS]
if os.environ.get("PIDS"): mine=[x for x in os.environ["PIDS"].split(",") if x]
SOS=butter(8,400,btype="low",fs=16000,output="sos")   # identical to p27b_extract.py
asr=pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",torch_dtype=torch.float16,device="cuda:0",chunk_length_s=30,batch_size=32)
out=[]; t0=time.time()
for pid in mine:
    x,_=librosa.load(f"{CUT}/{pid}.wav",sr=16000)
    xf=sosfiltfilt(SOS,x).astype(np.float32)
    row={"pid":pid,"dur_s":round(len(x)/16000,2)}
    for tag,w in (("unfiltered",x),("lp400",xf)):
        r=asr({"raw":w,"sampling_rate":16000},generate_kwargs={"language":"english","task":"transcribe","max_new_tokens":200})
        row[tag]=r["text"]
    out.append(row); print(pid,row["dur_s"],len(row["unfiltered"].split()),len(row["lp400"].split()),f"{time.time()-t0:.0f}s",flush=True)
json.dump({"model":"openai/whisper-large-v3","filter":"butter(8,400,low,fs=16000) sosfiltfilt, same as p27b_extract.py","excerpt":"whole window (up to 900 s), max_new_tokens 200 per 30 s chunk","selection":"numpy default_rng(0).choice(275,20,replace=False) over windows_full.csv order","pids":pids,"rows":out},open(f"/workspace/b2f_transcripts_{SH}.json","w"),indent=1)
open(f"/workspace/B2F_DONE_{SH}","w").write("done")
print("B2 DONE",time.time()-t0,flush=True)
