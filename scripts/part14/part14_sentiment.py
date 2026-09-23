"""PART 14 step 1: cardiffnlp sentiment over all 6,214 E-DAIC segments."""
import os, json, time
import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P="/Users/chaitanyaparwatkar/Desktop/af2_run/speech-health/results/health/paradox"
R="/Users/chaitanyaparwatkar/Desktop/release/edaic_rerun"
MID="cardiffnlp/twitter-roberta-base-sentiment-latest"
S=pd.read_csv(f"{P}/daic_segments_scored.csv")
print("segments %d, speakers %d"%(len(S),S.pid.nunique()),flush=True)
dev="mps" if torch.backends.mps.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(MID)
mod=AutoModelForSequenceClassification.from_pretrained(MID).to(dev).eval()
id2=mod.config.id2label
print("labels:",id2,"device",dev,flush=True)
txt=S.text.fillna("").astype(str).tolist()
probs=np.zeros((len(txt),3),dtype=np.float32)
B=64; t0=time.time()
for i in range(0,len(txt),B):
    b=txt[i:i+B]
    enc=tok(b,return_tensors="pt",truncation=True,max_length=512,padding=True).to(dev)
    with torch.no_grad(): lg=mod(**enc).logits.float().cpu()
    probs[i:i+len(b)]=torch.softmax(lg,-1).numpy()
    if (i//B)%20==0: print("  %d/%d %.0fs"%(i+len(b),len(txt),time.time()-t0),flush=True)
lab={v.lower():k for k,v in id2.items()}
S["p_neg"]=probs[:,lab.get("negative",0)]
S["p_neu"]=probs[:,lab.get("neutral",1)]
S["p_pos"]=probs[:,lab.get("positive",2)]
S["clearly_positive"]=(S.p_pos>=0.70).astype(int)
S["clearly_negative"]=(S.p_neg>=0.70).astype(int)
S.to_csv(f"{R}/part14_segments_sentiment.csv",index=False)
print("\nclearly positive: %d  clearly negative: %d  neither: %d"%(
    S.clearly_positive.sum(),S.clearly_negative.sum(),
    len(S)-S.clearly_positive.sum()-S.clearly_negative.sum()),flush=True)
json.dump({"model":MID,"n_segments":int(len(S)),"threshold":0.70,
  "id2label":{str(k):v for k,v in id2.items()},
  "clearly_positive":int(S.clearly_positive.sum()),
  "clearly_negative":int(S.clearly_negative.sum()),
  "source":f"{P}/daic_segments_scored.csv","device":dev,
  "minutes":round((time.time()-t0)/60,2)},open(f"{R}/part14_sentiment.json","w"),indent=1)
print("PART14_SENTIMENT_DONE %.1f min"%((time.time()-t0)/60),flush=True)
