import os, csv, argparse
import numpy as np, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
MODEL="Qwen/Qwen2-Audio-7B-Instruct"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True)
    ap.add_argument("--tasks",default="read"); ap.add_argument("--tag",required=True); a=ap.parse_args()
    R="/project2/msoleyma_946/speech_health/results_chaitanya"
    rows=[r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in set(a.tasks.split(","))]
    proc=AutoProcessor.from_pretrained(MODEL)
    model=Qwen2AudioForConditionalGeneration.from_pretrained(MODEL,torch_dtype=torch.float16,device_map="cuda").eval()
    atid=model.config.audio_token_index
    store={}
    h=model.multi_modal_projector.register_forward_hook(lambda m,i,o: store.__setitem__("proj",o))
    feats=None; spk=[]; lab=[]
    for k,r in enumerate(rows):
        try:
            y,_=librosa.load(r["filepath"],sr=16000,mono=True)
            conv=[{"role":"user","content":[{"type":"audio","audio_url":"x"},{"type":"text","text":"Describe this speaker's voice."}]}]
            text=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
            inp=proc(text=text,audio=[y],sampling_rate=16000,return_tensors="pt"); inp={kk:v.to("cuda") for kk,v in inp.items()}
            with torch.no_grad(): out=model(**inp,output_hidden_states=True)
            pos=(inp["input_ids"][0]==atid)
            pv=store["proj"]; pv=pv.reshape(-1,pv.shape[-1])
            vecs=[pv.float().mean(0).cpu().numpy()]
            for hs in out.hidden_states: vecs.append(hs[0][pos].float().mean(0).cpu().numpy())
            if feats is None: feats=[[] for _ in vecs]
            for j,v in enumerate(vecs): feats[j].append(v)
            spk.append(r["speaker_id"]); lab.append(int(r["label"]))
            if (k+1)%20==0: print(f"  {k+1}/{len(rows)}",flush=True)
        except Exception as e: print("FAIL",r["filepath"],repr(e),flush=True)
    h.remove()
    spk=np.array(spk); lab=np.array(lab)
    stages=["projector"]+[f"llm_{i:02d}" for i in range(len(feats)-1)]
    ns=min(5,len(set(spk))); res=[]
    for j,name in enumerate(stages):
        X=np.stack(feats[j]); aucs=[]; bas=[]
        for tr,te in GroupKFold(ns).split(X,lab,spk):
            if len(set(lab[tr]))<2 or len(set(lab[te]))<2: continue
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
            clf.fit(X[tr],lab[tr]); p=clf.predict_proba(X[te])[:,1]
            try: aucs.append(roc_auc_score(lab[te],p))
            except Exception: pass
            bas.append(balanced_accuracy_score(lab[te],(p>=.5).astype(int)))
        res.append((name, round(float(np.mean(aucs)),3) if aucs else float("nan"), round(float(np.mean(bas)),3) if bas else float("nan")))
        print(f"  {name}: AUC {res[-1][1]} bAcc {res[-1][2]}",flush=True)
    with open(f"{R}/probing/localize_{a.tag}.csv","w",newline="") as f:
        csv.writer(f).writerows([["stage","auc","bacc"]]+res)
    print("SAVED localize_"+a.tag,flush=True)

if __name__=="__main__": main()
