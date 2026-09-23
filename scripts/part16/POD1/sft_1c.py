"""1c RETRAIN: the E-DAIC full-window projector fine-tune (sft_full), then score the Part 14
conflict/agreement clips OUT OF FOLD BY SPEAKER. Mirrors sft_projector.py exactly (projector only,
AdamW lr 1e-4, 3 epochs, 5 speaker-disjoint GroupKFold folds, 30 s window, bf16) and adds, inside
each fold, scoring of the Part 14 clips whose speaker is in that fold's HELD-OUT set.
No per-fold checkpoints were ever saved by sft_projector.py, so this is a retrain, not a reload.
Records BOTH p_yes conventions per clip:
  p_yes_sft   = softmax over the two logits [" Yes"," No"]        (sft_projector.py convention)
  p_yes_paper = P(Yes)/(P(Yes)+P(No)) over all single-token Yes/No variants (paper convention)
usage: sft_1c.py"""
import os, csv, sys, copy, json, time, datetime, numpy as np, torch, librosa
from sklearn.model_selection import GroupKFold
MID="Qwen/Qwen2.5-Omni-7B"; EPOCHS, LR, WIN = 3, 1e-4, 30
P="Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
dev="cuda"; t0=time.time()
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc=Proc.from_pretrained(MID); full=Model.from_pretrained(MID, dtype=torch.bfloat16)
net=full.thinker
for a in ("talker","token2wav"):
    if hasattr(full,a): delattr(full,a)
if hasattr(net,"visual"): del net.visual
net=net.to(dev)
proj=net.audio_tower.proj
tok=proc.tokenizer
_orig=proj.forward
def _fp32_forward(x): return _orig(x.float()).to(torch.bfloat16)
proj.forward=_fp32_forward
for p in net.parameters(): p.requires_grad=False
proj_init=copy.deepcopy(proj.state_dict())
def wid(w): return tok(" "+w, add_special_tokens=False).input_ids[0]
YES1, NO1 = wid("Yes"), wid("No")
def idset(ws):
    s=set()
    for w in ws:
        t=tok.encode(w, add_special_tokens=False)
        if len(t)==1: s.add(t[0])
    return sorted(s)
YESV, NOV = idset(["Yes"," Yes","yes"," yes","YES"]), idset(["No"," No","no"," no","NO"])
train_rows=list(csv.DictReader(open("/workspace/mf_full_rebuilt.csv")))
p14=list(csv.DictReader(open("/workspace/mf_p14_pod.csv")))
y=np.array([int(r["label"]) for r in train_rows]); spk=np.array([str(r["speaker"]) for r in train_rows])
print(f"train {len(train_rows)} clips / {len(set(spk))} speakers | part14 {len(p14)} clips / "
      f"{len(set(r['speaker'] for r in p14))} speakers | {time.time()-t0:.0f}s", flush=True)
def inputs_for(path):
    x,_=librosa.load(path, sr=16000); x=x[:16000*WIN]
    conv=[{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text=proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp=proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp={k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
    inp.pop("use_audio_in_video",None)
    for _k,_v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype==torch.float32: inp[_k]=_v.to(torch.bfloat16)
    return inp
state_f="/workspace/out16/sft1c_state.json"
st=json.load(open(state_f)) if os.path.exists(state_f) else {"done":[],"oof":[0.0]*len(train_rows),"p14":{}}
oof=np.array(st["oof"]); p14res=st["p14"]
folds=list(GroupKFold(n_splits=5).split(np.zeros(len(train_rows)), y, groups=spk))
json.dump({str(f):sorted(set(spk[te].tolist())) for f,(tr,te) in enumerate(folds)},
          open("/workspace/out16/sft1c_folds.json","w"), indent=1)
for fold,(tr,te) in enumerate(folds):
    if fold in st["done"]:
        print(f"fold {fold} already done, skip", flush=True); continue
    test_spk=set(spk[te].tolist())
    proj.load_state_dict(proj_init)
    for p in proj.parameters(): p.data=p.data.float(); p.requires_grad=True
    opt=torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
    net.train()
    for ep in range(EPOCHS):
        order=np.random.RandomState(fold*10+ep).permutation(tr); tot=nb=0
        for i in order:
            out=net(**inputs_for(train_rows[i]["path"]))
            two=out.logits[0,-1,[YES1,NO1]].float()
            target=torch.tensor(0 if y[i]==1 else 1, device=dev)
            loss=torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad(); tot+=float(loss.detach()); nb+=1
        print(f"fold {fold} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    net.eval()
    with torch.no_grad():
        for i in te:
            lg=net(**inputs_for(train_rows[i]["path"])).logits[0,-1]
            oof[i]=float(torch.softmax(lg[[YES1,NO1]].float(),-1)[0])
        n14=0
        for r in p14:
            if r["speaker"] not in test_spk: continue
            lg=net(**inputs_for(r["path"])).logits[0,-1]
            ps=float(torch.softmax(lg[[YES1,NO1]].float(),-1)[0])
            pr=torch.softmax(lg.float(),-1).cpu().numpy()
            py,pn=float(pr[YESV].sum()),float(pr[NOV].sum())
            p14res[r["id"]]=[ps, py/(py+pn+1e-12), py+pn, fold]; n14+=1
    st["done"].append(fold); st["oof"]=oof.tolist(); st["p14"]=p14res
    json.dump(st, open(state_f,"w"))
    print(f"fold {fold} eval done: {len(te)} edaic + {n14} part14 clips  {(time.time()-t0)/60:.1f} min", flush=True)
with open("/workspace/out16/p14_o25_sftfull.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["id","speaker","arm","label","p_yes","answer_mass","p_yes_sft","fold","prompt"])
    for r in p14:
        v=p14res.get(r["id"])
        if v is None: continue
        w.writerow([r["id"],r["speaker"],r["arm"],r["label"],f"{v[1]:.10f}",f"{v[2]:.10f}",f"{v[0]:.10f}",int(v[3]),P])
with open("/workspace/out16/sft1c_edaic_oof.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["path","speaker","label","p_yes_sft"])
    for r,p in zip(train_rows,oof): w.writerow([r["path"],r["speaker"],r["label"],p])
from sklearn.metrics import roc_auc_score
A=float(roc_auc_score(y,oof))
json.dump({"checkpoint":MID,"RETRAIN":True,
  "why_retrain":"sft_projector.py never saved per-fold model checkpoints; only _state.json (oof scores) and _oof.csv",
  "train_clip_list":"/workspace/mf_full_rebuilt.csv (order recovered from sft_full_oof.csv)",
  "score_clip_list":"/workspace/mf_p14_pod.csv","prompt":P,"epochs":EPOCHS,"lr":LR,
  "trained":"projector only (thinker.audio_tower.proj), fp32 params, rest frozen",
  "folds":5,"fold_rule":"sklearn GroupKFold(n_splits=5) grouped by speaker over mf_full_rebuilt.csv row order",
  "window_seconds":WIN,"dtype":"bfloat16","n_train":len(train_rows),"n_scored_part14":len(p14res),
  "edaic_oof_auc_retrain":round(A,4),"edaic_oof_auc_original_sft_full":0.6421,
  "minutes":round((time.time()-t0)/60,1),"date":datetime.date.today().isoformat()},
  open("/workspace/out16/p14_o25_sftfull.json","w"), indent=1)
print(f"RESULT retrain edaic oof AUC={A:.4f} (original sft_full 0.6421) part14 scored {len(p14res)} "
      f"{(time.time()-t0)/60:.1f} min", flush=True)
