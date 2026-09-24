"""ADReSS-2020 standard split: probe trained on the 108 train recordings, evaluated on the 48 test ones,
layer picked by cross validation inside train only. Plus the model's own answers on all 156."""
import csv, os, numpy as np, torch, librosa, time
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev).eval()
tower = model.model.audio_tower; tok = proc.tokenizer
rows = list(csv.DictReader(open(f"{D}/adress2020/adress2020_manifest.csv")))
npz = f"{D}/adress2020/adress2020_encoder_states.npz"
if not os.path.exists(npz):
    feats = []
    for i, r in enumerate(rows):
        x,_ = librosa.load(f"{D}/{r['segment_path']}", sr=16000)
        inp = proc.feature_extractor([x], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            out = tower(inp["input_features"].to(dev).half(), output_hidden_states=True)
        feats.append(torch.stack([h.mean(dim=1).squeeze(0).float().cpu() for h in out.hidden_states]).numpy())
        if (i+1)%50==0: print(f"extract {i+1}/{len(rows)}", flush=True)
    np.savez_compressed(npz, feats=np.stack(feats), spk=[r["spk"] for r in rows],
                        label=[int(r["label"]) for r in rows], split=[r["split"] for r in rows])
z = np.load(npz, allow_pickle=True)
F, y, split = z["feats"], z["label"].astype(int), np.array([str(s) for s in z["split"]])
tr, te = split=="train", split=="test"
cv = []
for L in range(F.shape[1]):
    a = []
    for i1, i2 in StratifiedKFold(5, shuffle=True, random_state=42).split(F[tr][:,L,:], y[tr]):
        Xtr = F[tr][:,L,:]; sc = StandardScaler().fit(Xtr[i1])
        c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr[i1]), y[tr][i1])
        a.append(roc_auc_score(y[tr][i2], c.decision_function(sc.transform(Xtr[i2]))))
    cv.append(np.mean(a))
Lbest = int(np.argmax(cv))
with open(f"{D}/adress2020/adress2020_probe.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["layer","train_cv_auc","test_auc","test_acc"])
    for L in range(F.shape[1]):
        sc = StandardScaler().fit(F[tr][:,L,:])
        c = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(F[tr][:,L,:]), y[tr])
        s = c.decision_function(sc.transform(F[te][:,L,:]))
        w.writerow([L, f"{cv[L]:.3f}", f"{roc_auc_score(y[te], s):.3f}", f"{accuracy_score(y[te], (s>0).astype(int)):.3f}"])
        if L == Lbest:
            print(f"PROBE layer {L} chosen by train CV ({cv[L]:.3f}): TEST AUC {roc_auc_score(y[te], s):.3f}, TEST ACC {accuracy_score(y[te], (s>0).astype(int)):.3f}", flush=True)
def wids(ws):
    s=set()
    for wd in ws:
        for v in (wd," "+wd):
            e=tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)
YES, NO = wids(["Yes","yes","YES"]), wids(["No","no","NO"])
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
ps = []
for i, r in enumerate(rows):
    x,_ = librosa.load(f"{D}/{r['segment_path']}", sr=16000)
    conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
    with torch.no_grad(): o = model(**inp)
    p = torch.softmax(o.logits[0,-1].float().cpu(), -1); py, pn = float(p[YES].sum()), float(p[NO].sum())
    ps.append(py/(py+pn) if py+pn>0 else 0.5)
ps = np.array(ps)
with open(f"{D}/adress2020/adress2020_answers.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["spk","split","label","p_yes"])
    for r, p in zip(rows, ps): w.writerow([r["spk"], r["split"], r["label"], f"{p:.4f}"])
print(f"ANSWER AUC test {roc_auc_score(y[te], ps[te]):.3f}, train {roc_auc_score(y[tr], ps[tr]):.3f}, all {roc_auc_score(y, ps):.3f}", flush=True)
