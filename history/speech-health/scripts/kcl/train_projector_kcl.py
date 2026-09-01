#!/usr/bin/env python3
"""Projector retrain on kcl read speech, same recipe as alzheimers: plug layer 24,
freeze everything else, train only the projector, speaker disjoint folds."""
import csv, os, time, types, copy, glob
import numpy as np, torch, librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from sklearn.model_selection import GroupKFold
D = os.path.expanduser("~/Desktop/Hard drive data for paper")
PLUG, EPOCHS, LR = 24, 3, 1e-4
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev)
tok = proc.tokenizer
tower = model.model.audio_tower
proj = model.model.multi_modal_projector
_orig = proj.forward
proj.forward = lambda x: _orig(x.float()).to(torch.float16)
orig_forward = tower.forward
def plugged_forward(self, *args, **kwargs):
    kwargs["output_hidden_states"] = True
    with torch.no_grad():
        out = orig_forward(*args, **kwargs)
    x = out.hidden_states[PLUG]
    if getattr(self, "avg_pooler", None) is not None and x.shape[1] != out.last_hidden_state.shape[1]:
        x = self.avg_pooler(x.permute(0,2,1)).permute(0,2,1)
    if getattr(self, "layer_norm", None) is not None:
        x = self.layer_norm(x)
    out.last_hidden_state = x.detach()
    return out
tower.forward = types.MethodType(plugged_forward, tower)
for p in model.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
def wid(w): return tok(" "+w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
P = ("Based only on this recording, does this speaker show signs of Parkinson's disease? "
     "Answer with one word, Yes or No.")
wavs = {os.path.basename(p): p for p in glob.glob(f"{D}/KCL/extracted/**/*.wav", recursive=True)}
rows = [r for r in csv.DictReader(open(f"{D}/KCL/kcl_local_scores.csv"))
        if r["task"] == "read" and r["file"] in wavs]
y = np.array([int(r["label"]) for r in rows])
spk = np.array([r["speaker"] for r in rows])
print(f"{len(rows)} clips, {len(set(spk))} speakers, {y.sum()} pd", flush=True)
def inputs_for(r):
    x, _ = librosa.load(wavs[r["file"]], sr=16000, duration=30)
    conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    return {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
oof = np.zeros(len(rows))
t0 = time.time()
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk)):
    proj.load_state_dict(proj_init)
    for p in proj.parameters():
        p.data = p.data.float(); p.requires_grad = True
    opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
    model.train()
    for ep in range(EPOCHS):
        order = np.random.RandomState(fold*10+ep).permutation(tr)
        tot, nb = 0.0, 0
        for i in order:
            out = model(**inputs_for(rows[i]))
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i]==1 else 1, device=dev)
            loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss); nb += 1
        print(f"fold {fold} epoch {ep} loss {tot/nb:.4f}  {(time.time()-t0)/60:.0f} min", flush=True)
    model.eval()
    with torch.no_grad():
        for i in te:
            out = model(**inputs_for(rows[i]))
            oof[i] = float(torch.softmax(out.logits[0, -1, [YES, NO]].float(), -1)[0])
    np.save(f"{D}/KCL/kcl_projector_oof.npy", oof)
    print(f"fold {fold} eval done", flush=True)
def rank_auc(yy, pp):
    yy, pp = np.asarray(yy,float), np.asarray(pp,float)
    order = np.argsort(pp); ranks = np.empty(len(pp)); ranks[order] = np.arange(1,len(pp)+1)
    n1, n0 = yy.sum(), (1-yy).sum()
    return (ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"KCL RETRAINED PROJECTOR overall AUC {rank_auc(y, oof):.3f}  (kcl answer baseline 0.66)", flush=True)
