#!/usr/bin/env python3
"""The trained version of the layer feed: plug encoder layer 24, freeze everything,
retrain only the projector on the diagnosis answer. Speaker disjoint folds."""
import csv, os, time, types, copy
import numpy as np, torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from sklearn.model_selection import GroupKFold

D = os.path.expanduser("~/Desktop/Hard drive data for paper/DementiaBank")
PLUG = 32
EPOCHS = 3
LR = 1e-4
dev = "mps" if torch.backends.mps.is_available() else "cpu"
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", dtype=torch.float16).to(dev)
tok = proc.tokenizer
tower = model.model.audio_tower
proj = model.model.multi_modal_projector
_orig_proj_forward = proj.forward
def _proj_fp32_forward(x):
    return _orig_proj_forward(x.float()).to(torch.float16)
proj.forward = _proj_fp32_forward
orig_forward = tower.forward

def plugged_forward(self, *args, **kwargs):
    kwargs["output_hidden_states"] = True
    with torch.no_grad():
        out = orig_forward(*args, **kwargs)
    hs = out.hidden_states[PLUG]
    x = hs
    if getattr(self, "avg_pooler", None) is not None and hs.shape[1] != out.last_hidden_state.shape[1]:
        x = self.avg_pooler(x.permute(0,2,1)).permute(0,2,1)
    if getattr(self, "layer_norm", None) is not None:
        x = self.layer_norm(x)
    out.last_hidden_state = x.detach()
    return out
tower.forward = types.MethodType(plugged_forward, tower)

for p in model.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())

def wid(w):
    return tok(" "+w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
P_MAIN = ("Based only on this recording, does this speaker show signs of dementia? "
          "Answer with one word, Yes or No.")

rows = [r for r in csv.DictReader(open(f"{D}/pitt_conflict_manifest.csv")) if r.get("segment_path")]
y = np.array([int(r["label"]) for r in rows])
spk = np.array([r["grp"]+r["spk"] for r in rows])
arm = np.array([r["set"] for r in rows])
print(f"{len(rows)} segments, device {dev}", flush=True)

def inputs_for(r):
    x, _ = librosa.load(f"{D}/{r['segment_path']}", sr=16000)
    conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P_MAIN}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    return {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}

oof = np.zeros(len(rows))
_part = f"{D}/projector_noplug_oof_partial.npy"
if os.path.exists(_part):
    oof = np.load(_part)
SKIP = int(os.environ.get("SKIP_FOLDS", "0"))
gkf = GroupKFold(n_splits=5)
t0 = time.time()
FOLD_LIMIT = int(os.environ.get("FOLDS", "5"))
for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(rows)), y, groups=spk)):
    if fold >= FOLD_LIMIT: break
    if fold < SKIP: continue
    proj.load_state_dict(proj_init)
    for p in proj.parameters():
        p.data = p.data.float(); p.requires_grad = True
    opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
    model.train()
    for ep in range(EPOCHS):
        order = np.random.RandomState(fold*10+ep).permutation(tr)
        tot, nb = 0.0, 0
        for i in order:
            inp = inputs_for(rows[i])
            out = model(**inp)
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i]==1 else 1, device=dev)
            loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss); nb += 1
        print(f"fold {fold} epoch {ep} loss {tot/nb:.4f}  {(time.time()-t0)/60:.0f} min", flush=True)
    model.eval()
    with torch.no_grad():
        for i in te:
            inp = inputs_for(rows[i])
            out = model(**inp)
            two = torch.softmax(out.logits[0, -1, [YES, NO]].float(), -1)
            oof[i] = float(two[0])
    np.save(f"{D}/projector_noplug_oof_partial.npy", oof)
    print(f"fold {fold} eval done", flush=True)

np.save(f"{D}/projector_noplug_oof.npy", oof)
def rank_auc(yy, pp):
    yy, pp = np.asarray(yy,float), np.asarray(pp,float)
    order = np.argsort(pp); ranks = np.empty(len(pp)); ranks[order] = np.arange(1,len(pp)+1)
    n1, n0 = yy.sum(), (1-yy).sum()
    return (ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)
print(f"RETRAINED PROJECTOR NO PLUG (layer 32) overall AUC {rank_auc(y, oof):.3f}  (baseline answer 0.618, untrained plug 0.488)", flush=True)
for a in ("agreement","conflict"):
    m = arm==a
    print(f"  {a}: {rank_auc(y[m], oof[m]):.3f}", flush=True)
