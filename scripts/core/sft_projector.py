"""Retrain only the projector on the diagnosis answer. Audio encoder and language model frozen.
  Qwen2-Audio   multi_modal_projector, encoder hidden state PLUG=32 as its input (the released plug layer)
  Qwen2.5-Omni  thinker.audio_tower.proj, the single connector, encoder final layer as its input
AdamW, lr 1e-4, 3 epochs, one clip per update, cross entropy over the Yes and No logits at the answer position,
projector in float32, 5 speaker-disjoint folds, out-of-fold scoring, resumable per fold.
usage: sft_projector.py MANIFEST(path,label,speaker[,set]) OUTPREFIX COND(ad|pd|mdd) [MODEL_ID]"""
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import csv, sys, time, types, copy, json, datetime, numpy as np, torch, librosa
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
MAN, OUTP, COND = sys.argv[1:4]
MID = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-Omni-7B"
PLUG, EPOCHS, LR = 32, 3, 1e-4
P = {"ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.",
     "pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."}[COND]
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
IS_OMNI = "Omni" in MID
DT = torch.bfloat16 if IS_OMNI else torch.float16
if IS_OMNI:
    from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    full = Model.from_pretrained(MID, dtype=DT)
    net = full.thinker
    for attr in ("talker", "token2wav"):
        if hasattr(full, attr): delattr(full, attr)
    if hasattr(net, "visual"): del net.visual
    net = net.to(dev)
    proj = net.audio_tower.proj
else:
    from transformers import AutoProcessor as Proc, Qwen2AudioForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    net = Model.from_pretrained(MID, dtype=DT).to(dev)
    tower = getattr(net, "audio_tower", None) or net.model.audio_tower
    proj = getattr(net, "multi_modal_projector", None) or net.model.multi_modal_projector
    orig_forward = tower.forward
    def plugged_forward(self, *a, **kw):                    # feed the projector from encoder layer PLUG, as released
        kw["output_hidden_states"] = True
        with torch.no_grad(): out = orig_forward(*a, **kw)
        x = out.hidden_states[PLUG]
        if getattr(self, "avg_pooler", None) is not None and x.shape[1] != out.last_hidden_state.shape[1]:
            x = self.avg_pooler(x.permute(0, 2, 1)).permute(0, 2, 1)
        if getattr(self, "layer_norm", None) is not None: x = self.layer_norm(x)
        out.last_hidden_state = x.detach(); return out
    tower.forward = types.MethodType(plugged_forward, tower)
tok = proc.tokenizer
_orig = proj.forward
def _fp32_forward(x): return _orig(x.float()).to(DT)
proj.forward = _fp32_forward
for p in net.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
has_set = "set" in rows[0]
print(f"MODEL {MID} | {len(rows)} clips, {len(set(spk))} speakers, device {dev}, projector {type(proj).__name__}", flush=True)
def inputs_for(r):
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    return inp
state_f = f"{OUTP}_state.json"
st = json.load(open(state_f)) if os.path.exists(state_f) else {"done": [], "oof": [0.0] * len(rows)}
oof = np.array(st["oof"]); t0 = time.time()
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk)):
    if fold in st["done"]: continue
    proj.load_state_dict(proj_init)
    for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
    opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
    net.train()
    for ep in range(EPOCHS):
        order = np.random.RandomState(fold * 10 + ep).permutation(tr); tot = nb = 0
        for i in order:
            out = net(**inputs_for(rows[i]))
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
            loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1
        print(f"fold {fold} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    net.eval()
    with torch.no_grad():
        for i in te:
            out = net(**inputs_for(rows[i]))
            oof[i] = float(torch.softmax(out.logits[0, -1, [YES, NO]].float(), -1)[0])
    st["done"].append(fold); st["oof"] = oof.tolist(); json.dump(st, open(state_f, "w"))
    print(f"fold {fold} eval done {(time.time()-t0)/60:.1f} min", flush=True)
with open(f"{OUTP}_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "speaker", "label", "p_yes", "set", "prompt"])
    for r, p in zip(rows, oof): w.writerow([r["path"], r["speaker"], r["label"], p, r.get("set", ""), P])
AUC = float(roc_auc_score(y, oof)); arms = {}
if has_set:
    for s in sorted(set(r["set"] for r in rows if r["set"])):
        m = np.array([r["set"] == s for r in rows])
        if len(set(y[m])) > 1: arms[s] = round(float(roc_auc_score(y[m], oof[m])), 4)
json.dump({"checkpoint": MID, "clip_list": os.path.abspath(MAN), "condition": COND, "prompt": P, "n": len(rows),
           "n_speakers": int(len(set(spk))), "folds": 5, "epochs": EPOCHS, "lr": LR, "trained": "projector only",
           "auc_oof": round(AUC, 4), "arms": arms, "window_seconds": 30, "dtype": str(DT).split(".")[-1],
           "minutes": round((time.time() - t0) / 60, 1), "date": datetime.date.today().isoformat()},
          open(f"{OUTP}_sft.json", "w"), indent=1)
print(f"RESULT sft projector {os.path.basename(MAN)} n={len(rows)} AUC={AUC:.4f} arms={arms} "
      f"{(time.time()-t0)/60:.1f} min", flush=True)
