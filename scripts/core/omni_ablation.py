"""Where does the recovered signal have to be written back in? Three arms on Qwen2.5-Omni, same folds and optimizer.
  proj   projector only (thinker.audio_tower.proj), language model frozen        [the paper's arm]
  lora   LoRA r=8 on the language model attention q,k,v,o; projector frozen
  both   LoRA on the language model plus the projector unfrozen
AdamW lr 1e-4, 3 epochs, one clip per update, cross entropy over the Yes and No logits at the answer position,
5 speaker-disjoint folds, out-of-fold scoring, per-fold weights saved.
usage: omni_ablation.py MANIFEST OUTPREFIX COND ARM(proj|lora|both)"""
import os, csv, sys, time, json, copy, datetime, numpy as np, torch, librosa
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
MAN, OUTP, COND, ARM = sys.argv[1:5]
OUTP = f"{OUTP}_{ARM}"          # arm-specific prefix so two arms never overwrite each other
MID = "Qwen/Qwen2.5-Omni-7B"; EPOCHS, LR, RANK = 3, 1e-4, 8
P = {"ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.",
     "pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
     "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."}[COND]
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
proc = Qwen2_5OmniProcessor.from_pretrained(MID)
full = Qwen2_5OmniForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16)
th = full.thinker
for a in ("talker", "token2wav"):
    if hasattr(full, a): delattr(full, a)
if hasattr(th, "visual"): del th.visual
th = th.to("cuda")
tok = proc.tokenizer
proj = th.audio_tower.proj
_orig = proj.forward
def _cast(x):                                   # match the projector's own dtype: fp32 when it trains, bf16 when frozen
    w = next(proj.parameters())
    return _orig(x.to(w.dtype)).to(torch.bfloat16)
proj.forward = _cast
for p in th.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
lora_init = None
if ARM in ("lora", "both"):
    from peft import LoraConfig, inject_adapter_in_model
    cfg = LoraConfig(r=RANK, lora_alpha=16, lora_dropout=0.0, bias="none",
                     target_modules=r"^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
    th = inject_adapter_in_model(cfg, th)          # language model attention only, audio tower untouched
    n_lora = sum(1 for n, _ in th.named_modules() if "lora_A" in n)
    print(f"LoRA injected into {n_lora} modules", flush=True)
    for n, p in th.named_parameters():
        p.requires_grad = "lora_" in n
    lora_init = {n: p.detach().clone() for n, p in th.named_parameters() if "lora_" in n}
def set_trainable():
    for n, p in th.named_parameters(): p.requires_grad = ("lora_" in n) if ARM in ("lora", "both") else False
    if ARM in ("proj", "both"):
        for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
    return [p for p in th.parameters() if p.requires_grad]
def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
arms_col = [r.get("set", "") for r in rows]
def inputs_for(r):
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    t = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=t, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}; inp.pop("use_audio_in_video", None)
    return inp
oof = np.zeros(len(rows)); t0 = time.time(); fold_min = []
os.makedirs(f"{OUTP}_weights", exist_ok=True)
for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk)):
    f0 = time.time()
    proj.load_state_dict(proj_init)
    if lora_init is not None:
        with torch.no_grad():
            for n, p in th.named_parameters():
                if "lora_" in n: p.copy_(lora_init[n])
    params = set_trainable()
    ntr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR)
    th.train()
    for ep in range(EPOCHS):
        order = np.random.RandomState(fold * 10 + ep).permutation(tr); tot = nb = 0
        for i in order:
            out = th(**inputs_for(rows[i]))
            two = out.logits[0, -1, [YES, NO]].float()
            loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), torch.tensor(0 if y[i] == 1 else 1, device="cuda").unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach()); nb += 1
        print(f"[{ARM}] fold {fold} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    th.eval()
    with torch.no_grad():
        for i in te:
            out = th(**inputs_for(rows[i]))
            oof[i] = float(torch.softmax(out.logits[0, -1, [YES, NO]].float(), -1)[0])
    sd = {n: p.detach().cpu() for n, p in th.named_parameters() if "lora_" in n}
    if ARM in ("proj", "both"): sd.update({f"proj.{k}": v.detach().cpu() for k, v in proj.state_dict().items()})
    torch.save(sd, f"{OUTP}_weights/fold{fold}.pt")
    fold_min.append((time.time() - f0) / 60)
    print(f"[{ARM}] fold {fold} done {fold_min[-1]:.1f} min, trainable params {ntr:,}", flush=True)
with open(f"{OUTP}_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "speaker", "label", "p_yes", "set", "prompt"])
    for r, p in zip(rows, oof): w.writerow([r["path"], r["speaker"], r["label"], p, r.get("set", ""), P])
AUC = float(roc_auc_score(y, oof)); arms = {}
for s in sorted(set(a for a in arms_col if a)):
    m = np.array([a == s for a in arms_col])
    if len(set(y[m])) > 1: arms[s] = round(float(roc_auc_score(y[m], oof[m])), 4)
json.dump({"checkpoint": MID, "arm": ARM, "clip_list": os.path.abspath(MAN), "prompt": P, "condition": COND,
           "n": len(rows), "auc_oof": round(AUC, 4), "arms": arms, "trainable_params": int(ntr),
           "lora_rank": RANK if ARM in ("lora", "both") else None, "epochs": EPOCHS, "lr": LR, "folds": 5,
           "minutes_per_fold": [round(m, 1) for m in fold_min], "window_seconds": 30, "dtype": "bfloat16",
           "date": datetime.date.today().isoformat()}, open(f"{OUTP}.json", "w"), indent=1)
print(f"RESULT ablation {ARM} {os.path.basename(MAN)} n={len(rows)} AUC={AUC:.4f} arms={arms} "
      f"trainable={ntr:,} {np.mean(fold_min):.1f} min/fold", flush=True)
