"""LoRA baseline on Pitt 468 for Qwen2.5-Omni.
LoRA r=8 alpha=16 dropout=0.05 on q_proj,k_proj,v_proj,o_proj of EVERY language-model layer.
Audio encoder and projector stay frozen. Loss, prompt and input construction copied verbatim
from sft_projector.py (the script that produced the paper's projector fine-tune).
Folds are LOADED from a saved folds csv, never recomputed here.
usage: lora_pitt.py <manifest.csv> <folds.csv> <outprefix> [MODEL_ID]"""
import os; os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import csv, sys, json, time, copy, datetime, numpy as np, torch, librosa

MAN, FOLDS, OUTP = sys.argv[1:4]
MID = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-Omni-7B"
EPOCHS, LR, RANK, ALPHA, DROP = 3, 1e-4, 8, 16, 0.05
COND = "ad"
# verbatim from sft_projector.py
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
dev = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16

from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=DT)
TOTAL_FULL = sum(p.numel() for p in full.parameters())
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev)
TOTAL_THINKER = sum(p.numel() for p in net.parameters())
tok = proc.tokenizer

# ---- projector param count, for the comparison in the report ----
proj = net.audio_tower.proj
PROJ_PARAMS = sum(p.numel() for p in proj.parameters())
PROJ_NAME = type(proj).__name__

# ---- pick the LM decoder layers explicitly, so the audio tower is never touched ----
lm = net.model                       # the language model inside the thinker
lm_prefix = None
for cand in ("model.layers", "layers"):
    obj = net
    ok = True
    for part in cand.split("."):
        if not hasattr(obj, part): ok = False; break
        obj = getattr(obj, part)
    if ok: lm_prefix = cand; layers = obj; break
assert lm_prefix is not None, "could not find the LM decoder layers"
targets = []
for name, mod in net.named_modules():
    if not name.startswith(lm_prefix + "."): continue
    if name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj") and isinstance(mod, torch.nn.Linear):
        targets.append(name)
N_LM_LAYERS = len(layers)
print(f"LM layers {N_LM_LAYERS} | LoRA target modules {len(targets)} | prefix {lm_prefix}", flush=True)
assert len(targets) == 4 * N_LM_LAYERS, f"expected 4 per layer, got {len(targets)}"
assert not any(".audio_tower." in t for t in targets), "audio tower must not be targeted"

from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROP, bias="none",
                 target_modules=targets, task_type=None)
net = get_peft_model(net, cfg)
for n_, p_ in net.named_parameters():
    p_.requires_grad = ("lora_" in n_)
    if p_.requires_grad: p_.data = p_.data.float()
LORA_PARAMS = sum(p.numel() for p in net.parameters() if p.requires_grad)
assert not any("audio_tower" in n_ for n_, p_ in net.named_parameters() if p_.requires_grad)
lora_init = copy.deepcopy({n_: p_.detach().clone() for n_, p_ in net.named_parameters() if p_.requires_grad})

counts = {"model_id": MID, "total_params_full_model": int(TOTAL_FULL),
          "total_params_thinker_audio_plus_lm": int(TOTAL_THINKER),
          "lora_trainable_params": int(LORA_PARAMS),
          "lora_pct_of_full_model": round(100.0 * LORA_PARAMS / TOTAL_FULL, 6),
          "lora_pct_of_thinker": round(100.0 * LORA_PARAMS / TOTAL_THINKER, 6),
          "projector_trainable_params": int(PROJ_PARAMS), "projector_module": PROJ_NAME,
          "projector_pct_of_full_model": round(100.0 * PROJ_PARAMS / TOTAL_FULL, 6),
          "projector_pct_of_thinker": round(100.0 * PROJ_PARAMS / TOTAL_THINKER, 6),
          "lm_layers": int(N_LM_LAYERS), "lora_target_modules": int(len(targets)),
          "rank": RANK, "alpha": ALPHA, "dropout": DROP}
json.dump(counts, open(f"{OUTP}_params.json", "w"), indent=1)
print("PARAMCOUNTS " + json.dumps(counts), flush=True)

# ---- answer tokens ----
def wid(w):  # exactly as sft_projector.py
    return tok(" " + w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
def variants(words):
    out = []
    for w in words:
        for s in (w, " " + w):
            t = tok(s, add_special_tokens=False).input_ids
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YES_SET, NO_SET = variants(["Yes", "yes", "YES"]), variants(["No", "no", "NO"])
print(f"YES={YES} NO={NO} YES_SET={YES_SET} NO_SET={NO_SET}", flush=True)

rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
fr = list(csv.DictReader(open(FOLDS)))
assert len(fr) == len(rows)
for a, b in zip(rows, fr): assert os.path.basename(a["path"]) == os.path.basename(b["path"]), "fold file order mismatch"
fold = np.array([int(r["fold"]) for r in fr])
print(f"{len(rows)} clips, {len(set(spk))} speakers, folds loaded from {os.path.abspath(FOLDS)}", flush=True)

def inputs_for(r):  # verbatim from sft_projector.py
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    return inp

state_f = f"{OUTP}_state.json"
st = json.load(open(state_f)) if os.path.exists(state_f) else {
    "done": [], "oof": [0.0] * len(rows), "oof_multi": [0.0] * len(rows),
    "mass": [0.0] * len(rows), "minutes_per_fold": {}, "steps_per_fold": {}}
oof = np.array(st["oof"]); oof_m = np.array(st["oof_multi"]); mass = np.array(st["mass"])
t0 = time.time(); torch.cuda.reset_peak_memory_stats()

for k in range(5):
    if k in st["done"]: continue
    tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
    with torch.no_grad():
        for n_, p_ in net.named_parameters():
            if n_ in lora_init: p_.data.copy_(lora_init[n_])
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR)
    net.train(); fold_t0 = time.time(); steps = 0
    for ep in range(EPOCHS):
        order = np.random.RandomState(k * 10 + ep).permutation(tr); tot = nb = 0
        for i in order:
            out = net(**inputs_for(rows[i]))
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
            loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1; steps += 1
            if steps == 50:
                per = (time.time() - fold_t0) / 50
                proj_total = per * (EPOCHS * len(tr) + len(te)) * 5 / 60
                print(f"PROBE 50 steps: {per:.3f} s/step -> projected {proj_total:.1f} min "
                      f"for all 5 folds ({EPOCHS} ep x {len(tr)} clips + {len(te)} eval each)", flush=True)
        print(f"fold {k} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    net.eval()
    with torch.no_grad():
        for i in te:
            out = net(**inputs_for(rows[i]))
            lg = out.logits[0, -1].float()
            oof[i] = float(torch.softmax(lg[[YES, NO]], -1)[0])
            pr = torch.softmax(lg, -1)
            py = float(pr[YES_SET].sum()); pn = float(pr[NO_SET].sum())
            oof_m[i] = py / (py + pn); mass[i] = py + pn
    st["done"].append(k); st["oof"] = oof.tolist(); st["oof_multi"] = oof_m.tolist()
    st["mass"] = mass.tolist()
    st["minutes_per_fold"][str(k)] = round((time.time() - fold_t0) / 60, 2)
    st["steps_per_fold"][str(k)] = steps
    st["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    json.dump(st, open(state_f, "w"))
    print(f"fold {k} done {st['minutes_per_fold'][str(k)]} min, {steps} steps, "
          f"peak {st['peak_gpu_gb']} GB, total {(time.time()-t0)/60:.1f} min", flush=True)

with open(f"{OUTP}_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "spk", "label", "arm", "fold", "p_yes", "answer_mass", "p_yes_multivariant"])
    for i, r in enumerate(rows):
        w.writerow([os.path.basename(r["path"]), r["speaker"], r["label"], r["set"], fold[i],
                    oof[i], mass[i], oof_m[i]])
json.dump({**counts, "clip_list": os.path.abspath(MAN), "folds_file": os.path.abspath(FOLDS),
           "condition": COND, "prompt": P, "n": len(rows), "n_speakers": int(len(set(spk))),
           "epochs": EPOCHS, "lr": LR, "batch_size": 1, "trained": "LoRA on LM q,k,v,o only; "
           "audio encoder and projector frozen", "window_seconds": 30, "dtype": "bfloat16",
           "minutes_per_fold": st["minutes_per_fold"], "steps_per_fold": st["steps_per_fold"],
           "peak_gpu_gb": st.get("peak_gpu_gb"), "minutes_total": round((time.time()-t0)/60, 2),
           "command": " ".join(["python3"] + sys.argv), "date": datetime.date.today().isoformat()},
          open(f"{OUTP}_lora.json", "w"), indent=1)
print(f"WROTE {OUTP}_oof.csv  total {(time.time()-t0)/60:.1f} min", flush=True)
