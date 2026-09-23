"""PART20 POD2: BALANCED LoRA on Pitt 468 for Qwen2.5-Omni.
Copy of part16/POD2/lora_pitt.py (LoRA r=8 alpha=16 dropout=0.05 on q,k,v,o of every LM layer; encoder and
projector frozen; CE over [' Yes',' No'] logits at the answer position; AdamW 1e-4; 3 epochs; one clip per update;
same per-epoch shuffle RandomState(k*10+ep)), with ONE change: each training clip's loss is multiplied by a
per-clip weight so that, inside every training fold and for each label y, conflict and agreement segments carry
equal total weight: w = (N_y/2)/N_{y,a}, then rescaled so the mean weight over the training fold is 1.
Folds LOADED from the saved file (release/folds/pitt_groupkfold5_pod.csv), never recomputed.
Engineering changes that do not change the math: processor outputs cached per clip; folds can be split over
several processes (each fold restarts from the same seeded LoRA init, as the reference restarts from lora_init);
torch seeds fixed (reference left them unseeded).
usage: balanced_lora_pitt.py <manifest.csv> <folds.csv> <outprefix> <comma folds> [MODEL_ID]"""
import os, csv, sys, json, time, datetime, numpy as np, torch, librosa

MAN, FOLDS, OUTP, FLIST = sys.argv[1:5]
MID = sys.argv[5] if len(sys.argv) > 5 else "Qwen/Qwen2.5-Omni-7B"
MYFOLDS = [int(x) for x in FLIST.split(",")]
EPOCHS, LR, RANK, ALPHA, DROP = 3, 1e-4, 8, 16, 0.05
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
dev = "cuda"; DT = torch.bfloat16

rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
arm = np.array([r["set"] for r in rows])
fr = list(csv.DictReader(open(FOLDS)))
assert len(fr) == len(rows) == 468
for a, b in zip(rows, fr):
    assert os.path.basename(a["path"]) == b["clip_id"], "fold file order mismatch"
    assert a["speaker"] == b["speaker_id"], "speaker mismatch"
fold = np.array([int(r["fold"]) for r in fr])
print(f"{len(rows)} clips, {len(set(spk))} speakers, folds loaded from {os.path.abspath(FOLDS)}; "
      f"fold sizes {np.bincount(fold).tolist()}; my folds {MYFOLDS}", flush=True)

def fold_weights(tr):
    w = np.zeros(len(rows)); table = []
    for lab in (0, 1):
        Ny = int((y[tr] == lab).sum())
        for a in ("conflict", "agreement"):
            m = tr[(y[tr] == lab) & (arm[tr] == a)]
            Nya = len(m); w[m] = (Ny / 2.0) / Nya
            table.append({"label": lab, "arm": a, "N_y": Ny, "N_ya": Nya, "w_raw": (Ny / 2.0) / Nya})
    scale = 1.0 / w[tr].mean(); w = w * scale
    for t in table:
        t["w_rescaled"] = t["w_raw"] * scale; t["total_weight"] = t["w_rescaled"] * t["N_ya"]
    return w, table, scale

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
layers = net.model.layers; lm_prefix = "model.layers"
targets = [n for n, m in net.named_modules() if n.startswith(lm_prefix + ".")
           and n.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj") and isinstance(m, torch.nn.Linear)]
N_LM_LAYERS = len(layers)
assert len(targets) == 4 * N_LM_LAYERS and not any(".audio_tower." in t for t in targets)
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
net = get_peft_model(net, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROP, bias="none",
                                     target_modules=targets, task_type=None))
for n_, p_ in net.named_parameters():
    p_.requires_grad = ("lora_" in n_)
    if p_.requires_grad: p_.data = p_.data.float()
LORA_PARAMS = sum(p.numel() for p in net.parameters() if p.requires_grad)
assert not any("audio_tower" in n_ for n_, p_ in net.named_parameters() if p_.requires_grad)
lora_init = {n_: p_.detach().clone() for n_, p_ in net.named_parameters() if p_.requires_grad}
counts = {"model_id": MID, "total_params_full_model": int(TOTAL_FULL), "total_params_thinker_audio_plus_lm": int(TOTAL_THINKER),
          "lora_trainable_params": int(LORA_PARAMS), "lora_pct_of_full_model": round(100.0 * LORA_PARAMS / TOTAL_FULL, 6),
          "lora_pct_of_thinker": round(100.0 * LORA_PARAMS / TOTAL_THINKER, 6), "lm_layers": int(N_LM_LAYERS),
          "lora_target_modules": len(targets), "rank": RANK, "alpha": ALPHA, "dropout": DROP}
print("PARAMCOUNTS " + json.dumps(counts), flush=True)
json.dump(counts, open(f"{OUTP}_params.json", "w"), indent=1)

def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
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

CACHE = {}
def inputs_for(i):  # same construction as lora_pitt.py / sft_projector.py; cached on CPU
    if i not in CACHE:
        x, _ = librosa.load(rows[i]["path"], sr=16000); x = x[:16000 * 30]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp.pop("use_audio_in_video", None)
        CACHE[i] = dict(inp)
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in CACHE[i].items()}

t0 = time.time(); torch.cuda.reset_peak_memory_stats()
for k in MYFOLDS:
    ck = f"{OUTP}_fold{k}.json"
    if os.path.exists(ck): print(f"fold {k} checkpoint exists, skip", flush=True); continue
    tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
    w, table, scale = fold_weights(tr)
    print(f"WEIGHTS fold {k} (train n={len(tr)}, rescale x{scale:.6f}):", flush=True)
    for t in table:
        print(f"  label={t['label']} arm={t['arm']:9s} N_y={t['N_y']:3d} N_ya={t['N_ya']:3d} "
              f"w_raw={t['w_raw']:.6f} w={t['w_rescaled']:.6f} total={t['total_weight']:.3f}", flush=True)
    assert abs(w[tr].mean() - 1) < 1e-9
    with torch.no_grad():
        for n_, p_ in net.named_parameters():
            if n_ in lora_init: p_.data.copy_(lora_init[n_])
    torch.manual_seed(1000 + k)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=LR)
    net.train(); fold_t0 = time.time(); steps = 0; ep_loss = []
    for ep in range(EPOCHS):
        order = np.random.RandomState(k * 10 + ep).permutation(tr); tot = nb = 0; totw = 0
        for i in order:
            out = net(**inputs_for(i))
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
            ce = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss = float(w[i]) * ce
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(ce.detach()); totw += float(loss.detach()); nb += 1; steps += 1
            if steps == 50:
                per = (time.time() - fold_t0) / 50
                print(f"PROBE fold {k} 50 steps: {per:.3f} s/step -> ~{per*(EPOCHS*len(tr)+len(te))/60:.1f} min this fold", flush=True)
        ep_loss.append({"epoch": ep, "mean_ce": tot / nb, "mean_weighted_ce": totw / nb})
        print(f"fold {k} epoch {ep} ce {tot/nb:.4f} wce {totw/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    net.eval(); res = []
    with torch.no_grad():
        for i in te:
            lg = net(**inputs_for(i)).logits[0, -1].float()
            p2 = float(torch.softmax(lg[[YES, NO]], -1)[0])
            pr = torch.softmax(lg, -1); py = float(pr[YES_SET].sum()); pn = float(pr[NO_SET].sum())
            res.append({"row": int(i), "p_yes_2logit": p2, "p_yes": py / (py + pn), "answer_mass": py + pn})
    adapter = {n_: p_.detach().cpu() for n_, p_ in net.named_parameters() if p_.requires_grad}
    torch.save(adapter, f"{OUTP}_fold{k}_adapter.pt")
    json.dump({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), "weight_table": table, "rescale": scale,
               "steps": steps, "epoch_loss": ep_loss, "minutes": round((time.time() - fold_t0) / 60, 2),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3), "results": res,
               "torch_seed_init": 0, "torch_seed_fold": 1000 + k, "command": " ".join(["python3"] + sys.argv),
               "finished_utc": datetime.datetime.utcnow().isoformat()}, open(ck + ".tmp", "w"), indent=1)
    os.replace(ck + ".tmp", ck)
    print(f"fold {k} done {(time.time()-fold_t0)/60:.2f} min, {steps} steps, peak "
          f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB", flush=True)
print(f"ALLMYFOLDS_DONE {MYFOLDS} {(time.time()-t0)/60:.1f} min", flush=True)
