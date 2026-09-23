"""Part 20 POD1. BALANCED projector fine-tune on Pitt 468, Qwen2.5-Omni.
Recipe copied from sft_projector.py (the script behind omni_final/omnisft_pitt_oof.csv):
  thinker.audio_tower.proj trained alone (float32 forward), encoder and LM frozen, bf16 model,
  AdamW lr 1e-4, 3 epochs, one clip per update, order = RandomState(fold*10+ep).permutation(train_idx),
  cross entropy over the [' Yes',' No'] logits at the answer position (logits[0,-1]), net.train() while training.
ONLY CHANGE: per-clip loss weight. Inside each training fold, for label y in {0,1}:
  w(clip of label y in arm a) = (N_y/2) / N_{y,a}; then rescaled so mean weight over the training fold = 1.
  --weights uniform gives w = 1 (the standard recipe, used only as a same-pod control).
Folds LOADED from release/folds/pitt_groupkfold5_pod.csv (never recomputed). Train index array = np.where(fold != k)[0],
which is the ascending array GroupKFold hands sft_projector.py.
Scores per test clip: p_yes (recipe: softmax over [' Yes',' No'] logits), p_yes_multi (all single-token Yes/yes/YES vs
No/no/NO, bare and leading space), answer_mass (that total from the full-vocabulary softmax).
Checkpoints: per epoch (projector + optimizer) and per fold (scores).
usage: balanced_ft_pitt.py MANIFEST FOLDS OUTDIR TAG WEIGHTS(balanced|uniform) FOLDLIST(e.g. 0,3)"""
import os, csv, sys, time, json, copy, datetime, numpy as np, torch, librosa
MAN, FOLDS, OUTD, TAG, WMODE, FL = sys.argv[1:7]
FOLDLIST = [int(x) for x in FL.split(",")]
MID = "Qwen/Qwen2.5-Omni-7B"
EPOCHS, LR = 3, 1e-4
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
dev = "cuda"; DT = torch.bfloat16
os.makedirs(OUTD, exist_ok=True)
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=DT)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev)
proj = net.audio_tower.proj
tok = proc.tokenizer
_orig = proj.forward
def _fp32_forward(x): return _orig(x.float()).to(DT)
proj.forward = _fp32_forward
for p in net.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
def single_ids(words):
    out = []
    for w in words:
        for s in (w, " " + w):
            ids = tok(s, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] not in out: out.append(ids[0])
    return out
YES_V, NO_V = single_ids(["Yes", "yes", "YES"]), single_ids(["No", "no", "NO"])
assert YES in YES_V and NO in NO_V
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([r["speaker"] for r in rows])
arm = np.array([r["set"] for r in rows])
fmap = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(FOLDS))}
fold = np.array([fmap[os.path.basename(r["path"])][0] for r in rows])
assert all(fmap[os.path.basename(r["path"])][1] == r["speaker"] for r in rows), "speaker mismatch manifest vs folds"
print(f"MODEL {MID} | {len(rows)} clips, {len(set(spk))} speakers | YES {YES} NO {NO} YES_V {YES_V} NO_V {NO_V} | "
      f"weights {WMODE} | folds {FOLDLIST} | proj {type(proj).__name__} {sum(p.numel() for p in proj.parameters())}", flush=True)
cache = {}
def inputs_for(i):
    if i not in cache:
        r = rows[i]
        x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp = dict(inp); inp.pop("use_audio_in_video", None)
        cache[i] = inp
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in cache[i].items()}
def weights_for(tr):
    w = np.ones(len(rows)); table = []
    if WMODE == "balanced":
        for lab in (0, 1):
            n_y = int((y[tr] == lab).sum())
            for a in ("conflict", "agreement"):
                m = tr[(y[tr] == lab) & (arm[tr] == a)]
                w[m] = (n_y / 2) / len(m)
        w[tr] = w[tr] / w[tr].mean()
    for lab in (0, 1):
        for a in ("conflict", "agreement"):
            m = tr[(y[tr] == lab) & (arm[tr] == a)]
            table.append({"label": lab, "arm": a, "n": int(len(m)), "weight": float(w[m][0]) if len(m) else float("nan"),
                          "total_weight": float(w[m].sum())})
    return w, table
t0 = time.time()
for k in FOLDLIST:
    fres = f"{OUTD}/{TAG}_fold{k}.json"
    if os.path.exists(fres): print(f"fold {k} already done, skip", flush=True); continue
    tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
    assert not (set(spk[tr]) & set(spk[te])), "speaker leak"
    w, table = weights_for(tr)
    print(f"fold {k} WEIGHT TABLE (train n={len(tr)}, mean weight {w[tr].mean():.6f}):", flush=True)
    for t in table: print(f"   label {t['label']} {t['arm']:9s} n={t['n']:3d} weight={t['weight']:.6f} total={t['total_weight']:.3f}", flush=True)
    proj.load_state_dict(proj_init)
    for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
    opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
    ck = f"{OUTD}/{TAG}_fold{k}_ckpt.pt"; ep0 = 0; losses = []
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev)
        proj.load_state_dict(c["proj"]); opt.load_state_dict(c["opt"]); ep0 = c["epoch"] + 1; losses = c["losses"]
        print(f"fold {k} RESUMED from epoch checkpoint {c['epoch']}", flush=True)
    torch.manual_seed(1000 + k)
    net.train()
    for ep in range(ep0, EPOCHS):
        order = np.random.RandomState(k * 10 + ep).permutation(tr); tot = totw = 0.0; nb = 0
        for i in order:
            out = net(**inputs_for(i))
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
            ce = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss = ce * float(w[i])
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(ce.detach()); totw += float(loss.detach()); nb += 1
        losses.append({"epoch": ep, "mean_ce": tot / nb, "mean_weighted_loss": totw / nb, "steps": nb})
        print(f"fold {k} epoch {ep} ce {tot/nb:.4f} weighted {totw/nb:.4f} steps {nb} {(time.time()-t0)/60:.1f} min "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
        torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "epoch": ep, "losses": losses}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    net.eval(); res = []
    with torch.no_grad():
        for i in te:
            lg = net(**inputs_for(i)).logits[0, -1].float()
            p2 = float(torch.softmax(lg[[YES, NO]], -1)[0])
            pr = torch.softmax(lg, -1); py = float(pr[YES_V].sum()); pn = float(pr[NO_V].sum())
            res.append({"row": int(i), "p_yes": p2, "p_yes_multi": py / (py + pn), "answer_mass": py + pn})
    json.dump({"fold": k, "weights_mode": WMODE, "weight_table": table, "losses": losses, "n_train": int(len(tr)),
               "n_test": int(len(te)), "scores": res, "minutes": (time.time() - t0) / 60,
               "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "date": datetime.datetime.utcnow().isoformat()},
              open(fres + ".tmp", "w"), indent=1)
    os.replace(fres + ".tmp", fres)
    print(f"fold {k} eval done n_test {len(te)} {(time.time()-t0)/60:.1f} min", flush=True)
print(f"PROCESS DONE folds {FOLDLIST} {(time.time()-t0)/60:.1f} min", flush=True)
