"""PART20 POD1b: the STANDARD Pitt projector fine-tune (sft_projector.py recipe unchanged) at a new seed.
Recipe copied from <local data dir>/scratch/p15/sft_projector.py (the script behind omni_final/omnisft_pitt_*):
  Qwen2.5-Omni-7B thinker, only thinker.audio_tower.proj trained (float32 forward), encoder + LM frozen, bf16,
  AdamW lr 1e-4, 3 epochs, one clip per update, cross entropy over the [' Yes',' No'] logits at the answer
  position, 30 s window, no class weighting.
Only differences from sft_projector.py:
  1. folds LOADED from release/folds/pitt_groupkfold5_pod.csv (matched by clip basename), not recomputed;
     GroupKFold(5) is also run here only to report agreement with the file.
  2. seed: sft_projector.py has no seed argument; its only RNG is np.random.RandomState(fold*10+ep)
     (clip order), i.e. offset 0. Here the order RNG is np.random.RandomState(SEED*1000 + fold*10 + ep),
     disjoint from the original's 0..42, and torch.manual_seed(SEED) is set.
  3. processor inputs precomputed once by precompute_inputs.py (same code path), loaded from INPUT_CACHE,
     spot-checked equal to fresh processing on 3 clips; speed only.
  4. extra scoring columns: p_yes_multivariant and answer_mass (full-vocab softmax). p_yes is the recipe's
     two-token softmax, identical to the original run's p_yes.
usage: std_ft_seed.py MANIFEST FOLDS SEED OUTPREFIX"""
import os, csv, sys, time, copy, json, datetime, numpy as np, torch, librosa
from sklearn.model_selection import GroupKFold
MAN, FOLDS, SEED, OUTP = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
MID = "Qwen/Qwen2.5-Omni-7B"
EPOCHS, LR = 3, 1e-4
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
dev = "cuda"; DT = torch.bfloat16
torch.manual_seed(SEED); torch.set_num_threads(8)
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=DT)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev)
proj = net.audio_tower.proj
tc = net.config
drop = {}
for sub in ("audio_config", "text_config"):
    c = getattr(tc, sub, None)
    if c is not None:
        for k_ in ("dropout", "attention_dropout", "activation_dropout", "encoder_layerdrop", "hidden_dropout"):
            if hasattr(c, k_): drop[f"{sub}.{k_}"] = getattr(c, k_)
print("DROPOUT_CONFIG " + json.dumps(drop), flush=True)
tok = proc.tokenizer
_orig = proj.forward
def _fp32_forward(x): return _orig(x.float()).to(DT)
proj.forward = _fp32_forward
for p in net.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
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
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
fmap = {r["clip_id"]: r for r in csv.DictReader(open(FOLDS))}
fold = np.array([int(fmap[os.path.basename(r["path"])]["fold"]) for r in rows])
assert all(fmap[os.path.basename(r["path"])]["speaker_id"] == r["speaker"] for r in rows)
gk = np.full(len(rows), -1)
for k_, (_, te_) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk)): gk[te_] = k_
FOLD_AGREE = int((gk == fold).sum())
print(f"MODEL {MID} | {len(rows)} clips, {len(set(spk))} speakers | folds from file {FOLDS} "
      f"sizes {np.bincount(fold).tolist()} | pod GroupKFold agrees on {FOLD_AGREE}/{len(rows)} | SEED {SEED}", flush=True)
_cache = {}
CACHE_F = os.environ.get("INPUT_CACHE", "")
if CACHE_F and os.path.exists(CACHE_F):
    C = torch.load(CACHE_F, weights_only=False)
    assert C["paths"] == [r["path"] for r in rows] and C["prompt"] == P
    _pre = dict(enumerate(C["inputs"]))
    print(f"INPUT CACHE loaded {CACHE_F}: {len(_pre)} clips", flush=True)
else:
    _pre = {}
def _fresh(i):
    r = rows[i]
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp.pop("use_audio_in_video", None)
    return dict(inp)
if _pre:
    for i_chk in (0, len(rows) // 2, len(rows) - 1):     # cache must equal fresh processing exactly
        fr_ = _fresh(i_chk)
        assert set(fr_) == set(_pre[i_chk]) and all(torch.equal(fr_[k], _pre[i_chk][k]) for k in fr_), f"cache mismatch clip {i_chk}"
    print("INPUT CACHE equals fresh processing on 3 spot-check clips (exact)", flush=True)
    _cache.update(_pre)
def inputs_for(i):
    if i not in _cache:
        r = rows[i]
        x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp.pop("use_audio_in_video", None)
        _cache[i] = dict(inp)
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in _cache[i].items()}
state_f = f"{OUTP}_state.json"
st = json.load(open(state_f)) if os.path.exists(state_f) else {
    "done": [], "oof": [0.0] * len(rows), "oof_multi": [0.0] * len(rows), "mass": [0.0] * len(rows),
    "minutes_per_fold": {}, "loss": {}}
if st["done"]: print(f"RESUMING from checkpoint, folds done {st['done']}", flush=True)
oof = np.array(st["oof"]); oof_m = np.array(st["oof_multi"]); mass = np.array(st["mass"])
t0 = time.time()
for k in range(5):
    if k in st["done"]: continue
    tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]; ft0 = time.time()
    proj.load_state_dict(proj_init)
    for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
    opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
    net.train(); steps = 0
    for ep in range(EPOCHS):
        order = np.random.RandomState(SEED * 1000 + k * 10 + ep).permutation(tr); tot = nb = 0
        for i in order:
            out = net(**inputs_for(i))
            two = out.logits[0, -1, [YES, NO]].float()
            target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
            loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1; steps += 1
            if steps % 200 == 0:
                print(f"  fold {k} ep {ep} step {steps} {(time.time()-ft0)/steps:.3f} s/step", flush=True)
        st["loss"][f"{k}_{ep}"] = tot / nb
        print(f"fold {k} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    net.eval()
    with torch.no_grad():
        for i in te:
            lg = net(**inputs_for(i)).logits[0, -1].float()
            oof[i] = float(torch.softmax(lg[[YES, NO]], -1)[0])
            pr = torch.softmax(lg, -1); py = float(pr[YES_SET].sum()); pn = float(pr[NO_SET].sum())
            oof_m[i] = py / (py + pn); mass[i] = py + pn
    st["done"].append(k); st["oof"] = oof.tolist(); st["oof_multi"] = oof_m.tolist(); st["mass"] = mass.tolist()
    st["minutes_per_fold"][str(k)] = round((time.time() - ft0) / 60, 2)
    st["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    json.dump(st, open(state_f + ".tmp", "w")); os.replace(state_f + ".tmp", state_f)
    print(f"fold {k} CHECKPOINT {st['minutes_per_fold'][str(k)]} min, peak {st['peak_gpu_gb']} GB", flush=True)
with open(f"{OUTP}_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "clip_id", "speaker", "label", "set", "fold", "p_yes", "p_yes_multivariant", "answer_mass", "seed", "prompt"])
    for i, r in enumerate(rows):
        w.writerow([r["path"], os.path.basename(r["path"]), r["speaker"], r["label"], r["set"], fold[i],
                    oof[i], oof_m[i], mass[i], SEED, P])
import transformers, sklearn
json.dump({"checkpoint": MID, "seed": SEED, "order_rng": f"np.random.RandomState({SEED}*1000 + fold*10 + epoch)",
           "torch_manual_seed": SEED, "clip_list": os.path.abspath(MAN), "folds_file": os.path.abspath(FOLDS),
           "pod_groupkfold_agreement": FOLD_AGREE, "prompt": P, "n": len(rows), "n_speakers": int(len(set(spk))),
           "folds": 5, "epochs": EPOCHS, "lr": LR, "trained": "projector only (thinker.audio_tower.proj)",
           "dropout_config": drop, "yes_id": YES, "no_id": NO, "yes_set": YES_SET, "no_set": NO_SET,
           "loss_per_fold_epoch": st["loss"], "minutes_per_fold": st["minutes_per_fold"],
           "peak_gpu_gb": st.get("peak_gpu_gb"), "transformers": transformers.__version__,
           "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
           "command": " ".join(["python3"] + sys.argv), "date": datetime.date.today().isoformat()},
          open(f"{OUTP}_run.json", "w"), indent=1)
print(f"WROTE {OUTP}_oof.csv total {(time.time()-t0)/60:.1f} min", flush=True)
