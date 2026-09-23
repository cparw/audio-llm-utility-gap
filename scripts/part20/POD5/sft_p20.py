"""PART20 POD5. Copy of p15/sft_projector.py, Qwen2.5-Omni branch, recipe unchanged (thinker.audio_tower.proj only,
float32 projector, AdamW lr 1e-4, 3 epochs, one clip per update, order RandomState(fold*10+ep).permutation(tr),
cross entropy over the ' Yes' and ' No' logits at the answer position, 30 s window, bf16 model).
Changed only in: folds are READ from FOLDFILE keyed by orig_clip_id (never recomputed); at evaluation the part20
rule p_yes (single-token Yes/yes/YES vs No/no/NO, bare and leading space, full softmax) and answer_mass are
recorded beside the recipe's two-logit p_yes; resumable per fold.
usage: sft_p20.py MANIFEST(path,label,speaker,orig_clip_id,set) OUTPREFIX FOLDFILE"""
import os, csv, sys, time, copy, json, datetime, numpy as np, torch, librosa
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))  # pod cgroup quota is 15.3 CPUs; logging/threads only, recipe unchanged
MAN, OUTP, FOLDF = sys.argv[1:4]
MID = "Qwen/Qwen2.5-Omni-7B"; EPOCHS, LR = 3, 1e-4
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
dev = "cuda"; DT = torch.bfloat16
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
def ids_rule(ws):
    out = []
    for w in ws:
        for f in (w, " " + w):
            t = tok.encode(f, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YR, NR = ids_rule(["Yes", "yes", "YES"]), ids_rule(["No", "no", "NO"])
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
fd = {r["clip_id"]: (int(r["fold"]), r["speaker_id"]) for r in csv.DictReader(open(FOLDF))}
fold_of = np.array([fd[r["orig_clip_id"]][0] for r in rows])
assert all(fd[r["orig_clip_id"]][1] == r["speaker"] for r in rows), "speaker mismatch vs fold file"
print(f"MODEL {MID} | {len(rows)} clips, {len(set(spk))} speakers | recipe ids yes {YES} no {NO} | rule ids {YR} {NR} | fold sizes {np.bincount(fold_of).tolist()}", flush=True)
def inputs_for(r):
    x, _ = librosa.load(r["path"], sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    inp.pop("use_audio_in_video", None)
    return inp
state_f = f"{OUTP}_state.json"
st = json.load(open(state_f)) if os.path.exists(state_f) else {"done": [], "oof": [0.0] * len(rows), "oof_rule": [0.0] * len(rows), "mass": [0.0] * len(rows), "loss": {}}
oof, oofr, mass = np.array(st["oof"]), np.array(st["oof_rule"]), np.array(st["mass"]); t0 = time.time()
for fold in range(5):
    tr, te = np.where(fold_of != fold)[0], np.where(fold_of == fold)[0]
    if fold in st["done"]: print("skip done fold", fold, flush=True); continue
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
            if nb % 50 == 0: print(f'  fold {fold} epoch {ep} step {nb}/{len(order)} {(time.time()-t0)/60:.1f} min', flush=True)
        st["loss"][f"{fold}_{ep}"] = tot / nb
        print(f"fold {fold} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    net.eval()
    with torch.no_grad():
        for i in te:
            out = net(**inputs_for(rows[i]))
            lg = out.logits[0, -1].float()
            oof[i] = float(torch.softmax(lg[[YES, NO]], -1)[0])
            pr = torch.softmax(lg, -1); py, pn = float(pr[YR].sum()), float(pr[NR].sum())
            oofr[i] = py / (py + pn); mass[i] = py + pn
    st["done"].append(fold); st["oof"] = oof.tolist(); st["oof_rule"] = oofr.tolist(); st["mass"] = mass.tolist()
    json.dump(st, open(state_f + ".tmp", "w")); os.replace(state_f + ".tmp", state_f)
    print(f"fold {fold} eval done {(time.time()-t0)/60:.1f} min", flush=True)
with open(f"{OUTP}_oof.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "orig_clip_id", "speaker", "label", "set", "fold", "p_yes", "p_yes_rule", "answer_mass_rule", "prompt"])
    for i, r in enumerate(rows): w.writerow([r["path"], r["orig_clip_id"], r["speaker"], r["label"], r["set"], int(fold_of[i]), oof[i], oofr[i], mass[i], P])
json.dump({"checkpoint": MID, "clip_list": os.path.abspath(MAN), "fold_file": os.path.abspath(FOLDF), "prompt": P, "n": len(rows),
           "n_speakers": int(len(set(spk))), "epochs": EPOCHS, "lr": LR, "trained": "projector only (thinker.audio_tower.proj)",
           "p_yes": "recipe: softmax over the ' Yes' and ' No' logits", "yes_id": YES, "no_id": NO, "rule_ids": [YR, NR],
           "losses": st["loss"], "minutes": round((time.time() - t0) / 60, 1), "date": datetime.date.today().isoformat()},
          open(f"{OUTP}_sft.json", "w"), indent=1)
print("SFT DONE", flush=True)
