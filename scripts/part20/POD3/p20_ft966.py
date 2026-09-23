"""Part 20 POD3: Qwen2.5-Omni projector fine-tune ON the 966 Part 14 E-DAIC clips (483 pairs), out of fold by speaker.
Recipe = sft_projector.py unchanged: thinker.audio_tower.proj only (fp32 params), rest frozen, AdamW lr 1e-4,
3 epochs, one clip per update, cross entropy over the [" Yes"," No"] logits at the answer position, order
np.random.RandomState(fold*10+ep).permutation(train_idx), no class weighting, bf16 model, 30 s window.
Folds: the saved Part 20 file part14_groupkfold5_pod.csv (fold column of the manifest), asserted equal to
sklearn GroupKFold(5) on this pod. One process runs ONE fold (so folds run in parallel on the GPU).
Only differences from sft_projector.py: (a) processor outputs are computed once per distinct clip and reused
(identical tensors), (b) float32 processor tensors are cast to bf16 as in part16 POD1 sft_1c.py,
(c) checkpoint after every epoch and after the fold, (d) eval records the paper p_yes too.
Eval per test row: p_yes (paper) = P(Yes)/(P(Yes)+P(No)) over single-token Yes/No variants at the first answer
position, answer_mass = P(Yes)+P(No); p_yes_sft = softmax over [" Yes"," No"] (sft_projector.py convention).
usage: p20_ft966.py MANIFEST OUTDIR FOLD"""
import os, csv, sys, copy, json, time, datetime, numpy as np, torch, librosa
from sklearn.model_selection import GroupKFold
MAN, OUTD, FOLD = sys.argv[1], sys.argv[2], int(sys.argv[3])
MID = "Qwen/Qwen2.5-Omni-7B"; EPOCHS, LR, WIN = 3, 1e-4, 30
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
dev = "cuda"; DT = torch.bfloat16; t0 = time.time()
rows = list(csv.DictReader(open(MAN)))
y = np.array([int(r["label"]) for r in rows]); spk = np.array([str(r["speaker"]) for r in rows])
fold_col = np.array([int(r["fold"]) for r in rows])
gk = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)), y, groups=spk))
for k, (tr_, te_) in enumerate(gk):
    assert np.array_equal(np.sort(te_), np.where(fold_col == k)[0]), f"saved fold {k} != pod GroupKFold"
te = np.where(fold_col == FOLD)[0]; tr = np.where(fold_col != FOLD)[0]
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID); full = Model.from_pretrained(MID, dtype=DT)
net = full.thinker
for a in ("talker", "token2wav"):
    if hasattr(full, a): delattr(full, a)
if hasattr(net, "visual"): del net.visual
net = net.to(dev); proj = net.audio_tower.proj; tok = proc.tokenizer
_orig = proj.forward
def _fp32_forward(x): return _orig(x.float()).to(DT)
proj.forward = _fp32_forward
for p in net.parameters(): p.requires_grad = False
proj_init = copy.deepcopy(proj.state_dict())
def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
YES1, NO1 = wid("Yes"), wid("No")
def ids(ws):
    s = set()
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: s.add(t[0])
    return sorted(s)
YESV, NOV = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
extra = {w: tok.encode(w, add_special_tokens=False) for w in [" YES", " NO"]}
print(f"FOLD {FOLD} train {len(tr)} test {len(te)} | yes {YESV} no {NOV} sft [{YES1},{NO1}] | ' YES'/' NO' enc {extra} | {time.time()-t0:.0f}s", flush=True)
cache = {}
def inputs_for(path):
    if path not in cache:
        x, _ = librosa.load(path, sr=16000); x = x[:16000 * WIN]
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
        inp.pop("use_audio_in_video", None)
        cache[path] = {k: v for k, v in inp.items()}
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in cache[path].items()}
    for k, v in list(inp.items()):
        if torch.is_tensor(v) and v.dtype == torch.float32: inp[k] = v.to(DT)
    return inp
ck = f"{OUTD}/ft966_fold{FOLD}_ckpt.pt"; res_f = f"{OUTD}/ft966_fold{FOLD}.json"
if os.path.exists(res_f): print(f"fold {FOLD} already done", flush=True); sys.exit(0)
proj.load_state_dict(proj_init)
for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR)
ep0, losses = 0, []
if os.path.exists(ck):
    c = torch.load(ck, map_location=dev); proj.load_state_dict(c["proj"]); opt.load_state_dict(c["opt"])
    ep0, losses = c["epochs_done"], c["losses"]; print(f"RESUME fold {FOLD} from epoch {ep0}", flush=True)
net.train()
for ep in range(ep0, EPOCHS):
    order = np.random.RandomState(FOLD * 10 + ep).permutation(tr); tot = nb = 0
    for j, i in enumerate(order):
        out = net(**inputs_for(rows[i]["path"]))
        two = out.logits[0, -1, [YES1, NO1]].float()
        target = torch.tensor(0 if y[i] == 1 else 1, device=dev)
        loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), target.unsqueeze(0))
        loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach()); nb += 1
        if j % 100 == 0: print(f"fold {FOLD} ep {ep} step {j}/{len(order)} run_loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    losses.append(tot / nb)
    torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "epochs_done": ep + 1, "losses": losses}, ck + ".tmp")
    os.replace(ck + ".tmp", ck)
    print(f"fold {FOLD} epoch {ep} loss {tot/nb:.4f} CKPT {(time.time()-t0)/60:.1f} min", flush=True)
net.eval(); res = {}
with torch.no_grad():
    for i in te:
        lg = net(**inputs_for(rows[i]["path"])).logits[0, -1]
        ps = float(torch.softmax(lg[[YES1, NO1]].float(), -1)[0])
        pr = torch.softmax(lg.float(), -1)
        py, pn = float(pr[YESV].sum()), float(pr[NOV].sum())
        res[int(i)] = [py / (py + pn), py + pn, ps]
json.dump({"fold": FOLD, "n_train": int(len(tr)), "n_test": int(len(te)), "epoch_losses": losses,
           "yes_ids": YESV, "no_ids": NOV, "sft_ids": [YES1, NO1], "space_caps_enc": extra,
           "minutes": round((time.time() - t0) / 60, 1), "res": res}, open(res_f + ".tmp", "w"))
os.replace(res_f + ".tmp", res_f)
print(f"FOLD_DONE {FOLD} {(time.time()-t0)/60:.1f} min", flush=True)
