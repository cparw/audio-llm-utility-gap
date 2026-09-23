"""Train the projector once on ALL clips of a source dataset, then score target datasets zero-shot with it.
usage: omni_transfer.py SRC_MANIFEST SRC_COND OUTPREFIX TARGET1:COND1 [TARGET2:COND2 ...]"""
import os, csv, sys, time, json, datetime, numpy as np, torch, librosa
from sklearn.metrics import roc_auc_score
SRC, SRC_COND, OUTP = sys.argv[1:4]; TARGETS = sys.argv[4:]
MID = "Qwen/Qwen2.5-Omni-7B"; EPOCHS, LR = 3, 1e-4
PR = {"ad": "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No.",
      "pd": "Based only on how this person's voice sounds, does this speaker show signs of Parkinson's disease? Answer with one word, Yes or No.",
      "mdd": "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."}
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
proc = Qwen2_5OmniProcessor.from_pretrained(MID)
full = Qwen2_5OmniForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16)
th = full.thinker
for a in ("talker", "token2wav"):
    if hasattr(full, a): delattr(full, a)
if hasattr(th, "visual"): del th.visual
th = th.to("cuda")
tok = proc.tokenizer; proj = th.audio_tower.proj
_orig = proj.forward
def _fp32(x): return _orig(x.float()).to(torch.bfloat16)
proj.forward = _fp32
for p in th.parameters(): p.requires_grad = False
for p in proj.parameters(): p.data = p.data.float(); p.requires_grad = True
def wid(w): return tok(" " + w, add_special_tokens=False).input_ids[0]
YES, NO = wid("Yes"), wid("No")
def inputs_for(path, prompt):
    x, _ = librosa.load(path, sr=16000); x = x[:16000 * 30]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": prompt}]}]
    t = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=t, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}; inp.pop("use_audio_in_video", None)
    return inp
rows = list(csv.DictReader(open(SRC))); y = np.array([int(r["label"]) for r in rows])
opt = torch.optim.AdamW([p for p in proj.parameters() if p.requires_grad], lr=LR); th.train(); t0 = time.time()
for ep in range(EPOCHS):
    order = np.random.RandomState(ep).permutation(len(rows)); tot = nb = 0
    for i in order:
        out = th(**inputs_for(rows[i]["path"], PR[SRC_COND]))
        two = out.logits[0, -1, [YES, NO]].float()
        loss = torch.nn.functional.cross_entropy(two.unsqueeze(0), torch.tensor(0 if y[i] == 1 else 1, device="cuda").unsqueeze(0))
        loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach()); nb += 1
    print(f"transfer source {os.path.basename(SRC)} epoch {ep} loss {tot/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
torch.save({k: v.detach().cpu() for k, v in proj.state_dict().items()}, f"{OUTP}_projector.pt")
th.eval(); out_rows = []
for spec in TARGETS:
    tman, tcond = spec.split(":")
    trows = list(csv.DictReader(open(tman))); res = []
    for r in trows:
        with torch.no_grad():
            pr = torch.softmax(th(**inputs_for(r["path"], PR[tcond])).logits[0, -1].float(), -1).cpu().numpy()
        py, pn = float(pr[[YES]].sum()), float(pr[[NO]].sum())
        res.append((os.path.basename(r["path"]), r["speaker"], r["label"], py / (py + pn + 1e-12), PR[tcond]))
    name = os.path.basename(tman).replace("mf_", "").replace(".csv", "")
    with open(f"{OUTP}_to_{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["clip", "speaker", "label", "p_yes", "prompt"]); w.writerows(res)
    a = float(roc_auc_score([int(x[2]) for x in res], [x[3] for x in res]))
    out_rows.append({"target": name, "n": len(res), "auc_after_transfer": round(a, 4), "file": f"{OUTP}_to_{name}.csv"})
    print(f"RESULT transfer {os.path.basename(SRC)} -> {name} n={len(res)} AUC={a:.4f}", flush=True)
json.dump({"checkpoint": MID, "source": os.path.abspath(SRC), "source_condition": SRC_COND, "epochs": EPOCHS, "lr": LR,
           "trained_on_all_clips_no_folds": True, "targets": out_rows, "minutes": round((time.time()-t0)/60, 1),
           "window_seconds": 30, "dtype": "bfloat16", "date": datetime.date.today().isoformat()},
          open(f"{OUTP}_transfer.json", "w"), indent=1)
