"""Re-score each fold's held-out rows from its saved final projector checkpoint (epochs_done == 3), recording
p_yes with the paper token list (Yes, Yes, yes, yes, YES / No, No, no, no, NO as in extract_probe_layers.py) AND
with the literal rule-3 list that also adds the single-token ' YES' and ' NO'. Also a determinism check against
the in-training eval in ft966_fold<k>.json. usage: p20_eval.py FOLD [FOLD ...]"""
import os, csv, sys, json, time, numpy as np, torch, librosa
MAN = "/workspace/p20/p14_ft966_manifest.csv"; O = "/workspace/scores/part20/POD3"
MID = "Qwen/Qwen2.5-Omni-7B"; WIN = 30; DT = torch.bfloat16; dev = "cuda"; t0 = time.time()
P = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."
rows = list(csv.DictReader(open(MAN))); fold_col = np.array([int(r["fold"]) for r in rows])
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID); full = Model.from_pretrained(MID, dtype=DT); net = full.thinker
for a in ("talker", "token2wav"):
    if hasattr(full, a): delattr(full, a)
if hasattr(net, "visual"): del net.visual
net = net.to(dev); proj = net.audio_tower.proj; tok = proc.tokenizer
_orig = proj.forward
def _fp32_forward(x): return _orig(x.float()).to(DT)
proj.forward = _fp32_forward
for p in net.parameters(): p.requires_grad = False
for p in proj.parameters(): p.data = p.data.float()   # as in training: projector params in fp32 before loading the fp32 checkpoint
def ids(ws):
    s = set()
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: s.add(t[0])
    return sorted(s)
Y5, N5 = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
Y6, N6 = ids(["Yes", " Yes", "yes", " yes", "YES", " YES"]), ids(["No", " No", "no", " no", "NO", " NO"])
YES1, NO1 = tok(" Yes", add_special_tokens=False).input_ids[0], tok(" No", add_special_tokens=False).input_ids[0]
def inputs_for(path):
    x, _ = librosa.load(path, sr=16000); x = x[:16000 * WIN]
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True); inp.pop("use_audio_in_video", None)
    inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}
    for k, v in list(inp.items()):
        if torch.is_tensor(v) and v.dtype == torch.float32: inp[k] = v.to(DT)
    return inp
for FOLD in [int(a) for a in sys.argv[1:]]:
    c = torch.load(f"{O}/ft966_fold{FOLD}_ckpt.pt", map_location=dev); assert c["epochs_done"] == 3
    proj.load_state_dict(c["proj"]); net.eval(); res = {}
    prev = json.load(open(f"{O}/ft966_fold{FOLD}.json"))["res"] if os.path.exists(f"{O}/ft966_fold{FOLD}.json") else {}
    with torch.no_grad():
        for i in np.where(fold_col == FOLD)[0]:
            lg = net(**inputs_for(rows[i]["path"])).logits[0, -1]; pr = torch.softmax(lg.float(), -1)
            y5, n5, y6, n6 = (float(pr[v].sum()) for v in (Y5, N5, Y6, N6))
            ps = float(torch.softmax(lg[[YES1, NO1]].float(), -1)[0])
            res[int(i)] = [y5 / (y5 + n5), y5 + n5, y6 / (y6 + n6), y6 + n6, ps]
    d = [abs(res[i][0] - prev[str(i)][0]) for i in res if str(i) in prev]
    json.dump({"fold": FOLD, "Y5": Y5, "N5": N5, "Y6": Y6, "N6": N6, "max_abs_diff_vs_training_eval": max(d) if d else None,
               "cols": ["p_yes_paper", "mass_paper", "p_yes_rule3_literal", "mass_rule3_literal", "p_yes_sft"], "res": res},
              open(f"{O}/ft966_fold{FOLD}_reeval.json", "w"))
    print(f"REEVAL fold {FOLD} n {len(res)} max|diff| vs training eval {max(d) if d else None} {(time.time()-t0)/60:.1f} min", flush=True)
