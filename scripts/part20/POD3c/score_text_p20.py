"""PART20 POD3c: transcript-only Yes/No scoring at the first answer position.
Same input construction as p15/text_score.py (the script behind p14_o25_text.csv / p14_q2a_text.csv):
user turn = 'Transcript: "<text>"\n\n<PROMPT>', no system message, add_generation_prompt=True.
Two p_yes columns are written:
  p_yes      = rule-3 set: Yes/yes/YES + No/no/NO, bare and leading-space, single-token encodings only
  p_yes_ts   = text_score.py set (adds Yeah/yeah and Nope/nope), kept only to validate against earlier runs
usage: score_text_p20.py MODEL_ID IN.csv OUT.csv [LIMIT]
Resumes from OUT.csv.partial (checkpoint every 25 rows)."""
import os, sys, csv, json, time, torch
MID, INP, OUT = sys.argv[1:4]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
P = ("Based only on this transcript of what the person said, does this speaker show signs of "
     "depression? Answer with one word, Yes or No.")
dev = "cuda"
t0 = time.time()
if "Qwen3-Omni" in MID:
    from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    full = Model.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
    net = full.thinker
    for a in ("talker", "code2wav", "token2wav"):
        if hasattr(full, a): delattr(full, a)
    if hasattr(net, "visual"): del net.visual
else:
    from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID)
    full = Model.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda")
    net = full.thinker
    for a in ("talker", "token2wav"):
        if hasattr(full, a): delattr(full, a)
    if hasattr(net, "visual"): del net.visual
net = net.eval(); torch.cuda.empty_cache()
tok = proc.tokenizer
print(f"loaded {MID} in {time.time()-t0:.0f}s; gpu mem {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

def single(words):
    out = {}
    for w in words:
        for form in (w, " " + w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1: out[form] = t[0]
    return out
YES_R = single(["Yes", "yes", "YES"]); NO_R = single(["No", "no", "NO"])
YES_T = single(["Yes", "yes", "YES", "Yeah", "yeah"]); NO_T = single(["No", "no", "NO", "Nope", "nope"])
print("rule3 yes", YES_R, "no", NO_R, flush=True)
print("text_score yes", YES_T, "no", NO_T, flush=True)
yr, nr = sorted(set(YES_R.values())), sorted(set(NO_R.values()))
yt, nt = sorted(set(YES_T.values())), sorted(set(NO_T.values()))

rows = list(csv.DictReader(open(INP)))
if LIMIT: rows = rows[:LIMIT]
hdr = ["speaker_id", "id", "set", "label", "p_yes", "answer_mass", "p_yes_ts", "answer_mass_ts", "top1_token", "prompt"]
part = OUT + ".partial"; done = []
if os.path.exists(part):
    done = list(csv.reader(open(part)))[1:]
    print(f"RESUME from checkpoint: {len(done)} rows already scored", flush=True)
res = done[:]
example = None
t1 = time.time()
for i in range(len(done), len(rows)):
    r = rows[i]
    txt = (r.get("text") or "").strip()
    msg = [{"role": "user", "content": [{"type": "text", "text": f'Transcript: "{txt}"\n\n{P}'}]}]
    s = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    if example is None: example = s
    inp = proc(text=s, return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items() if torch.is_tensor(v)}
    with torch.no_grad():
        lg = net(**inp).logits[0, -1].float()
    pr = torch.softmax(lg, -1)
    py, pn = float(pr[yr].sum()), float(pr[nr].sum())
    pyt, pnt = float(pr[yt].sum()), float(pr[nt].sum())
    top = tok.decode([int(lg.argmax())])
    res.append([r.get("speaker", ""), r["id"], r.get("set", ""), r["label"], repr(py / (py + pn)), repr(py + pn),
                repr(pyt / (pyt + pnt)), repr(pyt + pnt), top, P])
    if (i + 1) % 25 == 0 or i + 1 == len(rows):
        with open(part, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(res)
        print(f"{i+1}/{len(rows)} p_yes {py/(py+pn):.4f} mass {py+pn:.4f} top1 {top!r} {time.time()-t1:.0f}s", flush=True)
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(res)
json.dump({"model": MID, "prompt": P, "rendered_example": example, "yes_ids_rule3": YES_R, "no_ids_rule3": NO_R,
           "yes_ids_textscore": YES_T, "no_ids_textscore": NO_T, "n": len(res), "input": os.path.abspath(INP),
           "dtype": "bfloat16", "transformers": __import__("transformers").__version__, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "seconds": round(time.time() - t0, 1)},
          open(OUT.replace(".csv", "_run.json"), "w"), indent=1)
os.remove(part) if os.path.exists(part) else None
print("FINISHED", OUT, flush=True)
