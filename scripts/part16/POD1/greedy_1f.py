"""1f: greedy generation, max_new_tokens=4, audio prompt, on the Part 14 clips.
Captures BOTH the generated string and the logit-based p_yes at the FIRST answer position
(scores[0] of the greedy run IS the first answer position), so the two decisions are
guaranteed to come from the same forward pass.
usage: greedy_1f.py MANIFEST OUTCSV [MODEL_ID]"""
import os, csv, sys, json, time, datetime, numpy as np, torch, librosa
MANIFEST, OUTCSV = sys.argv[1:3]
MID = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-Omni-7B"
P = os.environ.get("PROMPT_OVERRIDE",
    "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No.")
WIN = float(os.environ.get("WINDOW_S", "30"))
MAXNEW = int(os.environ.get("MAX_NEW", "4"))
dev = "cuda"
t0 = time.time()
from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
proc = Proc.from_pretrained(MID)
full = Model.from_pretrained(MID, dtype=torch.bfloat16)
net = full.thinker
for attr in ("talker", "token2wav"):
    if hasattr(full, attr): delattr(full, attr)
if hasattr(net, "visual"): del net.visual
net = net.to(dev).eval()
tok = proc.tokenizer
def ids(words):
    s = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: s.add(t[0])          # keep only single-token encodings
    return sorted(s)
YES, NO = ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])
print(f"MODEL {MID} loaded {time.time()-t0:.0f}s | yes ids {YES} | no ids {NO}", flush=True)
rows = list(csv.DictReader(open(MANIFEST)))
fh = open(OUTCSV, "w", newline=""); w = csv.writer(fh)
w.writerow(["id","spk","arm","label","generated_text","first_word","is_yes_no",
            "p_yes","answer_mass","logit_decision","agrees"])
import re
for i, r in enumerate(rows):
    x, _ = librosa.load(r["path"], sr=16000)
    x = x[:int(16000*WIN)] if WIN > 0 else x
    conv = [{"role":"user","content":[{"type":"audio","audio":x},{"type":"text","text":P}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[x], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
    inp.pop("use_audio_in_video", None)
    for _k,_v in list(inp.items()):
        if torch.is_tensor(_v) and _v.dtype == torch.float32: inp[_k] = _v.to(torch.bfloat16)
    with torch.no_grad():
        g = net.generate(**inp, max_new_tokens=MAXNEW, do_sample=False, num_beams=1,
                         output_scores=True, return_dict_in_generate=True)
    n_in = inp["input_ids"].shape[1]
    new_ids = g.sequences[0][n_in:]
    gen = tok.decode(new_ids, skip_special_tokens=True)
    pr = torch.softmax(g.scores[0][0].float(), dim=-1).cpu().numpy()   # FIRST answer position
    py, pn = float(pr[YES].sum()), float(pr[NO].sum())
    mass = py + pn
    p_yes = py / (mass + 1e-12)
    fw_raw = gen.strip().split()[0] if gen.strip() else ""
    fw = re.sub(r"[^A-Za-z]", "", fw_raw).lower()
    is_yn = fw in ("yes","no")
    dec = "Yes" if p_yes > 0.5 else "No"
    agrees = (fw == dec.lower()) if is_yn else ""
    w.writerow([r["id"], r["speaker"], r["arm"], r["label"], gen, fw_raw, int(is_yn),
                f"{p_yes:.10f}", f"{mass:.10f}", dec, agrees])
    if i % 50 == 0 or i == len(rows)-1:
        fh.flush()
        print(f"{i+1}/{len(rows)} {r['id']} gen={gen!r} fw={fw!r} p_yes={p_yes:.3f} mass={mass:.3f} {time.time()-t0:.0f}s", flush=True)
fh.close()
json.dump({"checkpoint":MID,"clip_list":os.path.abspath(MANIFEST),"prompt":P,"n":len(rows),
           "window_seconds":WIN,"max_new_tokens":MAXNEW,"decoding":"greedy do_sample=False num_beams=1",
           "dtype":"bfloat16","device":dev,"yes_ids":YES,"no_ids":NO,
           "p_yes_position":"scores[0] = first generated/answer position",
           "command":f"WINDOW_S={WIN} MAX_NEW={MAXNEW} python3 greedy_1f.py {MANIFEST} {OUTCSV} {MID}",
           "date":datetime.date.today().isoformat()}, open(OUTCSV.replace(".csv",".json"),"w"), indent=1)
print(f"DONE {len(rows)} rows -> {OUTCSV}  {time.time()-t0:.0f}s", flush=True)
