"""Part 4: one documented prompt, text-only condition, Yes/No logits at the first answer position.
usage: text_score.py TRANSCRIPTS.csv OUT.csv COND(ad|pd|mdd) [MODEL_ID]
Input needs columns: id,label,speaker,text   (extra columns ignored)."""
import os, csv, sys, json, time, datetime, numpy as np, torch
MAN, OUT, COND = sys.argv[1:4]
MID = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-Omni-7B"
DIS = {"ad": "dementia", "pd": "Parkinson's disease", "mdd": "depression"}[COND]
# ONE template for all seven datasets; only the disease word changes, exactly as the audio prompts do.
P = ("Based only on this transcript of what the person said, does this speaker show signs of "
     f"{DIS}? Answer with one word, Yes or No.")
P = os.environ.get("PROMPT_OVERRIDE", P)
dev = "cuda" if torch.cuda.is_available() else "cpu"
IS_Q3 = "Qwen3-Omni" in MID; IS_OMNI = "Omni" in MID
if IS_Q3:
    from transformers import Qwen3OmniMoeProcessor as Proc, Qwen3OmniMoeForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID); full = Model.from_pretrained(MID, dtype=torch.bfloat16)
    net = full.thinker
    for a in ("talker","code2wav","token2wav"):
        if hasattr(full,a): delattr(full,a)
    if hasattr(net,"visual"): del net.visual
    net = net.to(dev).eval(); tok = proc.tokenizer
elif IS_OMNI:
    from transformers import Qwen2_5OmniProcessor as Proc, Qwen2_5OmniForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID); full = Model.from_pretrained(MID, dtype=torch.bfloat16)
    net = full.thinker
    for a in ("talker","token2wav"):
        if hasattr(full,a): delattr(full,a)
    if hasattr(net,"visual"): del net.visual
    net = net.to(dev).eval(); tok = proc.tokenizer
else:
    from transformers import AutoProcessor as Proc, Qwen2AudioForConditionalGeneration as Model
    proc = Proc.from_pretrained(MID); net = Model.from_pretrained(MID, dtype=torch.float16).to(dev).eval()
    tok = proc.tokenizer

def ids_for(words):
    out = []
    for w in words:
        for form in (w, " " + w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1: out.append(t[0])
    return sorted(set(out))
YES = ids_for(["Yes","yes","YES","Yeah","yeah"]); NO = ids_for(["No","no","NO","Nope","nope"])

rows = list(csv.DictReader(open(MAN))); res = []; t0 = time.time()
for i, r in enumerate(rows):
    txt = (r.get("text") or "").strip()
    msg = [{"role":"user","content":[{"type":"text","text": f'Transcript: "{txt}"\n\n{P}'}]}]
    s = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    inp = proc(text=s, return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items() if torch.is_tensor(v)}
    with torch.no_grad():
        lg = net(**inp).logits[0, -1].float()
    pr = torch.softmax(lg, -1)
    py = float(pr[YES].sum()); pn = float(pr[NO].sum())
    res.append((r.get("speaker",""), r.get("id",""), r["label"], py/(py+pn+1e-12), py+pn, P))
    if (i+1) % 100 == 0 or i+1 == len(rows):
        print(f"{i+1}/{len(rows)} p_yes {res[-1][3]:.3f} mass {res[-1][4]:.3f} {time.time()-t0:.0f}s", flush=True)
with open(OUT,"w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["speaker_id","id","label","p_yes","answer_mass","prompt"]); w.writerows(res)
y = np.array([int(x[2]) for x in res]); p = np.array([x[3] for x in res])
def auc(y,s):
    import numpy as np
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1,len(s)+1)
    n1 = (y==1).sum(); n0 = (y==0).sum()
    return float("nan") if n1==0 or n0==0 else (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
A = auc(y,p)
json.dump({"model":MID,"prompt":P,"condition":COND,"transcripts":os.path.abspath(MAN),
           "n":len(res),"auc":round(A,4),"median_mass":round(float(np.median([x[4] for x in res])),4),
           "script":os.path.abspath(__file__),"date":str(datetime.date.today())},
          open(OUT.replace(".csv",".json"),"w"), indent=1)
print(f"RESULT text {os.path.basename(MAN)} n={len(res)} AUC={A:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
