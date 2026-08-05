"""
INTERVENTION 1b: training-free LEXICAL DEBIASING at inference.

If the failure is that the model answers from the words, the fix does not have to
be a better instruction. Score the same question twice:

    p_audio : the recording, no transcript          (what the benchmark does)
    p_text  : the transcript only, no audio         (the pure lexical prior)

and decide on the residual
    s(alpha) = logit(p_audio) - alpha * logit(p_text)

alpha = 0 is the current benchmark. alpha > 0 subtracts off exactly the component
of the answer that the words alone can explain. Nothing is trained; the second
forward pass is text-only and cheap.

This script produces p_text for the 390 conflict/agreement segments. The alpha
sweep and the nested evaluation are in pt_contrastive.py.
"""
import os, sys, json, time, argparse, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
PX = f"{R}/committed/paradox"
OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)

Q = "does this speaker show signs of depression"
TMPL = ("Here is a transcript of what a person said: \"%s\"\n"
        "Based on this, " + Q + "? Answer with one word, Yes or No.")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="omni", choices=["omni", "qwen2audio"])
A = ap.parse_args()


def wids(tok, ws):
    s = set()
    for w in ws:
        for v in (w, " " + w):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)


if A.model == "omni":
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    MID = "Qwen/Qwen2.5-Omni-7B"
    SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
           "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
    proc = Qwen2_5OmniProcessor.from_pretrained(MID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16, device_map="cuda").eval()
    try: model.disable_talker()
    except Exception: pass
    tok = proc.tokenizer
    YES, NO = wids(tok, ["Yes", "yes", "YES"]), wids(tok, ["No", "no", "NO"])

    def score_text(t):
        conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "text", "text": t}]}]
        txt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=txt, return_tensors="pt", padding=True)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, return_audio=False)
        p = torch.softmax(g.scores[0][0].float(), dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        return py / (py + pn + 1e-9), py + pn
else:
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    MID = "Qwen/Qwen2-Audio-7B-Instruct"
    proc = AutoProcessor.from_pretrained(MID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16, device_map="cuda").eval()
    tok = proc.tokenizer
    YES, NO = wids(tok, ["Yes", "yes", "YES"]), wids(tok, ["No", "no", "NO"])

    def score_text(t):
        conv = [{"role": "user", "content": [{"type": "text", "text": t}]}]
        txt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = tok(txt, return_tensors="pt")
        inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(lg, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        return py / (py + pn + 1e-9), py + pn

print("yes", YES, "no", NO, flush=True)
M = pd.read_csv(f"{PX}/manifest.csv")
t0 = time.time()
rows = []
for i, r in M.iterrows():
    t = str(r.text) if isinstance(r.text, str) else ""
    try:
        p, m = score_text(TMPL % t.replace('"', "'"))
    except Exception as e:
        print("FAIL", r.seg_uid, repr(e)[:120], flush=True); p, m = np.nan, np.nan
    rows.append(dict(seg_uid=r.seg_uid, set=r["set"], pair_id=r.pair_id,
                     speaker=r.speaker_id, label=r.label,
                     words_say_depressed=r.words_say_depressed, nw=r.nw,
                     p_text=p, mass_text=m))
    if (i + 1) % 50 == 0:
        print("  %d/%d %.0fs" % (i + 1, len(M), time.time() - t0), flush=True)
D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/textonly_{A.model}_clips.csv", index=False)

from sklearn.metrics import roc_auc_score
C, G = D[D.set == "conflict"], D[D.set == "agreement"]
print("\ntext-only P(yes): conflict meanP %.3f  agreement meanP %.3f" % (C.p_text.mean(), G.p_text.mean()))
print("text-only AUC vs CLINICAL LABEL : conflict %.3f  agreement %.3f"
      % (roc_auc_score(C.label, C.p_text), roc_auc_score(G.label, G.p_text)))
print("text-only AUC vs LEXICAL direction: conflict %.3f  agreement %.3f"
      % (roc_auc_score(C.words_say_depressed, C.p_text),
         roc_auc_score(G.words_say_depressed, G.p_text)))
print("wrote", f"{OUT}/textonly_{A.model}_clips.csv")
print("JOB_DONE", flush=True)
