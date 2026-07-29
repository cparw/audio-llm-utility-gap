"""
Redesigned reliance test: NO text is ever supplied to the model.
Per Soleymani (Jul 17): audio LLMs self-transcribe, so injecting a transcript is
redundant/confounding. Instead we hold the QUESTION fixed and ablate the AUDIO,
so we can tell whether the verdict rides on lexical content or on acoustics.

Conditions (same plain prompt every time):
  orig     : untouched audio (words + acoustics both available)
  lowpass  : low-pass @ cutoff Hz -> words unintelligible, prosody/voice-quality kept
  reversed : time-reversed -> words destroyed, spectral/voice stats kept
  noise    : heavy additive noise at target SNR -> degrades acoustics, words mostly gone
(vowel clips are selected via --tasks vowel: zero lexical content by construction)

Read: if the model truly listens, AUC stays above chance when words are destroyed
(lowpass/reversed/vowel). If it reads, AUC collapses to chance there.
"""
import os, csv, argparse, json
import numpy as np, torch, librosa
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score

DIAG = {"parkinsons": "does this speaker show signs of Parkinson's disease",
        "depression": "does this speaker show signs of depression"}

def lowpass(y, sr, cutoff):
    b, a = butter(6, cutoff / (sr / 2.0), btype="low")
    return filtfilt(b, a, y).astype(np.float32)

def add_noise(y, snr_db):
    p = np.mean(y ** 2) + 1e-12
    n = np.random.RandomState(0).randn(len(y)).astype(np.float32)
    n *= np.sqrt(p / (10 ** (snr_db / 10.0)) / (np.mean(n ** 2) + 1e-12))
    return (y + n).astype(np.float32)

def make(cond, y, sr, cutoff, snr):
    if cond == "orig":     return y
    if cond == "lowpass":  return lowpass(y, sr, cutoff)
    if cond == "reversed": return np.flip(y).copy()
    if cond == "noise":    return add_noise(y, snr)
    raise ValueError(cond)

def load_model(model_id):
    if "Omni" in model_id:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        proc = Qwen2_5OmniProcessor.from_pretrained(model_id)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda").eval()
        try: model.disable_talker()
        except Exception: pass
        kind = "omni"
    else:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        proc = AutoProcessor.from_pretrained(model_id)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda").eval()
        kind = "qwen2audio"
    tok = proc.tokenizer
    def wids(ws):
        s = set()
        for w in ws:
            for v in (w, " " + w):
                e = tok(v, add_special_tokens=False).input_ids
                if e: s.add(e[0])
        return sorted(s)
    return proc, model, kind, wids(["Yes", "yes", "YES"]), wids(["No", "no", "NO"])

SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
       "perceiving auditory and visual inputs, as well as generating text and speech.")

def score(proc, model, kind, yes, no, audio, prompt):
    if kind == "omni":
        conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, return_audio=False, use_audio_in_video=False)
        lg = g.scores[0][0].float()
    else:
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[audio], sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
    p = torch.softmax(lg, dim=-1)
    py, pn = float(p[yes].sum()), float(p[no].sum())
    return py / (py + pn + 1e-9), py + pn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tasks", default="read")
    ap.add_argument("--cond", default="parkinsons")
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--conditions", default="orig,lowpass,reversed,noise")
    ap.add_argument("--cutoff", type=float, default=400.0)
    ap.add_argument("--snr", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    R = "/project2/msoleyma_946/speech_health/results_chaitanya"
    keep = set(a.tasks.split(","))
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in keep]
    if a.limit: rows = rows[:a.limit]
    conds = a.conditions.split(",")
    proc, model, kind, yes, no = load_model(a.model)
    # ONE plain question. No transcript, no injected text, ever.
    prompt = f"Based only on this recording, {DIAG[a.cond]}? Answer with one word, Yes or No."
    print("PROMPT:", prompt, flush=True)

    out = []
    for i, r in enumerate(rows):
        try:
            y, sr = librosa.load(r["filepath"], sr=16000, mono=True)
            rec = dict(speaker=r["speaker_id"], label=int(r["label"]), task=r["task_type"])
            for c in conds:
                s, m = score(proc, model, kind, yes, no, make(c, y, sr, a.cutoff, a.snr), prompt)
                rec[f"p_{c}"] = s
                rec[f"mass_{c}"] = m
            out.append(rec)
            if (i + 1) % 25 == 0: print(f"  {i+1}/{len(rows)}", flush=True)
        except Exception as e:
            print("FAIL", r["filepath"], repr(e), flush=True)
    if not out:
        print("NO OUTPUT"); return

    yl = np.array([o["label"] for o in out])
    summ = {"tag": a.tag, "model": a.model, "cond": a.cond, "tasks": a.tasks,
            "n": len(out), "speakers": len(set(o["speaker"] for o in out)),
            "prompt": prompt, "cutoff_hz": a.cutoff, "noise_snr_db": a.snr,
            "note": "no text supplied to the model in any condition"}
    for c in conds:
        try: summ[f"AUC_{c}"] = round(roc_auc_score(yl, [o[f"p_{c}"] for o in out]), 3)
        except Exception: summ[f"AUC_{c}"] = float("nan")
        summ[f"meanp_{c}"] = round(float(np.mean([o[f"p_{c}"] for o in out])), 3)
        summ[f"mass_{c}"] = round(float(np.mean([o[f"mass_{c}"] for o in out])), 3)

    os.makedirs(f"{R}/ablation", exist_ok=True)
    with open(f"{R}/ablation/{a.tag}_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    json.dump(summ, open(f"{R}/ablation/{a.tag}_summary.json", "w"), indent=2)
    print("=== SUMMARY", a.tag, "===", flush=True)
    print(json.dumps(summ, indent=2), flush=True)

if __name__ == "__main__":
    main()
