"""Chunk-shuffle control for the DAIC / Qwen2.5-Omni reversal result.

Why this exists
---------------
Reversing a 12-minute interview produces a signal no human ever hears, so the
0.21 AUC drop under reversal could mean "long-range time structure matters" or
it could just mean "the model was handed an impossible input". Cutting the audio
into chunks and shuffling their order keeps every chunk natural forward speech
and keeps the voice untouched; only the ordering is destroyed.

Two things this script fixes relative to the earlier attempt
-----------------------------------------------------------
1. Qwen2.5-Omni has no usable top-level forward(); calling model(**inp) raises
   TypeError("_forward_unimplemented() ... 'input_ids'"). We use the same
   generate(max_new_tokens=1, output_scores=True) convention that
   code/audio_ablation_bench.py used to produce abl_dcaps_omni_summary.json.
2. The Omni feature extractor caps audio at chunk_length=300 s. DAIC clips run
   ~12.7 min, so the model only ever sees the first 5 minutes. We therefore
   truncate to the first 300 s BEFORE building any condition, so all conditions
   present exactly the same acoustic content and differ only in ordering.

Conditions (identical 300 s of content in every one):
  orig    untouched first 300 s
  shuf5s  that window cut into 5 s chunks, order shuffled
  shuf1s  that window cut into 1 s chunks, order shuffled
  rev300  that window time-reversed  <- re-run of "reversed" WITHOUT the content
          shift that contaminated the original reversed number (reversing the
          full 12.7 min then truncating showed the model the LAST 5 minutes)

Same fixed question every time. No transcript is ever supplied.
"""
import os, csv, json, argparse
import numpy as np, torch, librosa
from sklearn.metrics import roc_auc_score

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
SR = 16000
CAP_SEC = 300.0  # Qwen2.5-Omni feature extractor chunk_length

SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
       "perceiving auditory and visual inputs, as well as generating text and speech.")
PROMPT = ("Based only on this recording, does this speaker show signs of depression? "
          "Answer with one word, Yes or No.")


# ---------------------------------------------------------------- conditions
def shuffle_chunks(y, sec, rng):
    n = int(sec * SR)
    parts = [y[i:i + n] for i in range(0, len(y), n)]
    if len(parts) < 3:
        return y.copy()
    order = rng.permutation(len(parts))
    return np.concatenate([parts[i] for i in order]).astype(np.float32)


def make(cond, y, rng):
    if cond == "orig":   return y
    if cond == "shuf5s": return shuffle_chunks(y, 5.0, rng)
    if cond == "shuf1s": return shuffle_chunks(y, 1.0, rng)
    if cond == "rev300": return np.flip(y).copy()
    raise ValueError(cond)


# ---------------------------------------------------------------- model
def load_model(model_id):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    proc = Qwen2_5OmniProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer

    def wids(ws):
        s = set()
        for w in ws:
            for v in (w, " " + w):
                e = tok(v, add_special_tokens=False).input_ids
                if e:
                    s.add(e[0])
        return sorted(s)

    return proc, model, wids(["Yes", "yes", "YES"]), wids(["No", "no", "NO"])


def score(proc, model, yes, no, audio):
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio},
                                         {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True,
               use_audio_in_video=False)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True,
                           return_dict_in_generate=True, return_audio=False,
                           use_audio_in_video=False)
    lg = g.scores[0][0].float()
    p = torch.softmax(lg, dim=-1)
    py, pn = float(p[yes].sum()), float(p[no].sum())
    return py / (py + pn + 1e-9), py + pn


# ---------------------------------------------------------------- bootstrap
def auc_safe(y, s):
    if len(set(y.tolist())) < 2:
        return np.nan
    return roc_auc_score(y, s)


def paired_bootstrap(y, S, conds, n=2000, seed=0):
    """One clip per speaker, so an ordinary resample over rows is correct.
    Every condition is evaluated on the SAME draw, and each difference from
    orig is taken within that draw, so the diff CI is properly paired."""
    rng = np.random.RandomState(seed)
    N = len(y)
    draws = {c: [] for c in conds}
    diffs = {c: [] for c in conds if c != "orig"}
    kept = 0
    for _ in range(n):
        ii = rng.randint(0, N, size=N)
        yy = y[ii]
        if len(set(yy.tolist())) < 2:
            continue
        kept += 1
        a = {c: auc_safe(yy, S[c][ii]) for c in conds}
        for c in conds:
            draws[c].append(a[c])
        for c in diffs:
            diffs[c].append(a[c] - a["orig"])
    out = {"n_draws_used": kept}
    for c in conds:
        d = np.array([v for v in draws[c] if v == v])
        out[c] = {"auc": round(float(auc_safe(y, S[c])), 4),
                  "auc_lo": round(float(np.percentile(d, 2.5)), 4),
                  "auc_hi": round(float(np.percentile(d, 97.5)), 4),
                  "above_chance": bool(np.percentile(d, 2.5) > 0.5)}
    for c, dd in diffs.items():
        d = np.array([v for v in dd if v == v])
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[c]["diff_vs_orig"] = round(float(auc_safe(y, S[c]) - auc_safe(y, S["orig"])), 4)
        out[c]["diff_mean_boot"] = round(float(np.mean(d)), 4)
        out[c]["diff_lo"] = round(float(lo), 4)
        out[c]["diff_hi"] = round(float(hi), 4)
        out[c]["diff_significant"] = bool(lo > 0 or hi < 0)
        out[c]["p_two_sided_boot"] = round(float(2 * min((d >= 0).mean(), (d <= 0).mean())), 4)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{R}/manifests/dcaps.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--tag", default="abl_dcaps_omni_shuffle")
    ap.add_argument("--conditions", default="orig,shuf5s,shuf1s,rev300")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    conds = a.conditions.split(",")
    rows = [r for r in csv.DictReader(open(a.manifest))]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} clips, conditions {conds}", flush=True)
    print("PROMPT:", PROMPT, flush=True)

    proc, model, yes, no = load_model(a.model)
    print("model loaded", flush=True)
    cap = int(CAP_SEC * SR)

    out = []
    for i, r in enumerate(rows):
        try:
            y, _ = librosa.load(r["filepath"], sr=SR, mono=True)
            full_sec = len(y) / SR
            y = y[:cap].astype(np.float32)       # everything below sees the SAME audio
            rng = np.random.RandomState(1000 + i)
            rec = {"speaker": r["speaker_id"], "label": int(r["label"]),
                   "full_sec": round(full_sec, 1), "used_sec": round(len(y) / SR, 1)}
            for c in conds:
                s, m = score(proc, model, yes, no, make(c, y, rng))
                rec[f"p_{c}"] = s
                rec[f"mass_{c}"] = m
            out.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)}", flush=True)
        except Exception as e:
            print("FAIL", r["filepath"], repr(e), flush=True)

    if not out:
        print("NO OUTPUT")
        return

    os.makedirs(f"{R}/ablation", exist_ok=True)
    with open(f"{R}/ablation/{a.tag}_clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    y = np.array([o["label"] for o in out])
    S = {c: np.array([o[f"p_{c}"] for o in out]) for c in conds}
    ci = paired_bootstrap(y, S, conds, n=a.boot)

    summ = {"tag": a.tag, "model": a.model, "n": len(out),
            "speakers": len(set(o["speaker"] for o in out)),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "prompt": PROMPT, "cap_sec": CAP_SEC,
            "mean_full_sec": round(float(np.mean([o["full_sec"] for o in out])), 1),
            "mean_used_sec": round(float(np.mean([o["used_sec"] for o in out])), 1),
            "note": ("no text supplied in any condition; all conditions built from the SAME "
                     "first 300 s window, so they differ only in temporal ordering"),
            "bootstrap": ci}
    for c in conds:
        summ[f"meanp_{c}"] = round(float(np.mean(S[c])), 4)
        summ[f"mass_{c}"] = round(float(np.mean([o[f"mass_{c}"] for o in out])), 4)

    json.dump(summ, open(f"{R}/ablation/{a.tag}_summary.json", "w"), indent=2)
    print("=== SUMMARY", a.tag, "===", flush=True)
    print(json.dumps(summ, indent=2), flush=True)
    print("\n--- table ---", flush=True)
    for c in conds:
        d = ci[c]
        line = f"{c:8s} AUC {d['auc']:.3f}  95% CI [{d['auc_lo']:.3f}, {d['auc_hi']:.3f}]"
        if "diff_vs_orig" in d:
            sig = "SIGNIFICANT" if d["diff_significant"] else "n.s."
            line += (f"   vs orig {d['diff_vs_orig']:+.3f} "
                     f"[{d['diff_lo']:+.3f}, {d['diff_hi']:+.3f}] {sig}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
