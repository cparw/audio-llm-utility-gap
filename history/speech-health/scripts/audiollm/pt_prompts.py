"""
PART 3, INTERVENTION 1: prompt-level intervention on the DAIC conflict set.

The conflict/agreement set (195 matched pairs, 74 speakers, built by px_build.py)
is the place where the words and the clinical label disagree. Baseline finding:
Qwen2.5-Omni follows the words 71.3% of the time and the label 28.7%.

Here we re-score the identical 297 unique clips under a bank of prompt variants
that try, in increasingly explicit language, to push the model onto the acoustics.
No training. Output is one long CSV, scored downstream by pt_analyse.py.

Usage:  pt_prompts.py --model omni|qwen2audio [--limit N]
"""
import os, sys, json, argparse, warnings, time
import numpy as np, pandas as pd, torch, librosa
warnings.filterwarnings("ignore")

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
PX = f"{R}/committed/paradox"
OUT = f"{R}/final"
os.makedirs(OUT, exist_ok=True)

Q = "does this speaker show signs of depression"

# prompt bank. every one must elicit a single Yes/No token.
# flip=True means the question is polarity-inverted, so p_dep = 1 - p_yes.
PROMPTS = [
 ("P00_baseline", False,
  f"Based only on this recording, {Q}? Answer with one word, Yes or No."),
 ("P01_voice", False,
  f"Based only on how this person's voice sounds, {Q}? Answer with one word, Yes or No."),
 ("P02_ignore_words", False,
  "Ignore the meaning of the words completely. Judge only the sound of the voice. "
  f"{Q.capitalize()}? Answer with one word, Yes or No."),
 ("P03_words_misleading", False,
  "The words spoken in this recording may be misleading and must not be used. "
  "Base your judgement only on vocal acoustics. "
  f"{Q.capitalize()}? Answer with one word, Yes or No."),
 ("P04_acoustic_list", False,
  "Attend only to pitch variation, loudness, speaking rate, pauses and voice quality. "
  "Do not consider what is being said. "
  f"{Q.capitalize()}? Answer with one word, Yes or No."),
 ("P05_clinician", False,
  "You are a clinician assessing vocal biomarkers. Depressed speech is typically monotone, "
  "quiet, slow, and contains long pauses. Judge the voice, not the content. "
  f"{Q.capitalize()}? Answer with one word, Yes or No."),
 ("P06_foreign_language", False,
  "Assume you do not understand the language being spoken and can only hear how it sounds. "
  f"{Q.capitalize()}? Answer with one word, Yes or No."),
 ("P07_prosody_not_semantics", False,
  f"Based only on the prosody and vocal quality of this recording, and not on its semantic "
  f"content, {Q}? Answer with one word, Yes or No."),
 ("P08_symptom_acoustic", False,
  "Based only on how this person's voice sounds, does this speaker have flat affect, low "
  "energy, and a monotone or slowed voice? Answer with one word, Yes or No."),
 ("P09_polarity_flip", True,
  "Based only on how this person's voice sounds, does this speaker sound emotionally healthy "
  "and not depressed? Answer with one word, Yes or No."),
 ("P10_balanced_prior", False,
  f"Based only on this recording, {Q}? About half of the speakers in this study are "
  "depressed, so answer Yes about half the time. Answer with one word, Yes or No."),
 ("P11_ignore_words_prior", False,
  "Ignore the meaning of the words completely and judge only the sound of the voice. "
  f"{Q.capitalize()}? About half of the speakers in this study are depressed, so answer Yes "
  "about half the time. Answer with one word, Yes or No."),
]


def build_omni():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    MID = "Qwen/Qwen2.5-Omni-7B"
    SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
           "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
    proc = Qwen2_5OmniProcessor.from_pretrained(MID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16, device_map="cuda").eval()
    try:
        model.disable_talker(); print("talker disabled", flush=True)
    except Exception as e:
        print("disable_talker skipped", repr(e)[:80], flush=True)
    tok = proc.tokenizer
    YES, NO = wids(tok, ["Yes", "yes", "YES"]), wids(tok, ["No", "no", "NO"])
    print("yes_ids", YES, "no_ids", NO, flush=True)

    def score(audio, prompt):
        conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "audio", "audio": audio},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True,
                   use_audio_in_video=False)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=1, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, return_audio=False,
                               use_audio_in_video=False)
        p = torch.softmax(g.scores[0][0].float(), dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        return py / (py + pn + 1e-9), py + pn
    return score, MID


def build_qwen2audio():
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    MID = "Qwen/Qwen2-Audio-7B-Instruct"
    proc = AutoProcessor.from_pretrained(MID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16, device_map="cuda").eval()
    tok = proc.tokenizer
    YES, NO = wids(tok, ["Yes", "yes", "YES"]), wids(tok, ["No", "no", "NO"])
    print("yes_ids", YES, "no_ids", NO, flush=True)

    def score(audio, prompt):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[audio], sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        p = torch.softmax(lg, dim=-1)
        py, pn = float(p[YES].sum()), float(p[NO].sum())
        return py / (py + pn + 1e-9), py + pn
    return score, MID


def wids(tok, ws):
    s = set()
    for w in ws:
        for v in (w, " " + w):
            e = tok(v, add_special_tokens=False).input_ids
            if e: s.add(e[0])
    return sorted(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="omni", choices=["omni", "qwen2audio"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    M = pd.read_csv(f"{PX}/manifest.csv")
    if a.limit:
        M = M.head(a.limit)
    files = sorted(M.filepath.unique())
    print(f"rows {len(M)}  unique wavs {len(files)}  prompts {len(PROMPTS)}", flush=True)

    score, MID = (build_omni() if a.model == "omni" else build_qwen2audio())

    t0 = time.time()
    cache = {}          # filepath -> {prompt_id: (p_dep, mass)}
    for i, fp in enumerate(files):
        try:
            aud, _ = librosa.load(fp, sr=16000, mono=True)
        except Exception as e:
            print("LOADFAIL", fp, repr(e)[:100], flush=True); continue
        d = {}
        for pid, flip, ptxt in PROMPTS:
            try:
                py, mass = score(aud, ptxt)
                d[pid] = ((1.0 - py) if flip else py, mass)
            except Exception as e:
                print("SCOREFAIL", pid, fp, repr(e)[:120], flush=True)
                d[pid] = (float("nan"), float("nan"))
        cache[fp] = d
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  {el:.0f}s  eta {el/(i+1)*(len(files)-i-1):.0f}s", flush=True)

    rows = []
    for _, r in M.iterrows():
        if r.filepath not in cache: continue
        for pid, flip, ptxt in PROMPTS:
            p, mass = cache[r.filepath][pid]
            rows.append(dict(model=a.model, prompt_id=pid, seg_uid=r.seg_uid, set=r["set"],
                             pair_id=r.pair_id, speaker=r.speaker_id, label=r.label,
                             sev=r.sev, dur=r.clip_dur_s,
                             words_say_depressed=r.words_say_depressed,
                             p_dep=p, mass=mass))
    D = pd.DataFrame(rows)
    fo = f"{OUT}/prompt_intervention_{a.model}_clips.csv"
    D.to_csv(fo, index=False)
    json.dump({"model": MID, "n_rows": len(D), "n_wavs": len(cache),
               "prompts": {p[0]: p[2] for p in PROMPTS},
               "flip": [p[0] for p in PROMPTS if p[1]],
               "minutes": round((time.time() - t0) / 60, 1)},
              open(f"{OUT}/prompt_intervention_{a.model}_meta.json", "w"), indent=2)
    print("wrote", fo, len(D), "rows in %.1f min" % ((time.time() - t0) / 60), flush=True)

    # quick on-the-spot table so the log is readable even before the analysis job
    print("\n%-26s %8s %8s %8s %8s" % ("prompt", "meanP_c", "meanP_a", "AUC_conf", "AUC_agr"))
    for pid, flip, _ in PROMPTS:
        d = D[D.prompt_id == pid]
        c, g = d[d.set == "conflict"], d[d.set == "agreement"]
        def auc(x):
            from sklearn.metrics import roc_auc_score
            try: return roc_auc_score(x.label, x.p_dep)
            except Exception: return float("nan")
        print("%-26s %8.3f %8.3f %8.3f %8.3f" %
              (pid, c.p_dep.mean(), g.p_dep.mean(), auc(c), auc(g)))
    print("JOB_DONE", flush=True)


if __name__ == "__main__":
    main()
