"""Score the paradox segments + dcaps whole interviews with a Qwen audio model.

Scoring function is copied verbatim from
/scratch1/parwatka/pd_probing/code/audio_ablation_bench.py so the p_yes metric is
identical to every earlier run:  p = P(Yes)/(P(Yes)+P(No)) at the first answer
position, greedy, max_new_tokens=1.

Both prompts are scored on every clip:
  main  = "Based only on this recording, ..."          (the 0.835-AUC dcaps wording)
  voice = "Based only on how this person's voice sounds, ..."  (the reliance wording)

--max_sec truncates the waveform BEFORE the feature extractor so that every model
in the comparison sees the same window.
"""
import os, csv, json, argparse
import numpy as np, torch, librosa

P_MAIN = ("Based only on this recording, does this speaker show signs of depression? "
          "Answer with one word, Yes or No.")
P_VOICE = ("Based only on how this person's voice sounds, does this speaker show signs of "
           "depression? Answer with one word, Yes or No.")
SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
       "perceiving auditory and visual inputs, as well as generating text and speech.")


def load_model(model_id):
    if "Omni" in model_id:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        proc = Qwen2_5OmniProcessor.from_pretrained(model_id)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda").eval()
        try:
            model.disable_talker()
            print("talker disabled", flush=True)
        except Exception as e:
            print("disable_talker skipped", repr(e)[:80], flush=True)
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
                if e:
                    s.add(e[0])
        return sorted(s)
    return proc, model, kind, wids(["Yes", "yes", "YES"]), wids(["No", "no", "NO"])


def score(proc, model, kind, yes, no, audio, prompt):
    if kind == "omni":
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
    top1 = int(lg.argmax())
    word = 1 if top1 in set(yes) else (0 if top1 in set(no) else -1)
    return py / (py + pn + 1e-9), py + pn, word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", required=True)          # qwen2audio | omni
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--tasks", default="paradox_segment,interview")
    ap.add_argument("--out", default="/project2/msoleyma_946/speech_health/results_chaitanya/final")
    a = ap.parse_args()

    print("MODEL", a.model, "max_sec", a.max_sec, flush=True)
    print("P_MAIN :", P_MAIN, flush=True)
    print("P_VOICE:", P_VOICE, flush=True)

    keep = set(a.tasks.split(","))
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in keep]
    print("clips:", len(rows), flush=True)

    proc, model, kind, yes, no = load_model(a.model)
    print("yes_ids", yes, "no_ids", no, flush=True)

    out, nfail = [], 0
    for i, r in enumerate(rows):
        try:
            if a.max_sec > 0:
                y, _ = librosa.load(r["filepath"], sr=16000, mono=True, duration=a.max_sec)
            else:
                y, _ = librosa.load(r["filepath"], sr=16000, mono=True)
            pm, mm, wm = score(proc, model, kind, yes, no, y, P_MAIN)
            pv, mv, wv = score(proc, model, kind, yes, no, y, P_VOICE)
        except Exception as e:
            nfail += 1
            print("FAIL", r["filepath"], repr(e)[:160], flush=True)
            torch.cuda.empty_cache()
            continue
        out.append(dict(uid=r["uid"], filepath=r["filepath"], speaker_id=r["speaker_id"],
                        label=int(r["label"]), task_type=r["task_type"],
                        dur_used_s=round(len(y) / 16000.0, 3),
                        p_main=pm, mass_main=mm, word_main=wm,
                        p_voice=pv, mass_voice=mv, word_voice=wv))
        if i < 3 or (i + 1) % 50 == 0:
            print("  %d/%d %s p_main=%.4f p_voice=%.4f mass=%.3f"
                  % (i + 1, len(rows), r["uid"], pm, pv, mm), flush=True)

    assert out, "NO OUTPUT"
    os.makedirs(a.out, exist_ok=True)
    cp = os.path.join(a.out, "px_scores_%s.csv" % a.short)
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    json.dump(dict(model=a.model, short=a.short, max_sec=a.max_sec,
                   p_main=P_MAIN, p_voice=P_VOICE, n=len(out), n_failed=nfail,
                   metric="P(Yes)/(P(Yes)+P(No)) at first answer position, greedy",
                   mean_mass_main=float(np.mean([o["mass_main"] for o in out])),
                   mean_mass_voice=float(np.mean([o["mass_voice"] for o in out])),
                   n_word_other_main=int(sum(o["word_main"] < 0 for o in out)),
                   n_word_other_voice=int(sum(o["word_voice"] < 0 for o in out))),
              open(os.path.join(a.out, "px_scores_%s_meta.json" % a.short), "w"), indent=2)
    print("wrote", cp, len(out), "rows, failed", nfail, flush=True)
    print("JOB_DONE", flush=True)


if __name__ == "__main__":
    main()
