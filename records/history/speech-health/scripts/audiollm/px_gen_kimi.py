"""Kimi-Audio on the paradox segments + dcaps whole interviews, both prompts, fixed window.
score() / prep_audio() copied verbatim from /scratch1/parwatka/score_kimi_v2.py.
"""
import os, csv, json, glob, argparse
import numpy as np, torch, librosa, soundfile as sf
from kimia_infer.api.kimia import KimiAudio

P_MAIN = ("Based only on this recording, does this speaker show signs of depression? "
          "Answer with one word, Yes or No.")
P_VOICE = ("Based only on how this person's voice sounds, does this speaker show signs of "
           "depression? Answer with one word, Yes or No.")


def yes_no_ids(tok):
    def ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                e = None
                for call in (lambda: tok.encode(v, bos=False, eos=False),
                             lambda: tok.encode(v),
                             lambda: tok(v, add_special_tokens=False).input_ids):
                    try:
                        e = call(); break
                    except Exception:
                        continue
                if e:
                    s.add(int(e[0]))
        return sorted(s)
    return ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])


def prep_audio(path, max_sec, scratch):
    try:
        dur = float(sf.info(path).duration)
    except Exception:
        dur = None
    if max_sec <= 0 or (dur is not None and dur <= max_sec):
        return path, dur
    y, _ = librosa.load(path, sr=16000, mono=True, duration=max_sec)
    sf.write(scratch, y, 16000)
    return scratch, len(y) / 16000.0


@torch.inference_mode()
def score(m, path, yes, no, prompt):
    chats = [{"role": "user", "message_type": "text", "content": prompt},
             {"role": "user", "message_type": "audio", "content": path}]
    history = m.prompt_manager.get_prompt(chats, output_type="text")
    audio_ids, text_ids, cont_mask, _, _ = history.to_tensor()
    feats = history.continuous_feature
    dev = torch.cuda.current_device()
    audio_ids = audio_ids.to(dev); text_ids = text_ids.to(dev); cont_mask = cont_mask.to(dev)
    feats = [f.to(dev) for f in feats]
    pos = torch.arange(0, audio_ids.shape[1], device=dev).unsqueeze(0).long()
    _, text_logits, _ = m.alm.forward(
        input_ids=audio_ids, text_input_ids=text_ids, whisper_input_feature=feats,
        is_continuous_mask=cont_mask, position_ids=pos, past_key_values=None, return_dict=False)
    lg = text_logits[0, -1].float() if text_logits.dim() == 3 else text_logits[0].float()
    p = torch.softmax(lg, dim=-1)
    py, pn = float(p[yes].sum()), float(p[no].sum())
    top1 = int(lg.argmax())
    word = 1 if top1 in set(yes) else (0 if top1 in set(no) else -1)
    return py / (py + pn + 1e-9), py + pn, word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--tasks", default="paradox_segment,interview")
    ap.add_argument("--out", default="/project2/msoleyma_946/speech_health/results_chaitanya/final")
    a = ap.parse_args()

    print("MODEL kimi-audio-7b-instruct max_sec", a.max_sec, flush=True)
    print("P_MAIN :", P_MAIN, flush=True)
    print("P_VOICE:", P_VOICE, flush=True)

    S = glob.glob("/scratch1/parwatka/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*/")[0]
    m = KimiAudio(model_path=S, load_detokenizer=False)
    print("KIMI LOAD OK", flush=True)
    yes, no = yes_no_ids(m.prompt_manager.text_tokenizer)
    print("yes_ids", yes, "no_ids", no, flush=True)

    keep = set(a.tasks.split(","))
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in keep]
    print("clips:", len(rows), flush=True)

    tmpdir = os.environ.get("TMPDIR", "/scratch1/parwatka/tmp")
    os.makedirs(tmpdir, exist_ok=True)
    scratch = os.path.join(tmpdir, "kimi_px_%d.wav" % os.getpid())

    out, nfail = [], 0
    for i, r in enumerate(rows):
        try:
            p_use, dur = prep_audio(r["filepath"], a.max_sec, scratch)
            pm, mm, wm = score(m, p_use, yes, no, P_MAIN)
            pv, mv, wv = score(m, p_use, yes, no, P_VOICE)
        except Exception as e:
            nfail += 1
            print("FAIL", r["filepath"], repr(e)[:160], flush=True)
            torch.cuda.empty_cache()
            continue
        out.append(dict(uid=r["uid"], filepath=r["filepath"], speaker_id=r["speaker_id"],
                        label=int(r["label"]), task_type=r["task_type"],
                        dur_used_s=dur, p_main=pm, mass_main=mm, word_main=wm,
                        p_voice=pv, mass_voice=mv, word_voice=wv))
        if i < 3 or (i + 1) % 50 == 0:
            print("  %d/%d %s p_main=%.4f p_voice=%.4f mass=%.3f"
                  % (i + 1, len(rows), r["uid"], pm, pv, mm), flush=True)

    assert out, "NO OUTPUT"
    os.makedirs(a.out, exist_ok=True)
    cp = os.path.join(a.out, "px_scores_kimi.csv")
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    json.dump(dict(model="kimi-audio-7b-instruct", short="kimi", max_sec=a.max_sec,
                   p_main=P_MAIN, p_voice=P_VOICE, n=len(out), n_failed=nfail,
                   metric="P(Yes)/(P(Yes)+P(No)) at first answer position, greedy",
                   mean_mass_main=float(np.mean([o["mass_main"] for o in out])),
                   mean_mass_voice=float(np.mean([o["mass_voice"] for o in out])),
                   n_word_other_main=int(sum(o["word_main"] < 0 for o in out)),
                   n_word_other_voice=int(sum(o["word_voice"] < 0 for o in out))),
              open(os.path.join(a.out, "px_scores_kimi_meta.json"), "w"), indent=2)
    print("wrote", cp, len(out), "rows, failed", nfail, flush=True)
    print("JOB_DONE", flush=True)


if __name__ == "__main__":
    main()
