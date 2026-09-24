"""Kimi-Audio scored with the SAME metric as the Qwen models, generalised to any
manifest / task / condition.  See score_af3_v2.py for the metric definition.
"""
import os, csv, json, glob, argparse
import numpy as np, torch, librosa, soundfile as sf
from sklearn.metrics import roc_auc_score
from kimia_infer.api.kimia import KimiAudio

DIAG = {"parkinsons": "does this speaker show signs of Parkinson's disease",
        "depression": "does this speaker show signs of depression"}


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
        info = sf.info(path)
        dur = float(info.duration)
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
    return py / (py + pn + 1e-9), py + pn, word, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tasks", default="read")
    ap.add_argument("--cond", default="parkinsons")
    ap.add_argument("--tag", default="pcgita_read")
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--out", default="/scratch1/parwatka/scores")
    a = ap.parse_args()

    prompt = ("Based only on how this person's voice sounds, %s? "
              "Answer with one word, Yes or No." % DIAG[a.cond])
    print("PROMPT:", prompt, flush=True)
    print("max_sec:", a.max_sec, flush=True)

    S = glob.glob("/scratch1/parwatka/hf_cache/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*/")[0]
    m = KimiAudio(model_path=S, load_detokenizer=False)
    print("KIMI LOAD OK", flush=True)
    yes, no = yes_no_ids(m.prompt_manager.text_tokenizer)
    print("yes_ids", yes, "no_ids", no, flush=True)

    rows = [r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in set(a.tasks.split(","))]
    print("clips:", len(rows), flush=True)

    tmpdir = os.environ.get("TMPDIR", "/scratch1/parwatka/tmp")
    os.makedirs(tmpdir, exist_ok=True)
    scratch = os.path.join(tmpdir, "kimi_trunc_%d.wav" % os.getpid())

    out = []
    nfail = 0
    for i, r in enumerate(rows):
        try:
            p_use, dur = prep_audio(r["filepath"], a.max_sec, scratch)
            pv, mass, word, top1 = score(m, p_use, yes, no, prompt)
        except Exception as e:
            nfail += 1
            print("FAIL", r["filepath"], repr(e)[:120], flush=True)
            torch.cuda.empty_cache()
            continue
        out.append(dict(filepath=r["filepath"], speaker_id=r["speaker_id"],
                        label=int(r["label"]), task_type=r["task_type"],
                        p_yes=pv, answer_mass=mass, word_yes=word, top1_tok=top1,
                        dur_used_s=dur))
        if i < 3 or (i + 1) % 100 == 0:
            print(" ", i + 1, r["speaker_id"], "label", r["label"],
                  "p_yes=%.4f mass=%.3f word=%d" % (pv, mass, word), flush=True)

    if not out:
        print("NO OUTPUT", flush=True); return

    os.makedirs(a.out, exist_ok=True)
    cp = os.path.join(a.out, "%s_kimi_v2_perclip.csv" % a.tag)
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    v = np.array([o["p_yes"] for o in out]); mm = np.array([o["answer_mass"] for o in out])
    lab = np.array([o["label"] for o in out]); wd = np.array([o["word_yes"] for o in out])
    ok = wd >= 0
    q = lambda x, p: float(np.percentile(x, p))
    summ = dict(tag=a.tag, model="kimi-audio-7b-instruct", cond=a.cond, tasks=a.tasks,
                max_sec=a.max_sec, prompt=prompt,
                metric="p_yes = P(Yes)/(P(Yes)+P(No)) at first answer position",
                n_clips=len(out), n_failed=nfail,
                n_speakers=len(set(o["speaker_id"] for o in out)),
                n_pos=int((lab == 1).sum()), n_neg=int((lab == 0).sum()),
                p_yes_min=float(v.min()), p_yes_q1=q(v, 25), p_yes_median=float(np.median(v)),
                p_yes_q3=q(v, 75), p_yes_max=float(v.max()), p_yes_mean=float(v.mean()),
                frac_yes_prob=float((v > 0.5).mean()),
                frac_yes_word=float((wd[ok] == 1).mean()) if ok.sum() else None,
                n_word_parsed=int(ok.sum()), n_word_other=int((~ok).sum()),
                answer_mass_median=float(np.median(mm)), answer_mass_mean=float(mm.mean()),
                answer_mass_min=float(mm.min()),
                auc_prob=float(roc_auc_score(lab, v)) if len(set(lab)) > 1 else None,
                auc_word=(float(roc_auc_score(lab[ok], wd[ok]))
                          if ok.sum() and len(set(lab[ok].tolist())) > 1 else None))
    json.dump(summ, open(os.path.join(a.out, "%s_kimi_v2_summary.json" % a.tag), "w"), indent=2)
    print("=== KIMI SUMMARY %s ===" % a.tag, json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
