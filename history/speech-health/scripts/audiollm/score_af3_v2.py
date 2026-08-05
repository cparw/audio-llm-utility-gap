"""AF3 scored with the SAME metric as the Qwen models, generalised to any
manifest / task / condition.

  p_yes = P(Yes)/(P(Yes)+P(No)) from the softmax over the full vocabulary at the
  first answer position; mass = P(Yes)+P(No).
  word  = argmax token at that position (this IS the greedy first token, since
          do_sample=False), mapped to yes / no / other.

--max_sec truncates the audio, matching what the Qwen feature extractor does
implicitly (WhisperFeatureExtractor pads/truncates to 30 s), so the numbers stay
comparable with the qwen2audio / omni rows.
"""
import os, csv, json, glob, argparse
from collections import defaultdict
import numpy as np, torch, librosa, soundfile as sf
from sklearn.metrics import roc_auc_score

import llava
from llava.media import Sound
from llava import conversation as clib
from llava.mm_utils import process_sounds, process_sound_masks
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation

DIAG = {"parkinsons": "does this speaker show signs of Parkinson's disease",
        "depression": "does this speaker show signs of depression"}


def yes_no_ids(tok):
    def ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                try:
                    e = tok(v, add_special_tokens=False).input_ids
                except TypeError:
                    e = tok.encode(v)
                if e:
                    s.add(e[0])
        return sorted(s)
    return ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])


def _to_half_list(x, n):
    if isinstance(x, torch.Tensor):
        return [t.half() for t in torch.unbind(x, dim=0)]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, torch.Tensor):
                out.extend([t.half() for t in torch.unbind(v, dim=0)])
            else:
                out.append(v)
        return out
    return [None] * n


def prep_audio(path, max_sec, scratch):
    """Return (path_to_use, duration_used). Untouched original when short enough."""
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
def score(model, path, yes, no, prompt):
    conversation = [{"from": "human", "value": [Sound(path), prompt]}]
    media, media_meta = extract_media(conversation, model.config)
    media_config = defaultdict(dict)
    batch_tensor, _ = process_sounds(media["sound"], inference=True)
    if batch_tensor is None or batch_tensor.shape[0] == 0:
        raise ValueError("no audio windows")
    N = batch_tensor.shape[0]
    media["sound"] = _to_half_list(batch_tensor, N)
    media_meta["sound_feature_masks"] = _to_half_list(
        process_sound_masks(media_meta.get("sound_feature_masks", [None] * N)), N)
    media_meta["sound_embed_masks"] = _to_half_list(
        process_sound_masks(media_meta.get("sound_embed_masks", [None] * N)), N)

    input_ids = tokenize_conversation(conversation, model.tokenizer,
                                      add_generation_prompt=True).cuda().unsqueeze(0)
    gc = model.default_generation_config
    gc.do_sample = False
    gc.max_new_tokens = 1
    gc.output_scores = True
    gc.return_dict_in_generate = True
    out = model.generate(input_ids=input_ids, media=media, media_config=media_config,
                         media_meta=media_meta, generation_config=gc)
    lg = out.scores[0][0].float()
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

    S = glob.glob("/scratch1/parwatka/hf_cache/hub/models--nvidia--audio-flamingo-3/snapshots/*/")[0]
    model = llava.load(S, device_map=None).to("cuda").eval()
    clib.default_conversation = clib.conv_templates["auto"].copy()
    yes, no = yes_no_ids(model.tokenizer)
    print("yes_ids", yes, "no_ids", no, flush=True)

    rows = [r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in set(a.tasks.split(","))]
    print("clips:", len(rows), flush=True)

    tmpdir = os.environ.get("TMPDIR", "/scratch1/parwatka/tmp")
    os.makedirs(tmpdir, exist_ok=True)
    scratch = os.path.join(tmpdir, "af3_trunc_%d.wav" % os.getpid())

    out = []
    nfail = 0
    for i, r in enumerate(rows):
        try:
            p_use, dur = prep_audio(r["filepath"], a.max_sec, scratch)
            pv, mass, word, top1 = score(model, p_use, yes, no, prompt)
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
    cp = os.path.join(a.out, "%s_af3_v2_perclip.csv" % a.tag)
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    v = np.array([o["p_yes"] for o in out]); m = np.array([o["answer_mass"] for o in out])
    lab = np.array([o["label"] for o in out]); wd = np.array([o["word_yes"] for o in out])
    ok = wd >= 0
    q = lambda x, p: float(np.percentile(x, p))
    summ = dict(tag=a.tag, model="audio-flamingo-3", cond=a.cond, tasks=a.tasks,
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
                answer_mass_median=float(np.median(m)), answer_mass_mean=float(m.mean()),
                answer_mass_min=float(m.min()),
                auc_prob=float(roc_auc_score(lab, v)) if len(set(lab)) > 1 else None,
                auc_word=(float(roc_auc_score(lab[ok], wd[ok]))
                          if ok.sum() and len(set(lab[ok].tolist())) > 1 else None))
    json.dump(summ, open(os.path.join(a.out, "%s_af3_v2_summary.json" % a.tag), "w"), indent=2)
    print("=== AF3 SUMMARY %s ===" % a.tag, json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()
