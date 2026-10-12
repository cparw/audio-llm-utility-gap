"""AF3 on the paradox segments + dcaps whole interviews, both prompts, fixed window.
score() / prep_audio() copied verbatim from /scratch1/parwatka/score_af3_v2.py.
"""
import os, csv, json, glob, argparse
from collections import defaultdict
import numpy as np, torch, librosa, soundfile as sf

import llava
from llava.media import Sound
from llava import conversation as clib
from llava.mm_utils import process_sounds, process_sound_masks
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation

P_MAIN = ("Based only on this recording, does this speaker show signs of depression? "
          "Answer with one word, Yes or No.")
P_VOICE = ("Based only on how this person's voice sounds, does this speaker show signs of "
           "depression? Answer with one word, Yes or No.")


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
    return py / (py + pn + 1e-9), py + pn, word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--tasks", default="paradox_segment,interview")
    ap.add_argument("--out", default="/project2/msoleyma_946/speech_health/results_chaitanya/final")
    a = ap.parse_args()

    print("MODEL audio-flamingo-3 max_sec", a.max_sec, flush=True)
    print("P_MAIN :", P_MAIN, flush=True)
    print("P_VOICE:", P_VOICE, flush=True)

    S = glob.glob("/scratch1/parwatka/hf_cache/hub/models--nvidia--audio-flamingo-3/snapshots/*/")[0]
    model = llava.load(S, device_map=None).to("cuda").eval()
    clib.default_conversation = clib.conv_templates["auto"].copy()
    yes, no = yes_no_ids(model.tokenizer)
    print("yes_ids", yes, "no_ids", no, flush=True)

    keep = set(a.tasks.split(","))
    rows = [r for r in csv.DictReader(open(a.manifest)) if r["task_type"] in keep]
    print("clips:", len(rows), flush=True)

    tmpdir = os.environ.get("TMPDIR", "/scratch1/parwatka/tmp")
    os.makedirs(tmpdir, exist_ok=True)
    scratch = os.path.join(tmpdir, "af3_px_%d.wav" % os.getpid())

    out, nfail = [], 0
    for i, r in enumerate(rows):
        try:
            p_use, dur = prep_audio(r["filepath"], a.max_sec, scratch)
            pm, mm, wm = score(model, p_use, yes, no, P_MAIN)
            pv, mv, wv = score(model, p_use, yes, no, P_VOICE)
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
    cp = os.path.join(a.out, "px_scores_af3.csv")
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    json.dump(dict(model="audio-flamingo-3", short="af3", max_sec=a.max_sec,
                   p_main=P_MAIN, p_voice=P_VOICE, n=len(out), n_failed=nfail,
                   metric="P(Yes)/(P(Yes)+P(No)) at first answer position, greedy",
                   mean_mass_main=float(np.mean([o["mass_main"] for o in out])),
                   mean_mass_voice=float(np.mean([o["mass_voice"] for o in out])),
                   n_word_other_main=int(sum(o["word_main"] < 0 for o in out)),
                   n_word_other_voice=int(sum(o["word_voice"] < 0 for o in out))),
              open(os.path.join(a.out, "px_scores_af3_meta.json"), "w"), indent=2)
    print("wrote", cp, len(out), "rows, failed", nfail, flush=True)
    print("JOB_DONE", flush=True)


if __name__ == "__main__":
    main()
