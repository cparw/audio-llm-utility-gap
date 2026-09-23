"""PART20 POD4d: Kimi-Audio zero-shot p_yes, forward pass identical to <local data dir>/scratch/p15/kimi_probe.py
(same KimiAudio(load_detokenizer=False), same prompt_manager.get_prompt([text prompt, audio]), same alm.forward,
same librosa 16 kHz load + first-30 s cut written to a temp wav, softmax of text logits at the last position).
Hidden-state hooks dropped (zero-shot only; hooks do not touch logits).
p_yes columns:
  p_yes      : rule-3 set = bare + leading-space Yes/yes/YES and No/no/NO, single-token encodings only
  p_yes_ref  : exactly the reference kimi_probe.py set (["Yes"," Yes","yes"," yes","YES"] / No..., first token of each)
Checkpointed: rows appended to OUT.partial.csv, resumed by clip.
usage: kimi_zs.py MANIFEST(clip,path,speaker,label,set,orig_clip_id,fold) OUTCSV [MANIFEST2 OUTCSV2 ...] (model loaded once)"""
import os, sys, csv, glob, time, tempfile, hashlib
import numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, "/workspace/kimi_src")
from kimia_infer.api.kimia import KimiAudio
PAIRS = list(zip(sys.argv[1::2], sys.argv[2::2]))
P = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
CACHE = glob.glob("/workspace/hf/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*")[0]
t0 = time.time()
km = KimiAudio(model_path=CACHE, load_detokenizer=False)
alm = km.alm
print(f"Kimi loaded in {time.time()-t0:.0f}s from {CACHE} dtype {next(alm.parameters()).dtype}", flush=True)
def ids_for(words, single_only):
    s = set(); info = {}
    for w in words:
        t = km.prompt_manager._tokenize_text(w)
        info[w] = list(t) if t is not None else None
        if not t: continue
        if single_only and len(t) != 1: continue
        s.add(t[0])
    return sorted(s), info
YES_REF, iy = ids_for(["Yes", " Yes", "yes", " yes", "YES"], False)
NO_REF, inn = ids_for(["No", " No", "no", " no", "NO"], False)
YES, iy2 = ids_for(["Yes", " Yes", "yes", " yes", "YES", " YES"], True)
NO, in2 = ids_for(["No", " No", "no", " no", "NO", " NO"], True)
print("TOKENIZATION yes", iy2, "no", in2, flush=True)
print("ref ids yes", YES_REF, "no", NO_REF, "| rule3 ids yes", YES, "no", NO, flush=True)
TMP = tempfile.mkdtemp(prefix="kimi30_")
dev = torch.cuda.current_device()
cols = ["clip", "orig_clip_id", "speaker", "label", "set", "fold", "p_yes", "answer_mass", "p_yes_ref", "answer_mass_ref",
        "dur_file_s", "dur_heard_s", "wav_sha256", "prompt"]
for MAN, OUT in PAIRS:
    rows = list(csv.DictReader(open(MAN)))
    PART = OUT + ".partial.csv"
    done = set()
    if os.path.exists(PART):
        done = {r["clip"] for r in csv.DictReader(open(PART))}
        print(f"RESUME: {len(done)} clips already scored in {PART}", flush=True)
    cols = ["clip", "orig_clip_id", "speaker", "label", "set", "fold", "p_yes", "answer_mass", "p_yes_ref", "answer_mass_ref",
            "dur_file_s", "dur_heard_s", "wav_sha256", "prompt"]
    fh = open(PART, "a", newline=""); w = csv.writer(fh)
    if not done: w.writerow(cols); fh.flush()
    TMP = tempfile.mkdtemp(prefix="kimi30_")
    dev = torch.cuda.current_device()
    for n, r in enumerate(rows):
        if r["clip"] in done: continue
        x, _ = librosa.load(r["path"], sr=16000); dfile = len(x) / 16000; x = x[:16000 * 30]
        cut = os.path.join(TMP, "clip.wav"); sf.write(cut, x, 16000)
        chats = [{"role": "user", "message_type": "text", "content": P},
                 {"role": "user", "message_type": "audio", "content": cut}]
        history = km.prompt_manager.get_prompt(chats, output_type="text")
        aids, tids, cont_mask, _, _ = history.to_tensor()
        feat = history.continuous_feature
        aids = aids.to(dev); tids = tids.to(dev); cont_mask = cont_mask.to(dev)
        pos = torch.arange(0, aids.shape[1], device=dev).unsqueeze(0).long()
        with torch.no_grad():
            res = alm.forward(input_ids=aids, text_input_ids=tids, whisper_input_feature=feat,
                              is_continuous_mask=cont_mask, position_ids=pos, past_key_values=None)
            lg = res.logits if hasattr(res, "logits") else res
            text_logits = lg[1] if isinstance(lg, (tuple, list)) else lg
        pr = torch.softmax(text_logits[0, -1].float(), dim=-1).cpu().numpy()
        py, pn = float(pr[YES].sum()), float(pr[NO].sum())
        pyr, pnr = float(pr[YES_REF].sum()), float(pr[NO_REF].sum())
        h = hashlib.sha256(open(r["path"], "rb").read()).hexdigest()
        w.writerow([r["clip"], r["orig_clip_id"], r["speaker"], r["label"], r["set"], r["fold"],
                    repr(py / (py + pn)), repr(py + pn), repr(pyr / (pyr + pnr + 1e-12)), repr(pyr + pnr),
                    round(dfile, 4), round(len(x) / 16000, 4), h, P]); fh.flush()
        if n % 50 == 0 or n == len(rows) - 1:
            print(f"{n+1}/{len(rows)} {r['clip']} p_yes {py/(py+pn):.4f} mass {py+pn:.4f} {time.time()-t0:.0f}s", flush=True)
    fh.close()
    os.replace(PART, OUT)
    print(f"SCORED -> {OUT} {time.time()-t0:.0f}s", flush=True)
