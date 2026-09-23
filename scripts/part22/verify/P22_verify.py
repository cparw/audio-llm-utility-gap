#!/usr/bin/env python3
"""PART 22 independent verifier. Written from scratch; imports nothing from the pod job's scripts.

subcommands
  build   PID ...                 rebuild windows myself (stitch rule from windows_full.csv notes) -> /workspace/verify22/cut/
  tokens  PID ...                 processor token counts: default, 300 s cut, truncation off (several ways), input equality
  score   --mode M --pids ... --out CSV   model p_yes, M in {default, cut300, lift}
  auc     --scores CSV [--base CSV] --out JSON   AUC (pairs + trapezoid) and 2000-draw speaker bootstrap
  step1   MD                      check quoted code lines against the installed transformers source

conventions
  p_yes = P(Yes)/(P(Yes)+P(No)) at the first answer position (last input position), softmax in float32
  YES/NO id sets: RULE set = bare and leading-space Yes/yes/YES, No/no/NO, only strings that encode to ONE token
                  PAPER set = first token of Yes, ' Yes', yes, ' yes', YES / No, ' No', no, ' no', NO (as extract_full.py)
  bf16, sdpa default attention, Qwen/Qwen2.5-Omni-7B
"""
import os, sys, csv, json, time, hashlib, argparse
import numpy as np

V = "/workspace/verify22"
SRC = "/workspace/edaicfull"
MID = "Qwen/Qwen2.5-Omni-7B"
SR = 16000
PROMPT = "Based only on how this person's voice sounds, does this speaker show signs of depression? Answer with one word, Yes or No."


def sha1mb(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(1 << 20)).hexdigest()


def win_table():
    return {r["pid"]: r for r in csv.DictReader(open(f"{SRC}/windows_full.csv"))}


# ---------------------------------------------------------------- build
def my_window(pid, W):
    import soundfile as sf
    audio, sr = sf.read(f"{SRC}/wav/{pid}_AUDIO.wav", dtype="float32", always_2d=True)
    pieces = []
    with open(f"{SRC}/tr/{pid}_Transcript.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                a = float(row["Start_Time"]); b = float(row["End_Time"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (b > a):
                continue
            pieces.append(audio[int(a * sr):int(b * sr)])
    stitched = np.concatenate(pieces, axis=0) if pieces else audio[:0]
    r = W[pid]
    seg = stitched[int(float(r["start_s"]) * sr):int(float(r["end_s"]) * sr)]
    mono = seg[:, 0] if seg.shape[1] == 1 else seg.mean(axis=1)
    if sr != SR:
        import librosa
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    return mono, sr, len(stitched) / sr


def cmd_build(pids):
    import soundfile as sf
    os.makedirs(f"{V}/cut", exist_ok=True)
    W = win_table()
    out = []
    for pid in pids:
        m, sr, st_s = my_window(pid, W)
        p = f"{V}/cut/{pid}.wav"
        sf.write(p, m, SR, subtype="PCM_16")
        rec = dict(pid=pid, native_sr=sr, stitched_s=round(st_s, 4), src_dur_s=float(W[pid]["src_dur_s"]),
                   win_s=round(len(m) / SR, 4), win_dur_s=float(W[pid]["win_dur_s"]), n_samples=len(m))
        podcut = f"{SRC}/cut/{pid}.wav"
        if os.path.exists(podcut):
            a, _ = sf.read(podcut, dtype="int16"); b, _ = sf.read(p, dtype="int16")
            rec["pod_file_same_len"] = bool(len(a) == len(b))
            rec["pod_file_identical_int16"] = bool(len(a) == len(b) and np.array_equal(a, b))
            rec["pod_sha256_1MB"] = sha1mb(podcut)
        rec["mine_sha256_1MB"] = sha1mb(p)
        out.append(rec)
        print(json.dumps(rec), flush=True)
    return out


# ---------------------------------------------------------------- processor
def load_proc():
    from transformers import Qwen2_5OmniProcessor
    return Qwen2_5OmniProcessor.from_pretrained(MID)


def conv_text(proc, x):
    conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": PROMPT}]}]
    return proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def run_proc(proc, x, how):
    text = conv_text(proc, x)
    base = dict(text=text, audio=[x], sampling_rate=SR, return_tensors="pt", padding=True)
    if how == "default":
        return proc(**base)
    if how == "cut300":
        return proc(**dict(base, audio=[x[:300 * SR]]))
    if how == "trunc_off_toplevel":
        return proc(**base, truncation=False)
    if how == "trunc_off_audio_kwargs":
        return proc(**base, audio_kwargs={"truncation": False})
    if how == "maxlen900_audio_kwargs":
        return proc(**base, audio_kwargs={"max_length": 900 * SR})
    if how == "maxlen900_toplevel":
        return proc(**base, max_length=900 * SR)
    raise ValueError(how)


def expected_tokens(n_samples, max_len=300 * SR):
    # mask frames: a clip padded up to max_len keeps ceil(n/160) frames; an unpadded clip keeps n//160 (last frame dropped)
    F = -(-n_samples // 160) if n_samples < max_len else n_samples // 160
    return ((F - 1) // 2 + 1 - 2) // 2 + 1, F


def audio_id(proc):
    return proc.tokenizer.convert_tokens_to_ids("<|AUDIO|>")


def cmd_tokens(pids, wavdir):
    import librosa, torch
    proc = load_proc()
    AUD = audio_id(proc)
    fe = proc.feature_extractor
    res = dict(feature_extractor_class=type(fe).__name__, chunk_length=getattr(fe, "chunk_length", None),
               sampling_rate=fe.sampling_rate, n_samples=getattr(fe, "n_samples", None),
               nb_max_frames=getattr(fe, "nb_max_frames", None), audio_token_id=AUD, clips=[])
    print(json.dumps({k: v for k, v in res.items() if k != "clips"}), flush=True)
    for pid in pids:
        path = f"{wavdir}/{pid}.wav"
        x, _ = librosa.load(path, sr=SR)
        c = dict(pid=pid, file=path, sha256_1MB=sha1mb(path), dur_s=len(x) / SR, n_samples=len(x),
                 expected_full_tokens=expected_tokens(len(x))[0],
                 expected_300_tokens=expected_tokens(min(len(x), 300 * SR))[0])
        keep = {}
        for how in ["default", "cut300", "trunc_off_toplevel", "trunc_off_audio_kwargs",
                    "maxlen900_audio_kwargs", "maxlen900_toplevel"]:
            try:
                inp = run_proc(proc, x, how)
                ids = inp["input_ids"][0]
                fam = inp["feature_attention_mask"][0]
                c[how] = dict(n_audio_tokens=int((ids == AUD).sum()), seq_len=int(ids.shape[0]),
                              mask_frames=int(fam.sum()), feat_shape=list(inp["input_features"].shape))
                keep[how] = inp
            except Exception as e:
                c[how] = dict(error=repr(e)[:400])
        # exact input identity default vs cut300 (valid frames only, and full tensors)
        if "default" in keep and "cut300" in keep:
            a, b = keep["default"], keep["cut300"]
            fa = a["input_features"][0][:, a["feature_attention_mask"][0].bool()].float().numpy()
            fb = b["input_features"][0][:, b["feature_attention_mask"][0].bool()].float().numpy()
            c["default_vs_cut300"] = dict(
                input_ids_equal=bool(torch.equal(a["input_ids"], b["input_ids"])),
                mask_equal=bool(torch.equal(a["feature_attention_mask"], b["feature_attention_mask"])),
                features_full_tensor_equal=bool(a["input_features"].shape == b["input_features"].shape and
                                                torch.equal(a["input_features"], b["input_features"])),
                valid_features_equal=bool(fa.shape == fb.shape and np.array_equal(fa, fb)),
                valid_features_maxabs=float(np.abs(fa - fb).max()) if fa.shape == fb.shape else None)
        # lifted valid features: first 300 s identical to default?  (normalisation max can differ)
        for how in ["trunc_off_audio_kwargs", "trunc_off_toplevel", "maxlen900_audio_kwargs"]:
            if how in keep and "default" in keep:
                a, b = keep["default"], keep[how]
                fa = a["input_features"][0][:, a["feature_attention_mask"][0].bool()].float().numpy()
                fb = b["input_features"][0][:, b["feature_attention_mask"][0].bool()].float().numpy()
                n = min(fa.shape[1], fb.shape[1])
                c[f"default_vs_{how}_first_frames"] = dict(
                    n_frames_compared=n, equal=bool(np.array_equal(fa[:, :n], fb[:, :n])),
                    maxabs=float(np.abs(fa[:, :n] - fb[:, :n]).max()))
        res["clips"].append(c)
        print(json.dumps(c), flush=True)
    return res


# ---------------------------------------------------------------- model
def id_sets(tok):
    rule_yes, rule_no, multi = [], [], []
    for words, dst in [(["Yes", " Yes", "yes", " yes", "YES", " YES"], rule_yes),
                       (["No", " No", "no", " no", "NO", " NO"], rule_no)]:
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                dst.append(t[0])
            else:
                multi.append((w, t))
    paper_yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes", "yes", " yes", "YES"]})
    paper_no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No", "no", " no", "NO"]})
    return sorted(set(rule_yes)), sorted(set(rule_no)), paper_yes, paper_no, multi


def load_model():
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration
    m = Qwen2_5OmniForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16)
    th = m.thinker
    for a in ("talker", "token2wav"):
        if hasattr(m, a):
            delattr(m, a)
    if hasattr(th, "visual"):
        del th.visual
    return th.to("cuda").eval()


def cmd_score(mode, pids, wavdir, out):
    import librosa, torch
    proc = load_proc()
    AUD = audio_id(proc)
    RY, RN, PY, PN, multi = id_sets(proc.tokenizer)
    print(json.dumps(dict(rule_yes=RY, rule_no=RN, paper_yes=PY, paper_no=PN, multi_token=multi)), flush=True)
    net = load_model()
    done = set()
    if os.path.exists(out):
        done = {r["pid"] for r in csv.DictReader(open(out))}
    else:
        with open(out, "w", newline="") as fh:
            csv.writer(fh).writerow(["pid", "mode", "p_yes_rule", "mass_rule", "p_yes_paper", "mass_paper",
                                     "n_audio_tok", "seq_len", "dur_in_s", "sec", "wav_sha256_1MB", "error"])
    how = {"default": "default", "cut300": "cut300", "lift": "trunc_off_audio_kwargs"}[mode]
    for pid in pids:
        if pid in done:
            continue
        t0 = time.time()
        path = f"{wavdir}/{pid}.wav"
        x, _ = librosa.load(path, sr=SR)
        err = ""
        try:
            inp = run_proc(proc, x, how)
            n_aud = int((inp["input_ids"][0] == AUD).sum()); seq = int(inp["input_ids"].shape[1])
            feed = {}
            for k, v in inp.items():
                if not torch.is_tensor(v):
                    continue
                v = v.to("cuda")
                if v.dtype == torch.float32:
                    v = v.to(torch.bfloat16)
                feed[k] = v
            with torch.no_grad():
                lo = net(**feed).logits[0, -1].float()
            pr = torch.softmax(lo, dim=-1)
            ry, rn = float(pr[RY].sum()), float(pr[RN].sum())
            py, pn = float(pr[PY].sum()), float(pr[PN].sum())
            row = [pid, mode, ry / (ry + rn), ry + rn, py / (py + pn), py + pn, n_aud, seq,
                   min(len(x), 300 * SR) / SR if mode != "lift" else len(x) / SR, round(time.time() - t0, 2),
                   sha1mb(path), ""]
        except Exception as e:
            err = repr(e)[:300]
            row = [pid, mode, "", "", "", "", "", "", "", round(time.time() - t0, 2), sha1mb(path), err]
        with open(out, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        print(" ".join(map(str, row[:10])), err, flush=True)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- AUC
def auc_pairs(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def auc_trapz(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P, N = (y == 1).sum(), (y == 0).sum()
    tp, fp = [0.0], [0.0]
    for t in np.unique(s)[::-1]:
        tp.append(((s >= t) & (y == 1)).sum() / P); fp.append(((s >= t) & (y == 0)).sum() / N)
    tp, fp = np.array(tp), np.array(fp)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2))


def boot(spk, fns, draws=2000):
    spk = np.asarray(spk).astype(str)
    uniq = np.unique(spk)
    groups = [np.flatnonzero(spk == u) for u in uniq]
    rng = np.random.default_rng(0)
    st = {k: [] for k in fns}; bad = 0
    for _ in range(draws):
        d = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([groups[i] for i in d])
        v = {k: f(rows) for k, f in fns.items()}
        if any(np.isnan(z) for z in v.values()):
            bad += 1; continue
        for k, z in v.items():
            st[k].append(z)
    return {k: np.array(v) for k, v in st.items()}, len(uniq), bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("pids", nargs="*")
    ap.add_argument("--mode", default="default")
    ap.add_argument("--wavdir", default=f"{V}/cut")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pidfile", default=None)
    a = ap.parse_args()
    pids = a.pids
    if a.pidfile:
        pids = [l.strip() for l in open(a.pidfile) if l.strip()]
    if a.cmd == "build":
        r = cmd_build(pids)
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1)
    elif a.cmd == "tokens":
        r = cmd_tokens(pids, a.wavdir)
        if a.out:
            json.dump(r, open(a.out, "w"), indent=1)
    elif a.cmd == "score":
        cmd_score(a.mode, pids, a.wavdir, a.out)
    else:
        raise SystemExit("unknown cmd")
