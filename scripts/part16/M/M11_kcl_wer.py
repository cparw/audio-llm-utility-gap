#!/usr/bin/env python3
"""
PART16 / M11 - Does transcription quality explain the MDVR-KCL transcript result?

The MDVR-KCL read-passage REFERENCE TEXT is not on disk, so true WER cannot be
computed (see sidecar for the search and for a second reason it is ill-posed).
The instructed fallback - score Whisper large-v3 against the transcript the
paper's text condition actually used - turns out to be DEGENERATE: large-v3
reproduces that transcript exactly on all 37 clips (WER 0.0000 everywhere),
because that transcript IS large-v3 greedy output. That is reported as finding A.

Findings B-E are reference-free ASR-quality measures that CAN separate the groups:
  B  cross-system disagreement, Whisper large-v3 vs Whisper medium
  C  Whisper large-v3's own confidence: mean per-token logprob, and the
     zlib compression ratio (its standard repetition/degeneracy detector)
  D  words recovered per fixed 30 s window
  E  which passage the speaker was reading (the actual confound)
Each is reported by group with a 2000-draw speaker bootstrap, a paired
between-group difference, and Spearman against the transcript and audio p_yes.
"""
import os, sys, json, time, hashlib, re, zlib

HF_GDRIVE = "<local data dir>/DementiaBank/hf_cache/hub"
HF_LOCAL  = os.path.expanduser("~/.cache/huggingface/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, scipy.stats as st
import soundfile as sf

OUT   = "<local data dir>/Desktop/release/edaic_rerun/part16"
ROWS  = OUT + "/rows/M11.tsv"
CSV   = OUT + "/M11_kcl_wer.csv"
SIDE  = OUT + "/M11_kcl_wer.json"
DISC  = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"

CLIPDIR   = "<local data dir>/paper1_local_runs/kcl_read30b"
MANIFEST  = "<local data dir>/paper1_local_runs/kcl30b_manifest.csv"
REF_POD   = "<local data dir>/Desktop/release/part4_text/asr/asr_kcl.csv"
REF_LOCAL = "<local data dir>/paper1_local_runs/kcl_read30b_text.csv"
P_TEXT    = "<local data dir>/Desktop/release/podE_final/out/o25_kcl_text.csv"
P_AUDIO   = "<local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv"
SRCDIR    = "<local data dir>/KCL/extracted/26-29_09_2017_KCL/ReadText"
OFFSETS   = "scripts/audit_1a1c/kcl_offsets.json"
BIG, SMALL = "openai/whisper-large-v3", "openai/whisper-medium"
SEED, NBOOT = 0, 2000
CMD = "cd <local data dir>/Desktop/release/edaic_rerun/part16 && /usr/local/bin/python3 M11_kcl_wer.py"

def sha1mb(p):
    try:
        with open(p, "rb") as f: return hashlib.sha256(f.read(1024*1024)).hexdigest()
    except Exception as e: return "ERR:" + repr(e)[:60]

def disc(sev, msg):
    with open(DISC, "a") as f: f.write(f"\n- [{sev}] PART16 M11: {msg}\n")
    print(f"[DISCREPANCY {sev}] {msg}", flush=True)

def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))

# ---------------- normaliser (identical on both sides) ----------------
_NS = re.compile(r"[^a-z0-9 ]+"); _RU = re.compile(r"\d+|[a-z]+")
def make_norm(spelling):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    base = EnglishTextNormalizer(spelling)
    def norm(t):
        t = base("" if t is None or (isinstance(t, float) and pd.isna(t)) else str(t))
        return _RU.findall(_NS.sub(" ", t.lower()))
    return norm

def align_counts(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n+1, m+1), np.int32); D[:, 0] = np.arange(n+1); D[0, :] = np.arange(m+1)
    bt = np.zeros((n+1, m+1), np.int8); bt[1:, 0] = 1; bt[0, 1:] = 2
    for i in range(1, n+1):
        ri = ref[i-1]
        for j in range(1, m+1):
            sub = D[i-1, j-1] + (0 if ri == hyp[j-1] else 1)
            de = D[i-1, j] + 1; ins = D[i, j-1] + 1
            best, b = sub, 0
            if de < best: best, b = de, 1
            if ins < best: best, b = ins, 2
            D[i, j] = best; bt[i, j] = b
    S = Dl = I = 0; i, j = n, m
    while i > 0 or j > 0:
        b = bt[i, j]
        if b == 0:
            if ref[i-1] != hyp[j-1]: S += 1
            i -= 1; j -= 1
        elif b == 1: Dl += 1; i -= 1
        else: I += 1; j -= 1
    return S, Dl, I, n

# ---------------- bootstrap ----------------
def _cells(spks, rng): return rng.choice(spks, size=len(spks), replace=True)

def boot_mean(vals, spk, seed=SEED):
    rng = np.random.default_rng(seed)
    spks = np.asarray(sorted(set(spk))); vals = np.asarray(vals, float); spk = np.asarray(spk)
    by = {s: vals[spk == s] for s in spks}; out = []
    for _ in range(NBOOT):
        v = np.concatenate([by[s] for s in _cells(spks, rng)])
        if len(v): out.append(v.mean())
    if not out: return float("nan"), float("nan"), 0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)

def boot_paired_diff(vals, spk, grp, seed=SEED):
    rng = np.random.default_rng(seed)
    spks = np.asarray(sorted(set(spk)))
    vals = np.asarray(vals, float); spk = np.asarray(spk); grp = np.asarray(grp)
    bv = {s: vals[spk == s] for s in spks}; bg = {s: grp[spk == s][0] for s in spks}
    d = []
    for _ in range(NBOOT):
        pick = _cells(spks, rng)
        pv = np.concatenate([bv[s] for s in pick])
        pg = np.concatenate([np.repeat(bg[s], len(bv[s])) for s in pick])
        if (pg == "PD").sum() == 0 or (pg == "control").sum() == 0: continue
        d.append(pv[pg == "PD"].mean() - pv[pg == "control"].mean())
    if not d: return float("nan"), float("nan"), 0
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(d)

def boot_spearman(x, y, spk, seed=SEED):
    rng = np.random.default_rng(seed)
    spks = np.asarray(sorted(set(spk)))
    x = np.asarray(x, float); y = np.asarray(y, float); spk = np.asarray(spk)
    bx = {s: x[spk == s] for s in spks}; by = {s: y[spk == s] for s in spks}
    out = []
    for _ in range(NBOOT):
        pick = _cells(spks, rng)
        xx = np.concatenate([bx[s] for s in pick]); yy = np.concatenate([by[s] for s in pick])
        if len(np.unique(xx)) < 2 or len(np.unique(yy)) < 2: continue
        out.append(float(st.spearmanr(xx, yy).statistic))
    if not out: return float("nan"), float("nan"), 0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)

# ---------------- transcription ----------------
def transcribe(model_id, cache, clips, want_scores):
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(model_id, cache_dir=cache)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=cache,
                                                            dtype=torch.float32).to(dev).eval()
    DEC = dict(language="en", task="transcribe", do_sample=False, num_beams=1,
               temperature=0.0, max_new_tokens=440, condition_on_prev_tokens=False)
    texts, logp = {}, {}
    t0 = time.time()
    for i, c in enumerate(clips):
        x, sr = sf.read(os.path.join(CLIPDIR, c), dtype="float32")
        assert sr == 16000
        f = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float32)
        with torch.no_grad():
            o = model.generate(f, output_scores=want_scores, return_dict_in_generate=True, **DEC)
        ids = o.sequences
        texts[c] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if want_scores:
            sc = getattr(o, "scores", None) or getattr(o, "logits", None)
            if not sc:
                logp[c] = float("nan"); continue
            gen = ids[0, ids.shape[1]-len(sc):]
            lp = [float(torch.log_softmax(q[0].float(), -1)[t]) for q, t in zip(sc, gen)]
            logp[c] = float(np.mean(lp)) if lp else float("nan")
        if (i+1) % 10 == 0 or i == len(clips)-1:
            print(f"  [{model_id}] {i+1}/{len(clips)}  {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    return texts, logp, dev, (time.time()-t0)/60, DEC

NW = ["north wind", "traveller", "traveler", "cloak", "stronger"]
RB = ["scattering", "blue light", "atmospheric", "sunlight", "observer", "electromagnetic", "spectrum"]
def passage_of(t):
    t = re.sub(r"[^a-z ]", " ", (t or "").lower())
    a = sum(1 for w in NW if w in t); b = sum(1 for w in RB if w in t)
    if a == 0 and b == 0: return "neither"
    return "north_wind" if a > b else ("light_scattering" if b > a else "ambiguous")

# ================================================================= main
def main():
    t_all = time.time()
    srcs = [CLIPDIR, MANIFEST, REF_POD, REF_LOCAL, P_TEXT, P_AUDIO, OFFSETS]
    for p in srcs: print(("OK  " if os.path.exists(p) else "MISS") + "  " + p, flush=True)

    man = pd.read_csv(MANIFEST)
    man["clip"] = man["path"].map(os.path.basename)
    man["group"] = np.where(man["label"].astype(int) == 1, "PD", "control")
    clips = sorted(man["clip"].tolist())
    print(f"\nAUDIO ROOT (scored clips): {CLIPDIR}")
    print(f"SOURCE DATASET ROOT:       {SRCDIR}")
    print(f"n files = {len(man)}   labels: " +
          ", ".join(f"{g}={k}" for g, k in man.groupby('group').size().items()), flush=True)

    ref_pod = pd.read_csv(REF_POD); ref_loc = pd.read_csv(REF_LOCAL)
    ptext = pd.read_csv(P_TEXT); paudio = pd.read_csv(P_AUDIO)
    offs = json.load(open(OFFSETS))

    CACHE = OUT + "/M11_asr_cache.json"
    if os.path.exists(CACHE):
        cc = json.load(open(CACHE))
        big_txt, big_lp, sml_txt = cc["big_txt"], cc["big_lp"], cc["sml_txt"]
        dev, min_big, min_sml, DEC = cc["dev"], cc["min_big"], cc["min_sml"], cc["DEC"]
        print(f"\nreusing cached ASR from {CACHE}", flush=True)
    else:
        print(f"\n--- transcribing with {BIG} ---", flush=True)
        big_txt, big_lp, dev, min_big, DEC = transcribe(BIG, HF_GDRIVE, clips, True)
        print(f"--- transcribing with {SMALL} (cross-system control) ---", flush=True)
        sml_txt, _, _, min_sml, _ = transcribe(SMALL, HF_LOCAL, clips, False)
        json.dump(dict(big_txt=big_txt, big_lp=big_lp, sml_txt=sml_txt, dev=dev,
                       min_big=min_big, min_sml=min_sml, DEC=DEC), open(CACHE, "w"), indent=1)
    print(f"ASR runtime: large-v3 {min_big:.1f} min, medium {min_sml:.1f} min, device {dev}", flush=True)

    from transformers import WhisperTokenizerFast
    spelling = WhisperTokenizerFast.from_pretrained(BIG, cache_dir=HF_GDRIVE).english_spelling_normalizer
    norm = make_norm(spelling)

    pod = dict(zip(ref_pod["speaker_id"], ref_pod["text"])); loc = dict(zip(ref_loc["speaker_id"], ref_loc["text"]))
    pt = dict(zip(ptext["id"], ptext["p_yes"])); pa = dict(zip(paudio["clip"], paudio["p_yes"]))

    rows = []
    for _, r in man.iterrows():
        c, spk = r["clip"], r["speaker"]
        h = norm(big_txt[c]); m = norm(sml_txt[c])
        S, D_, I, N = align_counts(norm(pod.get(spk, "")), h)                 # A: vs paper transcript
        S2, D2, I2, N2 = align_counts(h, m)                                   # B: large-v3 vs medium
        Sl, Dl, Il, Nl = align_counts(norm(loc.get(spk, "")), h)              # 2nd stored copy
        txt = big_txt[c]
        comp = len(txt.encode()) / max(len(zlib.compress(txt.encode())), 1)
        rows.append(dict(
            file=c, spk=spk, group=r["group"], label=int(r["label"]),
            n_ref_words=N, S=S, D=D_, I=I, wer=(S+D_+I)/N if N else np.nan,
            p_yes_text=pt.get(c, np.nan), p_yes_audio=pa.get(c, np.nan),
            xsys_n_ref_words=N2, xsys_S=S2, xsys_D=D2, xsys_I=I2,
            xsys_disagree=(S2+D2+I2)/N2 if N2 else np.nan,
            mean_token_logprob=big_lp[c], compression_ratio=comp,
            n_words_30s=len(h), words_per_sec=len(h)/30.0,
            passage=passage_of(big_txt[c]), window_start_s=offs.get(c, [np.nan])[0],
            n_ref_words_local=Nl, wer_local=(Sl+Dl+Il)/Nl if Nl else np.nan,
            hyp_whisper_large_v3=big_txt[c], hyp_whisper_medium=sml_txt[c]))
    d = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    d.to_csv(CSV, index=False)
    n, nspk = len(d), d.spk.nunique()
    print(f"\nwrote {CSV}  ({n} rows)", flush=True)
    if d[["p_yes_text", "p_yes_audio"]].isna().any().any(): disc("HIGH", "p_yes missing after merge")

    tsv = []
    def emit(w, v, lo, hi, f):
        tsv.append(f"M11\t{w}\t{v:.4f}\t{'' if lo is None or np.isnan(lo) else format(lo,'.4f')}\t"
                   f"{'' if hi is None or np.isnan(hi) else format(hi,'.4f')}\t{n}\t{nspk}\t{f}")
    def line(s): print(s, flush=True)

    line("\n" + "="*104)
    line("M11  MDVR-KCL  does transcription quality explain the transcript result?")
    line("="*104)

    # ---- A
    line("\n[A] instructed fallback: Whisper large-v3 vs the transcript the text condition used")
    line(f"M11 A ASR reproduction WER vs stored transcript = {d.wer.mean():.4f}  "
         f"(exact match on {int((d.wer==0).sum())}/{n} clips)  n={n} n_speakers={nspk}  file={CSV}")
    line("    -> the stored transcript IS Whisper large-v3 greedy output, so this route is degenerate")
    line("       and carries NO information about ASR quality. Findings B-E are used instead.")
    emit("A_asr_reproduction_wer_vs_paper_transcript", float(d.wer.mean()), None, None, CSV)

    # ---- B, C, D as group contrasts
    METRICS = [("B_xsys_disagree",      "xsys_disagree",      "cross-system disagreement large-v3 vs medium"),
               ("C_mean_token_logprob", "mean_token_logprob", "Whisper large-v3 mean per-token logprob"),
               ("C_compression_ratio",  "compression_ratio",  "zlib compression ratio (repetition detector)"),
               ("D_n_words_30s",        "n_words_30s",        "words recovered in the 30 s window")]
    for key, col, label in METRICS:
        line(f"\n[{key.split('_')[0]}] {label}")
        mm = {}
        for g in ("PD", "control"):
            s = d[d.group == g]; mean = float(s[col].mean())
            lo, hi, k = boot_mean(s[col].values, s.spk.values); mm[g] = mean
            line(f"M11 {key} mean {g:<8} = {mean:.4f} [{lo:.4f}, {hi:.4f}]  n={len(s)} "
                 f"n_speakers={s.spk.nunique()} usable_draws={k}/{NBOOT}  file={CSV}")
            emit(f"{key}_mean_{g}", mean, lo, hi, CSV)
        dl, dh, dk = boot_paired_diff(d[col].values, d.spk.values, d.group.values)
        diff = mm["PD"] - mm["control"]
        line(f"M11 {key} PAIRED diff PD minus control = {diff:.4f} [{dl:.4f}, {dh:.4f}]  "
             f"n={n} n_speakers={nspk} usable_draws={dk}/{NBOOT}  file={CSV}")
        emit(f"{key}_diff_PD_minus_control", diff, dl, dh, CSV)
        for nm, pc in (("text", "p_yes_text"), ("audio", "p_yes_audio")):
            r = st.spearmanr(d[col], d[pc]); lo, hi, k = boot_spearman(d[col], d[pc], d.spk.values)
            line(f"M11 {key} Spearman vs p_yes {nm:<5} = {r.statistic:.4f} [{lo:.4f}, {hi:.4f}] "
                 f"p={r.pvalue:.4f} n={n} n_speakers={nspk} usable_draws={k}/{NBOOT}  file={CSV}")
            emit(f"{key}_spearman_vs_pyes_{nm}", float(r.statistic), lo, hi, CSV)
        a = auc(d.label, d[col])
        line(f"M11 {key} AUC from this measure alone = {a:.4f}  n={n} n_speakers={nspk}  file={CSV}")
        emit(f"{key}_auc_alone", a, None, None, CSV)

    # ---- E passage confound
    line("\n[E] which passage the speaker was reading (from the large-v3 transcript)")
    ct = pd.crosstab(d.passage, d.group)
    for p in ct.index:
        line(f"M11 E passage {p:<17} PD={int(ct.loc[p].get('PD',0)):>2}  "
             f"control={int(ct.loc[p].get('control',0)):>2}")
    nwflag = (d.passage == "north_wind").astype(int)
    a_p = auc(d.label, nwflag)
    line(f"M11 E AUC from passage identity alone (north_wind=1) = {a_p:.4f}  "
         f"n={n} n_speakers={nspk}  file={CSV}")
    emit("E_auc_from_passage_identity_alone", a_p, None, None, CSV)

    # ---- the two conditions, recomputed
    a_t, a_a = auc(d.label, d.p_yes_text), auc(d.label, d.p_yes_audio)
    line(f"\nM11 AUC recomputed, TEXT  condition = {a_t:.4f}  n={n} n_speakers={nspk}  file={P_TEXT}")
    line(f"M11 AUC recomputed, AUDIO condition = {a_a:.4f}  n={n} n_speakers={nspk}  file={P_AUDIO}")
    emit("auc_text_condition_recomputed", a_t, None, None, P_TEXT)
    emit("auc_audio_condition_recomputed", a_a, None, None, P_AUDIO)

    # ---- second stored transcript copy
    line(f"\n[F] large-v3 vs the OTHER stored transcript copy ({os.path.basename(REF_LOCAL)})")
    for g in ("PD", "control"):
        s = d[d.group == g]; mean = float(s.wer_local.mean())
        lo, hi, k = boot_mean(s.wer_local.values, s.spk.values)
        line(f"M11 F wer_vs_local_copy mean {g:<8} = {mean:.4f} [{lo:.4f}, {hi:.4f}]  n={len(s)} "
             f"n_speakers={s.spk.nunique()} usable_draws={k}/{NBOOT}  file={REF_LOCAL}")
        emit(f"F_wer_vs_local_copy_mean_{g}", mean, lo, hi, REF_LOCAL)

    with open(ROWS, "w") as f:
        f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n" + "\n".join(tsv) + "\n")
    line(f"\nwrote {ROWS}")

    side = dict(
        task="PART16 M11 MDVR-KCL: does transcription quality explain the transcript result",
        true_wer_possible=False,
        reference_text_search=dict(
            looked_in=[SRCDIR,
                       "zip listing of <local data dir>/KCL/26_29_09_2017_KCL.zip (79 entries, all .wav, no README or doc)",
                       "recursive grep of <local data dir>/Desktop/release for the passage wording",
                       "<local data dir>/paper1_local_runs"],
            result="no read-passage reference text file exists on disk",
            prior_audit_agrees="edaic_rerun/scripts_1a1c/report.py line 244: 'NO timed transcript exists for MDVR-KCL on disk. The KCL tree holds audio only'",
            second_reason_ill_posed=("speakers did not read one common passage (see finding E), and each scored "
                                     "clip is a 30 s window starting mid-passage at a measured per-file offset, "
                                     f"{OFFSETS}"),
            instructed_fallback="scored, reported as finding A, and found degenerate (WER 0.0000 on 37/37)"),
        models=dict(primary=BIG, cross_system_control=SMALL,
                    cache_primary=HF_GDRIVE, cache_control=HF_LOCAL),
        device=dev, dtype="float32",
        runtime_min=dict(large_v3=round(min_big, 2), medium=round(min_sml, 2),
                         total=round((time.time()-t_all)/60, 2)),
        decode_settings=DEC,
        normaliser=("transformers EnglishTextNormalizer built from the openai/whisper-large-v3 normalizer.json "
                    f"spelling map ({len(spelling)} entries): lowercase, punctuation removed, contractions "
                    "expanded, British->American spelling, whitespace collapsed. Then, identically on both "
                    "sides: any char not in [a-z0-9 ] becomes a space and each contiguous digit run is its own "
                    "token (5.7 -> '5 7', 30% -> '30')."),
        wer_formula="(S+D+I)/N, N = len(reference tokens), S/D/I from a Levenshtein backtrace with unit costs",
        passage_rule=dict(north_wind_keys=NW, light_scattering_keys=RB,
                          rule="more matched keys wins; 0-0 is 'neither', a tie is 'ambiguous'"),
        bootstrap=dict(draws=NBOOT, rng="numpy.random.default_rng(0), reseeded per cell",
                       resample="speakers with replacement (one clip per speaker here, so speaker = clip)",
                       percentiles=[2.5, 97.5],
                       paired="one speaker draw per replicate, both group means recomputed inside that draw"),
        folds="not applicable: no cross-validated model is fitted in this task",
        n=n, n_speakers=nspk, seed=SEED,
        prompt_text_condition=str(ptext["prompt"].iloc[0]),
        prompt_audio_condition=str(paudio["prompt"].iloc[0]),
        shell_command=CMD,
        sources={p: dict(sha256_first_1mb=sha1mb(p)) for p in
                 [MANIFEST, REF_POD, REF_LOCAL, P_TEXT, P_AUDIO, OFFSETS,
                  os.path.join(CLIPDIR, clips[0]), os.path.join(CLIPDIR, clips[-1])]},
        outputs=[CSV, SIDE, ROWS])
    with open(SIDE, "w") as f: json.dump(side, f, indent=1)
    line(f"wrote {SIDE}")
    line(f"TOTAL {(time.time()-t_all)/60:.1f} min")

if __name__ == "__main__":
    main()
