#!/usr/bin/env python3
"""
INDEPENDENT VERIFIER for PART16 M11.
Written from scratch. Does NOT import, exec or read the primary job's
M11_kcl_wer.py / M11_verify.py / M11_robust.py / M11_asr_cache.json.
Goes back to the ORIGINAL sources:
  audio    <local data dir>/paper1_local_runs/kcl_read30b/*.wav
  manifest <local data dir>/paper1_local_runs/kcl30b_manifest.csv
  text-cond scores <local data dir>/Desktop/release/podE_final/out/o25_kcl_text.csv
  audio-cond scores <local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv
  stored transcripts <local data dir>/Desktop/release/part4_text/asr/asr_kcl.csv

Stages:
  asr   -> re-decode all 37 clips with whisper-large-v3 and whisper-medium,
           write verify/M11_verify_asr_mine.json (MY cache, not theirs)
  stats -> recompute every headline number
"""
import sys, os, csv, json, math, re, unicodedata, time

RELEASE = "<local data dir>/Desktop/release"
P16     = os.path.join(RELEASE, "edaic_rerun/part16")
VDIR    = os.path.join(P16, "verify")
AUDIO   = "<local data dir>/paper1_local_runs/kcl_read30b"
MANIFEST= "<local data dir>/paper1_local_runs/kcl30b_manifest.csv"
TEXTSC  = os.path.join(RELEASE, "podE_final/out/o25_kcl_text.csv")
AUDSC   = os.path.join(RELEASE, "omni_final/omni_kcl_zeroshot_scores.csv")
ASRSTORE= os.path.join(RELEASE, "part4_text/asr/asr_kcl.csv")
LOCALTXT= "<local data dir>/paper1_local_runs/kcl_read30b_text.csv"
MYASR   = os.path.join(VDIR, "M11_verify_asr_mine.json")

def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

# ---------------------------------------------------------------- stage: asr
def stage_asr():
    os.environ.setdefault("HF_HOME", "<local data dir>/DementiaBank/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch, numpy as np, soundfile as sf
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    files = sorted(f for f in os.listdir(AUDIO) if f.endswith(".wav"))
    assert len(files) == 37, len(files)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    out = {"device": dev, "files": files, "models": {}}
    for tag, ckpt, hfhome in [
        ("large_v3", "openai/whisper-large-v3", "<local data dir>/DementiaBank/hf_cache"),
        ("medium",   "openai/whisper-medium",   os.path.expanduser("~/.cache/huggingface")),
    ]:
        os.environ["HF_HOME"] = hfhome
        t0 = time.time()
        proc = WhisperProcessor.from_pretrained(ckpt, cache_dir=os.path.join(hfhome, "hub"))
        mdl  = WhisperForConditionalGeneration.from_pretrained(
                   ckpt, cache_dir=os.path.join(hfhome, "hub"), dtype=torch.float32).to(dev).eval()
        res = {}
        for fn in files:
            wav, sr = sf.read(os.path.join(AUDIO, fn), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            assert sr == 16000, (fn, sr)
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            with torch.no_grad():
                g = mdl.generate(feats, language="en", task="transcribe",
                                 do_sample=False, num_beams=1, temperature=0.0,
                                 max_new_tokens=440, condition_on_prev_tokens=False,
                                 return_dict_in_generate=True, output_scores=True)
            seq = g.sequences[0]
            # token logprobs over the generated (non-forced) positions
            lps = []
            gen = seq[len(seq) - len(g.scores):]
            for i, sc in enumerate(g.scores):
                lp = torch.log_softmax(sc[0].float(), dim=-1)
                lps.append(float(lp[int(gen[i])]))
            txt = proc.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0].strip()
            res[fn] = {"text": txt,
                       "mean_token_logprob": (sum(lps)/len(lps)) if lps else float("nan"),
                       "n_gen_tokens": len(lps),
                       "dur_s": len(wav)/sr}
            print(f"  {tag} {fn} {len(txt.split()):4d}w lp={res[fn]['mean_token_logprob']:.4f}", flush=True)
        out["models"][tag] = res
        out.setdefault("runtime_min", {})[tag] = round((time.time()-t0)/60, 2)
        del mdl
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
        print(f"[{tag}] done in {out['runtime_min'][tag]} min", flush=True)
    with open(MYASR, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", MYASR)

if __name__ == "__main__":
    if sys.argv[1] == "asr":
        stage_asr()

# --------------------------------------------------------- independent AUC
def auc_pairs(scores, labels):
    """Mann-Whitney U by explicit concordant-pair count, ties = 0.5.
    NOT a rank-sum one-liner: O(n1*n0) double loop."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    c = 0.0
    for a in pos:
        for b in neg:
            if a > b:   c += 1.0
            elif a == b: c += 0.5
    return c / (len(pos) * len(neg))

def auc_trapezoid(scores, labels):
    """Second, fully separate route: trapezoid area under the empirical ROC."""
    P = sum(1 for l in labels if l == 1); N = len(labels) - P
    if P == 0 or N == 0: return float("nan")
    thr = sorted(set(scores), reverse=True)
    pts = [(0.0, 0.0)]
    for t in thr:
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        pts.append((fp / N, tp / P))
    pts.append((1.0, 1.0))
    a = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        a += (x1 - x0) * (y0 + y1) / 2.0
    return a

def spearman(x, y):
    """Own implementation: average ranks then Pearson on the ranks."""
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1): r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    n = len(x); mx = sum(rx)/n; my = sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return num/den if den else float("nan")

def stage_auc():
    man = rows(MANIFEST)
    lab = {r["speaker"]: int(r["label"]) for r in man}
    t = rows(TEXTSC); a = rows(AUDSC)
    print(f"manifest n={len(man)} n_spk={len(lab)} PD={sum(lab.values())} HC={len(lab)-sum(lab.values())}")
    print(f"o25_kcl_text n={len(t)}  omni_kcl_zeroshot n={len(a)}")
    # label agreement across the three files
    for nm, rr, sk, lk in [("text", t, "speaker_id", "label"), ("audio", a, "speaker", "label")]:
        bad = [r[sk] for r in rr if int(r[lk]) != lab[r[sk]]]
        print(f"  label mismatch vs manifest in {nm}: {len(bad)} {bad[:5]}")
    ts = [(r["speaker_id"], float(r["p_yes"]), lab[r["speaker_id"]]) for r in t]
    as_ = [(r["speaker"],    float(r["p_yes"]), lab[r["speaker"]])    for r in a]
    for nm, d in [("text", ts), ("audio", as_)]:
        sc = [x[1] for x in d]; ll = [x[2] for x in d]
        print(f"AUC_{nm}  pairs={auc_pairs(sc, ll):.6f}  trapz={auc_trapezoid(sc, ll):.6f}  n={len(d)}")
        print(f"   ties in p_yes: {len(sc)-len(set(sc))}")
    return lab, dict((s, p) for s, p, _ in ts), dict((s, p) for s, p, _ in as_)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "auc":
    stage_auc()

# --------------------------------------------------- passage identity (E)
NW  = ["north wind", "traveller", "traveler", "cloak", "stronger"]
LS  = ["scattering", "blue light", "atmospheric", "sunlight", "observer",
       "electromagnetic", "spectrum"]

def passage_of(text):
    t = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    n = sum(1 for k in NW if k in t)
    l = sum(1 for k in LS if k in t)
    if n == 0 and l == 0: return "neither"
    if n > l: return "north_wind"
    if l > n: return "light_scattering"
    return "ambiguous"

def stage_passage():
    import collections
    man = rows(MANIFEST); lab = {r["speaker"]: int(r["label"]) for r in man}
    st  = rows(ASRSTORE)
    print("asr_kcl.csv n =", len(st))
    pas = {r["speaker_id"]: passage_of(r["text"]) for r in st}
    nw  = {r["speaker_id"]: len(r["text"].split()) for r in st}
    ct = collections.Counter((pas[s], "PD" if lab[s] else "HC") for s in pas)
    for k in sorted(set(pas.values())):
        print(f"  {k:18s} PD={ct[(k,'PD')]:3d}  control={ct[(k,'HC')]:3d}")
    spk = sorted(pas)
    ll  = [lab[s] for s in spk]
    encodings = {
        "binary_is_north_wind": {"north_wind": 1.0, "light_scattering": 0.0, "neither": 0.0, "ambiguous": 0.0},
        "pd_rate_of_category":  None,
        "nw>light>neither":     {"north_wind": 2.0, "light_scattering": 1.0, "neither": 0.0, "ambiguous": 0.5},
        "nw>neither>light":     {"north_wind": 2.0, "light_scattering": 0.0, "neither": 1.0, "ambiguous": 0.5},
    }
    rate = {}
    for k in set(pas.values()):
        m = [lab[s] for s in spk if pas[s] == k]
        rate[k] = sum(m) / len(m)
    encodings["pd_rate_of_category"] = rate
    for nm, mp in encodings.items():
        sc = [mp[pas[s]] for s in spk]
        print(f"  AUC passage [{nm:22s}] pairs={auc_pairs(sc, ll):.6f} trapz={auc_trapezoid(sc, ll):.6f}")
    # word counts of the stored transcripts
    tiny = sorted(((nw[s], s) for s in spk))[:8]
    med  = sorted(nw[s] for s in spk)[len(spk)//2]
    print("  stored-transcript words: min8 =", tiny, " median =", med,
          " mean = %.1f" % (sum(nw.values())/len(nw)))
    return pas, lab, nw

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "passage":
    stage_passage()

# --------------------------------------------------- my own normaliser + DP
def norm_simple(s):
    """My own normaliser. No import of the primary job's code."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"(\d+)", r" \1 ", s)
    return [t for t in s.split() if t]

def norm_whisper(s):
    """The normaliser the sidecar documents: transformers EnglishTextNormalizer,
    then the same [a-z0-9 ] + digit-run pass. Library class, not their script."""
    global _EN
    try:
        _EN
    except NameError:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        import json as _j, glob as _g
        cand = _g.glob("<local data dir>/DementiaBank/hf_cache/hub/models--openai--whisper-large-v3/snapshots/*/normalizer.json")
        sp = _j.load(open(cand[0])) if cand else {}
        _EN = EnglishTextNormalizer(sp)
    return norm_simple(_EN(s))

def levenshtein_sdi(ref, hyp):
    """Own DP with backtrace. Unit costs. Returns S, D, I."""
    n, m = len(ref), len(hyp)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        ri = ref[i-1]
        for j in range(1, m+1):
            if ri == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j-1], d[i-1][j], d[i][j-1])
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1] and d[i][j] == d[i-1][j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i-1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

# ------------------------------------------------------------- bootstrap
def boot_ci(vals, rng_seed=0, draws=2000, stat=None):
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    v = np.asarray(vals, dtype=float); n = len(v)
    out = np.empty(draws)
    for b in range(draws):
        idx = rng.integers(0, n, n)
        out[b] = (stat or np.mean)(v[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def boot_ci_grouped(vals, labels, which, rng_seed=0, draws=2000, mode="all37"):
    """mode 'all37'  : resample ALL speakers, then take the subgroup mean
       mode 'within' : resample only the speakers inside that group"""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    v = np.asarray(vals, float); l = np.asarray(labels, int)
    if mode == "within":
        sub = v[l == which]; n = len(sub); out = np.empty(draws)
        for b in range(draws):
            out[b] = sub[rng.integers(0, n, n)].mean()
    else:
        n = len(v); out = np.full(draws, np.nan)
        for b in range(draws):
            idx = rng.integers(0, n, n)
            sel = v[idx][l[idx] == which]
            if len(sel): out[b] = sel.mean()
        out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def boot_ci_paired(vals, labels, rng_seed=0, draws=2000):
    """One speaker draw per replicate; both group means inside that draw."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    v = np.asarray(vals, float); l = np.asarray(labels, int); n = len(v)
    out = np.full(draws, np.nan)
    for b in range(draws):
        idx = rng.integers(0, n, n)
        vv, ll = v[idx], l[idx]
        a, c = vv[ll == 1], vv[ll == 0]
        if len(a) and len(c): out[b] = a.mean() - c.mean()
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def boot_ci_spearman(x, y, rng_seed=0, draws=2000):
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    out = np.full(draws, np.nan)
    for b in range(draws):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if len(set(xs.tolist())) > 1 and len(set(ys.tolist())) > 1:
            out[b] = spearman(xs.tolist(), ys.tolist())
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def f4(x): return "%.4f" % x

# ------------------------------------------------------------ main stats
def stage_stats():
    import numpy as np, collections
    man = rows(MANIFEST)
    lab = {r["speaker"]: int(r["label"]) for r in man}
    file_of = {r["speaker"]: os.path.basename(r["path"]) for r in man}
    spk = sorted(lab)
    assert len(spk) == 37

    p_text  = {r["speaker_id"]: float(r["p_yes"]) for r in rows(TEXTSC)}
    p_aud   = {r["speaker"]:    float(r["p_yes"]) for r in rows(AUDSC)}
    stored  = {r["speaker_id"]: r["text"]         for r in rows(ASRSTORE)}

    mine = json.load(open(MYASR))
    big  = mine["models"]["large_v3"]; med = mine["models"]["medium"]
    s2f  = {s: file_of[s] for s in spk}

    rec = {}
    for s in spk:
        fn = s2f[s]
        hb, hm = big[fn]["text"], med[fn]["text"]
        rs, hs = norm_simple(stored[s]), norm_simple(hb)
        S, D, I = levenshtein_sdi(rs, hs)
        rw, hw = norm_whisper(stored[s]), norm_whisper(hb)
        S2, D2, I2 = levenshtein_sdi(rw, hw)
        xa, xb = norm_whisper(hb), norm_whisper(hm)
        XS, XD, XI = levenshtein_sdi(xa, xb)
        xa1, xb1 = norm_simple(hb), norm_simple(hm)
        XS1, XD1, XI1 = levenshtein_sdi(xa1, xb1)
        rec[s] = dict(
            spk=s, file=fn, label=lab[s],
            n_ref_simple=len(rs), S=S, D=D, I=I,
            wer_simple=(S+D+I)/len(rs) if rs else float("nan"),
            n_ref_wn=len(rw), S_wn=S2, D_wn=D2, I_wn=I2,
            wer_wn=(S2+D2+I2)/len(rw) if rw else float("nan"),
            xsys_n=len(xa), xS=XS, xD=XD, xI=XI,
            xsys_disagree=(XS+XD+XI)/len(xa) if xa else float("nan"),
            xsys_disagree_simple=(XS1+XD1+XI1)/len(xa1) if xa1 else float("nan"),
            mean_token_logprob=big[fn]["mean_token_logprob"],
            n_words_30s=len(xa),
            passage=passage_of(stored[s]),
            p_text=p_text[s], p_aud=p_aud[s],
            hyp_big=hb, hyp_med=hm)

    L   = [rec[s]["label"] for s in spk]
    def col(k): return [rec[s][k] for s in spk]

    print("=" * 96)
    print("INDEPENDENT RECOMPUTE  (my own whisper run, my own normaliser, my own DP)")
    print("=" * 96)
    # --- A: WER vs the stored transcript
    w = col("wer_wn")
    print(f"A wer_vs_stored  mean={np.mean(w):.4f}  exact-zero clips = {sum(1 for x in w if x==0)}/37"
          f"   (simple-normaliser mean={np.mean(col('wer_simple')):.4f}, "
          f"exact-zero {sum(1 for x in col('wer_simple') if x==0)}/37)")

    # --- B / C / D
    out = {}
    for key, nm in [("xsys_disagree", "B_xsys_disagree"),
                    ("mean_token_logprob", "C_mean_token_logprob"),
                    ("n_words_30s", "D_n_words_30s")]:
        v = col(key)
        pd_ = [x for x, l in zip(v, L) if l == 1]
        hc_ = [x for x, l in zip(v, L) if l == 0]
        lo_a, hi_a = boot_ci_grouped(v, L, 1, mode="all37")
        lo_b, hi_b = boot_ci_grouped(v, L, 1, mode="within")
        lo_c, hi_c = boot_ci_grouped(v, L, 0, mode="all37")
        lo_d, hi_d = boot_ci_grouped(v, L, 0, mode="within")
        plo, phi = boot_ci_paired(v, L)
        print(f"{nm} mean PD      = {np.mean(pd_):.4f}  CI_all37[{lo_a:.4f},{hi_a:.4f}] CI_within[{lo_b:.4f},{hi_b:.4f}] n=16")
        print(f"{nm} mean control = {np.mean(hc_):.4f}  CI_all37[{lo_c:.4f},{hi_c:.4f}] CI_within[{lo_d:.4f},{hi_d:.4f}] n=21")
        print(f"{nm} PAIRED diff  = {np.mean(pd_)-np.mean(hc_):.4f}  CI[{plo:.4f},{phi:.4f}] n=37 draws 2000")
        for tgt, tn in [("p_text", "p_yes text"), ("p_aud", "p_yes audio")]:
            r = spearman(v, col(tgt))
            slo, shi = boot_ci_spearman(v, col(tgt))
            tstat = r*math.sqrt((37-2)/max(1e-12, 1-r*r))
            from scipy import stats as _st
            p = 2*_st.t.sf(abs(tstat), 35)
            print(f"{nm} Spearman vs {tn:12s} = {r:.4f}  CI[{slo:.4f},{shi:.4f}] p={p:.4f}")
        out[nm] = dict(pd=np.mean(pd_), hc=np.mean(hc_), diff=np.mean(pd_)-np.mean(hc_))

    # --- E and the two conditions
    pas = {s: rec[s]["passage"] for s in spk}
    enc = {"north_wind": 1.0, "light_scattering": 0.0, "neither": 0.0, "ambiguous": 0.0}
    print(f"E_auc_from_passage_identity_alone = {auc_pairs([enc[pas[s]] for s in spk], L):.4f}"
          f"  (trapz {auc_trapezoid([enc[pas[s]] for s in spk], L):.4f})")
    print(f"auc_text_condition_recomputed  = {auc_pairs(col('p_text'), L):.4f}")
    print(f"auc_audio_condition_recomputed = {auc_pairs(col('p_aud'), L):.4f}")
    print(f"auc_mean_token_logprob         = {auc_pairs(col('mean_token_logprob'), L):.4f}")

    # --- robustness on B
    v = col("xsys_disagree")
    pdv = [x for x, l in zip(v, L) if l == 1]; hcv = [x for x, l in zip(v, L) if l == 0]
    print(f"B_MEDIAN     PD {np.median(pdv):.4f}  control {np.median(hcv):.4f}  diff {np.median(pdv)-np.median(hcv):+.4f}")
    keep = [i for i, s in enumerate(spk) if rec[s]["xsys_n"] >= 20]
    pk = [v[i] for i in keep if L[i] == 1]; hk = [v[i] for i in keep if L[i] == 0]
    print(f"B_ge20w_MEAN PD {np.mean(pk):.4f}  control {np.mean(hk):.4f}  diff {np.mean(pk)-np.mean(hk):+.4f}  n={len(keep)}")
    num_p = sum(rec[s]["xS"]+rec[s]["xD"]+rec[s]["xI"] for s in spk if lab[s] == 1)
    den_p = sum(rec[s]["xsys_n"] for s in spk if lab[s] == 1)
    num_c = sum(rec[s]["xS"]+rec[s]["xD"]+rec[s]["xI"] for s in spk if lab[s] == 0)
    den_c = sum(rec[s]["xsys_n"] for s in spk if lab[s] == 0)
    print(f"B_POOLED     PD {num_p/den_p:.4f}  control {num_c/den_c:.4f}  diff {num_p/den_p-num_c/den_c:+.4f}")

    # --- clip-by-clip comparison against THEIR csv (audit only)
    th = {r["spk"]: r for r in rows(os.path.join(P16, "M11_kcl_wer.csv"))}
    print("-" * 96)
    print("PER-CLIP AUDIT vs their M11_kcl_wer.csv (their rows =", len(th), ")")
    dif = collections.Counter()
    worst = []
    for s in spk:
        for mine_k, their_k in [("xsys_disagree", "xsys_disagree"),
                                ("mean_token_logprob", "mean_token_logprob"),
                                ("n_words_30s", "n_words_30s"),
                                ("passage", "passage")]:
            a = rec[s][mine_k]; b = th[s][their_k]
            if mine_k == "passage":
                if a != b: dif[mine_k] += 1; worst.append((s, mine_k, a, b))
            else:
                b = float(b)
                if abs(a - b) > 1e-4:
                    dif[mine_k] += 1; worst.append((s, mine_k, round(a, 4), round(b, 4)))
    print("  clips differing:", dict(dif) or "NONE")
    for wsp in worst[:20]: print("   ", wsp)
    idt = sum(1 for s in spk if rec[s]["hyp_big"].strip() == (th[s]["hyp_whisper_large_v3"] or "").strip())
    idm = sum(1 for s in spk if rec[s]["hyp_med"].strip() == (th[s]["hyp_whisper_medium"] or "").strip())
    print(f"  my large-v3 hypothesis identical to theirs on {idt}/37 clips; medium {idm}/37")

    json.dump({s: {k: v for k, v in rec[s].items()} for s in spk},
              open(os.path.join(VDIR, "M11_verify_perclip_mine.json"), "w"), indent=1)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "stats":
    stage_stats()

# ------------------------------------------------- extra rows in their tsv
def stage_extra():
    import numpy as np, zlib
    man = rows(MANIFEST); lab = {r["speaker"]: int(r["label"]) for r in man}
    fof = {r["speaker"]: os.path.basename(r["path"]) for r in man}
    spk = sorted(lab); L = [lab[s] for s in spk]
    mine = json.load(open(MYASR)); big = mine["models"]["large_v3"]; med = mine["models"]["medium"]
    loc  = {r["speaker_id"]: r["text"] for r in rows(LOCALTXT)}
    stored = {r["speaker_id"]: r["text"] for r in rows(ASRSTORE)}

    xs, cr, nw, fw = [], [], [], []
    for s in spk:
        a, b = norm_whisper(big[fof[s]]["text"]), norm_whisper(med[fof[s]]["text"])
        S, D, I = levenshtein_sdi(a, b); xs.append((S+D+I)/len(a) if a else float("nan"))
        nw.append(len(a))
        t = big[fof[s]]["text"]
        cr.append(len(t.encode()) / len(zlib.compress(t.encode())))
        r2, h2 = norm_whisper(loc[s]), norm_whisper(big[fof[s]]["text"])
        S2, D2, I2 = levenshtein_sdi(r2, h2); fw.append((S2+D2+I2)/len(r2) if r2 else float("nan"))

    print("B_xsys_disagree_auc_alone      = %.4f" % auc_pairs(xs, L))
    print("D_n_words_30s_auc_alone        = %.4f" % auc_pairs(nw, L))
    print("C_compression_ratio_auc_alone  = %.4f" % auc_pairs(cr, L))
    for nm, v in [("C_compression_ratio", cr), ("F_wer_vs_local_copy", fw)]:
        p_ = [x for x, l in zip(v, L) if l == 1]; c_ = [x for x, l in zip(v, L) if l == 0]
        la, ha = boot_ci_grouped(v, L, 1, mode="within"); lc, hc = boot_ci_grouped(v, L, 0, mode="within")
        pl, ph = boot_ci_paired(v, L)
        print(f"{nm}_mean_PD      = {np.mean(p_):.4f} [{la:.4f},{ha:.4f}]")
        print(f"{nm}_mean_control = {np.mean(c_):.4f} [{lc:.4f},{hc:.4f}]")
        print(f"{nm}_diff         = {np.mean(p_)-np.mean(c_):.4f} [{pl:.4f},{ph:.4f}]")
        if nm.startswith("C_comp"):
            for tn, tk, tf in [("text", "p_text", TEXTSC), ("audio", "p_aud", AUDSC)]:
                pm = ({r["speaker_id"]: float(r["p_yes"]) for r in rows(TEXTSC)} if tn == "text"
                      else {r["speaker"]: float(r["p_yes"]) for r in rows(AUDSC)})
                y = [pm[s] for s in spk]
                lo, hi = boot_ci_spearman(v, y)
                print(f"{nm}_spearman_vs_pyes_{tn} = {spearman(v, y):.4f} [{lo:.4f},{hi:.4f}]")
    # medians / ge20 CIs that their tsv carries
    pdv = [x for x, l in zip(xs, L) if l == 1]; hcv = [x for x, l in zip(xs, L) if l == 0]
    lo, hi = boot_ci(pdv, stat=np.median); print(f"B_MEDIAN_PD      = {np.median(pdv):.4f} [{lo:.4f},{hi:.4f}] n=16")
    lo, hi = boot_ci(hcv, stat=np.median); print(f"B_MEDIAN_control = {np.median(hcv):.4f} [{lo:.4f},{hi:.4f}] n=21")
    kp = [x for x, n_, l in zip(xs, nw, L) if n_ >= 20 and l == 1]
    kc = [x for x, n_, l in zip(xs, nw, L) if n_ >= 20 and l == 0]
    lo, hi = boot_ci(kp); print(f"B_ge20w_PD       = {np.mean(kp):.4f} [{lo:.4f},{hi:.4f}] n={len(kp)}")
    lo, hi = boot_ci(kc); print(f"B_ge20w_control  = {np.mean(kc):.4f} [{lo:.4f},{hi:.4f}] n={len(kc)}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extra":
    stage_extra()

def stage_robci():
    """Chase the three robustness CIs that did not match: is it the
    resample convention (all-37 then subset) rather than arithmetic?"""
    import numpy as np
    man = rows(MANIFEST); lab = {r["speaker"]: int(r["label"]) for r in man}
    fof = {r["speaker"]: os.path.basename(r["path"]) for r in man}
    spk = sorted(lab); L = np.array([lab[s] for s in spk])
    mine = json.load(open(MYASR)); big = mine["models"]["large_v3"]; med = mine["models"]["medium"]
    xs, nwd, num, den = [], [], [], []
    for s in spk:
        a, b = norm_whisper(big[fof[s]]["text"]), norm_whisper(med[fof[s]]["text"])
        S, D, I = levenshtein_sdi(a, b)
        xs.append((S+D+I)/len(a) if a else float("nan")); nwd.append(len(a))
        num.append(S+D+I); den.append(len(a))
    xs = np.array(xs); nwd = np.array(nwd); num = np.array(num, float); den = np.array(den, float)

    def ci(fn, seed=0, draws=2000):
        rng = np.random.default_rng(seed); o = np.full(draws, np.nan)
        for b in range(draws):
            idx = rng.integers(0, 37, 37); o[b] = fn(idx)
        o = o[~np.isnan(o)]
        return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))

    for nm, which in [("MEDIAN_PD", 1), ("MEDIAN_control", 0)]:
        print(f"B_{nm} CI all37-subset  = [%.4f,%.4f]" % ci(
            lambda i, w=which: np.median(xs[i][L[i] == w]) if (L[i] == w).any() else np.nan))
    for nm, which in [("ge20w_PD", 1), ("ge20w_control", 0)]:
        print(f"B_{nm} CI all37-subset  = [%.4f,%.4f]" % ci(
            lambda i, w=which: xs[i][(L[i] == w) & (nwd[i] >= 20)].mean()
            if ((L[i] == w) & (nwd[i] >= 20)).any() else np.nan))
    for nm, which in [("POOLED_PD", 1), ("POOLED_control", 0)]:
        for mode, f in [("within", None), ("all37", None)]:
            pass
        def pooled(i, w=which):
            m = L[i] == w
            return num[i][m].sum()/den[i][m].sum() if m.any() and den[i][m].sum() else np.nan
        print(f"B_{nm} CI all37-subset  = [%.4f,%.4f]" % ci(pooled))
        sub = np.where(L == which)[0]
        def pooled_w(_, s=sub):
            pass
        rng = np.random.default_rng(0); o = np.empty(2000)
        for b in range(2000):
            j = sub[rng.integers(0, len(sub), len(sub))]
            o[b] = num[j].sum()/den[j].sum()
        print(f"B_{nm} CI within        = [%.4f,%.4f]" % (np.percentile(o, 2.5), np.percentile(o, 97.5)))

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "robci":
    stage_robci()

def stage_probe():
    """Adversarial probes on finding E."""
    import numpy as np, collections
    man = rows(MANIFEST); lab = {r["speaker"]: int(r["label"]) for r in man}
    fof = {r["speaker"]: os.path.basename(r["path"]) for r in man}
    spk = sorted(lab)
    mine = json.load(open(MYASR)); big = mine["models"]["large_v3"]
    stored = {r["speaker_id"]: r["text"] for r in rows(ASRSTORE)}
    pas = {s: passage_of(stored[s]) for s in spk}
    nwd = {s: len(norm_whisper(big[fof[s]]["text"])) for s in spk}
    print("clip word count x passage:")
    for s in sorted(spk, key=lambda x: nwd[x])[:10]:
        print(f"   {s} n={nwd[s]:4d} passage={pas[s]:17s} label={lab[s]}  '{stored[s][:52]}'")
    nei = [s for s in spk if pas[s] == "neither"]
    print("  'neither' speakers:", [(s, nwd[s], lab[s]) for s in nei])
    print("  overlap of 'neither' with the 6 clips <20 words:",
          sorted(set(nei) & set(s for s in spk if nwd[s] < 20)))
    enc = {"north_wind": 1.0, "light_scattering": 0.0, "neither": 0.0, "ambiguous": 0.0}
    keep = [s for s in spk if nwd[s] >= 20]
    print("  E on the 31 clips with >=20 words: %.4f (n=%d, PD=%d)" % (
        auc_pairs([enc[pas[s]] for s in keep], [lab[s] for s in keep]),
        len(keep), sum(lab[s] for s in keep)))
    print("  text-condition AUC on those same 31: %.4f" % auc_pairs(
        [float(r["p_yes"]) for r in rows(TEXTSC) if r["speaker_id"] in keep],
        [lab[r["speaker_id"]] for r in rows(TEXTSC) if r["speaker_id"] in keep]))
    # E as a leave-one-out (out-of-sample) score instead of in-sample
    rate_full = {}
    loo = []
    for s in spk:
        oth = [x for x in spk if x != s]
        rr = {}
        for k in set(pas[x] for x in oth):
            m = [lab[x] for x in oth if pas[x] == k]
            rr[k] = sum(m)/len(m)
        loo.append(rr.get(pas[s], 0.5))
    print("  E leave-one-out (out-of-sample category PD-rate): %.4f" % auc_pairs(loo, [lab[s] for s in spk]))

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "probe":
    stage_probe()

def stage_probe2():
    """Does passage identity actually explain the 0.7634? Stratify on it."""
    import numpy as np
    from scipy import stats as st
    man = rows(MANIFEST); lab = {r["speaker"]: int(r["label"]) for r in man}
    spk = sorted(lab)
    stored = {r["speaker_id"]: r["text"] for r in rows(ASRSTORE)}
    pas = {s: passage_of(stored[s]) for s in spk}
    pt  = {r["speaker_id"]: float(r["p_yes"]) for r in rows(TEXTSC)}
    pa  = {r["speaker"]:    float(r["p_yes"]) for r in rows(AUDSC)}

    a = sum(1 for s in spk if pas[s] == "north_wind" and lab[s] == 1)
    b = sum(1 for s in spk if pas[s] == "north_wind" and lab[s] == 0)
    c = sum(1 for s in spk if pas[s] != "north_wind" and lab[s] == 1)
    d = sum(1 for s in spk if pas[s] != "north_wind" and lab[s] == 0)
    orr, p = st.fisher_exact([[a, b], [c, d]])
    print(f"north_wind vs rest  PD/HC = {a}/{b} vs {c}/{d}   Fisher OR={orr:.3f} p={p:.4f}")

    for grp in ["north_wind", "NOT north_wind"]:
        sel = [s for s in spk if (pas[s] == "north_wind") == (grp == "north_wind")]
        ll = [lab[s] for s in sel]
        if len(set(ll)) < 2: continue
        print(f"  within {grp:15s} n={len(sel):2d} (PD={sum(ll)})  "
              f"AUC_text={auc_pairs([pt[s] for s in sel], ll):.4f}  "
              f"AUC_audio={auc_pairs([pa[s] for s in sel], ll):.4f}")
    # stratified (passage-weighted) AUC for the text condition
    tot = 0.0; wsum = 0.0
    for grp in [True, False]:
        sel = [s for s in spk if (pas[s] == "north_wind") == grp]
        ll = [lab[s] for s in sel]
        n1, n0 = sum(ll), len(ll) - sum(ll)
        if n1 and n0:
            tot += auc_pairs([pt[s] for s in sel], ll) * n1 * n0; wsum += n1 * n0
    print(f"  passage-STRATIFIED text AUC = {tot/wsum:.4f}   (unstratified 0.7634)")
    # how much of the text p_yes is explained by passage at all
    y = [pt[s] for s in spk]; x = [1.0 if pas[s] == "north_wind" else 0.0 for s in spk]
    print(f"  Spearman(p_yes_text, is_north_wind) = {spearman(x, y):.4f}")
    print(f"  AUC of is_north_wind for predicting p_yes_text above its median = "
          f"{auc_pairs(x, [1 if v > np.median(y) else 0 for v in y]):.4f}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "probe2":
    stage_probe2()
