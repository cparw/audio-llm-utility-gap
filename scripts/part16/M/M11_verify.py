#!/usr/bin/env python3
"""Independent recomputation of every M11 value from M11_kcl_wer.csv.
Deliberately different code: jiwer for edit distance, an index-based bootstrap
(pre-grouped row indices, one draw reused across both groups), AUC written separately."""
import os, re, numpy as np, pandas as pd, scipy.stats as st
os.environ["HF_HOME"] = "<local data dir>/DementiaBank/hf_cache"
CSV  = "scores/part16/M/M11_kcl_wer.csv"
ROWS = "reports/part16/rows/M11.tsv"
NB, SEED = 2000, 0
d = pd.read_csv(CSV)
out = {}

# 1. WER arithmetic re-derived from stored S/D/I/N
print(f"VERIFY A_wer_from_SDIN max|diff| = {float(np.abs((d.S+d.D+d.I)/d.n_ref_words - d.wer).max()):.10f}")
print(f"VERIFY B_xsys_from_SDIN max|diff| = "
      f"{float(np.abs((d.xsys_S+d.xsys_D+d.xsys_I)/d.xsys_n_ref_words - d.xsys_disagree).max()):.10f}")

# 2. jiwer, independent edit distance on the same normalised strings
try:
    import jiwer
    from transformers import WhisperTokenizerFast
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    tk = WhisperTokenizerFast.from_pretrained("openai/whisper-large-v3",
                                              cache_dir="<local data dir>/DementiaBank/hf_cache/hub")
    base = EnglishTextNormalizer(tk.english_spelling_normalizer)
    NS = re.compile(r"[^a-z0-9 ]+"); RU = re.compile(r"\d+|[a-z]+")
    def nm(t): return " ".join(RU.findall(NS.sub(" ", base("" if pd.isna(t) else str(t)).lower())))
    ref = {r.speaker_id: r.text for r in pd.read_csv(
        "<local data dir>/Desktop/release/part4_text/asr/asr_kcl.csv").itertuples()}
    a = [jiwer.process_words(nm(ref.get(r.spk, "")), nm(r.hyp_whisper_large_v3)) for r in d.itertuples()]
    b = [jiwer.process_words(nm(r.hyp_whisper_large_v3), nm(r.hyp_whisper_medium)) for r in d.itertuples()]
    for tag, o, cS, cD, cI, cW in (("A", a, d.S, d.D, d.I, d.wer),
                                   ("B", b, d.xsys_S, d.xsys_D, d.xsys_I, d.xsys_disagree)):
        S = np.array([x.substitutions for x in o]); D_ = np.array([x.deletions for x in o])
        I = np.array([x.insertions for x in o]); W = np.array([x.wer for x in o])
        print(f"VERIFY {tag}_jiwer  Smax|d|={int(np.abs(S-cS).max())}  Dmax|d|={int(np.abs(D_-cD).max())}  "
              f"Imax|d|={int(np.abs(I-cI).max())}  wer max|d|={float(np.abs(W-cW).max()):.10f}")
except Exception as e:
    print("VERIFY jiwer unavailable ->", repr(e)[:140])

# 3. independent AUC
def auc2(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y)-n1
    return float((st.rankdata(s)[y == 1].sum() - n1*(n1+1)/2.0)/(n1*n0))

# 4. independent bootstrap: one list of speaker draws, reused
spks = np.array(sorted(d.spk.unique()))
rowsof = {s: np.flatnonzero(d.spk.values == s) for s in spks}
rng = np.random.default_rng(SEED)
DRAWS = [np.concatenate([rowsof[s] for s in rng.choice(spks, len(spks), True)]) for _ in range(NB)]
isPD = (d.group.values == "PD"); isHC = (d.group.values == "control")

def ci(col, m):
    v = d[col].values.astype(float); o = [v[i[m[i]]].mean() for i in DRAWS if m[i].any()]
    return np.percentile(o, 2.5), np.percentile(o, 97.5), len(o)
def cidiff(col):
    v = d[col].values.astype(float); o = []
    for i in DRAWS:
        p, h = v[i[isPD[i]]], v[i[isHC[i]]]
        if len(p) and len(h): o.append(p.mean()-h.mean())
    return np.percentile(o, 2.5), np.percentile(o, 97.5), len(o)
def cisp(col, pc):
    x, y = d[col].values.astype(float), d[pc].values.astype(float); o = []
    for i in DRAWS:
        if len(np.unique(x[i])) > 1 and len(np.unique(y[i])) > 1:
            o.append(st.spearmanr(x[i], y[i]).statistic)
    return np.percentile(o, 2.5), np.percentile(o, 97.5), len(o)

print(f"VERIFY A_asr_reproduction_wer_vs_paper_transcript = {d.wer.mean():.4f}  exact={int((d.wer==0).sum())}/{len(d)}")
for key, col in (("B_xsys_disagree","xsys_disagree"), ("C_mean_token_logprob","mean_token_logprob"),
                 ("C_compression_ratio","compression_ratio"), ("D_n_words_30s","n_words_30s")):
    for g, m in (("PD", isPD), ("control", isHC)):
        lo, hi, k = ci(col, m)
        print(f"VERIFY {key}_mean_{g} = {d[col].values[m].mean():.4f} [{lo:.4f}, {hi:.4f}] usable={k}")
    lo, hi, k = cidiff(col)
    print(f"VERIFY {key}_diff_PD_minus_control = "
          f"{d[col].values[isPD].mean()-d[col].values[isHC].mean():.4f} [{lo:.4f}, {hi:.4f}] usable={k}")
    for nm2, pc in (("text","p_yes_text"), ("audio","p_yes_audio")):
        r = st.spearmanr(d[col], d[pc]); lo, hi, k = cisp(col, pc)
        print(f"VERIFY {key}_spearman_vs_pyes_{nm2} = {r.statistic:.4f} [{lo:.4f}, {hi:.4f}] p={r.pvalue:.4f} usable={k}")
    print(f"VERIFY {key}_auc_alone = {auc2(d.label, d[col]):.4f}")
print(f"VERIFY E_auc_from_passage_identity_alone = {auc2(d.label,(d.passage=='north_wind').astype(int)):.4f}")
print(f"VERIFY auc_text_condition_recomputed  = {auc2(d.label, d.p_yes_text):.4f}")
print(f"VERIFY auc_audio_condition_recomputed = {auc2(d.label, d.p_yes_audio):.4f}")
for g, m in (("PD", isPD), ("control", isHC)):
    lo, hi, k = ci("wer_local", m)
    print(f"VERIFY F_wer_vs_local_copy_mean_{g} = {d.wer_local.values[m].mean():.4f} [{lo:.4f}, {hi:.4f}] usable={k}")
