"""T6: Pitt transcript-only answer AUC by arm on the Sep 20 run (the run Table 1 prints).

Sources (read only):
  S20   Sep 20 Qwen2.5-Omni transcript run   overnight2/text_new/o25_pitt_text.csv
  S15   Sep 15 Qwen2.5-Omni transcript run   paper work/paper1_local_runs/omni_pitt_text.csv
  MAN   Pitt conflict manifest (arm = 'set', key = basename(segment_path))
  AUD   Qwen2.5-Omni audio zero-shot answer  omni_final/omni_pitt_zeroshot_scores.csv

AUC: rank formula, scipy.stats.rankdata (ties get the average rank).
Bootstrap: 2000 draws, a fresh numpy.random.default_rng(0) for every cell, speakers
(np.unique of the Sep 20 speaker_id) drawn with replacement, every clip of a drawn
speaker enters once per draw, percentile 2.5 / 97.5. PAIRED cells draw the speaker
list once per replicate and recompute both quantities inside that replicate.
Arm cells resample the speakers present in that arm; all-468 and cross-arm cells
resample all 228 speakers.
"""
import hashlib, json, os, sys, platform
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st

S20 = "<local data dir>/release/overnight2/text_new/o25_pitt_text.csv"
S15 = "<local data dir>/paper work/paper1_local_runs/omni_pitt_text.csv"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
AUD = "<local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv"
OUT = "<local data dir>/release/edaic_rerun/part17"
CSV_OUT = os.path.join(OUT, "T6_pitt_text_sep20_arms.csv")
JSON_OUT = os.path.join(OUT, "T6_pitt_text_sep20_arms.json")
TSV_OUT = os.path.join(OUT, "rows", "T6.tsv")
B = 2000
SEED = 0


def sha1mb(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read(1 << 20)).hexdigest()


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot(df, fn):
    """df: rows to resample (by df['spk']); fn(sub_df) -> scalar. Fresh rng(0) per call."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(df["spk"].values)
    pos = {u: np.flatnonzero(df["spk"].values == u) for u in uniq}
    vals = []; bad = 0
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[u] for u in pick])
        v = fn(df.iloc[idx])
        if np.isnan(v):
            bad += 1
        else:
            vals.append(v)
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), bad


def main():
    log = []
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)

    d20 = pd.read_csv(S20)
    d15 = pd.read_csv(S15)
    man = pd.read_csv(MAN, dtype={"spk": str})
    aud = pd.read_csv(AUD)

    # ---- 1. Sep 20 file checks
    P(f"S20 rows={len(d20)} unique id={d20['id'].nunique()} speakers={d20['speaker_id'].nunique()} "
      f"labels={d20['label'].value_counts().sort_index().to_dict()} p_yes NaN={int(d20['p_yes'].isna().sum())}")
    prompts = d20["prompt"].unique()
    P(f"S20 distinct prompt strings={len(prompts)}")
    for p in prompts:
        P(f"S20 prompt verbatim: {p!r}")
    assert len(d20) == 468 and d20["id"].nunique() == 468

    # ---- 2. join keys
    d20["key"] = d20["id"].map(os.path.basename)
    d15["key"] = d15["id"].map(os.path.basename)
    man["key"] = man["segment_path"].map(os.path.basename)
    aud["key"] = aud["clip"].map(os.path.basename)
    for n, d in [("S20", d20), ("S15", d15), ("MAN", man), ("AUD", aud)]:
        assert d["key"].is_unique, n
    man["spk_join"] = man["grp"] + man["spk"]

    j = d20.rename(columns={"speaker_id": "spk", "p_yes": "p_yes_sep20", "label": "label_s20"})[
        ["key", "id", "spk", "label_s20", "p_yes_sep20", "answer_mass"]]
    j = j.merge(man[["key", "set", "label", "grp", "spk", "spk_join", "segment_path"]]
                .rename(columns={"spk": "spk_manifest", "label": "label_man", "set": "arm"}),
                on="key", how="inner", validate="1:1")
    P(f"join S20 x manifest on basename: {len(j)} rows; arms={j['arm'].value_counts().to_dict()}")
    j = j.merge(d15[["key", "speaker", "label", "set", "p_yes"]]
                .rename(columns={"speaker": "spk_s15", "label": "label_s15", "set": "arm_s15",
                                 "p_yes": "p_yes_sep15"}), on="key", how="inner", validate="1:1")
    P(f"join + S15 on basename: {len(j)} rows")
    j = j.merge(aud[["key", "speaker", "label", "p_yes"]]
                .rename(columns={"speaker": "spk_aud", "label": "label_aud", "p_yes": "p_yes_audio"}),
                on="key", how="inner", validate="1:1")
    P(f"join + audio on basename: {len(j)} rows")
    assert len(j) == 468

    lab_ok = ((j["label_s20"] == j["label_man"]) & (j["label_s20"] == j["label_s15"])
              & (j["label_s20"] == j["label_aud"]))
    P(f"label agrees S20=manifest=S15=audio on {int(lab_ok.sum())}/468 rows")
    spk_ok_man = (j["spk"] == j["spk_join"])
    spk_ok_other = (j["spk"] == j["spk_s15"]) & (j["spk"] == j["spk_aud"])
    P(f"S20 speaker_id == manifest grp+spk on {int(spk_ok_man.sum())}/468; "
      f"== S15 speaker and audio speaker on {int(spk_ok_other.sum())}/468")
    arm_ok = (j["arm"] == j["arm_s15"])
    P(f"manifest arm == S15 'set' column on {int(arm_ok.sum())}/468")
    grp_lab = ((j["grp"] == "Dementia").astype(int) == j["label_s20"])
    P(f"manifest grp consistent with label on {int(grp_lab.sum())}/468")
    both = man.groupby("spk")["grp"].nunique()
    P(f"manifest bare spk numbers in both Control and Dementia: {sorted(both[both > 1].index.tolist())}")
    assert lab_ok.all() and spk_ok_man.all() and spk_ok_other.all() and arm_ok.all()

    j = j.rename(columns={"label_s20": "label"})
    j = j.sort_values(["arm", "spk", "id"]).reset_index(drop=True)
    conf = j[j["arm"] == "conflict"].reset_index(drop=True)
    agre = j[j["arm"] == "agreement"].reset_index(drop=True)
    P(f"n all={len(j)} spk={j['spk'].nunique()} | conflict n={len(conf)} spk={conf['spk'].nunique()} "
      f"labels={conf['label'].value_counts().sort_index().to_dict()} | agreement n={len(agre)} "
      f"spk={agre['spk'].nunique()} labels={agre['label'].value_counts().sort_index().to_dict()} | "
      f"speakers in both arms={len(set(conf['spk']) & set(agre['spk']))}")

    j[["id", "spk", "label", "arm", "p_yes_sep20", "p_yes_sep15", "p_yes_audio"]].to_csv(CSV_OUT, index=False)

    # ---- 3/4. cells
    def A(col, arm=None):
        def f(d):
            if arm is not None:
                d = d[d["arm"] == arm]
            return auc(d["label"].values, d[col].values)
        return f

    def minus(f, g):
        return lambda d: f(d) - g(d)

    rows = []
    def cell(cid, what, df, fn, file):
        v = fn(df)
        lo, hi, bad = boot(df, fn)
        rows.append(dict(id=cid, what=what, value=v, lo=lo, hi=hi, n=len(df),
                         n_spk=df["spk"].nunique(), file=file, degenerate_draws=bad))
        P(f"{cid}\t{v:.4f} [{lo:.4f}, {hi:.4f}]\tn={len(df)} n_speakers={df['spk'].nunique()}"
          f"\tdegenerate_draws={bad}\t{file}")
        return v

    runs = [("s20", "Sep 20 run", "p_yes_sep20", S20), ("s15", "Sep 15 run", "p_yes_sep15", S15)]
    for tag, lab, col, src in runs:
        cell(f"T6_{tag}_text_all468", f"Pitt Qwen2.5-Omni transcript-only answer AUC, all 468 ({lab})",
             j, A(col), src)
        cell(f"T6_{tag}_text_conflict", f"Pitt Qwen2.5-Omni transcript-only answer AUC, conflict arm ({lab})",
             conf, A(col), src)
        cell(f"T6_{tag}_text_agreement", f"Pitt Qwen2.5-Omni transcript-only answer AUC, agreement arm ({lab})",
             agre, A(col), src)
        cell(f"T6_{tag}_text_conflict_minus_agreement",
             f"Pitt Qwen2.5-Omni transcript-only PAIRED conflict minus agreement AUC ({lab})",
             j, minus(A(col, "conflict"), A(col, "agreement")), src)

    cell("T6_audio_all468", "Pitt Qwen2.5-Omni audio zero-shot answer AUC, all 468", j, A("p_yes_audio"), AUD)
    cell("T6_audio_conflict", "Pitt Qwen2.5-Omni audio zero-shot answer AUC, conflict arm", conf, A("p_yes_audio"), AUD)
    cell("T6_audio_agreement", "Pitt Qwen2.5-Omni audio zero-shot answer AUC, agreement arm", agre, A("p_yes_audio"), AUD)
    cell("T6_audio_conflict_minus_agreement", "Pitt Qwen2.5-Omni audio zero-shot PAIRED conflict minus agreement AUC",
         j, minus(A("p_yes_audio", "conflict"), A("p_yes_audio", "agreement")), AUD)

    for tag, lab, col, src in runs:
        cell(f"T6_{tag}_text_minus_audio_all468", f"PAIRED transcript ({lab}) minus audio AUC, all 468",
             j, minus(A(col), A("p_yes_audio")), CSV_OUT)
        cell(f"T6_{tag}_text_minus_audio_conflict", f"PAIRED transcript ({lab}) minus audio AUC, conflict arm",
             conf, minus(A(col), A("p_yes_audio")), CSV_OUT)
        cell(f"T6_{tag}_text_minus_audio_agreement", f"PAIRED transcript ({lab}) minus audio AUC, agreement arm",
             agre, minus(A(col), A("p_yes_audio")), CSV_OUT)
        cell(f"T6_{tag}_gap_text_minus_gap_audio",
             f"PAIRED (transcript {lab} arm gap) minus (audio arm gap), gap = conflict minus agreement",
             j, lambda d, c=col: (A(c, "conflict")(d) - A(c, "agreement")(d))
                                 - (A("p_yes_audio", "conflict")(d) - A("p_yes_audio", "agreement")(d)), CSV_OUT)

    cell("T6_s20_minus_s15_all468", "PAIRED Sep 20 minus Sep 15 transcript AUC, all 468",
         j, minus(A("p_yes_sep20"), A("p_yes_sep15")), CSV_OUT)
    cell("T6_s20_minus_s15_conflict", "PAIRED Sep 20 minus Sep 15 transcript AUC, conflict arm",
         conf, minus(A("p_yes_sep20"), A("p_yes_sep15")), CSV_OUT)
    cell("T6_s20_minus_s15_agreement", "PAIRED Sep 20 minus Sep 15 transcript AUC, agreement arm",
         agre, minus(A("p_yes_sep20"), A("p_yes_sep15")), CSV_OUT)
    cell("T6_s20_minus_s15_armgap", "PAIRED Sep 20 arm gap minus Sep 15 arm gap (transcript)",
         j, lambda d: (A("p_yes_sep20", "conflict")(d) - A("p_yes_sep20", "agreement")(d))
                      - (A("p_yes_sep15", "conflict")(d) - A("p_yes_sep15", "agreement")(d)), CSV_OUT)

    # gap ratios (point only)
    R = {r["id"]: r["value"] for r in rows}
    ratio20 = R["T6_s20_text_conflict_minus_agreement"] / R["T6_audio_conflict_minus_agreement"]
    ratio15 = R["T6_s15_text_conflict_minus_agreement"] / R["T6_audio_conflict_minus_agreement"]
    P(f"arm gap ratio transcript/audio: Sep 20 {ratio20:.4f}  Sep 15 {ratio15:.4f} (point only)")

    # ---- 6. run to run per-clip agreement
    x = j["p_yes_sep20"].values; y = j["p_yes_sep15"].values
    r_p = float(st.pearsonr(x, y)[0]); r_s = float(st.spearmanr(x, y)[0])
    ad = np.abs(x - y); imax = int(np.argmax(ad))
    mass15 = d15.set_index("key").loc[j["id"].map(os.path.basename), "mass"].values
    P(f"per-clip p_yes Sep20 vs Sep15: Pearson r={r_p:.4f} Spearman rho={r_s:.4f} max|diff|={ad.max():.4f} "
      f"(clip {j['id'].iloc[imax]}: {x[imax]:.4f} vs {y[imax]:.4f}) mean|diff|={ad.mean():.4f} "
      f"median|diff|={np.median(ad):.4f} clips |diff|>0.1: {int((ad > 0.1).sum())}/468")
    P(f"mean p_yes Sep20={x.mean():.4f} Sep15={y.mean():.4f}; median answer mass Sep20={np.median(j['answer_mass']):.4f} "
      f"Sep15={np.median(mass15):.4f}")
    for cid, what, v in [
        ("T6_perclip_pearson_s20_s15", "Per-clip p_yes Pearson r, Sep 20 vs Sep 15 transcript run", r_p),
        ("T6_perclip_spearman_s20_s15", "Per-clip p_yes Spearman rho, Sep 20 vs Sep 15 transcript run", r_s),
        ("T6_perclip_maxabsdiff_s20_s15", "Per-clip p_yes max abs diff, Sep 20 vs Sep 15 transcript run", float(ad.max())),
        ("T6_perclip_meanabsdiff_s20_s15", "Per-clip p_yes mean abs diff, Sep 20 vs Sep 15 transcript run", float(ad.mean())),
        ("T6_armgap_ratio_s20", "Point ratio transcript arm gap / audio arm gap, Sep 20 run", ratio20),
        ("T6_armgap_ratio_s15", "Point ratio transcript arm gap / audio arm gap, Sep 15 run", ratio15),
    ]:
        rows.append(dict(id=cid, what=what, value=v, lo=None, hi=None, n=468, n_spk=228, file=CSV_OUT,
                         degenerate_draws=None))

    # ---- sensitivity: cluster on manifest participant (227) rather than speaker id (228)
    jp = j.copy(); jp["spk"] = jp["spk_manifest"]
    sens = {}
    for cid, df, fn in [
        ("conflict", jp[jp["arm"] == "conflict"].reset_index(drop=True), A("p_yes_sep20")),
        ("agreement", jp[jp["arm"] == "agreement"].reset_index(drop=True), A("p_yes_sep20")),
        ("conflict_minus_agreement", jp, minus(A("p_yes_sep20", "conflict"), A("p_yes_sep20", "agreement"))),
    ]:
        lo, hi, bad = boot(df, fn)
        sens[cid] = dict(value=fn(df), lo=lo, hi=hi, n_clusters=int(df["spk"].nunique()))
        P(f"sensitivity (cluster = bare manifest spk) Sep 20 {cid}: {fn(df):.4f} [{lo:.4f}, {hi:.4f}] "
          f"clusters={df['spk'].nunique()}")

    # ---- fragment
    with open(TSV_OUT, "w") as f:
        f.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
        for r in rows:
            fmt = lambda v: "" if v is None else f"{v:.4f}"
            f.write(f"{r['id']}\t{r['what']}\t{fmt(r['value'])}\t{fmt(r['lo'])}\t{fmt(r['hi'])}\t"
                    f"{r['n']}\t{r['n_spk']}\t{r['file']}\n")

    side = {
        "task": "PART17 T6 Pitt transcript-only by arm on the Sep 20 run",
        "sources": {k: {"path": p, "sha256_first_1MB": sha1mb(p), "bytes": os.path.getsize(p)}
                    for k, p in [("sep20_text", S20), ("sep15_text", S15), ("manifest", MAN), ("audio_zeroshot", AUD)]},
        "sep20_prompt_verbatim": [str(p) for p in prompts],
        "sep15_prompt": "not recorded in the Sep 15 csv (no prompt column); the Sep 15 scoring script is lost per "
                        "paper1_local_runs/omni_text_score.py header",
        "join": "basename(id) = basename(manifest segment_path) = basename(Sep 15 id) = audio clip; "
                "Sep 20 speaker_id equals manifest grp + spk (e.g. 'Control' + '015' = 'Control015') on all rows",
        "n": int(len(j)), "n_speakers": int(j["spk"].nunique()),
        "n_conflict": int(len(conf)), "n_speakers_conflict": int(conf["spk"].nunique()),
        "n_agreement": int(len(agre)), "n_speakers_agreement": int(agre["spk"].nunique()),
        "seed": SEED, "bootstrap_draws": B, "bootstrap_percentiles": [2.5, 97.5],
        "bootstrap_unit": "speaker_id from the Sep 20 file (228); arm cells resample speakers present in that arm",
        "auc_method": "rank formula, scipy.stats.rankdata, ties get average rank",
        "sensitivity_cluster_manifest_spk_227": sens,
        "outputs": {"perclip_csv": CSV_OUT, "fragment": TSV_OUT, "sidecar": JSON_OUT, "script": os.path.abspath(__file__)},
        "command": "/usr/local/bin/python3 " + os.path.abspath(__file__),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(),
        "rows": rows,
        "log": log,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(side, f, indent=1, default=float)


if __name__ == "__main__":
    main()
