#!/usr/local/bin/python3
"""PART20 POD3c analysis: Qwen3-Omni transcript-only vs Qwen3-Omni audio on the 483 E-DAIC pairs.
AUC by rank formula (scipy rankdata, ties averaged). Bootstrap: 2000 draws, fresh
numpy.random.default_rng(0) per cell, speakers resampled with replacement, percentile 2.5/97.5.
Paired quantities: one speaker draw per replicate, both sides recomputed in that draw.
Degenerate replicates (one class only) are dropped and counted (n_valid in the sidecar)."""
import hashlib, json, os, sys
import numpy as np, pandas as pd, scipy.stats as st

MAC = "scores/part20/POD3c"
TEXT_RAW = os.path.join(MAC, "p14_q3o_text_raw.csv")
TEXT_RUN = os.path.join(MAC, "p14_q3o_text_raw_run.json")
AUDIO = "<local data dir>/release/edaic_rerun/part14/p14_q3o_zeroshot_scores.csv"
MANIFEST = "<local data dir>/release/edaic_rerun/part14_manifest_new.csv"
TRANSCRIPTS = "<local data dir>/scratch/p15/p14_text_input.csv"
O25_TEXT = "<local data dir>/release/edaic_rerun/part16/pod_sync/p14_o25_text.csv"
ROWS = "reports/part20/rows/POD3c_provisional.tsv"
NDRAW = 2000


def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest()


def auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def boot(df, fns):
    rng = np.random.default_rng(0)
    spk = df["spk"].to_numpy(); uniq = np.unique(spk)
    idx = {s: np.flatnonzero(spk == s) for s in uniq}
    acc = {k: [] for k in fns}
    for _ in range(NDRAW):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sub = df.iloc[np.concatenate([idx[s] for s in pick])]
        for k, f in fns.items():
            acc[k].append(f(sub))
    out = {}
    for k, v in acc.items():
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v)))
    return out


def main():
    t = pd.read_csv(TEXT_RAW)
    a = pd.read_csv(AUDIO)
    m = pd.read_csv(MANIFEST)
    tr = pd.read_csv(TRANSCRIPTS)
    assert len(t) == len(a) == len(m) == len(tr) == 966
    a_id = a["clip"].str.replace(".wav", "", regex=False).to_numpy()
    m_id = (m["set"] + "_" + m["speaker_id"].astype(str) + "_" + m["seg_uid"]).to_numpy()
    assert (t["id"].to_numpy() == a_id).all() and (t["id"].to_numpy() == m_id).all() and (t["id"].to_numpy() == tr["id"].to_numpy()).all()
    assert (t["label"].to_numpy() == a["label"].to_numpy()).all()
    assert (t["set"].to_numpy() == m["set"].to_numpy()).all()
    assert (t["speaker_id"].astype(str).to_numpy() == a["speaker"].astype(str).to_numpy()).all()
    df = pd.DataFrame({"id": t["id"], "spk": t["speaker_id"].astype(str), "set": t["set"], "seg_uid": m["seg_uid"],
                       "pair_id": m["pair_id"], "y": t["label"].astype(int), "pt": t["p_yes"].astype(float),
                       "mass_t": t["answer_mass"].astype(float), "pa": a["p_yes"].astype(float),
                       "mass_a": a["mass"].astype(float), "pt_ts": t["p_yes_ts"].astype(float)})
    # identical transcripts must give identical p_yes for repeated agreement rows
    g = df[df["set"] == "agreement"].groupby("seg_uid")
    assert g["pt"].nunique().max() == 1 and g["pa"].nunique().max() == 1
    df["first_agree"] = False
    ag = df[df["set"] == "agreement"]
    df.loc[ag.drop_duplicates("seg_uid", keep="first").index, "first_agree"] = True

    C = df[df["set"] == "conflict"].reset_index(drop=True)
    A483 = df[df["set"] == "agreement"].reset_index(drop=True)
    A311 = df[df["first_agree"]].reset_index(drop=True)
    assert len(C) == 483 and len(A483) == 483 and len(A311) == 311

    results = []

    def cell(rid, what, sub_df, fn, kind="auc"):
        v = fn(sub_df)
        lo, hi, nv = boot(sub_df, {"x": fn})["x"]
        results.append(dict(id=rid, what=what, value=v, lo=lo, hi=hi, n=len(sub_df), n_spk=sub_df["spk"].nunique(),
                            n_valid=nv, kind=kind))

    for col, mod in (("pt", "transcript"), ("pa", "audio")):
        for nm, d in (("conflict483", C), ("agreement483", A483), ("agreement311", A311)):
            cell(f"q3o_{mod}_{nm}", f"Qwen3-Omni {mod} AUC, {nm}", d, lambda s, c=col: auc(s["y"], s[c]))

    # paired conflict minus agreement, one speaker draw over the union of both arms
    def arm_diff(col, agree_mask):
        def f(s):
            c = s[s["set"] == "conflict"]; g_ = s[agree_mask(s)]
            return auc(c["y"], c[col]) - auc(g_["y"], g_[col])
        return f
    both483 = df.reset_index(drop=True)
    both311 = df[(df["set"] == "conflict") | df["first_agree"]].reset_index(drop=True)
    for col, mod in (("pt", "transcript"), ("pa", "audio")):
        cell(f"q3o_{mod}_conf_minus_agree483", f"Qwen3-Omni {mod}, conflict483 minus agreement483 (paired)",
             both483, arm_diff(col, lambda s: s["set"] == "agreement"), "diff")
        cell(f"q3o_{mod}_conf_minus_agree311", f"Qwen3-Omni {mod}, conflict483 minus agreement311 (paired)",
             both311, arm_diff(col, lambda s: s["first_agree"]), "diff")
    # paired transcript minus audio, per arm
    for nm, d in (("conflict483", C), ("agreement483", A483), ("agreement311", A311)):
        cell(f"q3o_text_minus_audio_{nm}", f"Qwen3-Omni transcript minus audio AUC, {nm} (paired)", d,
             lambda s: auc(s["y"], s["pt"]) - auc(s["y"], s["pa"]), "diff")

    # agreement between transcript and audio p_yes (no bootstrap; descriptive)
    desc = {}
    for nm, d in (("all966", df), ("distinct794", df[(df["set"] == "conflict") | df["first_agree"]]),
                  ("conflict483", C), ("agreement483", A483), ("agreement311", A311)):
        r, pv = st.pearsonr(d["pt"], d["pa"])
        same = float(((d["pt"] >= 0.5) == (d["pa"] >= 0.5)).mean())
        desc[nm] = dict(n=len(d), n_spk=int(d["spk"].nunique()), pearson_r=float(r), pearson_p=float(pv),
                        same_decision_share=same, text_yes_rate=float((d["pt"] >= 0.5).mean()),
                        audio_yes_rate=float((d["pa"] >= 0.5).mean()),
                        median_mass_text=float(d["mass_t"].median()), min_mass_text=float(d["mass_t"].min()))

    # sensitivity: text_score.py token set (adds Yeah/Nope) vs rule-3 set
    sens = {"max_abs_p_yes_ts_minus_rule3": float((df["pt_ts"] - df["pt"]).abs().max()),
            "conflict483_auc_ts": auc(C["y"], C["pt_ts"]), "agreement483_auc_ts": auc(A483["y"], A483["pt_ts"]),
            "agreement311_auc_ts": auc(A311["y"], A311["pt_ts"]), "all966_auc_rule3": auc(df["y"], df["pt"]),
            "all966_auc_ts": auc(df["y"], df["pt_ts"])}

    # per-clip csv
    out_csv = os.path.join(MAC, "p14_q3o_text.csv")
    pc = df[["spk", "id", "set", "seg_uid", "pair_id", "y", "pt", "mass_t", "pt_ts", "pa", "mass_a", "first_agree"]].rename(
        columns={"spk": "speaker_id", "y": "label", "pt": "p_yes_text", "mass_t": "answer_mass_text",
                 "pt_ts": "p_yes_text_textscore_tokens", "pa": "p_yes_audio_q3o", "mass_a": "answer_mass_audio_q3o",
                 "first_agree": "in_agreement311_distinct"})
    pc["prompt_text"] = t["prompt"]
    pc["prompt_audio"] = a["prompt"]
    pc.to_csv(out_csv, index=False)

    run = json.load(open(TEXT_RUN))
    srcs = [TEXT_RAW, AUDIO, MANIFEST, TRANSCRIPTS, O25_TEXT]
    side = {
        "result": "Qwen3-Omni-30B-A3B-Instruct transcript-only zero-shot on E-DAIC part14 pairs, beside Qwen3-Omni audio",
        "per_clip_csv": out_csv, "n": 966, "n_speakers": int(df["spk"].nunique()),
        "arms": {"conflict483": [len(C), int(C["spk"].nunique())], "agreement483": [len(A483), int(A483["spk"].nunique())],
                 "agreement311_distinct_seg_uid_first_occurrence": [len(A311), int(A311["spk"].nunique())]},
        "model_id": run["model"], "dtype": run["dtype"], "transformers": run["transformers"], "torch": run["torch"], "gpu": run["gpu"],
        "prompt_verbatim": run["prompt"],
        "user_turn_template": 'Transcript: "<text>"\\n\\n<prompt>  (no system message, add_generation_prompt=True; same as p15/text_score.py)',
        "rendered_example_first_clip": run["rendered_example"],
        "p_yes": "P(Yes)/(P(Yes)+P(No)) at the first answer position; Yes/yes/YES and No/no/NO, bare and leading-space, single-token encodings only",
        "yes_ids": run["yes_ids_rule3"], "no_ids": run["no_ids_rule3"],
        "audio_prompt_verbatim": str(a["prompt"].iloc[0]),
        "seed": 0, "bootstrap": "2000 draws, fresh numpy.random.default_rng(0) per cell, speakers with replacement, percentile 2.5/97.5; paired = one speaker draw per replicate",
        "auc_method": "rank formula, scipy.stats.rankdata, ties averaged",
        "decision_threshold_same_decision": "p_yes >= 0.5 counts as Yes",
        "commands": {
            "pod": "HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python3 /workspace/p20/score_text_p20.py Qwen/Qwen3-Omni-30B-A3B-Instruct /workspace/p20/p14_text_input.csv /workspace/scores/part20/POD3c/p14_q3o_text_raw.csv",
            "validation": "HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 python3 /workspace/p20/score_text_p20.py Qwen/Qwen2.5-Omni-7B /workspace/p20/p14_text_input.csv /workspace/p20/val_o25_first25.csv 25",
            "mac": "python3 " + os.path.abspath(__file__)},
        "sources": {p: {"sha256_first_1MB": sha1mb(p)} for p in srcs},
        "pod_transcript_copy": "/workspace/p20/p14_text_input.csv (sha256 full file 281805c57a39eb44c3dfab0bebca79c9ed87bbc41030608cd96d395bd55783c9, identical to the Mac source)",
        "results": results, "text_vs_audio": desc, "token_set_sensitivity": sens,
    }
    json.dump(side, open(out_csv.replace(".csv", ".json"), "w"), indent=1)

    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    new = not os.path.exists(ROWS)
    with open(ROWS, "a") as fh:
        if new:
            fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
        for r in results:
            if r["kind"] == "auc":
                vs = "above 0.5" if r["value"] > 0.5 else "below 0.5"
            else:
                vs = "interval excludes zero" if (r["lo"] > 0 or r["hi"] < 0) else "interval includes zero"
            two = f"{r['value']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
            fh.write(f"{r['id']}\t{r['what']}\t{r['value']:.4f}\t{r['lo']:.4f}\t{r['hi']:.4f}\t{r['n']}\t{r['n_spk']}\t{out_csv}\t{two}\t{vs}\n")
            print(f"PASTE {r['id']}: {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']} n_spk={r['n_spk']} valid_draws={r['n_valid']} file={out_csv} | {two} {vs}")
    for k, v in desc.items():
        print(f"PASTE text_vs_audio {k}: pearson_r={v['pearson_r']:.4f} same_decision={v['same_decision_share']:.4f} n={v['n']} n_spk={v['n_spk']} text_yes_rate={v['text_yes_rate']:.4f} audio_yes_rate={v['audio_yes_rate']:.4f} median_mass_text={v['median_mass_text']:.4f} min_mass_text={v['min_mass_text']:.4f}")
    print("SENS", json.dumps(sens))


if __name__ == "__main__":
    main()
