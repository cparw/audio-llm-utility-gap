#!/usr/local/bin/python3
"""
PART16 / M1 - Pitt transcript-only, Qwen2.5-Omni, split by arm.

Recomputes from per-clip scores with the rank formula. Nothing is copied out of a json.
Bootstrap: 2000 draws, numpy.random.default_rng(0) reseeded per cell, SPEAKERS resampled
with replacement, percentile 2.5 / 97.5. Paired quantities draw the speaker list ONCE per
replicate and recompute both sides inside that draw.
"""
import hashlib, json, os, sys
import numpy as np
import pandas as pd
import scipy.stats as st

OUT = "<local data dir>/Desktop/release/edaic_rerun/part16"
ROWS = os.path.join(OUT, "rows", "M1.tsv")
DISC = "<local data dir>/Desktop/release/edaic_rerun/DISCREPANCIES.md"

TEXT_PRIMARY = "<local data dir>/paper1_local_runs/omni_pitt_text.csv"
TEXT_ALT     = "<local data dir>/Desktop/release/overnight2/text_new/o25_pitt_text.csv"
TEXT_ALT2    = "<local data dir>/Desktop/release/podD2_final/out/o25_pitt_text.csv"
AUDIO        = "<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv"
MANIFEST     = "manifests/pitt_conflict_manifest_468.csv"
SEP18_BOOT   = "<local data dir>/Desktop/release/overnight/bootstrap_intervals.csv"

NDRAW = 2000


def sha256_1mb(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest()


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pct(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def boot_speaker(df, fns, seed=0, ndraw=NDRAW):
    """fns: dict name -> function(sub_dataframe)->float.
    One speaker draw per replicate, every fn evaluated inside that same draw."""
    rng = np.random.default_rng(seed)
    spk = df["spk"].to_numpy()
    uniq = np.unique(spk)
    idx_by_spk = {s: np.flatnonzero(spk == s) for s in uniq}
    acc = {k: [] for k in fns}
    for _ in range(ndraw):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_spk[s] for s in pick])
        sub = df.iloc[rows]
        for k, fn in fns.items():
            try:
                acc[k].append(fn(sub))
            except Exception:
                acc[k].append(float("nan"))
    return acc


def load_text(path):
    t = pd.read_csv(path)
    if "id" in t.columns:          # Sep 15 local-run layout
        t = t.rename(columns={"id": "name", "speaker": "spk"})
    elif "speaker_id" in t.columns:  # Sep 20 pod layout
        t = t.rename(columns={"speaker_id": "spk", "id": "name"})
    t["key"] = t["name"].astype(str).str.replace(r"^segments/", "", regex=True)
    return t


def main():
    # ---------- load ----------
    txt = load_text(TEXT_PRIMARY)
    alt = load_text(TEXT_ALT)
    aud = pd.read_csv(AUDIO).rename(columns={"clip": "key", "speaker": "spk"})
    man = pd.read_csv(MANIFEST)
    man["key"] = man["segment_path"].astype(str).str.replace(r"^segments/", "", regex=True)

    # ---------- join to the arm labels ----------
    j = txt[["key", "spk", "label", "p_yes"]].rename(columns={"p_yes": "p_yes_text"})
    j = j.merge(man[["key", "set", "label", "spk", "grp"]].rename(
        columns={"set": "arm", "label": "label_man", "spk": "spk_man"}), on="key", how="inner")
    joined_manifest = len(j)
    set_agree = int((txt.set_index("key")["set"].reindex(j["key"]).to_numpy() == j["arm"].to_numpy()).sum())
    label_agree = int((j["label"] == j["label_man"]).sum())

    j = j.merge(aud[["key", "p_yes_text" if False else "p_yes", "label", "spk"]].rename(
        columns={"p_yes": "p_yes_audio", "label": "label_aud", "spk": "spk_aud"}), on="key", how="inner")
    joined_audio = len(j)
    spk_agree = int((j["spk"] == j["spk_aud"]).sum())
    label_aud_agree = int((j["label"] == j["label_aud"]).sum())

    j = j.merge(alt[["key", "p_yes"]].rename(columns={"p_yes": "p_yes_text_alt"}), on="key", how="left")
    j = j.sort_values("key").reset_index(drop=True)

    n_all = len(j)
    n_spk_all = j["spk"].nunique()
    conf = j[j["arm"] == "conflict"].reset_index(drop=True)
    agre = j[j["arm"] == "agreement"].reset_index(drop=True)

    print("=" * 100)
    print("M1  Pitt transcript-only, Qwen2.5-Omni, split by arm")
    print("=" * 100)
    print(f"join: text->manifest {joined_manifest}/468, arm column agrees on {set_agree}/468, "
          f"label agrees on {label_agree}/468")
    print(f"join: +audio {joined_audio}/468, speaker agrees on {spk_agree}/468, "
          f"label agrees on {label_aud_agree}/468")
    print(f"n_all={n_all} n_spk_all={n_spk_all} | conflict n={len(conf)} n_spk={conf['spk'].nunique()} "
          f"| agreement n={len(agre)} n_spk={agre['spk'].nunique()}")
    print()

    results = []   # (rowid, what, value, lo, hi, n, n_spk, file)

    def cell(rowid, what, value, lo, hi, n, nspk, f, usable=None):
        results.append((rowid, what, value, lo, hi, n, nspk, f, usable))

    # ---------- whole-set AUCs (unpaired, own speaker universe) ----------
    for tag, col, srcfile in [("text", "p_yes_text", TEXT_PRIMARY),
                              ("audio", "p_yes_audio", AUDIO)]:
        for armname, sub in [("all468", j), ("conflict", conf), ("agreement", agre)]:
            v = auc(sub["label"], sub[col])
            acc = boot_speaker(sub, {"a": lambda d, c=col: auc(d["label"], d[c])}, seed=0)
            lo, hi, usable = pct(acc["a"])
            cell(f"M1_{tag}_{armname}",
                 f"Pitt Qwen2.5-Omni {'transcript-only' if tag=='text' else 'audio zero-shot'} "
                 f"answer AUC, {armname} arm",
                 v, lo, hi, len(sub), sub["spk"].nunique(), srcfile, usable)

    # ---------- paired conflict minus agreement (one speaker draw over all 228) ----------
    def arm_diff(col):
        def f(d):
            c = d[d["arm"] == "conflict"]; a = d[d["arm"] == "agreement"]
            return auc(c["label"], c[col]) - auc(a["label"], a[col])
        return f

    for tag, col, srcfile in [("text", "p_yes_text", TEXT_PRIMARY),
                              ("audio", "p_yes_audio", AUDIO)]:
        v = arm_diff(col)(j)
        acc = boot_speaker(j, {"d": arm_diff(col)}, seed=0)
        lo, hi, usable = pct(acc["d"])
        cell(f"M1_{tag}_conflict_minus_agreement",
             f"Pitt Qwen2.5-Omni {'transcript-only' if tag=='text' else 'audio zero-shot'} "
             f"paired conflict minus agreement AUC",
             v, lo, hi, n_all, n_spk_all, srcfile, usable)

    # ---------- paired transcript minus audio, per arm (both inside one draw) ----------
    for armname, sub in [("all468", j), ("conflict", conf), ("agreement", agre)]:
        v = auc(sub["label"], sub["p_yes_text"]) - auc(sub["label"], sub["p_yes_audio"])
        acc = boot_speaker(sub, {"d": lambda d: auc(d["label"], d["p_yes_text"]) -
                                                auc(d["label"], d["p_yes_audio"])}, seed=0)
        lo, hi, usable = pct(acc["d"])
        cell(f"M1_text_minus_audio_{armname}",
             f"Pitt Qwen2.5-Omni paired transcript minus audio AUC, {armname} arm",
             v, lo, hi, len(sub), sub["spk"].nunique(),
             os.path.join(OUT, "M1_pitt_text_arms.csv"), usable)

    # ---------- the competing Sep 20 transcript file, same three arms ----------
    for armname, sub in [("all468", j), ("conflict", conf), ("agreement", agre)]:
        v = auc(sub["label"], sub["p_yes_text_alt"])
        acc = boot_speaker(sub, {"a": lambda d: auc(d["label"], d["p_yes_text_alt"])}, seed=0)
        lo, hi, usable = pct(acc["a"])
        cell(f"M1_ALT_text_{armname}",
             f"ALT Pitt Qwen2.5-Omni transcript-only answer AUC, {armname} arm "
             f"(Sep 20 pod rerun, different prompt run)",
             v, lo, hi, len(sub), sub["spk"].nunique(), TEXT_ALT, usable)

    # ---------- print ----------
    for rowid, what, v, lo, hi, n, nspk, f, usable in results:
        print(f"{rowid:38s} {what}")
        print(f"    AUC={v:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={n}  n_speakers={nspk}  "
              f"usable_draws={usable}/{NDRAW}")
        print(f"    file: {f}")
    print()

    # ---------- per-clip csv ----------
    per = j[["key", "spk", "label", "arm", "p_yes_text", "p_yes_audio", "p_yes_text_alt"]].copy()
    per = per.rename(columns={"key": "name"})
    per_path = os.path.join(OUT, "M1_pitt_text_arms.csv")
    per.to_csv(per_path, index=False)

    # ---------- rows fragment ----------
    with open(ROWS, "w") as fh:
        fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
        for rowid, what, v, lo, hi, n, nspk, f, _u in results:
            fh.write(f"{rowid}\t{what}\t{v:.4f}\t{lo:.4f}\t{hi:.4f}\t{n}\t{nspk}\t{f}\n")

    # ---------- sidecar ----------
    srcs = [TEXT_PRIMARY, TEXT_ALT, TEXT_ALT2, AUDIO, MANIFEST, SEP18_BOOT]
    sidecar = {
        "task": "PART16 / M1 - Pitt transcript-only, Qwen2.5-Omni, split by arm",
        "date_run": "2026-09-22",
        "source_files": {p: {"sha256_first_1mb": sha256_1mb(p),
                             "mtime": pd.Timestamp(os.path.getmtime(p), unit="s", tz="UTC")
                                       .tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M:%S %Z"),
                             "bytes": os.path.getsize(p)} for p in srcs},
        "text_file_used": TEXT_PRIMARY,
        "text_file_used_reason": (
            "This is the per-clip file the Sep 18 bootstrap tables point at. "
            "overnight/bootstrap_intervals.csv rows transcript_auc / pitt_conflict_auc / "
            "pitt_agreement_auc / pitt_conflict_minus_agreement for Qwen2.5-Omni pitt all name "
            "this exact path, and overnight/bootstrap_conflict_detail.csv row "
            "'Qwen2.5-Omni,transcript' names it too. Its run log "
            "<local data dir>/paper1_local_runs/logs/omni_pitt_text.log records "
            "'RESULT omni text pitt_text.csv n=468 AUC=0.697 agreement=0.799 conflict=0.455'."),
        "n": int(n_all),
        "n_speakers": int(n_spk_all),
        "n_conflict": int(len(conf)), "n_speakers_conflict": int(conf["spk"].nunique()),
        "n_agreement": int(len(agre)), "n_speakers_agreement": int(agre["spk"].nunique()),
        "join": {"text_to_manifest": joined_manifest, "arm_column_agreement": set_agree,
                 "label_agreement_manifest": label_agree, "plus_audio": joined_audio,
                 "speaker_agreement_audio": spk_agree, "label_agreement_audio": label_aud_agree},
        "seed": 0, "bootstrap_draws": NDRAW, "bootstrap_unit": "speaker, with replacement",
        "bootstrap_percentiles": [2.5, 97.5],
        "model_id": "Qwen/Qwen2.5-Omni-7B",
        "prompt_text": ("Based only on this transcript of what the person said, does this speaker "
                        "show signs of dementia? Answer with one word, Yes or No."),
        "prompt_audio": ("Based only on this recording, does this speaker show signs of dementia? "
                         "Answer with one word, Yes or No."),
        "folds": ("not applicable - no model is fitted in this task; both streams are prompted "
                  "zero-shot answer probabilities read off disk, so no fold assignment is used"),
        "auc_method": "rank formula, scipy.stats.rankdata, ties get average rank",
        "shell_command": "/usr/local/bin/python3 " + os.path.join(OUT, "M1_pitt_text_arms.py"),
        "outputs": {"per_clip_csv": per_path, "rows_tsv": ROWS,
                    "sidecar": os.path.join(OUT, "M1_pitt_text_arms.sidecar.json")},
        "values": [{"id": r[0], "what": r[1], "value": round(r[2], 6),
                    "lo": round(r[3], 6), "hi": round(r[4], 6),
                    "n": r[5], "n_spk": r[6], "file": r[7], "usable_draws": r[8]}
                   for r in results],
        "candidate_files_found": [
            {"path": TEXT_PRIMARY, "mtime": "2026-09-15 15:14", "n_rows": 468,
             "columns": "id,speaker,label,set,p_yes,mass", "used": True,
             "note": "Sep 18 bootstrap tables point here; carries its own set column"},
            {"path": TEXT_ALT, "mtime": "2026-09-20 02:38", "n_rows": 468,
             "columns": "speaker_id,id,label,p_yes,answer_mass,prompt", "used": False,
             "note": "Sep 20 pod rerun, same prompt string, json auc 0.6629; reported as ALT rows"},
            {"path": TEXT_ALT2, "mtime": "2026-09-20 02:38", "n_rows": 468,
             "columns": "speaker_id,id,label,p_yes,answer_mass,prompt", "used": False,
             "note": "byte-identical copy of the previous (md5 c5b37ecc5c7de852caa87ec7a5b16e21)"},
            {"path": "scores/pitt_all/o25_pitt_all_text.csv",
             "mtime": "2026-09-22 18:46", "n_rows": 708, "used": False,
             "note": "different clip set (pitt_all, 708 rows), not the 468 conflict manifest"},
            {"path": "<local data dir>/paper1_local_runs/rebuild/omni_pitt_text_v1s_strip.csv",
             "mtime": "2026-09-19 08:04", "n_rows": 468, "used": False,
             "note": "Sep 19 prompt-variant rebuild v1s, auc 0.6377 per summary json"},
            {"path": "<local data dir>/paper1_local_runs/rebuild/omni_pitt_text_v3s_strip.csv",
             "mtime": "2026-09-19 08:09", "n_rows": 468, "used": False,
             "note": "Sep 19 prompt-variant rebuild v3s, auc 0.6442 per summary json"},
            {"path": "<local data dir>/paper1_local_runs/rebuild/omni_pitt_text_v7s_strip.csv",
             "mtime": "2026-09-19 08:15", "n_rows": 468, "used": False,
             "note": "Sep 19 prompt-variant rebuild v7s"},
            {"path": "<local data dir>/Desktop/release/overnight2/text_new/q3o_pitt_text.csv",
             "mtime": "2026-09-20 02:44", "n_rows": 468, "used": False,
             "note": "Qwen3-Omni, not Qwen2.5-Omni"},
        ],
    }
    with open(os.path.join(OUT, "M1_pitt_text_arms.sidecar.json"), "w") as fh:
        json.dump(sidecar, fh, indent=1)

    # ---------- cross-check against the Sep 18 published intervals ----------
    pub = pd.read_csv(SEP18_BOOT)
    pub = pub[(pub["model"] == "Qwen2.5-Omni") & (pub["dataset"] == "pitt")]
    print("cross-check vs Sep 18 overnight/bootstrap_intervals.csv:")
    want = {"transcript_auc": "M1_text_all468",
            "pitt_conflict_auc": "M1_text_conflict",
            "pitt_agreement_auc": "M1_text_agreement",
            "pitt_conflict_minus_agreement": "M1_text_conflict_minus_agreement"}
    byid = {r[0]: r for r in results}
    lines = []
    for k, rid in want.items():
        row = pub[(pub["number"] == k) & (pub["stream"].isin(["text", "transcript"]))]
        if len(row) == 0:
            continue
        pv = float(row.iloc[0]["value"])
        mine = byid[rid][2]
        flag = "MATCH" if abs(pv - mine) < 5e-4 else "DIFF"
        print(f"  {k:32s} published {pv:.4f}  recomputed {mine:.4f}  {flag}")
        lines.append((k, pv, mine, flag))
    print()
    return results, lines, sidecar


if __name__ == "__main__":
    main()
