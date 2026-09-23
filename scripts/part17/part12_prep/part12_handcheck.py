#!/usr/local/bin/python3
"""Part 12 hand-check values, recomputed from the raw per-clip / summary files.

Every value written here was computed by this script from a file it opened in this run.
Output: handcheck_values.csv (same columns as value_index.csv), read by build_value_index.py.

What it recomputes
  1. Qwen2.5-Omni transcript-only AUC on Pitt 468 from BOTH runs (Sep 15 local, Sep 20 pod), by arm.
  2. Qwen2.5-Omni transcript-only AUC on ADReSS-2020 from both runs (Sep 15 local, Sep 20 pod).
  3. Qwen2.5-Omni projector fine-tune OOF AUC on Pitt by arm (omnisft_pitt_oof.csv, read only).
  4. Pitt mean age, dementia minus control, several definitions (for "seven years").
  5. eGeMAPS baseline AUC per dataset from the per-clip OOF files, and encoder-probe minus eGeMAPS margins.
  6. Nested encoder / LM / answer-state probe means straight from the *_nested_repeats.json files
     (and the Pitt 5-repeat npz), for Qwen2.5-Omni, Qwen2-Audio and Qwen3-Omni.
  7. Zero-shot answer AUC recomputed from per-clip score files for the three Qwen models.
Nothing is written under omni_final.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12common import (RELEASE, NEW_PREFIX, PREP, auc_rank, read_csv, SIX_DATASETS)  # noqa: E402

OUT = os.path.join(PREP, "handcheck_values.csv")
P = NEW_PREFIX.rstrip("/")
rows = []
log = []


def add(key, model, dataset, stream, arm, value, n, n_spk, estimator, file, lo="", hi="", note=""):
    rows.append(dict(key=key, model=model, dataset=dataset, stream=stream, arm=arm,
                     value=("%.4f" % value) if isinstance(value, float) else value, lo=lo, hi=hi,
                     n=n, n_spk=n_spk, estimator=estimator, file=file, note=note))


def col(r, *names):
    for nm in names:
        if nm in r and r[nm] != "":
            return r[nm]
    return None


def auc_file(path, arm_of=None, want_arm=None, label_col=("label",), score_col=("p_yes",), spk_col=("speaker", "speaker_id", "spk")):
    rs = read_csv(path)
    ys, ss, spk = [], [], set()
    for r in rs:
        if want_arm is not None:
            a = arm_of(r)
            if a != want_arm:
                continue
        y = col(r, *label_col)
        s = col(r, *score_col)
        if y is None or s is None:
            continue
        ys.append(int(float(y)))
        ss.append(float(s))
        spk.add(col(r, *spk_col))
    return auc_rank(ys, ss), len(ys), len(spk)


# ------------------------------------------------------------------ 1. Pitt transcript-only, two runs
runA = P + "/omni_pitt_text.csv"
runB = RELEASE + "/overnight2/text_new/o25_pitt_text.csv"
armA = {r["id"]: r["set"] for r in read_csv(runA)}
for arm in (None, "conflict", "agreement"):
    a, n, ns = auc_file(runA, arm_of=lambda r: r["set"], want_arm=arm)
    add("hc_pitt_text_runA_%s" % (arm or "all"), "Qwen2.5-Omni", "pitt", "transcript", arm or "all", a, n, ns,
        "text only, Run A (local MPS, file mtime Sep 15), rank AUC recomputed here", runA,
        note="Run A; the Sep 18 bootstrap tables point at this file")
    b, n, ns = auc_file(runB, arm_of=lambda r: armA.get(r["id"]), want_arm=arm)
    add("hc_pitt_text_runB_%s" % (arm or "all"), "Qwen2.5-Omni", "pitt", "transcript", arm or "all", b, n, ns,
        "text only, Run B (pod /workspace/text_score.py, file mtime Sep 20), rank AUC recomputed here; arms joined from Run A 'set' by id",
        runB, note="Run B; master_lookup and table1_omni25.csv cite this run's json (0.6629)")
jB = json.load(open(RELEASE + "/overnight2/text_new/o25_pitt_text.json"))
log.append("o25_pitt_text.json auc field = %s (date %s)" % (jB.get("auc"), jB.get("date")))

# ------------------------------------------------------------------ 2. ADReSS-2020 transcript-only, two runs
a20_new = RELEASE + "/overnight2/text_new/o25_adress2020_text.csv"
a20_old = P + "/omni_adress2020_text.csv"
v, n, ns = auc_file(a20_new)
add("hc_adress2020_text_sep20", "Qwen2.5-Omni", "adress2020", "transcript", "all", v, n, ns,
    "text only, pod run Sep 20 (o25_adress2020_text.csv), rank AUC recomputed here", a20_new,
    note="json alongside says auc %s" % json.load(open(a20_new.replace(".csv", ".json"))).get("auc"))
v, n, ns = auc_file(a20_old)
add("hc_adress2020_text_sep15", "Qwen2.5-Omni", "adress2020", "transcript", "all", v, n, ns,
    "text only, local run Sep 15 (omni_adress2020_text.csv), rank AUC recomputed here", a20_old,
    note="older local run; not cited by master_lookup")
for split in ("train", "test"):
    v, n, ns = auc_file(a20_old, arm_of=lambda r: r.get("set"), want_arm=split)
    add("hc_adress2020_text_sep15_%s" % split, "Qwen2.5-Omni", "adress2020", "transcript", split, v, n, ns,
        "text only, local run Sep 15, %s split only" % split, a20_old)

# ------------------------------------------------------------------ 3. Pitt projector fine-tune by arm (read only file)
sft = RELEASE + "/omni_final/omnisft_pitt_oof.csv"
for arm in (None, "conflict", "agreement"):
    v, n, ns = auc_file(sft, arm_of=lambda r: r["set"], want_arm=arm)
    add("hc_pitt_sft_%s" % (arm or "all"), "Qwen2.5-Omni", "pitt", "projector_ft", arm or "all", v, n, ns,
        "projector only fine tune, 5 fold OOF, rank AUC recomputed here", sft)
js = json.load(open(RELEASE + "/omni_final/omnisft_pitt_sft.json"))
log.append("omnisft_pitt_sft.json auc_oof %s arms %s" % (js.get("auc_oof"), js.get("arms")))

# ------------------------------------------------------------------ 4. Pitt age gap ("seven years")
man468 = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
manall = "<local data dir>/DementiaBank/pitt_manifest.csv"
r468 = read_csv(man468)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def age_gap(rs, grp_key, spk_key, pos, neg, age_key="age"):
    clip = {pos: [], neg: []}
    spk = {pos: {}, neg: {}}
    for r in rs:
        g = r[grp_key]
        if g not in clip or r.get(age_key, "") in ("", None):
            continue
        a = float(r[age_key])
        clip[g].append(a)
        spk[g].setdefault(r[spk_key], []).append(a)
    cg = mean(clip[pos]) - mean(clip[neg])
    sp = {g: mean([mean(v) for v in spk[g].values()]) for g in spk}
    first = {g: mean([v[0] for v in spk[g].values()]) for g in spk}
    return (cg, mean(clip[pos]), mean(clip[neg]), len(clip[pos]) + len(clip[neg]),
            sp[pos] - sp[neg], sp[pos], sp[neg], len(spk[pos]) + len(spk[neg]),
            first[pos] - first[neg])


g = age_gap(r468, "grp", "spk", "Dementia", "Control")
add("hc_pitt468_age_gap_clip", "age", "pitt", "age_gap_years", "all", g[0], g[3], "",
    "Pitt 468 segments, clip-level mean age dementia %.2f minus control %.2f" % (g[1], g[2]), man468)
add("hc_pitt468_age_gap_speaker", "age", "pitt", "age_gap_years", "all", g[4], g[3], g[7],
    "Pitt 468 segments, speaker-level (mean age per speaker id 'spk'+group) dementia %.2f minus control %.2f" % (g[5], g[6]), man468,
    note="speaker key = (group, spk); participant 172 appears in both groups (PART17 T3 note)")
rall = [r for r in read_csv(manall) if r.get("task") == "cookie"]
for sub, keep in (("all_dx", lambda r: True),
                  ("probable_possible_AD", lambda r: r["group"] == "Control" or r["diagnosis"] in ("ProbableAD", "PossibleAD"))):
    rr = [r for r in rall if keep(r)]
    g2 = age_gap(rr, "group", "speaker_id", "Dementia", "Control")
    add("hc_pittall_cookie_age_gap_rec_%s" % sub, "age", "pitt_cookie_all", "age_gap_years", "all", g2[0], g2[3], g2[7],
        "all Pitt cookie recordings in pitt_manifest.csv (%s), recording-level dementia %.2f minus control %.2f" % (sub, g2[1], g2[2]), manall)
    add("hc_pittall_cookie_age_gap_spk_%s" % sub, "age", "pitt_cookie_all", "age_gap_years", "all", g2[4], g2[3], g2[7],
        "all Pitt cookie recordings (%s), speaker-level mean-of-visits dementia %.2f minus control %.2f" % (sub, g2[5], g2[6]), manall)
    add("hc_pittall_cookie_age_gap_first_%s" % sub, "age", "pitt_cookie_all", "age_gap_years", "all", g2[8], g2[3], g2[7],
        "all Pitt cookie recordings (%s), speaker-level age at first listed visit, dementia minus control" % sub, manall)

# ------------------------------------------------------------------ 5. eGeMAPS baseline and margins
eg_sum = RELEASE + "/overnight/egemaps_baseline.csv"
eg = {r["dataset"]: r for r in read_csv(eg_sum)}
for ds in ["pcgita", "neurovoz", "kcl", "pitt", "adresso", "adress2020", "edaic"]:
    f = RELEASE + "/overnight/perclip/egemaps_%s_oof.csv" % ds
    v, n, ns = auc_file(f, score_col=("p_egemaps",))
    add("hc_egemaps_%s" % ds, "eGeMAPS", ds, "egemaps", "all", v, n, ns,
        "eGeMAPS 88 features + logistic regression, speaker-disjoint OOF, rank AUC recomputed here (summary csv says %s)" % eg[ds]["egemaps_auc"], f)

# ------------------------------------------------------------------ 6. nested probe means from the json files themselves
nested_files = {
    "Qwen2.5-Omni": {ds: RELEASE + "/omni_final/omni_%s_nested_repeats.json" % ds for ds in SIX_DATASETS + ["edaic"]},
    "Qwen2-Audio": {ds: RELEASE + "/omni_final/q2a_%s_nested_repeats.json" % ds for ds in SIX_DATASETS + ["edaic"]},
    "Qwen3-Omni": {ds: RELEASE + "/overnight2/%s/q3o_%s_nested_repeats.json" % ("final" if ds in ("pitt", "edaic") else "q3o_new", ds)
                   for ds in SIX_DATASETS + ["edaic"]},
}
nested_files["Qwen2.5-Omni"]["edaic_full"] = RELEASE + "/edaic_rerun/variants/o25_full_nested_repeats.json"
smap = {"enc": "encoder_probe", "llm": "lm_probe", "ans": "answer_state_probe", "proj": "projector_probe"}
enc = {}
for m, d in nested_files.items():
    for ds, f in d.items():
        if not os.path.exists(f):
            log.append("MISSING " + f)
            continue
        j = json.load(open(f))
        for k, st in smap.items():
            if k not in j or not isinstance(j[k], dict):
                continue
            est = "five split nested, mean of %s repeats (json field %s.mean)" % (j[k].get("repeats"), k)
            note = ""
            if m == "Qwen2.5-Omni" and ds == "pitt" and k == "enc":
                note = "SUPERSEDED by the part10 5-repeat npz mean (0.7706); master_lookup row 126"
            add("hc_nested_%s_%s_%s" % (m, ds, k), m, ds, st, "all", float(j[k]["mean"]), j[k].get("n", ""), "", est, f, note=note)
            if k == "enc":
                enc[(m, ds)] = float(j[k]["mean"])
# Pitt 5-repeat npz for Qwen2.5-Omni
try:
    import numpy as np
    npz = RELEASE + "/overnight2/part10/pitt_enc_nested5_oof.npz"
    z = np.load(npz, allow_pickle=True)
    keys = list(z.keys())
    oof = z["oof"] if "oof" in keys else None
    y = z["y"] if "y" in keys else (z["label"] if "label" in keys else None)
    if oof is not None and y is not None:
        per = [auc_rank(list(y), list(oof[i])) for i in range(oof.shape[0])]
        v = sum(per) / len(per)
        add("hc_pitt_enc_5rep_npz", "Qwen2.5-Omni", "pitt", "encoder_probe", "all", v, oof.shape[1], "",
            "five split nested, mean of per-repeat AUCs recomputed here: " + ", ".join("%.4f" % p for p in per), npz)
        enc[("Qwen2.5-Omni", "pitt")] = v
    else:
        log.append("npz keys %s: could not find oof/label arrays" % keys)
except Exception as e:  # noqa: BLE001
    log.append("npz read failed: %r" % (e,))

# encoder minus eGeMAPS margins (Omni, canonical encoder values)
for ds in ["pcgita", "neurovoz", "kcl", "pitt", "adresso", "adress2020", "edaic"]:
    e_row = [r for r in rows if r["key"] == "hc_egemaps_%s" % ds][0]
    ev = float(e_row["value"])
    pv = enc.get(("Qwen2.5-Omni", ds))
    if pv is None:
        continue
    add("hc_enc_minus_egemaps_%s" % ds, "Qwen2.5-Omni", ds, "enc_minus_egemaps", "all", pv - ev, e_row["n"], e_row["n_spk"],
        "Qwen2.5-Omni nested encoder probe (5-repeat mean %.4f) minus eGeMAPS OOF AUC %.4f" % (pv, ev), e_row["file"])
old_ac = "<local data dir>/paper work/Hard drive data for paper/acoustic_baseline.csv"
if os.path.exists(old_ac):
    for r in read_csv(old_ac):
        ds = {"pitt": "pitt", "adresso": "adresso", "adress2020": "adress2020", "kcl": "kcl"}[r["dataset"].split()[0]]
        add("hc_egemaps_sep5_%s" % ds, "eGeMAPS", ds, "egemaps_sep5", "all", float(r["egemaps_logreg_auc"]), r["n"], "",
            "OLDER eGeMAPS baseline (file dated Sep 5, mean over folds, std %s); NOT the Sep 18 per-clip OOF run" % r["std"], old_ac)

# ------------------------------------------------------------------ 7. zero-shot recomputed from per-clip files
zs_files = {
    "Qwen2.5-Omni": {ds: RELEASE + "/omni_final/omni_%s_zeroshot_scores.csv" % ds for ds in SIX_DATASETS + ["edaic"]},
    "Qwen2-Audio": {ds: P + "/probe2/%s_zeroshot_scores.csv" % ds for ds in SIX_DATASETS + ["edaic"]},
    "Qwen3-Omni": {ds: RELEASE + "/overnight2/q3o_new/q3o_%s_zeroshot_scores.csv" % ds for ds in SIX_DATASETS + ["edaic"]},
}
for m, d in zs_files.items():
    for ds, f in d.items():
        if not os.path.exists(f):
            log.append("MISSING " + f)
            continue
        v, n, ns = auc_file(f)
        add("hc_zs_%s_%s" % (m, ds), m, ds, "zero_shot", "all", v, n, ns, "zero shot answer, rank AUC recomputed here from per-clip p_yes", f)

# ------------------------------------------------------------------ 8. speaking rate alone on the Pitt segments ("reaches 0.92")
seg = RELEASE + "/edaic_rerun/pitt_all_segments.csv"
if os.path.exists(seg):
    rs = read_csv(seg)
    for sub, keep in (("all708", lambda r: True), ("kept468", lambda r: r.get("tertile") in ("top", "bottom", "0", "2") or r.get("orig_set") in ("conflict", "agreement"))):
        rr = [r for r in rs if keep(r) and r.get("rate") not in ("", None)]
        a = auc_rank([int(r["label"]) for r in rr], [float(r["rate"]) for r in rr])
        if a is None:
            continue
        add("hc_pitt_rate_alone_%s" % sub, "recording", "pitt_all" if sub == "all708" else "pitt", "speaking_rate_alone", "all",
            max(a, 1 - a), len(rr), len({r["spk"] for r in rr}),
            "speaking rate alone, clip-level rank AUC, direction-free max(auc, 1-auc) (raw auc %.4f), %s segments" % (a, sub), seg,
            note="tertile values seen: %s" % sorted({r.get("tertile") for r in rs})[:5])

# ------------------------------------------------------------------ 9. E-DAIC conflict-set counts (pool before pairing vs paired)
m2 = RELEASE + "/edaic_rerun/part16/M2_distilbert_sentiment.csv"
if os.path.exists(m2):
    rs = read_csv(m2)
    for arm_col, lab in (("arm_roberta", "pool before pairing"), ("arm_roberta_selected", "paired (Part 14 manifest)")):
        c1 = sum(1 for r in rs if r[arm_col] == "conflict" and r["label_y"] == "1")
        c0 = sum(1 for r in rs if r[arm_col] == "conflict" and r["label_y"] == "0")
        spk = len({r["pid"] for r in rs if r[arm_col] == "conflict"})
        tag = "pool" if arm_col == "arm_roberta" else "paired"
        add("hc_edaic_conflict_%s_dep_pos" % tag, "count", "edaic_conflict_v2", "conflict_count_" + tag, "conflict", float(c1), c1, spk,
            "E-DAIC conflict segments, PHQ-8 >= 15 speaking positively (label 1), %s (RoBERTa arm column %s)" % (lab, arm_col), m2)
        add("hc_edaic_conflict_%s_ctrl_neg" % tag, "count", "edaic_conflict_v2", "conflict_count_" + tag, "conflict", float(c0), c0, spk,
            "E-DAIC conflict segments, PHQ-8 <= 4 speaking negatively (label 0), %s (RoBERTa arm column %s)" % (lab, arm_col), m2)
        add("hc_edaic_conflict_%s_total" % tag, "count", "edaic_conflict_v2", "conflict_count_" + tag, "conflict", float(c0 + c1), c0 + c1, spk,
            "E-DAIC conflict segments total, %s" % lab, m2)

bac = "<local data dir>/DementiaBank/build_ad_conflict.py"
if os.path.exists(bac):
    for i, line in enumerate(open(bac), start=1):
        if "caught by the validity battery at AUC" in line:
            import re as _re
            mo = _re.search(r"AUC ([0-9.]+)", line)
            add("hc_rate_shortcut_code_comment", "recording", "pitt", "speaking_rate_alone", "all", float(mo.group(1).rstrip(".")), "", "",
                "CODE COMMENT, not a result file: build_ad_conflict.py line %d says the rate shortcut was 'caught by the validity battery at AUC %s'" % (i, mo.group(1).rstrip(".")),
                bac, note="ALT: no result file on disk carries this number")

with open(OUT, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
with open(OUT.replace(".csv", ".log"), "w") as fo:
    fo.write("\n".join(log) + "\n")
print("wrote %d rows to %s" % (len(rows), OUT))
for l in log:
    print("  " + l)
