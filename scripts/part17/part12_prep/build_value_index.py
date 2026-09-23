#!/usr/local/bin/python3
"""A. Build value_index.csv: every citable value with its CURRENT file path.

Sources (all read only):
  master/master_lookup.csv                      stale G-Drive prefixes rewritten, existence checked
  edaic_rerun/part16/rows/*.tsv, part17/rows/*.tsv   verified tonight  -> verified_tonight = yes
  bootstrap/bootstrap_cis.csv
  edaic_rerun/part15/{A_backbone,B_text_conflict,C_intervals,F_channel_residual,H_readout_arms}.csv
  supplementary: omni_final/omnisft_*_sft.json + abl*.json (read only), omni/pitt_probe_estimators.csv,
                 omni/pitt_agematched.csv, omni/readout_direction.csv, overnight/egemaps_baseline.csv,
                 part12_prep/handcheck_values.csv (recomputed by part12_handcheck.py)
Status marks: master/SUPERSEDED.csv when present, plus explicit rules taken from DISCREPANCIES.md
(entries tagged SUPERSEDED / RETRACTED and the contested-value entries). Every DISCREPANCIES heading
tagged SUPERSEDED/RETRACTED that no rule covers is printed as UNHANDLED so a new tag is never missed.

Usage: build_value_index.py [--out value_index.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12common import (RELEASE, EDAIC, PREP, resolve_path, rewrite_path, first_path, to_float,  # noqa: E402
                       canon_model, canon_dataset, canon_stream, canon_arm, is_delta, canon_metric, canon_level)

FIELDS = ["key", "model", "dataset", "stream/arm", "value", "lo", "hi", "n", "n_spk", "estimator", "file",
          "source_table", "verified_tonight",
          # extra columns used by match_tex.py
          "stream", "arm", "status", "status_note", "description", "is_delta", "file_exists",
          "file_as_recorded", "path_rewritten", "row_id", "metric", "level"]

rows = []
log = []


def fmt(x):
    v = to_float(x)
    if v is None:
        return ""
    return ("%.4f" % v).rstrip("0").rstrip(".") if abs(v) < 1e6 and v != int(v) else ("%d" % v if v == int(v) else str(v))


def add(model, dataset, stream, arm, value, lo, hi, n, n_spk, estimator, file_rec, source_table, verified,
        description, row_id, delta=None):
    v = to_float(value)
    if v is None:
        return
    fpath = resolve_path(first_path(file_rec)) if file_rec else ""
    _, rew = rewrite_path(first_path(file_rec) if file_rec else "")
    rows.append({
        "key": "|".join([model or "?", dataset or "?", stream or "?", arm or "all"]),
        "model": model or "", "dataset": dataset or "", "stream/arm": "%s/%s" % (stream or "?", arm or "all"),
        "value": repr(v) if abs(v) >= 1e4 else ("%.6g" % v), "lo": fmt(lo), "hi": fmt(hi),
        "n": fmt(n), "n_spk": fmt(n_spk), "estimator": (estimator or "").strip(), "file": fpath,
        "source_table": source_table, "verified_tonight": verified,
        "stream": stream or "", "arm": arm or "all", "status": "current", "status_note": "",
        "description": (description or "").strip(), "is_delta": "yes" if (delta if delta is not None else is_delta(description)) else "no",
        "file_exists": "yes" if (fpath and os.path.exists(fpath)) else ("no" if fpath else ""),
        "file_as_recorded": file_rec or "", "path_rewritten": "yes" if rew else "no", "row_id": row_id,
        "metric": "auc" if source_table == "master_lookup" else canon_metric((description or "").split(" | ")[0] + " " + (estimator or "")),
        "level": canon_level(description + " " + (estimator or "")),
    })


MASTER_STREAM = {"zero shot answer": "zero_shot", "transcript only": "transcript", "encoder probe": "encoder_probe",
                 "LM probe": "lm_probe", "answer-state probe": "answer_state_probe", "projector probe": "projector_probe",
                 "projector fine tune": "projector_ft"}
MASTER_DS = {"edaic30": "edaic", "edaicfull": "edaic_full", "conflict": "edaic_conflict_390",
             "edaic_conflict": "edaic_conflict_390"}


def src_master():
    f = RELEASE + "/master/master_lookup.csv"
    with open(f, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            model = canon_model(r["model"]) or r["model"]
            ds = MASTER_DS.get(r["dataset"], r["dataset"])
            arm = "all"
            mo = re.match(r"^(pitt|adresso|adress2020)_(conflict|agreement)$", ds)
            if mo:
                ds, arm = mo.group(1), mo.group(2)
            add(model, ds, MASTER_STREAM.get(r["stream"], canon_stream(r["stream"])), arm, r["auc"], "", "",
                r["n"], "", r["estimator"], r["source"], "master_lookup", "no",
                "%s %s %s" % (r["model"], r["dataset"], r["stream"]), "master_lookup.csv:%d" % i, delta=False)


def parse_rowtext(rid, what, fname):
    txt = " ".join([rid or "", what or ""])
    model = canon_model(txt) or canon_model(fname)
    ds = canon_dataset(txt) or canon_dataset(fname)
    stream = canon_stream(what or "") or canon_stream(rid or "") or canon_stream(fname)
    arm = canon_arm(txt)
    # structured M8 rows: "Model | A_edaic_part14 | conflict | src=..."
    if "|" in (what or ""):
        parts = [p.strip() for p in what.split("|")]
        model = canon_model(parts[0]) or model
        if any("edaic_part14" in p.lower() for p in parts):
            ds = "edaic_conflict_v2"
        if any(p.lower() == "b_pitt" for p in parts):
            ds = "pitt"
        stream = stream or "zero_shot"
    # tonight's table 2 rows
    if rid.startswith("T1_"):
        ds = "edaic_conflict_v2"
        stream = "transcript" if "transcript" in what else "zero_shot"
    if rid.startswith("M8") and not stream:
        stream = "zero_shot"
    if rid.startswith("1a_") or rid.startswith("VALIDATION") or rid.startswith("1b_") or rid.startswith("1e") or rid.startswith("1d_") or rid.startswith("1c_"):
        ds = "edaic_conflict_v2" if not rid.startswith("1c_edaic") else "edaic_full"
        model = model or "Qwen2.5-Omni"
    if rid.startswith("EDAICFULL") or rid.startswith("edaic_full") or rid.startswith("MEDAIC_full"):
        ds = "edaic_full"
    if rid.startswith("edaic_first30") or rid.startswith("MEDAIC_w30"):
        ds = "edaic"
    if (rid.startswith("MEDAIC") or rid.startswith("edaic_")) and not model:
        model = "Qwen2.5-Omni"
    if rid.startswith("M5") or rid.startswith("T3_"):
        model = model or "Qwen2-Audio"
        ds = ds or "pitt"
    if rid.startswith("3b") or rid.startswith("Q3O"):
        model = "Qwen3-Omni"
        ds = "pitt"
    return model, ds, stream, arm


OMNI_DEFAULT_FILES = ("POD2", "POD4", "POD4B", "M7", "M10", "T5", "EDAICFULL", "MEDAIC", "M3")


def src_rows():
    files = sorted(glob.glob(EDAIC + "/part16/rows/*.tsv")) + sorted(glob.glob(EDAIC + "/part17/rows/*.tsv"))
    for f in files:
        tag = os.path.relpath(f, EDAIC)
        with open(f, newline="") as fh:
            lines = [l.rstrip("\n") for l in fh if l.strip()]
        if not lines:
            continue
        header = lines[0].split("\t")
        body = lines[1:]
        if header[0] != "id":
            header = ["id", "what", "value", "lo", "hi", "n", "n_spk", "file"]
            body = lines
        base = os.path.dirname(os.path.dirname(f))
        for j, l in enumerate(body, start=2 if lines[0].startswith("id\t") else 1):
            parts = l.split("\t")
            r = dict(zip(header, parts + [""] * (len(header) - len(parts))))
            rid, what = r.get("id", ""), r.get("what", "")
            model, ds, stream, arm = parse_rowtext(rid, what, os.path.basename(r.get("file", "")))
            if not model and os.path.splitext(os.path.basename(f))[0] in OMNI_DEFAULT_FILES:
                model = "Qwen2.5-Omni"
            frec = r.get("file", "")
            fr = first_path(frec)
            if fr and not os.path.isabs(fr):
                cands = [os.path.join(base, os.path.basename(os.path.dirname(f)).replace("rows", ""), fr),
                         os.path.join(base, fr), os.path.join(EDAIC, fr), os.path.join(RELEASE, fr)]
                pod = re.match(r"(POD\w+|EDAICFULL|M\d+)", os.path.basename(f))
                if pod:
                    cands.insert(0, os.path.join(base, os.path.splitext(os.path.basename(f))[0], fr))
                    cands.insert(0, os.path.join(EDAIC, "part16", "POD4B", fr))
                for c in cands:
                    if os.path.exists(c):
                        frec = c
                        break
            add(model, ds, stream, arm, r.get("value"), r.get("lo"), r.get("hi"), r.get("n"), r.get("n_spk"),
                what, frec, tag, "yes", "%s: %s" % (rid, what), "%s:%d" % (tag, j))


def src_bootstrap():
    f = RELEASE + "/bootstrap/bootstrap_cis.csv"
    with open(f, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            name = r["name"]
            model = canon_model(name)
            ds = canon_dataset(name)
            stream = canon_stream(name + " " + r["estimator"])
            if re.search(r"\benc probe", name):
                stream = "encoder_probe"
            elif re.search(r"\bans probe", name):
                stream = "answer_state_probe"
            elif re.search(r"\bllm probe", name):
                stream = "lm_probe"
            elif re.search(r"\bproj probe", name):
                stream = "projector_probe"
            if not stream and "paired speaker bootstrap" in r["estimator"]:
                stream = "zero_shot"
            add(model, ds, stream, canon_arm(name), r["auc"], r["ci_lo"], r["ci_hi"], r["n"], r["n_speakers"],
                r["estimator"], r["source"], "bootstrap_cis", "no", name + (" | " + r["note"] if r.get("note") else ""),
                "bootstrap_cis.csv:%d" % i)


def src_part15():
    d = EDAIC + "/part15"
    for i, r in enumerate(csv.DictReader(open(d + "/A_backbone.csv")), start=2):
        m = canon_model(r["model"])
        add(m, "pitt", "projector_ft", "all", r["auc_finetune_oof"], r["ft_lo"], r["ft_hi"], r["n_clips"], r["n_speakers"],
            "projector only fine tune OOF", r["finetune_file"], "part15/A_backbone.csv", "no", "%s Pitt projector fine-tune" % m, "A_backbone.csv:%d" % i)
        add(m, "pitt", "zero_shot", "all", r["auc_zeroshot"], r["zs_lo"], r["zs_hi"], r["n_clips"], r["n_speakers"],
            "zero shot", r["zeroshot_file"], "part15/A_backbone.csv", "no", "%s Pitt zero-shot" % m, "A_backbone.csv:%d" % i)
        add(m, "pitt", "projector_ft_gain", "all", r["gain"], r["gain_lo"], r["gain_hi"], r["n_clips"], r["n_speakers"],
            "paired fine-tune minus zero-shot", r["finetune_file"], "part15/A_backbone.csv", "no", "%s Pitt projector gain paired" % m, "A_backbone.csv:%d" % i, delta=True)
    for i, r in enumerate(csv.DictReader(open(d + "/B_text_conflict.csv")), start=2):
        m = canon_model(r["model"])
        add(m, "edaic_conflict_v2", "transcript", "conflict", r["text_conflict"], r["text_conflict_lo"], r["text_conflict_hi"], r["n_conflict"], r["n_speakers"],
            "transcript-only, 483 conflict rows", d + "/partB", "part15/B_text_conflict.csv", "no", "%s E-DAIC text conflict" % m, "B_text_conflict.csv:%d" % i)
        add(m, "edaic_conflict_v2", "transcript", "agreement", r["text_agreement"], r["text_agreement_lo"], r["text_agreement_hi"], r["n_agreement"], r["n_speakers"],
            "transcript-only, 483 agreement rows (repeats included)", d + "/partB", "part15/B_text_conflict.csv", "no", "%s E-DAIC text agreement" % m, "B_text_conflict.csv:%d" % i)
        if r.get("pearson_vs_audio"):
            add(m, "edaic_conflict_v2", "text_audio_pearson", "all", r["pearson_vs_audio"], "", "", 966, r["n_speakers"],
                "Pearson r transcript p_yes vs audio p_yes", d + "/partB", "part15/B_text_conflict.csv", "no", "%s text vs audio pearson" % m, "B_text_conflict.csv:%d" % i, delta=False)
            add(m, "edaic_conflict_v2", "text_audio_same_decision", "all", r["same_yes_no"], "", "", 966, r["n_speakers"],
                "share identical Yes/No", d + "/partB", "part15/B_text_conflict.csv", "no", "%s text vs audio same decision" % m, "B_text_conflict.csv:%d" % i, delta=False)
    for i, r in enumerate(csv.DictReader(open(d + "/C_intervals.csv")), start=2):
        txt = r["name"] + " " + r["quantity"]
        add(canon_model(r["name"]), canon_dataset(txt) or canon_dataset(r["source"]), canon_stream(r["quantity"]) or "zero_shot",
            canon_arm(r["quantity"]), r["value"], r["ci_lo"], r["ci_hi"], r["n"], r["n_speakers"], r["quantity"], r["source"],
            "part15/C_intervals.csv", "no", txt + ("" if r.get("canonical", "True") == "True" else " [canonical=False]"), "C_intervals.csv:%d" % i)
    for i, r in enumerate(csv.DictReader(open(d + "/F_channel_residual.csv")), start=2):
        for col, st in (("auc_before", "encoder_probe_600subset"), ("auc_after", "encoder_probe_channel_residual"),
                        ("auc_features_only", "channel_features_only"), ("drop", "channel_residual_drop")):
            lo = r.get(col + "_lo", "") if col != "auc_features_only" else ""
            hi = r.get(col + "_hi", "") if col != "auc_features_only" else ""
            add("Qwen2-Audio", r["dataset"], st, "all", r[col], lo, hi, r["n_used"], r["n_speakers"],
                "part15 F channel residual (%s)" % col, r["states"], "part15/F_channel_residual.csv", "no",
                "%s %s" % (r["dataset"], col), "F_channel_residual.csv:%d" % i, delta=(col == "drop"))
    for i, r in enumerate(csv.DictReader(open(d + "/H_readout_arms.csv")), start=2):
        st = "answer_state_probe"
        add(canon_model(r["model"]), "pitt", st, canon_arm(r["arm"]), r["value"], r["lower"], r["upper"], r["n_clips"], r["n_speakers"],
            "%s, %s" % (r["stream"], r["cv"]), r["file"], "part15/H_readout_arms.csv", "no", "%s %s %s" % (r["row"], r["stream"], r["arm"]),
            "H_readout_arms.csv:%d" % i, delta=("minus" in r["row"]))


def src_supplement():
    # projector / LoRA fine-tune jsons (read only)
    for f in sorted(glob.glob(RELEASE + "/omni_final/omnisft_*_sft.json")) + sorted(glob.glob(RELEASE + "/omni_final/abl*.json")):
        j = json.load(open(f))
        ds = canon_dataset(j.get("clip_list", "")) or canon_dataset(os.path.basename(f))
        b = os.path.basename(f)
        st = "projector_ft" if b.startswith("omnisft") else ("lora_ft" if "lora" in b else "lora_projector_ft")
        oof = f.replace("_sft.json", "_oof.csv") if b.startswith("omnisft") else f
        add("Qwen2.5-Omni", ds, st, "all", j.get("auc_oof"), "", "", j.get("n"), j.get("n_speakers"),
            "%s, %s folds, %s epochs, lr %s, json auc_oof" % (j.get("trained", st), j.get("folds"), j.get("epochs"), j.get("lr")),
            f, "omni_final_json", "no", "%s %s auc_oof" % (b, ds), b)
        for arm, v in (j.get("arms") or {}).items():
            add("Qwen2.5-Omni", ds, st, arm, v, "", "", "", "", "%s arm from json" % st, f, "omni_final_json", "no", "%s %s arm %s" % (b, ds, arm), b + ":" + arm)
        for k in ("epochs", "lr", "folds"):
            if k in j:
                add("Qwen2.5-Omni", ds, st + "_config_" + k, "all", j[k], "", "", j.get("n"), j.get("n_speakers"), "fine-tune config " + k, f,
                    "omni_final_json", "no", "%s %s config %s" % (b, ds, k), b + ":" + k, delta=False)
    f = RELEASE + "/omni/pitt_probe_estimators.csv"
    for i, r in enumerate(csv.DictReader(open(f)), start=2):
        meas = r["measure"]
        m = "age" if meas.startswith("age") else "Qwen2.5-Omni"
        st = "age_alone" if meas.startswith("age") else ("encoder_probe" if "probe" in meas else "zero_shot")
        ds = "pitt" if r["subset"] == "full" else "pitt_agematched"
        add(m, ds, st if r["level"] == "clip" else st + "_speaker_level", "all", r["auc"], "", "", r["n_clips"], r["n_speakers"],
            "%s, %s level, %s" % (meas, r["level"], r["subset"]), resolve_path(r["source"], [RELEASE]), "omni/pitt_probe_estimators.csv", "no",
            "%s %s %s" % (meas, r["level"], r["subset"]), "pitt_probe_estimators.csv:%d" % i, delta=False)
    f = RELEASE + "/omni/pitt_agematched.csv"
    for i, r in enumerate(csv.DictReader(open(f)), start=2):
        meas = r["measure"]
        m = "age" if r["model"] == "age" else canon_model(r["model"])
        st = {"nested_encoder_probe": "encoder_probe", "zero_shot_answer": "zero_shot", "age_alone": "age_alone"}.get(meas, meas)
        ds = "pitt" if r["scope"] == "full468" else "pitt_agematched_273"
        add(m, ds, st if r["level"] == "clip" else st + "_speaker_level", "all", r["auc"], r["ci_lo"], r["ci_hi"], r["n"], "",
            "%s, %s level, %s (Sep 19 age-matched run)" % (meas, r["level"], r["scope"]), r["file"], "omni/pitt_agematched.csv", "no",
            "%s %s %s %s" % (r["model"], meas, r["level"], r["scope"]), "pitt_agematched.csv:%d" % i, delta=False)
    f = RELEASE + "/omni/readout_direction.csv"
    for i, r in enumerate(csv.DictReader(open(f)), start=2):
        ds = r["dataset"]
        for col, st in (("cosine", "readout_cosine"), ("cosine_random", "readout_cosine_random_floor"), ("auc_probe", "answer_state_probe"),
                        ("auc_d", "along_readout_direction"), ("auc_probe_without_d", "answer_state_probe_minus_readout")):
            add("Qwen2.5-Omni", ds, st, "all", r[col], r["ci_lo"] if col == "cosine" else "", r["ci_hi"] if col == "cosine" else "",
                r["n"], r["n_speakers"], "readout direction (%s), part10" % r["direction_used"], f, "omni/readout_direction.csv", "no",
                "%s %s" % (ds, col), "readout_direction.csv:%d" % i, delta=False)
    f = RELEASE + "/overnight/egemaps_baseline.csv"
    for i, r in enumerate(csv.DictReader(open(f)), start=2):
        add("eGeMAPS", r["dataset"], "egemaps", "all", r["egemaps_auc"], "", "", r["n"], r["n_speakers"],
            "eGeMAPS 88 features logistic regression (Sep 18 run)", f, "overnight/egemaps_baseline.csv", "no",
            "%s eGeMAPS" % r["dataset"], "egemaps_baseline.csv:%d" % i, delta=False)
    # projector transfer across datasets (trained on one set, scored on another)
    for f in sorted(glob.glob(RELEASE + "/overnight2/part6/transfer_*_transfer.json")) + sorted(glob.glob(RELEASE + "/omni_final/transfer_*_transfer.json")):
        j = json.load(open(f))
        src = canon_dataset(j.get("source", "")) or canon_dataset(os.path.basename(f))
        for t in j.get("targets", []):
            add("Qwen2.5-Omni", canon_dataset(t.get("target", "")), "projector_transfer", "all", t.get("auc_after_transfer"), "", "", t.get("n"), "",
                "projector trained on all %s clips (no folds), scored on %s" % (src, t.get("target")), f, "transfer_json", "no",
                "projector transfer %s -> %s" % (src, t.get("target")), os.path.basename(f) + ":" + str(t.get("target")), delta=False)
    # recording-only confound features (speaker level)
    f = EDAIC + "/confounds/confound_auc.csv"
    if os.path.exists(f):
        for i, r in enumerate(csv.DictReader(open(f)), start=2):
            add("recording", r["dataset"], "confound_" + r["feature"], "all", r["strength"], "", "", r["n_clips"], r["n_speakers"],
                "recording-only feature %s, speaker-level AUC, direction-free strength max(auc, 1-auc)" % r["feature"], f,
                "confounds/confound_auc.csv", "no", "%s %s" % (r["dataset"], r["feature"]), "confound_auc.csv:%d" % i, delta=False)
    f = PREP + "/handcheck_values.csv"
    if os.path.exists(f):
        for i, r in enumerate(csv.DictReader(open(f)), start=2):
            add(r["model"], r["dataset"], r["stream"], r["arm"], r["value"], r["lo"], r["hi"], r["n"], r["n_spk"],
                r["estimator"], r["file"], "part12_handcheck", "yes", r["key"] + (" | " + r["note"] if r.get("note") else ""),
                "handcheck_values.csv:%d" % i, delta=("gap" in r["stream"] or "minus" in r["stream"]))
    else:
        log.append("WARNING handcheck_values.csv missing; run part12_handcheck.py first")


# ----------------------------------------------------------------------------- status rules
def v4(r):
    return round(float(r["value"]), 4)


def mark(r, status, note):
    order = {"current": 0, "ALT": 1, "CONTESTED": 2, "SUPERSEDED": 3, "RETRACTED": 4}
    if order.get(status, 0) >= order.get(r["status"], 0):
        r["status"] = status
    r["status_note"] = (r["status_note"] + " | " if r["status_note"] else "") + note


RULES_COVER = {  # DISCREPANCIES heading substring -> rule id that applies it
    "M7 first pass used a SUPERSEDED Pitt probe file": "S1,S2",
    "quoted Pitt encoder-probe figure 0.7706 not found": "retracted claim, carries no value (0.7706 is correct)",
    "claim that \"0.7706 exists in no file\" is WRONG": "retraction of the claim above; no value to mark",
    "POD4's 4b used the SUPERSEDED Pitt encoder baseline": "S3",
}


def apply_rules():
    for r in rows:
        d, m, st, arm, f = r["dataset"], r["model"], r["stream"], r["arm"], r["file"] + " " + r["file_as_recorded"]
        val = v4(r)
        desc = r["description"].lower()
        # S1 Pitt encoder probe: single split 0.7969 and 4-repeat json 0.7761 superseded by 5-repeat 0.7706
        if m == "Qwen2.5-Omni" and d.startswith("pitt") and st.startswith("encoder_probe") and \
                ("omni_pitt_enc_nested_oof.csv" in f or (d == "pitt" and val in (0.7761, 0.7969))):
            mark(r, "SUPERSEDED", "S1: Pitt encoder probe from single-split OOF csv or the 0.7761 json mean; current is the part10 5-repeat mean 0.7706 (master_lookup row 126, DISCREPANCIES M7/POD4 entries)")
        # S2 the paired gap computed on the superseded probe
        if r["row_id"].startswith("part16/rows/M7.tsv") and "pitt" in desc and ("gap" in desc or "minus" in desc) and "rep5" not in desc:
            mark(r, "SUPERSEDED", "S2: gap built on the single-split probe (0.1391); corrected +0.1128 in M7fix")
        # S1b the averaged-probability AUC of the five Pitt repeats is not used (authors' decision 2026-09-23, master row 126)
        if m == "Qwen2.5-Omni" and d == "pitt" and st == "encoder_probe" and val == 0.7917:
            mark(r, "SUPERSEDED", "S1b: 0.7917 is the AUC of the probabilities averaged over the five repeats; NOT used (master_lookup row 126, SUPERSEDED.csv row 4)")
        # S3 POD4 arm A encoder baseline
        if "POD4_4b_A_enc" in r["description"]:
            mark(r, "SUPERSEDED", "S3: single-split arm A encoder value; must not be quoted beside the 5-repeat 0.7706 (DISCREPANCIES POD4 4b)")
        # S4 E-DAIC Part 14 agreement arm computed on 483 rows with repeats
        if d == "edaic_conflict_v2" and arm == "agreement" and "NEW_311distinct" not in r["description"] and "311" not in r["n"]:
            mark(r, "CONTESTED", "S4: 483 agreement rows hold only 311 distinct segments; PART17 T1 311-distinct value differs at 2 dp")
        if d == "edaic_conflict_v2" and arm == "agreement" and "NEW_311distinct" in r["description"]:
            mark(r, "CONTESTED", "S4: PART17 T1 value on the 311 distinct agreement segments; the paper's Table 2 used all 483 rows (repeats included); not yet decided which is printed")
        # S5 Pitt transcript-only, two runs
        if m == "Qwen2.5-Omni" and d == "pitt" and st == "transcript":
            mark(r, "CONTESTED", "S5: two Omni Pitt transcript runs disagree: Run A Sep 15 local 0.6975 (bootstrap tables, M1 headline) vs Run B Sep 20 pod 0.6629 (master_lookup, table1_omni25); cause not established")
        # S6 Qwen3-Omni Pitt zero-shot rerun
        if m == "Qwen3-Omni" and d == "pitt" and st == "zero_shot" and r["arm"] == "all":
            mark(r, "CONTESTED", "S6: shipped 0.7619 vs 2026-09-22 rerun 0.7673 on the same 468 clips; unresolved")
        # S7 ALT copies
        if "[alt" in desc or "canonical=false" in desc or "alt_overnight_copy" in desc or r["row_id"].startswith("part16/rows/M1.tsv") and "ALT" in r["description"]:
            mark(r, "ALT", "S7: alternative copy/run, not the canonical file")
        if "ad_scores_omni.csv" in f or "ad_scores_qwen2audio.csv" in f or "paradox/omni_clip_scores.csv" in f:
            mark(r, "ALT", "S7: older source the Sep 14 tex traced to; current Table 2 uses the omni_final / Part 14 files")
        # S8 six transcript-only master values that do not reproduce from their per-clip files
        s8 = {("Qwen3-Omni", "pcgita"): 0.5325, ("Qwen2.5-Omni", "pcgita"): 0.5218, ("Qwen3-Omni", "kcl"): 0.4688,
              ("Qwen2.5-Omni", "kcl"): 0.7634, ("Qwen3-Omni", "neurovoz"): 0.5664, ("Qwen2.5-Omni", "neurovoz"): 0.5372}
        if r["source_table"] == "master_lookup" and st == "transcript" and (m, d) in s8:
            mark(r, "CONTESTED", "S8: stored AUC does not reproduce from its per-clip file; rank-formula recompute gives %.4f" % s8[(m, d)])
        # S9 Q3O by-arm probe estimators
        if m == "Qwen3-Omni" and d == "pitt" and arm in ("conflict", "agreement") and ("3b" in r["row_id"] or "Q3Ofix" in r["row_id"] or "POD3" in r["row_id"]):
            mark(r, "CONTESTED", "S9: three estimators in circulation for the Q3O by-arm probe; choose and label one")
        # S10 abl_pitt_both json vs csv
        if r["row_id"].startswith("abl_pitt_both.json"):
            mark(r, "CONTESTED", "S10: abl_pitt_both.json auc_oof 0.7872 but its csv recomputes to 0.7696 (abl2 rerun)")
        # S11 part15 B claims a dedup that did not happen
        if r["source_table"] == "part15/B_text_conflict.csv" and arm == "agreement":
            mark(r, "CONTESTED", "S11: B_text_conflict agreement used 483 rows with repeats; de-duplicated 0.9620 (Omni)")
        # S12 KCL / other: nothing
        # S13 old age-matched Sep 19 run (273 clips) vs pitt_probe_estimators (275 / 129)
        if r["source_table"] == "omni/pitt_agematched.csv" and d != "pitt":
            mark(r, "ALT", "S13: Sep 19 age-matched join gives 273 clips / 128 speakers; pitt_probe_estimators.csv (Sep 20) gives 275 / 129")
        if "hc_rate_shortcut_code_comment" in r["description"]:
            mark(r, "ALT", "value quoted in a code comment (build_ad_conflict.py), no result file")
        # TA batch is never canonical
        if "ashutosh_batch2" in f:
            mark(r, "ALT", "TA batch run on another computer; never canonical")


def apply_superseded_csv():
    """master/SUPERSEDED.csv (columns file, superseded_by, reason, decided_on as of 2026-09-23).

    The superseded values are the numbers written in 'reason' BEFORE the phrase that introduces the
    replacement. A row is marked when its file is the superseded file (full path, or same basename when
    the value also matches) and, when the reason scopes the entry ("as the source of ... only") or lists
    values, its value is one of the listed superseded values."""
    f = RELEASE + "/master/SUPERSEDED.csv"
    if not os.path.exists(f):
        log.append("master/SUPERSEDED.csv not present at build time")
        return
    sup = list(csv.DictReader(open(f, newline="")))
    log.append("master/SUPERSEDED.csv present: %d rows, columns %s" % (len(sup), list(sup[0].keys()) if sup else []))
    cut = re.compile(r"replaced by|canonical is|the paper value is|decided estimator is|per-clip csv recomputes to|"
                     r"POD3 per-clip file gives|the json remains|superseded by", re.I)
    total = 0
    for k, s in enumerate(sup, start=2):
        spath = resolve_path(s.get("file", ""))
        reason = s.get("reason", "")
        m = cut.search(reason)
        part_a = reason[:m.start()] if m else reason
        vals = {round(float(x), 4) for x in re.findall(r"(?<![\d.])[01]\.\d{3,4}(?!\d)", part_a)}
        scoped = bool(re.search(r"\bonly\b|as the source of|as the .* source", reason))
        hits = 0
        for r in rows:
            full = (r["file"] == spath)
            base = os.path.basename(r["file"]) == os.path.basename(spath)
            if not (full or base):
                continue
            inval = round(float(r["value"]), 4) in vals
            if not full and not inval:
                continue                      # basename-only hit needs the value too
            if (scoped or vals) and not inval:
                continue
            mark(r, "SUPERSEDED", "master/SUPERSEDED.csv row %d (%s): %s -> %s" % (k, s.get("decided_on", ""), reason[:220], s.get("superseded_by", "")))
            hits += 1
        total += hits
        log.append("SUPERSEDED.csv row %d %s: superseded values %s, scoped=%s, index rows marked %d" % (
            k, os.path.basename(spath), sorted(vals), scoped, hits))
    log.append("SUPERSEDED.csv marked %d index rows" % total)


def scan_discrepancies():
    f = EDAIC + "/DISCREPANCIES.md"
    for i, line in enumerate(open(f), start=1):
        if line.startswith("## ") and re.search(r"SUPERSEDED|RETRACTED", line, re.I):
            cov = [v for k, v in RULES_COVER.items() if k in line]
            log.append("DISCREPANCIES.md:%d %s -> %s" % (i, line.strip()[:120], cov[0] if cov else "UNHANDLED, review by hand"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PREP, "value_index.csv"))
    a = ap.parse_args()
    src_master()
    src_rows()
    src_bootstrap()
    src_part15()
    src_supplement()
    apply_rules()
    apply_superseded_csv()
    scan_discrepancies()
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    miss = [r for r in rows if r["file_exists"] == "no"]
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    rew = sum(1 for r in rows if r["path_rewritten"] == "yes")
    with open(a.out.replace(".csv", "_build.log"), "w") as fo:
        fo.write("rows %d, status %s, stale prefixes rewritten %d, files not found %d\n" % (len(rows), st, rew, len(miss)))
        for l in log:
            fo.write(l + "\n")
        fo.write("\nFILES NOT FOUND (path as resolved | as recorded | row):\n")
        for r in miss:
            fo.write("%s | %s | %s\n" % (r["file"], r["file_as_recorded"], r["row_id"]))
    print("value_index: %d rows -> %s ; status %s ; rewritten %d ; missing files %d" % (len(rows), a.out, st, rew, len(miss)))
    for l in log:
        if "UNHANDLED" in l or "WARNING" in l or "SUPERSEDED.csv" in l:
            print("  " + l)


if __name__ == "__main__":
    main()
