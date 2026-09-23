#!/usr/local/bin/python3
"""C. Propose the best file value for each paper number and mark MATCH / MISMATCH / UNMATCHED.

Point values: context (model, dataset, stream, arm) is read from the table row + column, or from the
clause / sentence / previous sentence around the number. Candidates are value_index rows compatible
with that context. MATCH when round(file_value, printed_decimals) == paper value for the best
candidate that is not SUPERSEDED; MISMATCH when compatible candidates exist but none rounds to the
paper value (the reason names the best one and any superseded or alternative value that would round);
UNMATCHED when no compatible candidate exists.

Ranges: both ends must equal round(min) and round(max) of a NAMED SET of canonical values
(for example "Qwen2.5-Omni encoder probe, 5-repeat means, PD+AD six datasets"). The matched set, or the
closest compatible set when nothing matches, is named in the reason. Changes ("from A to B") are two
points. Word numbers are checked only for "N years" (age gap) and "N of (the) six models ... below chance".

Usage: match_tex.py NUMBERS.csv VALUE_INDEX.csv --tex TEX [--added ADDED.csv] [--red-spec SPEC.csv]
                    [--include-comments] --out MATCHES.csv [--paper-vs-omni OUT.csv]
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p12common import rnd, to_float, SIX_DATASETS, PD, AD, MODELS  # noqa: E402

ALL7 = SIX_DATASETS + ["edaic"]
ALL7F = SIX_DATASETS + ["edaic_full"]
GROUPS = {"PD": PD, "AD": AD, "PD+AD": SIX_DATASETS, "ALL7 (E-DAIC first 30 s)": ALL7, "ALL7 (E-DAIC full)": ALL7F,
          "PD+AD without MDVR-KCL": [d for d in SIX_DATASETS if d != "kcl"],
          "ALL7 without MDVR-KCL (E-DAIC first 30 s)": [d for d in ALL7 if d != "kcl"]}
EDAIC_FAMILY = ["edaic_full", "edaic", "edaic_mid300", "edaic_mid30"]
PROBES = ["encoder_probe", "lm_probe", "answer_state_probe", "projector_probe"]
LABEL_DS = {"pc-gita": "pcgita", "neurovoz": "neurovoz", "mdvr-kcl": "kcl", "kcl": "kcl", "e-daic": "edaic",
            "pitt": "pitt", "adresso": "adresso", "adress-2020": "adress2020"}
SRC_PREF = {"master_lookup": 0.30, "part12_handcheck": 0.28, "omni_final_json": 0.22, "bootstrap_cis": 0.15}
STATUS_PREF = {"current": 0.5, "CONTESTED": 0.3, "ALT": 0.1, "SUPERSEDED": 0.0, "RETRACTED": 0.0}

MODEL_PAT = [("Qwen2.5-Omni", r"Qwen2\.5-Omni|\bomni\b"), ("Qwen3-Omni", r"Qwen3-Omni"), ("Qwen2-Audio", r"Qwen2-Audio"),
             ("Audio Flamingo 2", r"Audio Flamingo 2\b|\bAF2\b"), ("Audio Flamingo 3", r"Audio Flamingo 3\b|\bAF3\b"),
             ("Kimi-Audio", r"Kimi(-Audio)?")]
MODEL_GROUP_PAT = [(["Qwen2-Audio", "Qwen2.5-Omni", "Qwen3-Omni"], r"three Qwen models|Qwen models"),
                   (["Audio Flamingo 2", "Audio Flamingo 3"], r"Audio Flamingo models|both Audio Flamingo|Audio Flamingo 2 and 3"),
                   (list(MODELS), r"every model|all six models|of the six models|of six models|six Audio LLMs")]
DS_PAT = [("adress2020", r"ADReSS-2020"), ("adresso", r"ADReSSo"), ("pcgita", r"PC-GITA"), ("neurovoz", r"NeuroVoz|Neurovoz"),
          ("kcl", r"MDVR-KCL|\bKCL\b"), ("pitt_all", r"\b708\b"), ("pitt_agematched", r"age(?:[- ]and[- ]sex)?[- ]matched"),
          ("pitt", r"\bPitt\b"), ("edaic", r"E-DAIC|depression|\bMDD\b|interviews?\b")]
DOMINANT_DS = ["pitt_all", "pitt_agematched"]
GROUP_PAT = [("PD+AD", r"Parkinson's and Alzheimer's|PD and AD|PD\+AD"), ("ALL7", r"(all )?seven datasets"),
             ("PD", r"Parkinson's|\bPD\b|read-speech"), ("AD", r"Alzheimer's|\bAD\b|picture description")]
# order matters: earlier patterns win where matches overlap
STREAM_PAT = [("prompt_variant", r"wordings?|prompt variants?"), ("speaking_rate_alone", r"speaking rate"),
              ("enc_minus_egemaps", r"eGeMAPS"), ("age_alone", r"[Aa]ge alone"), ("age_gap_years", r"years older|older than"),
              ("readout_cosine", r"[Cc]osine"), ("answer_state_probe_minus_readout", r"[Rr]emoving the readout direction"),
              ("confound", r"[Rr]ecording features|quietest frames|such feature|spectral centroid"),
              ("projector_transfer", r"projector trained on|One trained on"),
              ("projector_ft", r"fine-tuned answer|fine-tun\w*|Fine-tun\w*|retrained projector|retrain\w*|projector alone"),
              ("transcript", r"transcript\b|transcript alone|\bText\b|words alone"),
              ("lm_probe", r"language model|language-model"), ("answer_state_probe", r"answer state"),
              ("encoder_probe", r"encoder probe|\bencoder\b"), ("probe", r"\bprobes?\b"),
              ("zero_shot", r"zero-shot answer|zero-shot|zero shot|\banswers?\b|Zero-shot|\bAudio\b")]
DOMINANT_STREAMS = ["prompt_variant", "speaking_rate_alone", "enc_minus_egemaps", "age_alone", "age_gap_years", "readout_cosine",
                    "answer_state_probe_minus_readout", "confound", "projector_transfer"]
ARM_PAT = [("conflict", r"conflict"), ("agreement", r"agreement|\bAgree\.?|agree\b")]
METHOD_PAT = r"iterations|learning rate|epochs?\b|\bfolds?\b|C\s*=|probability of at least|at least \d+ words|PHQ-8 of|capped at|minutes|hidden states|stages|\bwindow\b|first \d+\s*s\b|hears \d+|\d+\s*s of speech|or more|or less|median segment|top or bottom third|Answer with one word"
BOUND_UP = r"(below|under|less than|at most|never above)\s*\$?$"
BOUND_LO = r"(above|over|more than|at least|exceeds?)\s*\$?$"


MODEL_MASK = re.compile(r"Qwen2-Audio|Kimi-Audio|Qwen2\.5-Omni|Qwen3-Omni(-30B-A3B)?|Audio Flamingo \d( and \d)?|Audio LLMs?|Audio language")


def nomodel(text):
    return MODEL_MASK.sub(lambda mo: " " * len(mo.group()), text)


def mentions(pats, text):
    """All pattern hits, earlier-listed patterns winning where hits overlap."""
    taken = []
    out = []
    for name, pat in pats:
        for mo in re.finditer(pat, text):
            a, b = mo.start(), mo.end()
            if any(a < tb and ta < b for ta, tb in taken):
                continue
            taken.append((a, b))
            out.append((a, name))
    out.sort()
    return out


def nearest(ms, pos, clause_span=None):
    pre = [m for m in ms if m[0] < pos and (clause_span is None or m[0] >= clause_span[0])]
    if pre:
        return [pre[-1][1]]
    return []


def following(ms, pos, clause_span=None):
    fol = [m for m in ms if m[0] > pos and (clause_span is None or m[0] <= clause_span[1])]
    return [fol[0][1]] if fol else []


def fmtv(v, dp=4):
    return ("%." + str(dp) + "f") % v if v is not None else ""


class Index:
    def __init__(self, path):
        self.rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r["v"] = to_float(r["value"])
                r["n_i"] = to_float(r["n"])
                r["nspk_i"] = to_float(r["n_spk"])
                self.rows.append(r)
        self.by = defaultdict(list)
        for r in self.rows:
            self.by[(r["model"], r["dataset"], r["stream"], r["arm"])].append(r)

    def canonical(self, model, ds, stream, arm="all"):
        """Preferred non-superseded, non-delta value for a key."""
        cands = [r for r in self.by.get((model, ds, stream, arm), []) if r["status"] != "SUPERSEDED" and r["is_delta"] == "no"
                 and r.get("metric", "auc") in ("auc", "") and r.get("level", "clip") == "clip"]
        if not cands:
            return None
        cands.sort(key=lambda r: (-STATUS_PREF.get(r["status"], 0), -SRC_PREF.get(r["source_table"], 0.1)))
        return cands[0]


def item_context(it, flags):
    """Return dict: models, datasets, streams, arm, delta, bound, abs, best, except_models, notes."""
    ctx = dict(models=[], datasets=[], group="", streams=[], arm="", delta=False, bound="", absval=False, best=False,
               method=False, notes=[], model_default=False)
    if it["table"]:
        row, col, grp = it["table_row"], it["table_col"], it["table_group"]
        m = [n for n, p in MODEL_PAT if re.search(p, row)]
        if m:
            ctx["models"] = m[:1]
        d = [n for n, p in DS_PAT if re.search(p, row)]
        if it["table"] == "tab:conflict":
            ctx["datasets"] = ["edaic_conflict_v2"] if "Depression" in col else (["pitt"] if "Alzheimer" in col else [])
            ctx["streams"] = ["zero_shot"]
            ctx["arm"] = "conflict" if "Conflict" in col else ("agreement" if "Agree" in col else "")
        else:
            ctx["datasets"] = d[:1]
            if "Text" in col:
                ctx["streams"] = ["transcript"]
            elif "Fine-tuned" in col or "Fine" in col:
                ctx["streams"] = ["projector_ft"]
            elif "Zero-shot" in col or "Audio" in col:
                ctx["streams"] = ["zero_shot"]
            elif "probe" in col.lower():
                ctx["streams"] = ["encoder_probe"]
            if "[clips]" in col or "[speakers]" in col:
                ctx["streams"] = ["count_" + ("n" if "[clips]" in col else "n_spk")]
            if not ctx["models"]:
                cap = it.get("table_caption", "")
                cm = [n for n, p in MODEL_PAT if re.search(p, cap)]
                ctx["models"] = cm[:1] or ["Qwen2.5-Omni"]
        if ctx["datasets"] == ["edaic"]:
            ctx["datasets"] = EDAIC_FAMILY
        return ctx
    sent, clause, prev = it["context"], it["clause"], it["prev_context"]
    para = it.get("para_before", "") or ""
    # position of the number inside the sentence
    pos = sent.find(it["text"].replace("$", ""))
    if pos < 0:
        pos = sent.find(it["text"].split()[0])
    cpos = sent.find(clause[:30]) if clause else -1
    cspan = (cpos, cpos + len(clause)) if cpos >= 0 else None
    # ---- models: "except", groups, respectively mapping, nearest preceding, sentence, previous sentence, default
    mm = mentions(MODEL_PAT, sent)
    if re.search(r"except", sent):
        ex = [n for n, p in MODEL_PAT if re.search(r"except\s+(?:" + p + ")", sent)]
        ctx["models"] = [m for m in MODELS if m not in ex]
        ctx["notes"].append("every model except " + ", ".join(ex))
    else:
        for grp_models, gp in MODEL_GROUP_PAT:
            if re.search(gp, sent):
                ctx["models"] = grp_models
                break
        if not ctx["models"]:
            pre = [m for m in mm if m[0] < pos]
            nums_after = [mo.start() for mo in re.finditer(r"\d\.\d+", sent)]
            if len(pre) >= 2 and re.search(r"\band\b", sent[pre[-2][0]:pre[-1][0] + 1]):
                last = pre[-1][0]
                brk = re.search(r",|\bwhile\b|;", sent[last:])
                end = last + brk.start() if brk else len(sent)
                after = [p for p in nums_after if last < p < end]
                if len(after) == 2 and pos in after:
                    ctx["models"] = [pre[-2:][after.index(pos)][1]]
                    ctx["notes"].append("respectively mapping")
            if not ctx["models"]:
                ctx["models"] = nearest(mm, pos) or ([m for _, m in mm][:1]) or [n for _, n in mentions(MODEL_PAT, prev)][-1:]
    if not ctx["models"]:
        ctx["models"] = ["Qwen2.5-Omni"]
        ctx["model_default"] = True
    # ---- datasets: dominant ("708", "age matched"), "X on <Dataset>", nearest in clause, sentence, groups, previous text
    after = sent[pos + len(it["text"]):pos + len(it["text"]) + 30] if pos >= 0 else ""
    dsm = mentions(DS_PAT, sent)
    dom = [n for _, n in dsm if n in DOMINANT_DS]
    fo = re.match(r"\s*on (?:the )?([A-Z][\w\-]+)", after)
    if dom:
        ctx["datasets"] = dom[:1]
    elif fo and any(re.fullmatch(p, fo.group(1)) for _, p in DS_PAT):
        ctx["datasets"] = [n for n, p in DS_PAT if re.fullmatch(p, fo.group(1))][:1]
    else:
        near_sent = [m for m in dsm if pos - 60 <= m[0] < pos]
        ctx["datasets"] = nearest(dsm, pos, cspan) or ([near_sent[-1][1]] if near_sent else []) or following(dsm, pos, cspan) or nearest(dsm, pos)
        if not ctx["datasets"] and len({n for _, n in dsm}) == 1:
            ctx["datasets"] = [dsm[0][1]]
    grp = [n for _, n in mentions(GROUP_PAT, sent)]
    if grp:
        ctx["group"] = grp[0]
        gds = GROUPS.get(grp[0]) if grp[0] != "ALL7" else ALL7F
        if not ctx["datasets"]:
            ctx["datasets"] = list(gds)
        elif re.search(r"\band\b", sent) and set(ctx["datasets"]) & {"edaic"} and grp[0] in ("AD", "PD"):
            ctx["datasets"] = list(dict.fromkeys(ctx["datasets"] + list(gds)))   # "the Alzheimer's and E-DAIC sets"
            ctx["notes"].append("dataset group + named set")
    if not ctx["datasets"]:
        for src, lab in ((prev, "previous sentence"), (para, "paragraph")):
            pm = mentions(DS_PAT, src)
            pg = [n for _, n in mentions(GROUP_PAT, src)]
            if pm:
                ctx["datasets"] = [pm[-1][1]]
                ctx["notes"].append("dataset from " + lab)
                break
            if pg:
                ctx["group"] = ctx["group"] or pg[-1]
                ctx["datasets"] = list(GROUPS.get(pg[-1], [])) if pg[-1] != "ALL7" else ALL7F
                ctx["notes"].append("dataset group from " + lab)
                break
    if not ctx["group"] and it["kind"] == "range":
        pg = [n for _, n in mentions(GROUP_PAT, prev)]
        if pg and re.search(r"\bthey\b|same probes|On Qwen|On Audio|On Kimi", sent):
            ctx["group"] = pg[0]
            ctx["notes"].append("group from previous sentence")
    if "edaic" in ctx["datasets"]:
        ctx["datasets"] = [d for d in ctx["datasets"] if d != "edaic"] + (["edaic_conflict_v2"] if re.search(r"conflict|agree", sent) else EDAIC_FAMILY)
    # ---- streams: dominant in clause, nearest preceding in clause, preceding in sentence, following in clause, previous sentence
    sm = mentions(STREAM_PAT, nomodel(sent))
    cl_dom = [n for p_, n in sm if n in DOMINANT_STREAMS and (cspan is None or cspan[0] <= p_ <= cspan[1])]
    sen_dom = [n for p_, n in sm if n in DOMINANT_STREAMS]
    if cl_dom:
        st = cl_dom[:1]
    elif sen_dom and not [n for p_, n in sm if n not in DOMINANT_STREAMS and cspan and cspan[0] <= p_ <= cspan[1]]:
        st = sen_dom[:1]
    else:
        st = nearest(sm, pos, cspan) or nearest(sm, pos) or following(sm, pos, cspan)
    if not st and re.search(r"against\s*\$?$", sent[max(0, pos - 12):pos]):
        st = ["zero_shot"]
        ctx["notes"].append("'probe against answer' convention")
    if not st:
        st = [n for _, n in mentions(STREAM_PAT, nomodel(prev))][:1]
        if st:
            ctx["notes"].append("stream from previous sentence")
    if not st and (it["section"].lower().startswith("conflict") or re.search(r"conflict|agree", sent)):
        st = ["zero_shot"]
        ctx["notes"].append("conflict-test default stream")
    if re.search(r"at the language model", after):
        st = ["lm_probe"]
    ctx["streams"] = st
    if st == ["probe"]:
        ctx["streams"] = list(PROBES)
    if re.search(r"\bbest\b", sent[max(0, pos - 60):pos]) and set(ctx["streams"]) & set(PROBES):
        ctx["best"] = True
    # ---- arms
    am = mentions(ARM_PAT, sent)
    a = nearest(am, pos, cspan) or following(am, pos, cspan)
    ctx["arm"] = a[0] if a else ""
    # ---- delta / bound / magnitude / method constant
    if it["text"].lstrip("$").startswith(("+", "-", "−")) or re.search(r"\bdrop(s)? by\b|\bby\s*$|beats|minus|gap", sent[max(0, pos - 25):pos]):
        ctx["delta"] = True
    before = sent[max(0, pos - 25):pos]
    if re.search(BOUND_UP, before):
        ctx["bound"] = "upper"
    elif re.search(BOUND_LO, before):
        ctx["bound"] = "lower"
    if re.search(r"in magnitude|absolute", sent):
        ctx["absval"] = True
    if re.search(METHOD_PAT, clause) and it["kind"] in ("integer", "sci", "point", "percent"):
        ctx["method"] = True
    return ctx


def compatible(r, ctx, kind):
    if r["v"] is None:
        return False
    st = r["stream"]
    special = ("age", "eGeMAPS", "recording")
    if ctx["streams"]:
        want = set(ctx["streams"])
        ok = st in want
        if "confound" in want and st.startswith("confound_"):
            ok = True
        if "zero_shot" in want and st == "zero_shot":
            ok = True
        if not ok:
            return False
    if r["model"] not in special and ctx["models"] and r["model"] not in ctx["models"]:
        return False
    if ctx["datasets"] and r["dataset"] not in ctx["datasets"]:
        return False
    if ctx["arm"]:
        if r["arm"] != ctx["arm"]:
            return False
    elif r["arm"] not in ("all", "") and not ctx["delta"]:
        return False
    if ctx["delta"] != (r["is_delta"] == "yes") and not st.startswith(("enc_minus", "age_gap")):
        return False
    if "_config_" in st or st.startswith("count_"):
        return False
    if (r["stream"].endswith("_speaker_level") or r.get("level") == "speaker") and not st.startswith("confound_"):
        return False
    want_metric = "cosine" if "readout_cosine" in (ctx["streams"] or []) else "auc"
    if st in ("readout_cosine", "readout_cosine_random_floor"):
        return want_metric == "cosine"
    if r.get("metric", "auc") != want_metric and not st.startswith(("age_gap", "confound_", "enc_minus")):
        return False
    return True


def rank_key(r, ctx):
    s = STATUS_PREF.get(r["status"], 0) + SRC_PREF.get(r["source_table"], 0.1) + (0.2 if r["verified_tonight"] == "yes" else 0)
    if r["dataset"] in EDAIC_FAMILY and ctx.get("edaic_pref"):
        s += 1.0 if r["dataset"] == ctx["edaic_pref"] else 0
    if ctx["streams"] and r["stream"] == ctx["streams"][0]:
        s += 0.4
    return -s


TIER = {"current": 0, "CONTESTED": 1, "ALT": 2, "SUPERSEDED": 3, "RETRACTED": 3}


def match_point(it, ctx, idx, dp, pv):
    if ctx["method"] and it["kind"] in ("integer", "sci", "point", "percent"):
        return method_match(it, ctx, idx, pv)
    if "prompt_variant" in (ctx["streams"] or []):
        return dict(status="UNMATCHED", reason="prompt-wording sensitivity figure; no averaged prompt-variant delta is indexed (omni_final/omni_*_p2.json and _p3.json hold the per-wording runs)")
    cands = [r for r in idx.rows if compatible(r, ctx, it["kind"])]
    if not cands:
        hint = [r for r in idx.rows if r["v"] is not None and abs(r["v"]) <= 1.5 and r["model"] in (ctx["models"] or MODELS)
                and rnd(r["v"], dp) == pv and r["status"] != "SUPERSEDED" and r.get("metric") == "auc"][:3]
        h = "; hint: equal after rounding: " + " / ".join("%s=%s (%s)" % (r["key"], r["value"], os.path.basename(r["file"])) for r in hint) if hint else ""
        return dict(status="UNMATCHED", reason="no indexed value for context %s%s" % (ctx_str(ctx), h))
    cands.sort(key=lambda r: (TIER.get(r["status"], 3), rank_key(r, ctx)))
    val = (lambda r: abs(r["v"])) if ctx["absval"] else (lambda r: r["v"])
    if ctx["bound"]:
        live = [r for r in cands if r["status"] not in ("SUPERSEDED", "ALT")] or cands
        ext = max(live, key=val) if ctx["bound"] == "upper" else min(live, key=val)
        ok = val(ext) < pv if ctx["bound"] == "upper" else val(ext) > pv
        return dict(status="MATCH" if ok else "MISMATCH", best=ext,
                    reason="bound claim (%s %s): %s over %d compatible values is %s (%s, %s)" % (
                        "below" if ctx["bound"] == "upper" else "above", it["text"], "max" if ctx["bound"] == "upper" else "min",
                        len(live), fmtv(val(ext)), ext["key"], ext["estimator"][:60]))
    tier0 = TIER.get(cands[0]["status"], 3)
    top_tier = [r for r in cands if TIER.get(r["status"], 3) == tier0]
    good = [r for r in top_tier if rnd(val(r), dp) == pv]
    lower = [r for r in cands if TIER.get(r["status"], 3) > tier0 and rnd(val(r), dp) == pv]
    top = top_tier[0]
    others = sorted({"%s (%s %s, %s)" % (r["value"], r["key"].split("|")[1], r["estimator"][:40], r["status"])
                     for r in top_tier if rnd(val(r), dp) != pv})
    note_best = ""
    if ctx["best"]:
        dss = {r["dataset"] for r in top_tier}
        mx = []
        for d in dss:
            for m in ctx["models"]:
                for stp in PROBES:
                    c = idx.canonical(m, d, stp)
                    if c:
                        mx.append(c)
        if mx:
            b = max(mx, key=lambda r: r["v"])
            note_best = "; 'best probe' check: the highest probe on %s is %s = %s" % (b["dataset"], b["stream"], b["value"])
    if good:
        best = good[0]
        why = "context %s; tier %s" % (ctx_str(ctx), best["status"])
        if others:
            why += "; same-tier values that round differently: " + ", ".join(others[:4])
        if best["status"] != "current":
            why += "; NOTE %s" % best["status_note"][:200]
        return dict(status="MATCH", best=best, reason=why + note_best)
    why = "context %s; best %s candidate %s = %s (%s) rounds to %s" % (ctx_str(ctx), top["status"], top["key"], top["value"],
                                                                        top["estimator"][:60], fmtv(rnd(val(top), dp), dp))
    if others:
        why += "; same-tier values: " + ", ".join(others[:4])
    if lower:
        why += "; paper value equals lower-tier %s %s = %s (%s)" % (lower[0]["status"], lower[0]["key"], lower[0]["value"], os.path.basename(lower[0]["file"]))
    return dict(status="MISMATCH", best=top, reason=why + note_best)


def method_match(it, ctx, idx, pv):
    sent = it["context"]
    if it["kind"] == "sci" or re.search(r"fine-tun|projector|Fine-tuning|speaker-disjoint folds", sent):
        key = "lr" if it["kind"] == "sci" or "learning rate" in it["clause"] else ("folds" if "fold" in it["clause"] else ("epochs" if "epoch" in it["clause"] else ""))
        if key:
            rs = [r for r in idx.rows if r["stream"] == "projector_ft_config_" + key and r["status"] != "SUPERSEDED"]
            vals = sorted({r["v"] for r in rs})
            if rs:
                st = "MATCH" if all(abs(v - pv) < 1e-12 for v in vals) else "MISMATCH"
                return dict(status=st, best=rs[0], reason="fine-tune config '%s' across %d omnisft/abl json files: %s" % (key, len(rs), vals))
    return dict(status="UNMATCHED", reason="method/design constant, not a result; no file value indexed", method=True)


def match_count(it, ctx, idx, pv):
    ds = ctx["datasets"]
    col = ctx["streams"][0] if ctx["streams"] and ctx["streams"][0].startswith("count_") else ""
    clause = it["clause"] + " " + it["context"]
    if not col:
        if re.search(r"speakers?\b|participants", it["clause"]):
            col = "count_n_spk"
        elif re.search(r"clips|segments|recordings|pairs|training|test", it["clause"]):
            col = "count_n"
    rows = [r for r in idx.rows if (not ds or r["dataset"] in ds) and r["status"] != "SUPERSEDED"]
    if "pitt" in ds and re.search(r"age(?:[- ]and[- ]sex)? matched", clause):
        rows = [r for r in idx.rows if r["dataset"] == "pitt_agematched"] or rows
    if not ds and not rows:
        return dict(status="UNMATCHED", reason="no dataset in context for a count")
    fields = ["n_i", "nspk_i"] if not col else (["n_i"] if col == "count_n" else ["nspk_i", "n_i"])
    hit = [r for r in rows if any(r[f] is not None and abs(r[f] - pv) < 1e-9 for f in fields)]
    if hit:
        hit.sort(key=lambda r: rank_key(r, ctx))
        return dict(status="MATCH", best=hit[0], reason="count: %d index rows for %s carry %s = %d (e.g. %s)" % (
            len(hit), ",".join(ds) or "any dataset", "/".join(f.replace("_i", "").replace("nspk", "n_spk") for f in fields), pv, hit[0]["row_id"]),
            count_field=fields[0])
    if it["table"] and ds:
        vals = defaultdict(int)
        for r in rows:
            if r["source_table"] == "master_lookup" and r[fields[0]] is not None:
                vals[int(r[fields[0]])] += 1
        if vals:
            common = max(vals.items(), key=lambda kv: kv[1])[0]
            return dict(status="MISMATCH", reason="table count: master_lookup %s for %s is %d (%d rows); paper %d" % (
                fields[0], ",".join(ds), common, vals[common], pv))
    if ctx["method"]:
        return dict(status="UNMATCHED", reason="method/design constant, not a result; no file value indexed", method=True)
    return dict(status="UNMATCHED", reason="count %d not found as n or n_spk in index rows for %s" % (pv, ",".join(ds) or "(no dataset)"))


def ctx_str(ctx):
    return "[model=%s%s; data=%s; stream=%s; arm=%s%s%s]" % (
        "/".join(ctx["models"])[:60], " (default)" if ctx["model_default"] else "", "/".join(ctx["datasets"])[:60] or (ctx["group"] or "any"),
        "/".join(ctx["streams"])[:60] or "any", ctx["arm"] or "all", "; delta" if ctx["delta"] else "",
        "; " + ", ".join(ctx["notes"]) if ctx["notes"] else "")


# ----------------------------------------------------------------------------- named sets for ranges
def named_sets(idx):
    sets = []
    for m in MODELS:
        for st in ["zero_shot", "transcript", "encoder_probe", "lm_probe", "answer_state_probe", "projector_probe", "projector_ft"]:
            for g, dss in GROUPS.items():
                vals = []
                for d in dss:
                    r = idx.canonical(m, d, st)
                    if r is None:
                        break
                    vals.append((d, r))
                else:
                    sets.append(dict(name="%s %s, %s" % (m, st, g), model=[m], stream=st, group=g, vals=vals))
        for g, dss in GROUPS.items():
            vals = []
            for d in dss:
                best = None
                for st in PROBES:
                    r = idx.canonical(m, d, st)
                    if r and (best is None or r["v"] > best["v"]):
                        best = r
                if best is None:
                    break
                vals.append((d, best))
            else:
                sets.append(dict(name="%s best probe (max over encoder/LM/answer-state/projector), %s" % (m, g), model=[m], stream="best_probe", group=g, vals=vals))
    # single-split encoder (Qwen2.5-Omni, M7 per-clip OOF csv)
    ss = {r["dataset"]: r for r in idx.rows if r["row_id"].startswith("part16/rows/M7.tsv") and r["stream"] == "encoder_probe" and r["is_delta"] == "no"}
    for g, dss in GROUPS.items():
        if all(d in ss for d in dss):
            sets.append(dict(name="Qwen2.5-Omni encoder probe SINGLE-SPLIT OOF (M7), %s" % g, model=["Qwen2.5-Omni"], stream="encoder_probe_single", group=g,
                             vals=[(d, ss[d]) for d in dss]))
    # encoder probe minus eGeMAPS
    em = {r["dataset"]: r for r in idx.rows if r["stream"] == "enc_minus_egemaps"}
    for g, dss in GROUPS.items():
        if all(d in em for d in dss):
            sets.append(dict(name="Qwen2.5-Omni encoder probe (5-repeat) minus eGeMAPS (Sep 18 OOF), %s" % g, model=["Qwen2.5-Omni"], stream="enc_minus_egemaps",
                             group=g, vals=[(d, em[d]) for d in dss]))
    # Pitt conflict drop for the three Qwen models (agreement minus conflict)
    drop, drop2 = [], []
    for m in ["Qwen2-Audio", "Qwen2.5-Omni", "Qwen3-Omni"]:
        rs = [r for r in idx.rows if r["model"] == m and r["dataset"] == "pitt" and r["stream"] == "zero_shot" and r["arm"] == "conflict-agreement"
              and r["row_id"].startswith("part16/rows/M8.tsv")]
        if rs:
            drop.append((m, dict(rs[0], v=-rs[0]["v"])))
        c = [r for r in idx.rows if r["model"] == m and r["dataset"] == "pitt" and r["arm"] == "conflict" and r["row_id"].startswith("part16/rows/M8.tsv")]
        a = [r for r in idx.rows if r["model"] == m and r["dataset"] == "pitt" and r["arm"] == "agreement" and r["row_id"].startswith("part16/rows/M8.tsv")]
        if c and a:
            drop2.append((m, dict(a[0], v=rnd(a[0]["v"], 2) - rnd(c[0]["v"], 2), value="%.2f-%.2f" % (rnd(a[0]["v"], 2), rnd(c[0]["v"], 2)))))
    if len(drop) == 3:
        sets.append(dict(name="Pitt agreement minus conflict, three Qwen models, unrounded paired AUCs (M8)", model=["Qwen2-Audio", "Qwen2.5-Omni", "Qwen3-Omni"],
                         stream="arm_drop", group="Pitt", vals=drop))
    if len(drop2) == 3:
        sets.append(dict(name="Pitt agreement minus conflict, three Qwen models, from the 2-dp table cells", model=["Qwen2-Audio", "Qwen2.5-Omni", "Qwen3-Omni"],
                         stream="arm_drop_rounded", group="Pitt", vals=drop2))
    return sets


def set_minmax(s):
    vs = [(d, r["v"]) for d, r in s["vals"]]
    lo = min(vs, key=lambda x: x[1])
    hi = max(vs, key=lambda x: x[1])
    return lo, hi


def match_range(it, ctx, sets, idx):
    lo_p, hi_p = float(it["value"]), float(it["value2"])
    dp1 = int(it["decimals"] or 2)
    dp2 = int(it["decimals2"] or dp1)
    sent = it["context"]
    labels = it["range_labels"].split("|") if it["range_labels"] else []
    want_models = ctx["models"]
    want_group = ctx["group"]
    if not want_group and ctx["datasets"]:
        dd = set(ctx["datasets"])
        want_group = next((g for g, v in GROUPS.items() if set(v) == dd), "")
    want_streams = set(ctx["streams"])
    if re.search(r"eGeMAPS", sent):
        want_streams = {"enc_minus_egemaps"}
    if re.search(r"drop by", sent) and re.search(r"conflict", sent):
        want_streams = {"arm_drop", "arm_drop_rounded"}
    if want_streams & set(PROBES):
        want_streams |= {"best_probe", "encoder_probe_single"}

    def compat(s):
        if want_streams and s["stream"] not in want_streams:
            return False
        if want_models and not set(s["model"]) & set(want_models):
            return False
        if want_group:
            if want_group == "ALL7":
                return s["group"].startswith("ALL7")
            if want_group == "PD+AD":
                return s["group"].startswith("PD+AD")
            return s["group"] == want_group or s["group"] == "Pitt"
        return True

    comp = [s for s in sets if compat(s)]
    hits = []
    for s in comp:
        (dlo, vlo), (dhi, vhi) = set_minmax(s)
        if rnd(vlo, dp1) == lo_p and rnd(vhi, dp2) == hi_p:
            lab_ok = True
            if labels and labels[0]:
                lab_ok = LABEL_DS.get(labels[0].lower()) == dlo and (not labels[1] or LABEL_DS.get(labels[1].lower()) == dhi)
            hits.append((s, dlo, vlo, dhi, vhi, lab_ok))
    desc = lambda s: "; ".join("%s %s" % (d, fmtv(r["v"])) for d, r in s["vals"])  # noqa: E731
    strict = [h for h in hits if "without" not in h[0]["name"]]
    if hits and not strict and want_group in ("ALL7", "PD+AD"):
        s, dlo, vlo, dhi, vhi, lab_ok = hits[0]
        full = [s2 for s2 in comp if "without" not in s2["name"] and s2["stream"] == s["stream"]]
        fx = ""
        if full:
            (a1, v1), (b1, v2) = set_minmax(full[0])
            fx = "; over the set the sentence names ('%s') the range is %s (%s) to %s (%s)" % (full[0]["name"], fmtv(v1), a1, fmtv(v2), b1)
        return dict(status="MISMATCH", reason="range matches only a REDUCED set '%s' (min %s %s, max %s %s)%s; members of reduced set: %s" % (
            s["name"], fmtv(vlo), dlo, fmtv(vhi), dhi, fx, desc(s)),
            file_value="%s to %s" % (fmtv(set_minmax(full[0])[0][1]), fmtv(set_minmax(full[0])[1][1])) if full else "%s to %s" % (fmtv(vlo), fmtv(vhi)),
            file_path=" ; ".join(sorted({os.path.basename(r["file"]) for _, r in (full[0] if full else s)["vals"]}))[:300],
            estimator=(full[0] if full else s)["name"], best_rows=[r for _, r in s["vals"]])
    if hits:
        hits = strict or hits
        hits.sort(key=lambda h: (not h[5], "SINGLE" in h[0]["name"], "best probe" in h[0]["name"], "without" in h[0]["name"]))
        s, dlo, vlo, dhi, vhi, lab_ok = hits[0]
        others = [h[0]["name"] for h in hits[1:]][:3]
        miss = []
        for s2 in comp:
            if s2 is s or any(s2 is h[0] for h in hits):
                continue
            (a, va), (b, vb) = set_minmax(s2)
            miss.append("%s gives %s to %s" % (s2["name"], fmtv(rnd(va, dp1), dp1), fmtv(rnd(vb, dp2), dp2)))
        reason = "range matches SET '%s': min %s (%s) max %s (%s); members: %s" % (s["name"], fmtv(vlo), dlo, fmtv(vhi), dhi, desc(s))
        if labels and not lab_ok:
            reason += "; WARNING endpoint labels in text (%s) do not name the min/max datasets" % "/".join(labels)
        if others:
            reason += "; also matches: " + " | ".join(others)
        if miss:
            reason += "; compatible sets that do NOT match: " + " | ".join(miss[:4])
        return dict(status="MATCH" if lab_ok else "MISMATCH", reason=reason, file_value="%s to %s" % (fmtv(vlo), fmtv(vhi)),
                    file_path=" ; ".join(sorted({os.path.basename(r["file"]) for _, r in s["vals"]}))[:300], estimator=s["name"],
                    best_rows=[r for _, r in s["vals"]])
    if comp:
        comp.sort(key=lambda s: (abs(rnd(set_minmax(s)[0][1], dp1) - lo_p) + abs(rnd(set_minmax(s)[1][1], dp2) - hi_p),
                                 "SINGLE" in s["name"], "best probe" in s["name"]))
        s = comp[0]
        (dlo, vlo), (dhi, vhi) = set_minmax(s)
        near = ["%s gives %s to %s" % (s2["name"], fmtv(rnd(set_minmax(s2)[0][1], dp1), dp1), fmtv(rnd(set_minmax(s2)[1][1], dp2), dp2)) for s2 in comp[1:4]]
        return dict(status="MISMATCH", reason="no compatible named set matches both ends; closest SET '%s' gives %s (%s) to %s (%s); members: %s%s" % (
            s["name"], fmtv(vlo), dlo, fmtv(vhi), dhi, desc(s), ("; others: " + " | ".join(near)) if near else ""),
            file_value="%s to %s" % (fmtv(vlo), fmtv(vhi)), file_path=" ; ".join(sorted({os.path.basename(r["file"]) for _, r in s["vals"]}))[:300],
            estimator=s["name"], best_rows=[r for _, r in s["vals"]])
    # interval-like: maybe [lo, hi] of a point
    return dict(status="UNMATCHED", reason="no named set compatible with context %s (streams %s, group %s)" % (ctx_str(ctx), sorted(want_streams), want_group or "any"))


def match_word(it, ctx, idx):
    pv = float(it["value"])
    sent = it["context"]
    if it["unit"] == "word_years" or re.search(r"years", sent[sent.find(it["text"]):sent.find(it["text"]) + 20]):
        rs = [r for r in idx.rows if r["stream"] == "age_gap_years"]
        pitt = [r for r in rs if r["dataset"] == "pitt"] or rs
        good = [r for r in pitt if rnd(r["v"], 0) == pv]
        allv = "; ".join("%s=%s (%s)" % (r["key"].split("|")[1], fmtv(r["v"], 2), r["estimator"][:60]) for r in rs)
        if good:
            return dict(status="MATCH", best=good[0], reason="age gap %s years rounds to %d under '%s'; all definitions: %s" % (
                fmtv(good[0]["v"], 2), pv, good[0]["estimator"][:80], allv))
        return dict(status="MISMATCH", best=pitt[0] if pitt else None, reason="no age-gap definition rounds to %d: %s" % (pv, allv))
    if re.search(r"below chance", sent):
        out = []
        from_sent = not any(n.startswith("dataset") for n in ctx["notes"])
        dsl = [d for d in ctx["datasets"] if d in ("edaic_conflict_v2", "pitt")] if from_sent else []
        for ds in (dsl or ["edaic_conflict_v2", "pitt"]):
            dsx = "edaic_conflict_v2" if ds in EDAIC_FAMILY else ds
            below = []
            for m in MODELS:
                r = [x for x in idx.rows if x["model"] == m and x["dataset"] == dsx and x["arm"] == "conflict" and x["stream"] == "zero_shot"
                     and x["status"] != "SUPERSEDED" and (x["row_id"].startswith("part16/rows/M8.tsv") or x["row_id"].startswith("part17/rows/T1.tsv"))]
                if r and r[0]["v"] < 0.5:
                    below.append("%s %s" % (m, fmtv(r[0]["v"])))
            out.append((dsx, len(below), below))
        ok = [d for d, n, b in out if n == pv]
        st = "MATCH" if ok else "MISMATCH"
        scope = "" if len(out) == 1 else ("; the sentence names no dataset: holds on %s only" % ",".join(ok) if ok else "")
        return dict(status=st, reason="count of models with conflict-arm AUC < 0.5 (M8/T1 primary files)" + scope + ": " + "; ".join(
            "%s: %d of 6 (%s)" % (d, n, ", ".join(b)) for d, n, b in out))
    return dict(status="UNMATCHED", reason="word number without a checkable pattern")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers")
    ap.add_argument("value_index")
    ap.add_argument("--tex", required=True)
    ap.add_argument("--added", default="")
    ap.add_argument("--red-spec", default="")
    ap.add_argument("--include-comments", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper-vs-omni", default="")
    a = ap.parse_args()
    idx = Index(a.value_index)
    sets = named_sets(idx)
    tex = open(a.tex, encoding="utf-8").read()
    flags = dict(whole_interview=bool(re.search(r"whole interview|capped at 15 minutes|full interview", tex)))
    items = list(csv.DictReader(open(a.numbers, newline="")))
    added_lines, added_texts = set(), set()
    norm = lambda t: re.sub(r"\s+", " ", t or "").strip()  # noqa: E731
    if a.added and os.path.exists(a.added):
        for r in csv.DictReader(open(a.added, newline="")):
            if r["status"] not in ("added", "edited"):
                continue
            if r["unit"] == "table_row":
                for ln in range(int(r["line_start"]), int(r["line_end"]) + 1):
                    added_lines.add(ln)
            else:
                added_texts.add(norm(r["text"]))
    red_lines = {}
    if a.red_spec and os.path.exists(a.red_spec):
        for r in csv.DictReader(open(a.red_spec, newline="")):
            red_lines.setdefault(r["span_id"], r)
    out = []
    for it in items:
        if it["in_comment"] == "yes" and not a.include_comments:
            continue
        ctx = item_context(it, flags)
        ctx["edaic_pref"] = "edaic_full" if flags["whole_interview"] else "edaic"
        kind = it["kind"]
        res_list = []
        if kind == "range":
            res_list.append((it["text"], match_range(it, ctx, sets, idx)))
        elif kind == "change":
            for k, (v, dp) in enumerate(((it["value"], it["decimals"]), (it["value2"], it["decimals2"]))):
                sub = dict(it, text=str(v))
                c2 = dict(ctx)
                if k == 1 and re.search(r"trained on", it["context"]):
                    c2["streams"] = ["projector_transfer"]
                elif k == 0:
                    c2["streams"] = ["zero_shot"]
                r = match_point(sub, c2, idx, int(dp or 2), rnd(float(v), int(dp or 2)))
                r["reason"] = ("before: " if k == 0 else "after: ") + r["reason"]
                res_list.append(("%s (%s)" % (v, "before" if k == 0 else "after"), r))
        elif kind.startswith("word"):
            res_list.append((it["text"], match_word(it, ctx, idx)))
        elif kind == "integer" and not ctx["method"] or (kind == "integer" and it["table"]):
            res_list.append((it["text"], match_count(it, ctx, idx, float(it["value"]))))
        elif kind == "percent":
            v = float(it["value"]) / 100.0
            if re.search(r"identical|same (Yes|decision)", it["context"]):
                r = match_point(dict(it), dict(ctx, streams=["text_audio_same_decision"]), idx, int(it["decimals"] or 0) + 2, rnd(v, int(it["decimals"] or 0) + 2))
            elif ctx["method"]:
                r = dict(status="UNMATCHED", reason="method/design constant, not a result; no file value indexed", method=True)
            else:
                r = dict(status="UNMATCHED", reason="percentage with no indexed file value for context %s" % ctx_str(ctx))
            res_list.append((it["text"], r))
        else:
            dp = int(it["decimals"] or 0)
            res_list.append((it["text"], match_point(it, ctx, idx, dp, rnd(float(it["value"]), dp) if kind != "sci" else float(it["value"]))))
        for label, res in res_list:
            best = res.get("best")
            fv = res.get("file_value") or (best["value"] if best else "")
            if res.get("count_field") and best:
                fv = best["n"] if res["count_field"] == "n_i" else best["n_spk"]
            fp = res.get("file_path") or (best["file"] if best else "")
            est = res.get("estimator") or (best["estimator"] if best else "")
            grp = "other"
            if it["in_red"] == "yes":
                grp = "red"
            elif (it["table"] and int(it["line"]) in added_lines) or (not it["table"] and (
                    norm(it["context"]) in added_texts or any(len(t) > 20 and norm(it["context"]).startswith(t) for t in added_texts))):
                grp = "added_23sep"
            out.append(dict(order_group=grp, line=it["line"], context=(it["table_key"] + " :: " if it["table_key"] else "") + it["context"][:400],
                            paper_value=label, file_value=fv, file_path=fp, estimator=est[:200], status=res["status"],
                            reason=res["reason"][:900], item_id=it["item_id"], kind=kind, section=it["section"],
                            value_key=best["key"] if best else "", value_status=best["status"] if best else "",
                            value_row=best["row_id"] if best else "", method_constant="yes" if res.get("method") else "no",
                            in_comment=it["in_comment"]))
    F = ["order_group", "line", "context", "paper_value", "file_value", "file_path", "estimator", "status", "reason", "item_id",
         "kind", "section", "value_key", "value_status", "value_row", "method_constant", "in_comment"]
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=F)
        w.writeheader()
        w.writerows(out)
    cnt = defaultdict(int)
    for r in out:
        cnt[(r["order_group"], r["status"])] += 1
    print("match_tex: %d rows -> %s ; %s" % (len(out), a.out, dict(cnt)))


if __name__ == "__main__":
    main()
