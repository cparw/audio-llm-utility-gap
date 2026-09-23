#!/usr/local/bin/python3
"""E (script half). Check the authors' red spans.

Spans come from red_spans_spec.csv (the eight spans the authors named, located by table cell or regex)
plus any \\textcolor{red} / \\color{red} markup the extractor found. For each number inside a span it
reports the matcher's verdict AND every compatible file value in value_index.csv, all tiers
(current / CONTESTED / ALT / SUPERSEDED), so two disagreeing runs are both visible.
Red markup without numbers is checked only when the spec names a claim_check.

Usage: red_span_check.py NUMBERS.csv MATCHES.csv VALUE_INDEX.csv --spec red_spans_spec.csv --out red_spans.csv
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_tex as M  # noqa: E402
from p12common import rnd  # noqa: E402


def candidates(it, idx):
    ctx = M.item_context(it, {})
    ctx["edaic_pref"] = "edaic_full"
    rows = []
    for r in idx.rows:
        rr = dict(r)
        if M.compatible(rr, ctx, it["kind"]):
            rows.append(rr)
    rows.sort(key=lambda r: (M.TIER.get(r["status"], 3), M.rank_key(r, ctx)))
    seen, out = set(), []
    for r in rows:
        k = (r["value"], r["status"], os.path.basename(r["file"]))
        if k in seen:
            continue
        seen.add(k)
        out.append("%s [%s; %s; %s]" % (r["value"], r["status"], r["estimator"][:70], r["file"]))
    return out[:8]


def ft_gain_by_condition(idx):
    rows = [r for r in idx.rows if r["source_table"] == "part15/C_intervals.csv"]
    parts, bad = [], []
    for r in rows:
        d = r["description"]
        if "projector gain, paired" in d:
            ds = d.split()[1] if len(d.split()) > 1 else "?"
            parts.append("%s %+.4f [%s, %s]" % (ds, float(r["value"]), r["lo"], r["hi"]))
            if ds == "edaic" and not (float(r["lo"] or 0) > 0):
                bad.append("E-DAIC first-30 s projector gain %+.4f [%s, %s] does not exclude zero" % (float(r["value"]), r["lo"], r["hi"]))
        if "full-window drop" in d:
            parts.append("E-DAIC full window: zero-shot minus fine-tune %+.4f [%s, %s]" % (float(r["value"]), r["lo"], r["hi"]))
            if float(r["value"]) > 0:
                bad.append("E-DAIC full window: fine-tuning LOWERS the AUC by %.4f [%s, %s] (zero-shot 0.8285 vs fine-tuned 0.6421)" % (
                    float(r["value"]), r["lo"], r["hi"]))
    status = "MISMATCH" if bad else "MATCH"
    return status, "; ".join(bad) if bad else "all gains positive", "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers")
    ap.add_argument("matches")
    ap.add_argument("value_index")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [r for r in csv.DictReader(open(a.numbers, newline="")) if r["in_comment"] == "no"]
    matches = list(csv.DictReader(open(a.matches, newline="")))
    by_item = {}
    for m in matches:
        by_item.setdefault(m["item_id"], []).append(m)
    idx = M.Index(a.value_index)
    spec = list(csv.DictReader(open(a.spec, newline="")))
    red_markup = []
    rs_path = os.path.splitext(a.numbers)[0] + "_redspans.csv"
    if os.path.exists(rs_path):
        red_markup = list(csv.DictReader(open(rs_path, newline="")))
    out, flagged = [], set()
    for sp in spec:
        hits = []
        if sp["where"] == "table":
            hits = [it for it in items if it["table"] == sp["table"] and it["table_row"] == sp["table_row"]
                    and it["table_col"].startswith(sp["table_col"])]
        elif sp["where"] == "prose":
            rx = re.compile(sp["regex"])
            for it in items:
                if it["table"]:
                    continue
                for mo in rx.finditer(it["context"]):
                    seg = mo.group(0)
                    if it["text"] in seg or seg in it["text"] or it["text"].split()[0] in seg:
                        hits.append(it)
                        break
        elif sp["where"] == "markup":
            for rm in red_markup:
                if re.search(sp["regex"], rm["text"]):
                    if sp["claim_check"] == "ft_gain_by_condition":
                        st, why, detail = ft_gain_by_condition(idx)
                    else:
                        st, why, detail = "UNMATCHED", "no claim check defined", ""
                    out.append(dict(span_id=sp["span_id"], label=sp["label"], line=rm["line_start"], found_text=rm["text"][:300],
                                    in_red_markup="yes", paper_value="(claim)", status=st, file_value=detail[:700],
                                    file_path=idx.rows and "edaic_rerun/part15/C_intervals.csv", estimator="paired speaker bootstrap, 2000 draws",
                                    candidates="", note=why, item_id="claim_L%s" % rm["line_start"]))
            continue
        if not hits:
            out.append(dict(span_id=sp["span_id"], label=sp["label"], line="", found_text="NOT FOUND in this tex", in_red_markup="",
                            paper_value="", status="UNMATCHED", file_value="", file_path="", estimator="", candidates="",
                            note="span not located; check the wording in the tex", item_id=""))
            continue
        for it in hits:
            flagged.add(it["item_id"])
            for m in by_item.get(it["item_id"], []):
                cands = candidates(it, idx) if it["kind"] not in ("range",) else []
                out.append(dict(span_id=sp["span_id"], label=sp["label"], line=it["line"],
                                found_text=(it["table_key"] or it["context"])[:300], in_red_markup=it["in_red"],
                                paper_value=m["paper_value"], status=m["status"], file_value=m["file_value"], file_path=m["file_path"],
                                estimator=m["estimator"], candidates=" || ".join(cands), note=m["reason"][:700], item_id=it["item_id"]))
    # red markup numbers not covered by the spec
    for it in items:
        if it["in_red"] == "yes" and it["item_id"] not in flagged:
            flagged.add(it["item_id"])
            for m in by_item.get(it["item_id"], []):
                out.append(dict(span_id="MARKUP", label="red markup", line=it["line"], found_text=it["context"][:300], in_red_markup="yes",
                                paper_value=m["paper_value"], status=m["status"], file_value=m["file_value"], file_path=m["file_path"],
                                estimator=m["estimator"], candidates=" || ".join(candidates(it, idx)), note=m["reason"][:700], item_id=it["item_id"]))
    F = ["span_id", "label", "line", "found_text", "in_red_markup", "paper_value", "status", "file_value", "file_path", "estimator",
         "candidates", "note", "item_id"]
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=F)
        w.writeheader()
        w.writerows(out)
    with open(os.path.splitext(a.out)[0] + "_items.txt", "w") as fo:
        fo.write("\n".join(sorted(flagged)) + "\n")
    print("red_span_check: %d rows -> %s ; %s" % (len(out), a.out,
          {s: sum(1 for r in out if r["status"] == s) for s in ("MATCH", "MISMATCH", "UNMATCHED")}))


if __name__ == "__main__":
    main()
