#!/usr/local/bin/python3
"""G (last step). Write paper_vs_omni.csv in the authors' order: red spans, then numbers in sentences added since
the baseline, then everything else. Also writes a summary of MATCH / MISMATCH / UNMATCHED per group.

Usage: assemble_part12.py MATCHES.csv RED_SPANS.csv --tex TEX --added-basis TEXT --out paper_vs_omni.csv --summary summary.txt
"""
import argparse
import csv
import os
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("matches")
ap.add_argument("red")
ap.add_argument("--tex", required=True)
ap.add_argument("--added-basis", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--summary", required=True)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.matches, newline="")))
red = list(csv.DictReader(open(a.red, newline="")))
red_ids = {r["item_id"]: r["span_id"] for r in red if r["item_id"]}
out = []
for r in rows:
    g = "red" if (r["item_id"] in red_ids or r["order_group"] == "red") else r["order_group"]
    out.append(dict(order_group=g, line=r["line"], context=r["context"], paper_value=r["paper_value"], file_value=r["file_value"],
                    file_path=r["file_path"], estimator=r["estimator"], status=r["status"],
                    red_span=red_ids.get(r["item_id"], ""), reason=r["reason"], value_status=r["value_status"],
                    method_constant=r["method_constant"], kind=r["kind"], section=r["section"], item_id=r["item_id"],
                    added_basis=a.added_basis if g == "added_23sep" else "", tex_file=a.tex))
for r in red:
    if r["paper_value"] == "(claim)":
        out.append(dict(order_group="red", line=r["line"], context=r["found_text"], paper_value="(claim, no number)", file_value=r["file_value"],
                        file_path=r["file_path"], estimator=r["estimator"], status=r["status"], red_span=r["span_id"], reason=r["note"],
                        value_status="", method_constant="no", kind="claim", section="", item_id=r["item_id"], added_basis="", tex_file=a.tex))
order = {"red": 0, "added_23sep": 1, "other": 2}
out.sort(key=lambda r: (order.get(r["order_group"], 3), int(r["line"] or 0)))
F = ["order_group", "line", "context", "paper_value", "file_value", "file_path", "estimator", "status",
     "red_span", "reason", "value_status", "method_constant", "kind", "section", "item_id", "added_basis", "tex_file"]
with open(a.out, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=F)
    w.writeheader()
    w.writerows(out)
c = Counter((r["order_group"], r["status"]) for r in out)
cm = Counter((r["order_group"], r["status"]) for r in out if r["method_constant"] == "yes")
lines = ["paper_vs_omni: %s" % a.out, "tex: %s" % a.tex, "added basis: %s" % a.added_basis, ""]
lines.append("%-12s %6s %9s %10s   (of UNMATCHED: method/design constants)" % ("group", "MATCH", "MISMATCH", "UNMATCHED"))
for g in ("red", "added_23sep", "other"):
    lines.append("%-12s %6d %9d %10d   (%d)" % (g, c[(g, "MATCH")], c[(g, "MISMATCH")], c[(g, "UNMATCHED")], cm[(g, "UNMATCHED")]))
lines.append("")
lines.append("MISMATCH rows:")
for r in out:
    if r["status"] == "MISMATCH":
        lines.append("  [%s] line %s  paper %s  file %s  %s" % (r["order_group"], r["line"], r["paper_value"], str(r["file_value"])[:40],
                                                             os.path.basename(str(r["file_path"]).split(" ; ")[0])[:80]))
open(a.summary, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:10]))
