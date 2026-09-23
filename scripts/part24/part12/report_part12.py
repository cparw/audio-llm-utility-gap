#!/usr/local/bin/python3
"""Print every MISMATCH and UNMATCHED row of a Part 12 paper_vs_omni.csv with its tex line
number and text, and attach the hand triage from triage_last_run.csv to each MISMATCH.

  report_part12.py PAPER_VS_OMNI.csv TRIAGE.csv --tex TEX --out FLAGGED.csv

Triage lookup, in order:
  1. same paper value and same sentence (whitespace normalised)   -> the saved verdict
  2. same paper value and same matcher file value, sentence edited -> 'PROBABLY <verdict>' (re-check)
  3. otherwise                                                     -> 'UNTRIAGED (new since the 06:38 run)'
"""
import argparse, re, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("triage")
ap.add_argument("--tex", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/omni_final/" not in a.out

d = pd.read_csv(a.csv)
tri = pd.read_csv(a.triage)
tex_lines = open(a.tex, encoding="utf-8", errors="replace").read().split("\n")

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()

tri["k1"] = tri.paper_value.map(norm) + "||" + tri.context.map(norm)
tri["k2"] = tri.paper_value.map(norm) + "||" + tri.matcher_file_value.map(norm)
k1 = {r.k1: r for r in tri.itertuples()}
k2 = {r.k2: r for r in tri.itertuples()}

def triage(r):
    t = k1.get(norm(r.paper_value) + "||" + norm(r.context))
    if t is not None:
        return t.triage, t.reason, t.recomputed
    t = k2.get(norm(r.paper_value) + "||" + norm(r.file_value))
    if t is not None:
        return f"PROBABLY {t.triage} (sentence edited; was line {t.line})", t.reason, t.recomputed
    return "UNTRIAGED (new since the 06:38 run)", "", ""

def texline(n):
    try:
        return tex_lines[int(n) - 1]
    except Exception:
        return ""

cnt = d.groupby(["order_group", "status"]).size().unstack(fill_value=0)
print("Part 12 status counts by group")
print(cnt.to_string())
print()

flag = d[d.status.isin(["MISMATCH", "UNMATCHED"])].copy()
flag["triage"] = ""; flag["triage_reason"] = ""; flag["triage_recomputed"] = ""
flag["tex_line_text"] = flag.line.map(texline)

mm = flag[flag.status == "MISMATCH"].sort_values(["line"])
print(f"MISMATCH rows: {len(mm)}")
for i, r in mm.iterrows():
    t, why, rec = triage(r)
    flag.loc[i, ["triage", "triage_reason", "triage_recomputed"]] = [t, why, rec]
    print(f"- tex line {int(r.line)} [{r.order_group}]  paper {r.paper_value}  |  matcher file {str(r.file_value)[:80]}  ({str(r.file_path).split('/')[-1][:60]})")
    print(f"    text: {norm(r.context)}")
    print(f"    TRIAGE: {t}")
    if why:
        print(f"    why: {why}")
    if rec:
        print(f"    recomputed: {rec}")
print()
tc = flag.loc[mm.index, "triage"].map(lambda s: s.split(" (")[0] if s.startswith(("UNTRIAGED", "PROBABLY")) else s)
print("MISMATCH triage counts: " + ", ".join(f"{k} {v}" for k, v in tc.value_counts().items()))
print()

um = flag[flag.status == "UNMATCHED"].sort_values(["line"])
print(f"UNMATCHED rows: {len(um)}  (method/design constants: {(um.method_constant == 'yes').sum()})")
for _, r in um.iterrows():
    mc = " [method constant]" if r.method_constant == "yes" else ""
    print(f"- tex line {int(r.line)} [{r.order_group}]  paper {r.paper_value}{mc}  |  {norm(r.reason)[:220]}")
    print(f"    text: {norm(r.context)}")

cols = ["status", "order_group", "line", "paper_value", "file_value", "file_path", "triage", "triage_reason",
        "triage_recomputed", "method_constant", "kind", "section", "item_id", "reason", "context", "tex_line_text"]
flag[cols].to_csv(a.out, index=False)
print(f"\nflagged rows with triage written to {a.out}")
