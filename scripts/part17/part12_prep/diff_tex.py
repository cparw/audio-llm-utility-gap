#!/usr/local/bin/python3
"""F. Sentence-level diff: units (prose sentences and table rows, comments excluded) present in NEW but not in BASE.

Each NEW unit gets the best-matching BASE unit (difflib ratio on normalised text):
  same    ratio >= 0.97
  edited  0.60 <= ratio < 0.97
  added   ratio < 0.60
For edited units the numbers that do not appear in the matched BASE unit are listed (new_numbers).
Output rows are written for edited and added units only (use --all to also write unchanged ones).

Usage: diff_tex.py NEW.tex BASE.tex --out added.csv [--all]
"""
import argparse
import csv
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_tex_numbers as X  # noqa: E402

NUM = re.compile(r"(?<![\w.])[+\-]?\d+(?:\.\d+)?")


def units(path):
    text = open(path, encoding="utf-8").read()
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    cm = X.comment_mask(text)
    mo = re.search(r"^[ \t]*\\begin\{document\}", text, re.M)
    d0 = mo.start() if mo else 0
    d1 = text.find("\\end{document}")
    d1 = len(text) if d1 < 0 else d1
    out = []
    tab_spans = [(m.start(), m.end()) for m in re.finditer(r"\\begin\{(tabular\*?)\}.*?\\end\{\1\}", text, re.S)]
    for (ra, rb, rlabel, cells, tlabel, caption, group) in X.parse_tables(text, cm):
        live = "".join(text[k] for k in range(ra, rb) if not cm[k])
        t = X.strip_tex(live)
        if t:
            out.append(dict(kind="table_row", a=ra, b=rb, text="[%s] %s" % (tlabel or "table", t)))
    blocks, cur = [], d0
    for a, b in sorted(tab_spans):
        if a > cur:
            blocks.append((cur, a))
        cur = max(cur, b)
    blocks.append((cur, d1))
    for a, b in blocks:
        for pm in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", text[a:b]):
            pa, pb = a + pm.start(), a + pm.end()
            for sa, sb in X.sentences(text, pa, pb):
                live = "".join(text[k] for k in range(sa, sb) if not cm[k])
                t = X.strip_tex(live)
                if len(re.sub(r"[^A-Za-z0-9]", "", t)) < 3:
                    continue
                out.append(dict(kind="sentence", a=sa, b=sb, text=t))
    for u in out:
        u["line_start"] = X.line_of(starts, u["a"])
        u["line_end"] = X.line_of(starts, max(u["a"], u["b"] - 1))
        u["norm"] = re.sub(r"\s+", " ", re.sub(r"\\[A-Za-z]+", " ", u["text"].lower())).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    new, base = units(a.new), units(a.base)
    base_norm = [b["norm"] for b in base]
    base_set = set(base_norm)
    rows = []
    counts = {"same": 0, "edited": 0, "added": 0}
    for u in new:
        if u["norm"] in base_set:
            ratio, best = 1.0, u["norm"]
            bt = u["text"]
        else:
            ratio, best, bt = 0.0, "", ""
            for b in base:
                sm = difflib.SequenceMatcher(None, u["norm"], b["norm"], autojunk=False)
                if sm.real_quick_ratio() < max(ratio, 0.3) or sm.quick_ratio() < max(ratio, 0.3):
                    continue
                r = sm.ratio()
                if r > ratio:
                    ratio, best, bt = r, b["norm"], b["text"]
        st = "same" if ratio >= 0.97 else ("edited" if ratio >= 0.60 else "added")
        counts[st] += 1
        new_nums = sorted(set(NUM.findall(u["text"])) - set(NUM.findall(bt))) if st != "same" else []
        if st != "same" or a.all:
            rows.append(dict(line_start=u["line_start"], line_end=u["line_end"], unit=u["kind"], status=st,
                             ratio="%.3f" % ratio, new_numbers=" ".join(new_nums), text=u["text"][:1500],
                             best_base_text=bt[:1500]))
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["line_start", "line_end", "unit", "status", "ratio", "new_numbers", "text", "best_base_text"])
        w.writeheader()
        w.writerows(rows)
    print("diff_tex: %s vs %s -> %s ; units same %d, edited %d, added %d" % (
        os.path.basename(a.new), os.path.basename(a.base), a.out, counts["same"], counts["edited"], counts["added"]))


if __name__ == "__main__":
    main()
