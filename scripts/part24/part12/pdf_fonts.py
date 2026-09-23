"""Smallest and largest font size in a PDF, with where they occur. Run with the part24 venv python (PyMuPDF).

  venv/bin/python pdf_fonts.py PDF [--json OUT.json]

Size is PyMuPDF's span size, which includes the text matrix, so text inside a scaled
figure is reported at the size it prints at. Whitespace-only spans are skipped.
"""
import argparse, collections, json
import pymupdf

ap = argparse.ArgumentParser()
ap.add_argument("pdf"); ap.add_argument("--json")
a = ap.parse_args()

doc = pymupdf.open(a.pdf)
chars = collections.Counter()
where = collections.defaultdict(list)
for pno, page in enumerate(doc, 1):
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            for s in ln["spans"]:
                t = s["text"].strip()
                if not t:
                    continue
                sz = round(s["size"], 2)
                chars[sz] += len(t)
                where[sz].append((pno, s["font"], t[:50]))

if not chars:
    print("no text spans found (scanned PDF?)")
    raise SystemExit(0)
lo, hi = min(chars), max(chars)
body = chars.most_common(1)[0][0]
print(f"pages {doc.page_count}; distinct sizes {len(chars)}; most common (body) {body} pt")
print(f"SMALLEST font size: {lo} pt  ({chars[lo]} chars)")
for p, f, t in where[lo][:5]:
    print(f"    page {p}  {f}  '{t}'")
print(f"LARGEST font size:  {hi} pt  ({chars[hi]} chars)")
for p, f, t in where[hi][:3]:
    print(f"    page {p}  {f}  '{t}'")
under9 = sorted(s for s in chars if s < 9.0)
if under9:
    print("sizes under 9 pt (size: chars, pages, first example):")
    for s in under9:
        pages = sorted({p for p, _, _ in where[s]})
        ex = "; ".join(f"p{p} {f} '{t}'" for p, f, t in where[s][:3])
        print(f"    {s} pt: {chars[s]} chars, pages {pages}, e.g. {ex}")
else:
    print("no text under 9 pt")
print("note: sizes a few percent under a round size (e.g. 8.9 for 9) usually come from font expansion or a scaled box")
if a.json:
    json.dump(dict(pdf=a.pdf, pages=doc.page_count, smallest=lo, largest=hi, body=body,
                   sizes={str(k): dict(chars=v, pages=sorted({p for p, _, _ in where[k]}),
                                       examples=where[k][:5]) for k, v in sorted(chars.items())},
                   pymupdf=pymupdf.__version__), open(a.json, "w"), indent=1)
