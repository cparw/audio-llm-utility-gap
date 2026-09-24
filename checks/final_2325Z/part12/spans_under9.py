"""Every PyMuPDF text span in a PDF with size < 9 pt: page, size, font, bbox, text.
  venv/bin/python spans_under9.py PDF OUT.csv
Whitespace-only spans skipped (same rule as part24 pdf_fonts.py)."""
import sys, csv, collections, pymupdf
pdf, out = sys.argv[1], sys.argv[2]
doc = pymupdf.open(pdf)
rows, allsz = [], collections.Counter()
for pno, page in enumerate(doc, 1):
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0: continue
        for ln in b["lines"]:
            for s in ln["spans"]:
                t = s["text"].strip()
                if not t: continue
                allsz[round(s["size"], 2)] += 1
                if s["size"] < 9.0:
                    rows.append(dict(page=pno, size=round(s["size"], 3), font=s["font"],
                                     x0=round(s["bbox"][0], 1), y0=round(s["bbox"][1], 1), text=t))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["page", "size", "font", "x0", "y0", "text"]); w.writeheader(); w.writerows(rows)
print(f"pymupdf {pymupdf.__version__}; pages {doc.page_count}; spans total {sum(allsz.values())}; spans < 9 pt {len(rows)}")
print("smallest span size:", min(allsz))
print("spans < 8.8 pt (every one):")
for r in sorted(rows, key=lambda r: (r['size'], r['page'], r['y0'], r['x0'])):
    if r["size"] < 8.8:
        print(f"  p{r['page']}  {r['size']:.2f} pt  {r['font']:<22} y={r['y0']:<6} '{r['text']}'")
print("spans 8.8 to < 9 pt, count by page:", dict(collections.Counter(r['page'] for r in rows if r['size'] >= 8.8)))
print("spans 8.8 to < 9 pt, count by font:", dict(collections.Counter(r['font'] for r in rows if r['size'] >= 8.8)))
