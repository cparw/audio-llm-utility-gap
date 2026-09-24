"""Independent smallest-font measure: per character (rawdict), all pages, plus the
text-trace route (page.get_texttrace) which reads the size straight from the content stream ops."""
import sys, collections, pymupdf
doc = pymupdf.open(sys.argv[1])
print("pymupdf", pymupdf.__version__, "pages", doc.page_count)
glob = []
for pno, page in enumerate(doc, 1):
    chars = []
    for b in page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT)["blocks"]:
        if b.get("type") != 0: continue
        for ln in b["lines"]:
            for s in ln["spans"]:
                for c in s["chars"]:
                    if c["c"].strip():
                        chars.append((s["size"], s["font"], c["c"], round(c["bbox"][0], 1), round(c["bbox"][1], 1)))
    tr = [(t["size"], t["font"]) for t in page.get_texttrace() if t["chars"]]
    trmin = min(tr)[0] if tr else None
    if not chars:
        print(f"p{pno}: no text"); continue
    mn = min(chars)
    under9 = [c for c in chars if c[0] < 9.0 - 1e-6]
    print(f"p{pno}: chars {len(chars)}  min rawdict {mn[0]:.3f} pt ({mn[1]}, '{mn[2]}' at x={mn[3]} y={mn[4]})  min texttrace {trmin:.3f}  chars<9pt {len(under9)}")
    sizes = collections.Counter(round(c[0], 2) for c in under9)
    if sizes: print("     under 9 pt by size:", dict(sorted(sizes.items())))
    glob += chars
g = min(glob)
print(f"SMALLEST overall: {g[0]:.4f} pt font {g[1]} char '{g[2]}'")
smallest = sorted({round(c[0], 2) for c in glob})[:6]
for sz in smallest:
    hits = [c for c in glob if round(c[0], 2) == sz]
    print(f"  {sz} pt: {len(hits)} chars, fonts {sorted({h[1] for h in hits})}, text '{''.join(h[2] for h in hits)[:60]}'")
imgs = [(pno, len(p.get_images(full=True))) for pno, p in enumerate(doc, 1)]
print("raster images per page (text inside them cannot be measured):", imgs)
