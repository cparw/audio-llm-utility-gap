"""Font size audit of the compiled paper, own PyMuPDF code (fonts track, 24 Sep).

usage: python3 fonts_under9.py FINAL.pdf OUTDIR
writes: OUTDIR/raw_operators.txt   every Tf size and Tm/cm scale in the page streams and the figure XObject
        OUTDIR/spans_under_9p0bp.csv   every non-blank PyMuPDF span with size < 9.0 bp, with class
        OUTDIR/spans_below_9texpt.csv  the subset whose nominal (Tf) size is below 9 TeX pt = 8.9664 bp
"""
import sys, re, csv, collections, fitz

PDF, OUT = sys.argv[1], sys.argv[2]
doc = fitz.open(PDF)
TEX9 = 9 * 72 / 72.27            # 9 TeX pt in PDF big points = 8.9664
raw_lines = []

# ---------- 1. raw operators ----------
num = r"[-+]?\d*\.?\d+"
def scan(stream, fonts, tag):
    tf = collections.Counter(); tm = collections.Counter(); cm = collections.Counter()
    for m in re.finditer(r"/([\w.+-]+)\s+(%s)\s+Tf" % num, stream):
        tf[(fonts.get(m.group(1), m.group(1)), float(m.group(2)))] += 1
    for m in re.finditer(r"(%s)\s+(%s)\s+(%s)\s+(%s)\s+%s\s+%s\s+Tm" % ((num,) * 6), stream):
        a, b, c, d = map(float, m.groups()); tm[(round(a, 3), b, c, d)] += 1
    for m in re.finditer(r"(%s)\s+(%s)\s+(%s)\s+(%s)\s+%s\s+%s\s+cm" % ((num,) * 6), stream):
        a, b, c, d = map(float, m.groups()); cm[(round(a, 5), b, c, round(d, 5))] += 1
    raw_lines.append(f"== {tag}")
    for k, v in sorted(tf.items(), key=lambda x: x[0][1]): raw_lines.append(f"  Tf size {k[1]:<8} font {k[0]:<32} x{v}")
    a_vals = [k[0] for k in tm]; d_vals = sorted({k[3] for k in tm}); bc = sorted({(k[1], k[2]) for k in tm})
    if tm: raw_lines.append(f"  Tm: {sum(tm.values())} ops, horizontal a in [{min(a_vals)}, {max(a_vals)}], vertical d values {d_vals}, (b,c) values {bc}")
    for k, v in cm.items(): raw_lines.append(f"  cm {k} x{v}")
    return tf, tm

tf_sizes = set()
for pno, page in enumerate(doc, 1):
    fonts = {f[4]: f[3] for f in page.get_fonts(full=True)}
    tf, _ = scan(page.read_contents().decode("latin-1"), fonts, f"page {pno} content stream")
    tf_sizes |= {k[1] for k in tf}
    for xref, name, invoker, bbox in page.get_xobjects():
        if doc.xref_get_key(xref, "Subtype")[1] != "/Form": continue
        xs = doc.xref_stream(xref)
        if b"Tf" not in xs: continue
        # font resources of the form xobject
        xfonts = {}
        fr = doc.xref_get_key(xref, "Resources/Font")
        for m in re.finditer(r"/([\w.+-]+)\s+(\d+)\s+0\s+R", fr[1] if fr[0] != "null" else ""):
            bf = doc.xref_get_key(int(m.group(2)), "BaseFont")[1]
            xfonts[m.group(1)] = bf.lstrip("/")
        scan(xs.decode("latin-1"), xfonts, f"page {pno} form XObject xref {xref} ({name}) bbox {tuple(round(v,2) for v in bbox)}")
open(f"{OUT}/raw_operators.txt", "w").write("\n".join(raw_lines) + "\n")

# ---------- 2. regions ----------
def blocks(page):
    return [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]

def block_text(b):
    return " ".join("".join(s["text"] for s in l["spans"]) for l in b["lines"])

regions = collections.defaultdict(list)   # pno -> [(class, Rect)]
ref_started = False
for pno, page in enumerate(doc, 1):
    hl = [d["rect"] for d in page.get_drawings() if d["rect"].height < 1.5 and d["rect"].width > 20]
    dr = [d["rect"] for d in page.get_drawings()]
    for b in blocks(page):
        t = block_text(b).strip(); r = fitz.Rect(b["bbox"])
        if re.match(r"^(Fig\.|Table)\s*\d+\.", t):
            regions[pno].append(("caption", r))
            if t.startswith("Table"):
                below = [h for h in hl if h.y0 >= r.y1 - 1 and h.y0 < r.y1 + 200 and h.x0 >= r.x0 - 5 and h.x1 <= r.x1 + 15]
                if below:
                    regions[pno].append(("table", fitz.Rect(r.x0 - 2, r.y1 + 0.5, r.x1 + 15, max(h.y1 for h in below) + 1)))
        if re.match(r"^\d+\.\s*REFERENCES", t):
            regions[pno].append(("reference", fitz.Rect(r.x0 - 10, r.y1 + 0.5, page.rect.x1, page.rect.y1)))
            ref_started = True
    if ref_started and not any(c == "reference" for c, _ in regions[pno]):
        regions[pno].append(("reference", fitz.Rect(page.rect)))
    # the one float that is vector graphics: union of the drawings that are not table rules
    fig = [d for d in dr if not (d.height < 1.5 and d.width > 20 and any(c == "table" and fitz.Rect(d).intersects(rr) for c, rr in regions[pno]))]
    big = [d for d in fig if d.width < 300]
    if len(big) > 20:
        u = fitz.Rect(big[0])
        for d in big[1:]: u |= d
        regions[pno].append(("figure_text", u + (-15, -5, 15, 5)))

def classify(pno, span, nominal):
    r = fitz.Rect(span["bbox"]); f = span["font"]
    if f.startswith("CM") and nominal < 7: return "math_sub_or_superscript"
    if nominal < 7 and not f.startswith("STIX"): return "math_sub_or_superscript"   # \text{} inside a script
    for c, rr in regions[pno]:
        if c in ("figure_text",) and rr.contains(r) and f.startswith("STIX"): return c
    for c, rr in regions[pno]:
        if c in ("caption", "table") and rr.intersects(r): return c
    if pno == 1 and abs(nominal - 7.9701) < 0.01 and r.y0 > 700: return "footnote"
    for c, rr in regions[pno]:
        if c == "reference" and rr.intersects(r): return c
    if f.startswith("CM"): return "math_inline_or_display_9pt"
    return "body_text"

KNOWN = sorted(tf_sizes)
def nominal_of(span):
    s, f = span["size"], span["font"]
    if f.startswith("STIX"):       # figure text: uniform scale, no expansion
        return s
    return min(KNOWN, key=lambda k: abs(k - s))

rows = []
for pno, page in enumerate(doc, 1):
    for b in blocks(page):
        for l in b["lines"]:
            for s in l["spans"]:
                if not s["text"].strip(): continue
                nom = nominal_of(s)
                expa = (s["size"] / nom) ** 2
                rows.append(dict(page=pno, size_bp=round(s["size"], 3), nominal_bp=round(nom, 4),
                                 nominal_texpt=round(nom * 72.27 / 72, 3), expansion=round(expa, 4),
                                 font=s["font"], cls=classify(pno, s, nom),
                                 x0=round(s["bbox"][0], 1), y0=round(s["bbox"][1], 1),
                                 x1=round(s["bbox"][2], 1), y1=round(s["bbox"][3], 1), text=s["text"]))

under = [r for r in rows if r["size_bp"] < 9.0]
small = [r for r in rows if r["nominal_bp"] < TEX9 - 0.005]
cols = list(rows[0].keys())
for name, data in (("spans_under_9p0bp.csv", under), ("spans_below_9texpt.csv", small)):
    with open(f"{OUT}/{name}", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(data)

print("regions:", {p: [(c, [round(v) for v in r]) for c, r in v] for p, v in regions.items()})
print("Tf sizes in page streams:", KNOWN)
print("all non-blank spans:", len(rows), " spans < 9.0 bp:", len(under), " spans with nominal < 9 TeX pt:", len(small))
bad_exp = [r for r in rows if not r["font"].startswith("STIX") and not (0.979 <= r["expansion"] <= 1.021)]
print("spans whose size is not nominal x sqrt(a) with a in [0.98,1.02]:", len(bad_exp), bad_exp[:5])
print("\n-- spans < 9.0 bp by class and nominal size (count, chars)")
agg = collections.defaultdict(lambda: [0, 0])
for r in under:
    k = (r["cls"], r["nominal_bp"]); agg[k][0] += 1; agg[k][1] += len(r["text"])
for k, v in sorted(agg.items()): print(" ", k, v)
print("\n-- every span with nominal size below 9 TeX pt")
for r in small:
    print(f"  p{r['page']} {r['size_bp']:>6} bp (nominal {r['nominal_texpt']} TeX pt) {r['font']:<22} {r['cls']:<24} y={r['y0']:<6} {r['text']!r}")
