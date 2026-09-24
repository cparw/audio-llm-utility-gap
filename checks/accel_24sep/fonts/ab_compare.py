# Compare local tectonic builds: base vs footnote-at-\small, and base vs the Overleaf PDF.
import sys, fitz
def lines(pdf):
    d = fitz.open(pdf); out = []
    for pno, p in enumerate(d, 1):
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"]).strip()
                if t: out.append((pno, round(l["bbox"][0]) > 306, round(l["bbox"][1], 1), t, min(s["size"] for s in l["spans"] if s["text"].strip())))
    return d.page_count, out
def column_ends(ls):
    ends = {}
    for pno, right, y, t, sz in ls:
        k = (pno, right)
        if k not in ends or y > ends[k][0]: ends[k] = (y, t[:60])
    return ends
res = {}
for name in sys.argv[1:]:
    n, ls = lines(name); res[name] = (n, ls)
    print(f"{name}: pages {n}")
    foot = [x for x in ls if "Corresponding author" in x[3]]
    print("   footnote line:", foot)
    for k, v in sorted(column_ends(ls).items()): print("   end of page/col", k, v)
    p5 = [x for x in ls if x[0] == 5]
    print("   page 5 first line:", p5[:1], " non-reference lines on p5:", [x[3][:50] for x in p5 if not x[3].startswith("[") ][:0])
a, b = sys.argv[1], sys.argv[2]
ta = [(x[0], x[1], x[3]) for x in res[a][1]]; tb = [(x[0], x[1], x[3]) for x in res[b][1]]
print("identical line text and page/column placement (A vs B):", ta == tb)
if ta != tb:
    import difflib
    for d in list(difflib.unified_diff([str(x) for x in ta], [str(x) for x in tb], lineterm="", n=0))[:20]: print("  ", d)
