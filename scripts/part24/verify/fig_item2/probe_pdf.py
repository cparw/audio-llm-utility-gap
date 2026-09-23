import pymupdf, sys
p = sys.argv[1]
doc = pymupdf.open(p)
print("pages", doc.page_count)
pg = doc[0]
print("rect", pg.rect)
d = pg.get_text("rawdict")
for b in d["blocks"]:
    for l in b.get("lines", []):
        for s in l["spans"]:
            txt = "".join(c["c"] for c in s["chars"])
            print(repr(txt), s["font"], round(s["size"],4), [round(v,2) for v in s["bbox"]], "dir", l["dir"])
print("fonts", pg.get_fonts(full=True))
dr = pg.get_drawings()
print("n drawings", len(dr))
for i, g in enumerate(dr):
    items = g["items"]
    kinds = "".join(it[0] for it in items)
    print(i, g.get("type"), "n_items", len(items), kinds[:12], "color", g.get("color"), "fill", g.get("fill"), "dashes", g.get("dashes"), "w", g.get("width"), [round(v,2) for v in g["rect"]])
