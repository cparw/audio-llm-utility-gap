#!/usr/local/bin/python3
# indep_placement_scale.py: the scale pdflatex applied to the 244.08 bp wide Figure 1
# in an earlier compile of the same spconf layout (width=\columnwidth).
import PyPDF2, re, sys
p = sys.argv[1]
r = PyPDF2.PdfReader(p)
G = lambda o: o.get_object() if hasattr(o, "get_object") else o
for pi, pg in enumerate(r.pages):
    res = G(pg.get("/Resources"))
    if res is None or "/XObject" not in res:
        continue
    xo = G(res["/XObject"])
    data = pg.get_contents().get_data().decode("latin1")
    for k, v in xo.items():
        o = G(v)
        if o.get("/Subtype") != "/Form":
            continue
        w = float(o["/BBox"][2]) - float(o["/BBox"][0])
        i = data.find(k + " Do")
        cms = re.findall(r"([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) cm", data[max(0, i - 200):i])
        s = 1.0
        for c in cms:
            s *= float(c[0])
        print(f"{p}\n  page {pi + 1} {k} BBox width {w:.4f} bp, cm chain {cms}\n  scale {s:.5f}, placed width {w * s:.4f} bp, 9 pt text prints at {9 * s:.4f} pt")
