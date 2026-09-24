# Native size of the included figure and the sizes of its text, before LaTeX scales it.
import sys, collections, fitz
d = fitz.open(sys.argv[1])
p = d[0]
print("figure page rect (bp)", p.rect, "width in", p.rect.width/72)
h = collections.Counter()
for b in p.get_text("dict")["blocks"]:
    if b["type"]: continue
    for l in b["lines"]:
        for s in l["spans"]:
            if s["text"].strip(): h[(round(s["size"],2), s["font"])] += 1; print(round(s["size"],2), s["font"], repr(s["text"]))
print(h)
