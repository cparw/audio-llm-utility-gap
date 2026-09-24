import sys, collections, fitz
doc = fitz.open(sys.argv[1])
print("pages", doc.page_count, "meta", doc.metadata)
hist = collections.Counter()
for pno, page in enumerate(doc, 1):
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b["type"] != 0: continue
        for l in b["lines"]:
            for s in l["spans"]:
                if not s["text"].strip(): continue
                hist[(round(s["size"],2), s["font"])] += len(s["text"])
for k, v in sorted(hist.items()):
    print(k, v)
for pno, page in enumerate(doc, 1):
    imgs = page.get_image_info(xrefs=True)
    print("page", pno, "rect", page.rect, "images", [(i["xref"], [round(x,1) for x in i["bbox"]]) for i in imgs], "drawings", len(page.get_drawings()))
