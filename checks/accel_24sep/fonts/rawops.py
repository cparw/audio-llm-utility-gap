# Raw PDF text operators: which Tf sizes are set and what horizontal scale the Tm matrices carry.
import sys, re, collections, fitz
doc = fitz.open(sys.argv[1])
for pno, page in enumerate(doc, 1):
    fonts = {f[4]: f[3] for f in page.get_fonts()}   # resource name -> basefont
    raw = page.read_contents().decode("latin-1")
    tf = collections.Counter()
    for m in re.finditer(r"/(\w+)\s+([\d.]+)\s+Tf", raw):
        tf[(fonts.get(m.group(1), m.group(1)), float(m.group(2)))] += 1
    tm = collections.Counter()
    for m in re.finditer(r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+Tm", raw):
        a, b, c, d = map(float, m.groups())
        tm[(round(a, 4), round(d, 4))] += 1
    print("page", pno)
    for k, v in sorted(tf.items(), key=lambda x: x[0][1]): print("  Tf", k, v)
    for k, v in sorted(tm.items()): print("  Tm a,d", k, v)
