#!/usr/local/bin/python3
# indep_check_fig1_whole.py (23 Sep 2026)
# Independent check of the figure track (item 3). Written from scratch; it does
# not import or read the track's verifier, and it uses no PyMuPDF, no
# matplotlib and no sklearn. The PDF is read with zlib and a small content
# stream interpreter written here. AUC is a rank (Mann-Whitney) AUC written here.
#
# usage: indep_check_fig1_whole.py FIG.pdf
import sys, os, re, zlib, csv, hashlib, math, glob, zipfile, subprocess

FIG = sys.argv[1]
POD = ("scores/part23/pull/"
       "p23-a-probes/p23a/out")
POD_ENC = f"{POD}/o25_whole_encoder_perlayer.csv"
POD_LLM = f"{POD}/o25_whole_llm_perlayer.csv"
A2 = f"{POD}/A2_edaic_whole_zeroshot_perclip.csv"
VERIFIED = ("scores/part23/verify/"
            "PART23_VERIFIED.tsv")
OVL = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z"
P24 = "scores/part24/fig"
UPLOAD = ("<local data dir>/"
          "The_Representation_Utility_Gap_in_Audio_LLMs____ICASSP (2).zip")
OLDZIP = "<local data dir>/paper1_submission_23sep/zip/figures"
# md5 values exactly as written in the track report
REPORTED = {
    f"{P24}/fig1_gap_whole.pdf": "e05e5188fc39554024e8d5ef3cb31020",
    f"{P24}/fig1_gap_whole_v4layout.pdf": "c3ee6d4d403e24aa68a1cdb202373973",
    f"{OVL}/figures/fig1_gap.pdf": "bbefb559a5768dcb07d11ed5d440da18",
    f"{OVL}/figures/fig1_gap1.pdf": "63c9ae9c510b0afee943dcaea0cc7343",
}
REPORTED_FIG_MD5 = "b178e9d020758acf4d1c496889ccd7d6"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- PDF reading
RAW = open(FIG, "rb").read()
OBJ = {}
for m in re.finditer(rb"(\d+) 0 obj(.*?)endobj", RAW, re.S):
    OBJ[int(m.group(1))] = m.group(2)


def stream_of(num):
    body = OBJ[num]
    m = re.search(rb"stream\r?\n(.*)endstream", body, re.S)
    data = m.group(1)
    if b"/FlateDecode" in body[:m.start()]:
        return zlib.decompress(data)
    return data


def tokenize(s):
    """PDF content stream tokens: numbers, names, strings, arrays, operators."""
    i, n = 0, len(s)
    out = []
    ws = b" \t\r\n\f\x00"
    while i < n:
        c = s[i:i + 1]
        if c in (b" ", b"\t", b"\r", b"\n", b"\f", b"\x00"):
            i += 1
        elif c == b"%":
            while i < n and s[i:i + 1] not in (b"\r", b"\n"):
                i += 1
        elif c == b"(":
            depth, j, buf = 1, i + 1, bytearray()
            while depth:
                ch = s[j:j + 1]
                if ch == b"\\":
                    nx = s[j + 1:j + 2]
                    esc = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b",
                           b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}
                    if nx in esc:
                        buf += esc[nx]; j += 2
                    elif nx.isdigit():
                        k = j + 1
                        while k < j + 4 and s[k:k + 1].isdigit():
                            k += 1
                        buf.append(int(s[j + 1:k], 8) & 255); j = k
                    else:
                        j += 2
                    continue
                if ch == b"(":
                    depth += 1
                elif ch == b")":
                    depth -= 1
                    if depth == 0:
                        break
                buf += ch; j += 1
            out.append(("str", bytes(buf))); i = j + 1
        elif c == b"<" and s[i + 1:i + 2] != b"<":
            j = s.index(b">", i)
            h = re.sub(rb"\s", b"", s[i + 1:j])
            if len(h) % 2:
                h += b"0"
            out.append(("str", bytes.fromhex(h.decode()))); i = j + 1
        elif c in (b"[", b"]"):
            out.append(("delim", c.decode())); i += 1
        elif s[i:i + 2] in (b"<<", b">>"):
            out.append(("delim", s[i:i + 2].decode())); i += 2
        elif c == b"/":
            j = i + 1
            while j < n and s[j:j + 1] not in ws and s[j:j + 1] not in b"/[]()<>":
                j += 1
            out.append(("name", s[i + 1:j].decode())); i = j
        else:
            j = i
            while j < n and s[j:j + 1] not in ws and s[j:j + 1] not in b"/[]()<>":
                j += 1
            tok = s[i:j].decode("latin1")
            try:
                out.append(("num", float(tok)))
            except ValueError:
                out.append(("op", tok))
            i = j
    return out


def matmul(a, b):
    """PDF matrices [a b c d e f]; returns a x b (a applied first)."""
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def apply(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


# ToUnicode map of every font on the page
def parse_tounicode(num):
    cm = stream_of(num).decode("latin1")
    table = {}
    for blk in re.findall(r"beginbfchar(.*?)endbfchar", cm, re.S):
        for a, b in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            table[int(a, 16)] = chr(int(b, 16))
    for blk in re.findall(r"beginbfrange(.*?)endbfrange", cm, re.S):
        for line in blk.strip().splitlines():
            m = re.fullmatch(r"\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(.*?)\s*", line)
            a, b, rest = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            if rest.startswith("["):          # array form: one destination per code
                dst = re.findall(r"<([0-9A-Fa-f]+)>", rest)
                assert len(dst) == b - a + 1, line
                for k, d in zip(range(a, b + 1), dst):
                    table[k] = chr(int(d, 16))
            else:                             # offset form
                c = int(re.fullmatch(r"<([0-9A-Fa-f]+)>", rest).group(1), 16)
                for k in range(a, b + 1):
                    table[k] = chr(c + k - a)
    return table


page_num = [k for k, v in OBJ.items() if re.search(rb"/Type\s*/Page\b", v)]
assert len(page_num) == 1, page_num
page = OBJ[page_num[0]]
mb = [float(v) for v in re.search(rb"/MediaBox\s*\[([^\]]*)\]", page).group(1).split()]
contents = int(re.search(rb"/Contents\s+(\d+) 0 R", page).group(1))
res_num = int(re.search(rb"/Resources\s+(\d+) 0 R", page).group(1))
res = OBJ[res_num]
fontdict = OBJ[int(re.search(rb"/Font\s+(\d+) 0 R", res).group(1))]
FONTS = {}
for name, num in re.findall(rb"/(\w+)\s+(\d+) 0 R", fontdict):
    fobj = OBJ[int(num)]
    base = re.search(rb"/BaseFont\s*/([^\s/]+)", fobj).group(1).decode()
    sub = re.search(rb"/Subtype\s*/(\w+)", fobj).group(1).decode()
    tu = re.search(rb"/ToUnicode\s+(\d+) 0 R", fobj)
    two_byte = b"Identity-H" in fobj
    FONTS[name.decode()] = dict(base=base, subtype=sub,
                                tounicode=parse_tounicode(int(tu.group(1))) if tu else {},
                                two_byte=two_byte, obj=int(num))
xobjdict = OBJ[int(re.search(rb"/XObject\s+(\d+) 0 R", res).group(1))]
XOBJ = {n.decode(): int(k) for n, k in re.findall(rb"/(\w+)\s+(\d+) 0 R", xobjdict)}


def decode_text(font, raw):
    f = FONTS[font]
    if f["two_byte"]:
        codes = [raw[k] * 256 + raw[k + 1] for k in range(0, len(raw) - 1, 2)]
    else:
        codes = list(raw)
    return "".join(f["tounicode"].get(c, chr(c)) for c in codes)


def run(stream, ctm0, texts, paths, where):
    toks = tokenize(stream)
    st = dict(ctm=list(ctm0), lw=1.0, dash=([], 0.0), RG=(0, 0, 0), rg=(0, 0, 0))
    stack = []
    ops = []
    path, cur = [], None
    tm = tlm = None
    font = size = None
    for kind, val in toks:
        if kind != "op":
            ops.append((kind, val)); continue
        op = val
        nums = [v for k, v in ops if k == "num"]
        if op == "q":
            stack.append(dict(st, ctm=list(st["ctm"])))
        elif op == "Q":
            st = stack.pop()
        elif op == "cm":
            st["ctm"] = matmul(nums[-6:], st["ctm"])
        elif op == "w":
            st["lw"] = nums[-1]
        elif op == "d":
            arr, inarr = [], False
            for k, v in ops:
                if k == "delim" and v == "[":
                    inarr = True
                elif k == "delim" and v == "]":
                    inarr = False
                elif k == "num" and inarr:
                    arr.append(v)
            st["dash"] = (arr, nums[-1])
        elif op == "RG":
            st["RG"] = tuple(nums[-3:])
        elif op == "G":
            st["RG"] = (nums[-1],) * 3
        elif op == "rg":
            st["rg"] = tuple(nums[-3:])
        elif op == "g":
            st["rg"] = (nums[-1],) * 3
        elif op == "m":
            cur = [apply(st["ctm"], *nums[-2:])]; path.append(cur)
        elif op == "l":
            cur.append(apply(st["ctm"], *nums[-2:]))
        elif op == "c":
            cur.append(apply(st["ctm"], *nums[-2:]))
        elif op == "re":
            x, y, w, h = nums[-4:]
            path.append([apply(st["ctm"], x, y), apply(st["ctm"], x + w, y),
                         apply(st["ctm"], x + w, y + h), apply(st["ctm"], x, y + h)])
        elif op in ("S", "s", "B", "B*", "b", "b*"):
            for sp in path:
                paths.append(dict(pts=sp, lw=st["lw"] * math.sqrt(abs(
                    st["ctm"][0] * st["ctm"][3] - st["ctm"][1] * st["ctm"][2])),
                    dash=st["dash"], RG=st["RG"], where=where))
            path, cur = [], None
        elif op in ("f", "F", "f*", "n", "W", "W*"):
            if op != "W" and op != "W*":
                path, cur = [], None
        elif op == "BT":
            tm = [1, 0, 0, 1, 0, 0]; tlm = list(tm)
        elif op == "Tf":
            font = [v for k, v in ops if k == "name"][-1]; size = nums[-1]
        elif op == "Td":
            tlm = matmul([1, 0, 0, 1, nums[-2], nums[-1]], tlm); tm = list(tlm)
        elif op == "Tm":
            tlm = list(nums[-6:]); tm = list(tlm)
        elif op in ("Tj", "TJ", "'", '"'):
            raw = b"".join(v for k, v in ops if k == "str")
            m = matmul(tm, st["ctm"])
            # rendered size = Tf size times the scale the text space gets on the page
            vscale = math.hypot(m[2], m[3])
            hscale = math.hypot(m[0], m[1])
            texts.append(dict(text=decode_text(font, raw), font=FONTS[font]["base"],
                              tf=size, size=size * vscale, hsize=size * hscale,
                              origin=apply(m, 0, 0), fill=st["rg"],
                              rot=math.degrees(math.atan2(m[1], m[0])), where=where))
        elif op == "Do":
            name = [v for k, v in ops if k == "name"][-1]
            sub = stream_of(XOBJ[name])
            run(sub, st["ctm"], texts, paths, f"xobject {name}")
        ops = []


texts, paths = [], []
run(stream_of(contents), [1, 0, 0, 1, 0, 0], texts, paths, "page")
# also look inside every XObject once on its own, so text hidden in one is not missed
xo_texts = []
for name, num in XOBJ.items():
    t2, p2 = [], []
    run(stream_of(num), [1, 0, 0, 1, 0, 0], t2, p2, f"xobject {name}")
    xo_texts += t2

out = []
P = out.append
P(f"indep_check_fig1_whole.py  python {sys.version.split()[0]}  (zlib + own PDF interpreter; no PyMuPDF, matplotlib or sklearn)")
P(f"figure {FIG}")
P(f"  md5 {md5(FIG)}  (track report says {REPORTED_FIG_MD5}: {'same' if md5(FIG) == REPORTED_FIG_MD5 else 'DIFFERENT'})")
P(f"  MediaBox {mb}  width {mb[2] - mb[0]:.4f} bp = {(mb[2] - mb[0]) / 72:.4f} in, height {(mb[3] - mb[1]) / 72:.4f} in")

# ---------------------------------------------------------------- check 1 fonts
P("")
P("CHECK 1  smallest font size")
P(f"  fonts in resources: " + ", ".join(f"{k}={v['base']} ({v['subtype']})" for k, v in FONTS.items()))
embedded = []
for k, v in FONTS.items():
    body = OBJ[v["obj"]]
    desc = re.search(rb"/DescendantFonts\s*\[\s*(\d+) 0 R", body)
    fo = OBJ[int(desc.group(1))] if desc else body
    fd = OBJ[int(re.search(rb"/FontDescriptor\s+(\d+) 0 R", fo).group(1))]
    embedded.append((k, bool(re.search(rb"/FontFile[23]?\s", fd))))
P(f"  font program embedded: {embedded}")
type3 = [k for k, v in OBJ.items() if re.search(rb"/Subtype\s*/Type3", v)]
P(f"  Type 3 fonts in file: {len(type3)}")
P(f"  text drawn inside XObjects (markers): {len(xo_texts)}")
P(f"  text runs on the page: {len(texts)}")
for t in sorted(texts, key=lambda t: (t['size'], t['text'])):
    P(f"    {t['size']:.4f} pt (Tf {t['tf']:g}, rot {t['rot']:.0f})  {t['font']}  {t['text']!r}")
smin = min(t["size"] for t in texts)
smax = max(t["size"] for t in texts)
P(f"  smallest {smin:.4f} pt, largest {smax:.4f} pt, runs under 9 pt: {sum(t['size'] < 9 - 1e-9 for t in texts)}")
# second tool: poppler pdffonts, only for font type and embedding
pf = subprocess.run(["/opt/homebrew/bin/pdffonts", FIG], capture_output=True, text=True).stdout
for line in pf.strip().splitlines():
    P(f"  pdffonts | {line}")
# printed size if placed with width=\columnwidth (tex line 39 style)
sty = open(f"{OVL}/spconf.sty").read()
tw = float(re.search(r"^\\textwidth\s+([\d.]+)truemm", sty, re.M).group(1))
cs = float(re.search(r"^\\columnsep\s+([\d.]+)truemm", sty, re.M).group(1))
colw_bp = (tw - cs) / 2 / 25.4 * 72
scale = colw_bp / (mb[2] - mb[0])
P(f"  spconf.sty \\textwidth {tw} truemm, \\columnsep {cs} truemm -> \\columnwidth {(tw - cs) / 2} mm = {colw_bp:.4f} bp")
P(f"  placed at width=\\columnwidth: scale {scale:.6f}, smallest prints at {smin * scale:.4f} pt, largest at {smax * scale:.4f} pt")
check1 = smin >= 9.0 - 1e-9 and not type3 and not xo_texts and all(e for _, e in embedded)

# ---------------------------------------------------------------- curves
# identify E-DAIC by the colour of its in-plot label, not by any script value
lab = [t for t in texts if t["text"] == "E-DAIC"]
assert len(lab) == 1
ecol = lab[0]["fill"]
P("")
P(f"E-DAIC label fill colour {tuple(round(c, 6) for c in ecol)}")


def same(c1, c2):
    return all(abs(a - b) < 1e-6 for a, b in zip(c1, c2))


col_paths = [p for p in paths if same(p["RG"], ecol) and p["where"] == "page"]
curves = [p for p in col_paths if len(p["pts"]) > 10]
hlines = [p for p in col_paths if len(p["pts"]) == 2 and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-9]
P(f"  stroked page paths in that colour: {len(col_paths)} (polylines over 10 points {len(curves)}, horizontal 2 point lines {len(hlines)})")
assert len(curves) == 1 and len(hlines) == 1
curve, ans = curves[0], hlines[0]

# y calibration from the y tick marks and their labels (text), least squares
# tick marks: 2 point horizontal black segments ending on the left spine
spine_x = min(p["pts"][0][0] for p in paths if len(p["pts"]) == 2
              and abs(p["pts"][0][0] - p["pts"][1][0]) < 1e-9
              and abs(p["pts"][0][1] - p["pts"][1][1]) > 50)
yt = [p for p in paths if len(p["pts"]) == 2 and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-9
      and abs(max(q[0] for q in p["pts"]) - spine_x) < 1e-6
      and 0 < abs(p["pts"][0][0] - p["pts"][1][0]) < 5]
ylabels = [t for t in texts if re.fullmatch(r"\d\.\d", t["text"])]
pairs = []
for p in yt:
    y = p["pts"][0][1]
    # label whose glyph box centre is nearest in y (centre = origin + half cap height)
    t = min(ylabels, key=lambda t: abs(t["origin"][1] + 0.35 * t["size"] - y))
    pairs.append((y, float(t["text"])))
pairs.sort()


def fit(pp):
    n = len(pp)
    mx = sum(a for a, _ in pp) / n; my = sum(b for _, b in pp) / n
    sxx = sum((a - mx) ** 2 for a, _ in pp)
    a = sum((a - mx) * (b - my) for a, b in pp) / sxx
    return a, my - a * mx


ya, yb = fit(pairs)
yres = max(abs(ya * y + yb - v) for y, v in pairs)
ya2, yb2 = fit([pairs[0], pairs[-1]])        # second calibration: end ticks only
P(f"  y calibration A: {len(pairs)} tick marks paired with their labels {[v for _, v in pairs]}, max residual {yres:.2e}")
P(f"  y calibration B: end ticks only ({pairs[0][1]} and {pairs[-1][1]})")


def yv(y, cal="A"):
    return (ya * y + yb) if cal == "A" else (ya2 * y + yb2)


# x calibration from the x tick marks and labels: "0 16 | 0 14 28"
enc_rows = list(csv.DictReader(open(POD_ENC)))
llm_rows = list(csv.DictReader(open(POD_LLM)))
n_enc, n_llm = len(enc_rows), len(llm_rows)
xt = [p for p in paths if len(p["pts"]) == 2 and abs(p["pts"][0][0] - p["pts"][1][0]) < 1e-9
      and 0 < abs(p["pts"][0][1] - p["pts"][1][1]) < 5]
xt.sort(key=lambda p: p["pts"][0][0])
xlabels = [t for t in texts if re.fullmatch(r"\d+", t["text"])]
xpairs = []
for k, p in enumerate(xt):
    x = p["pts"][0][0]
    t = min(xlabels, key=lambda t: abs(t["origin"][0] + 0.25 * t["size"] * len(t["text"]) - x))
    xpairs.append((x, t["text"]))
# labels restart at 0 at the language model boundary
idx, seen_restart = [], False
for k, (x, lbl) in enumerate(xpairs):
    if k > 0 and int(lbl) < int(xpairs[k - 1][1]):
        seen_restart = True
    idx.append((x, int(lbl) + (n_enc if seen_restart else 0)))
xa, xb = fit(idx)
xres = max(abs(xa * x + xb - v) for x, v in idx)
P(f"  x calibration: tick labels {[l for _, l in xpairs]} -> plot index {[v for _, v in idx]} (n_enc {n_enc} from pod csv rows), max residual {xres:.2e}")

# pod values keyed by the layer/stage column (not file order)
pod_enc = {int(r["layer"]): float(r["auc_oof"]) for r in enc_rows}
pod_llm = {int(r["stage"]): float(r["auc_oof"]) for r in llm_rows}
assert sorted(pod_enc) == list(range(n_enc)) and sorted(pod_llm) == list(range(n_llm))
file_order_enc = [int(r["layer"]) for r in enc_rows] == list(range(n_enc))
file_order_llm = [int(r["stage"]) for r in llm_rows] == list(range(n_llm))

P("")
P("CHECK 2  E-DAIC curves vs pod csvs (auc_oof)")
P(f"  pod encoder {POD_ENC}  sha256 {sha256(POD_ENC)}  rows {n_enc}  file order is layer order: {file_order_enc}")
P(f"  pod llm     {POD_LLM}  sha256 {sha256(POD_LLM)}  rows {n_llm}  file order is stage order: {file_order_llm}")
P(f"  curve vertices {len(curve['pts'])}  (expected {n_enc + n_llm})")
rows = []
worst = {"encoder": [0, None, None, None], "llm": [0, None, None, None]}
worstB = {"encoder": 0, "llm": 0}
xoff = 0
used = set()
for (x, y) in curve["pts"]:
    fi = xa * x + xb
    i = round(fi)
    xoff = max(xoff, abs(fi - i))
    used.add(i)
    stream, j = ("encoder", i) if i < n_enc else ("llm", i - n_enc)
    pod = pod_enc[j] if stream == "encoder" else pod_llm[j]
    va, vb = yv(y, "A"), yv(y, "B")
    d = abs(va - pod)
    rows.append([stream, j, i, f"{va:.9f}", f"{vb:.9f}", repr(pod), f"{d:.3e}"])
    if d > worst[stream][0]:
        worst[stream] = [d, j, va, pod]
    worstB[stream] = max(worstB[stream], abs(vb - pod))
P(f"  every plot index 0..{n_enc + n_llm - 1} hit once: {sorted(used) == list(range(n_enc + n_llm))}; max |x - nearest integer| {xoff:.2e}")
for s in ("encoder", "llm"):
    d, j, va, pod = worst[s]
    P(f"  {s:<8} max |PDF - pod| cal A {d:.2e} (at {j}: PDF {va:.8f}, pod {pod:.8f}); cal B {worstB[s]:.2e}")
P(f"  dash of the curve {curve['dash']}, width {curve['lw']:.3f}")
check2 = (max(worst['encoder'][0], worst['llm'][0]) < 1e-6 and sorted(used) == list(range(n_enc + n_llm)))
# rounding floor: matplotlib writes 6 decimals in bp; 1e-6 bp in AUC units
P(f"  write precision floor: 1e-6 bp is {abs(ya) * 1e-6:.1e} AUC")
csv_out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "indep_check_fig1_whole_perlayer.csv")
with open(csv_out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["stream", "layer_or_stage", "plot_index", "pdf_value_calA", "pdf_value_calB", "pod_auc_oof", "abs_diff_calA"])
    w.writerows(rows)

# ---------------------------------------------------------------- check 3 answer line
P("")
P("CHECK 3  E-DAIC answer line")
(x0, y0), (x1, y1) = ans["pts"]
axis_right = max(p["pts"][1][0] for p in paths if len(p["pts"]) == 2
                 and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-9 and abs(p["pts"][0][0] - spine_x) < 1e-6
                 and p["pts"][1][0] - p["pts"][0][0] > 100)
P(f"  line y {y0:.6f} bp -> AUC cal A {yv(y0, 'A'):.6f}, cal B {yv(y0, 'B'):.6f}")
P(f"  x from {min(x0, x1):.4f} to {max(x0, x1):.4f} bp; axes from {spine_x:.4f} to {axis_right:.4f} bp")
P(f"  dash {ans['dash']}, width {ans['lw']:.3f} (curve width {curve['lw']:.3f})")
# the value it should be, recomputed here with a rank AUC
lab_, sc = [], []
for r in csv.DictReader(open(A2)):
    v = r["p_yes_whole"]
    m = re.fullmatch(r"np\.float64\((.*)\)", v)
    sc.append(float(m.group(1) if m else v)); lab_.append(int(r["label"]))
order = sorted(range(len(sc)), key=lambda k: sc[k])
ranks = [0.0] * len(sc)
k = 0
while k < len(order):
    j = k
    while j + 1 < len(order) and sc[order[j + 1]] == sc[order[k]]:
        j += 1
    for q in range(k, j + 1):
        ranks[order[q]] = (k + j) / 2 + 1
    k = j + 1
npos = sum(lab_); nneg = len(lab_) - npos
auc = (sum(r for r, l in zip(ranks, lab_) if l == 1) - npos * (npos + 1) / 2) / (npos * nneg)
P(f"  rank AUC of p_yes_whole on {os.path.basename(A2)}: {auc:.6f} (n {len(sc)}, positives {npos}) -> {auc:.4f}")
ver = [l.split("\t") for l in open(VERIFIED).read().splitlines()]
row = [r for r in ver if r[0] == "1_whole"][0]
P(f"  PART23_VERIFIED.tsv row 1_whole: claimed {row[1]}, recomputed {row[2]}, verified {row[5]}")
dotted = len(ans["dash"][0]) == 2 and ans["dash"][0][0] <= ans["lw"] * 1.01 + 1e-9
check3 = (abs(yv(y0, 'A') - 0.8366) < 5e-6 and abs(y0 - y1) < 1e-9
          and abs(min(x0, x1) - spine_x) < 1e-6 and abs(max(x0, x1) - axis_right) < 1e-6
          and round(auc, 4) == 0.8366)
P(f"  dash on length <= line width (dotted look): {dotted}")

# ---------------------------------------------------------------- check 4 md5
P("")
P("CHECK 4  no existing file overwritten")
ok4 = True
for pth, want in REPORTED.items():
    got = md5(pth)
    ok4 &= got == want
    P(f"  {got}  report {want}  {'same' if got == want else 'DIFFERENT'}  {pth}")
zf = zipfile.ZipFile(UPLOAD)
zmd5 = {n: hashlib.md5(zf.read(n)).hexdigest() for n in zf.namelist() if not n.endswith("/")}
P(f"  upload zip {UPLOAD}  md5 {md5(UPLOAD)}  members {len(zmd5)}")
for pth in sorted(glob.glob(f"{P24}/*.pdf")) + sorted(glob.glob(f"{OVL}/figures/*.pdf")):
    got = md5(pth)
    extra = ""
    if pth.startswith(OVL):
        member = "figures/" + os.path.basename(pth)
        zm = zmd5.get(member)
        om = md5(f"{OLDZIP}/{os.path.basename(pth)}") if os.path.exists(f"{OLDZIP}/{os.path.basename(pth)}") else None
        extra = f"upload zip {'same' if zm == got else 'DIFFERENT'}; zip/figures {'same' if om == got else 'DIFFERENT'}"
        ok4 &= zm == got
    else:
        extra = "listed in report" if pth in REPORTED else "NOT listed in report"
        ok4 &= pth in REPORTED
    P(f"  {got}  {extra}  {os.path.basename(pth)}")
# every file of the Overleaf copy against the upload
allsame = True
for n, zm in sorted(zmd5.items()):
    got = md5(f"{OVL}/{n}")
    if got != zm:
        allsame = False
        P(f"  DIFFERENT from upload: {n}")
P(f"  all {len(zmd5)} Overleaf copy files byte identical to the upload zip: {allsame}")
extra_files = [os.path.relpath(p, OVL) for p in glob.glob(f"{OVL}/**/*", recursive=True)
               if os.path.isfile(p) and not os.path.basename(p).startswith("._")
               and os.path.relpath(p, OVL) not in zmd5]
P(f"  files in the Overleaf copy that are not in the upload: {extra_files}")
ok4 &= allsame and not extra_files
check4 = ok4

P("")
for k, v in (("1 smallest font size >= 9 pt in the file", check1),
             ("2 E-DAIC curves equal the pod csvs", check2),
             ("3 answer line at 0.8366", check3),
             ("4 no existing file overwritten", check4)):
    P(f"RESULT check {k}: {'yes' if v else 'no'}")
print("\n".join(out))
