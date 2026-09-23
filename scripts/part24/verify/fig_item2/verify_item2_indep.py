# PART 24 item 2, independent verifier (second pass). Written from scratch; does not import
# or read make_fig1_whole.py or verify_fig1_whole.py. Reads only; writes only under this folder.
import hashlib, io, json, os, re, sys, tarfile, zipfile, datetime, platform
import numpy as np, pandas as pd, pymupdf, sklearn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = "<local data dir>/part24/fig"
P23 = "<local data dir>/part23"
POD = f"{P23}/pull/p23-a-probes/p23a"
SUB = "<local data dir>/paper1_submission_23sep"
MIRROR = "<local data dir>/release/overnight/curves/omni"
PDFS = {"whole": f"{FIG}/fig1_gap_whole.pdf", "whole_v4layout": f"{FIG}/fig1_gap_whole_v4layout.pdf",
        "current_overleaf": f"{SUB}/zip/figures/fig1_gap.pdf"}
out_lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out_lines.append(s)
def md5(b): return hashlib.md5(b).hexdigest()
def sha(b): return hashlib.sha256(b).hexdigest()
def rb(p):
    with open(p, "rb") as f: return f.read()
def mtime(p): return datetime.datetime.fromtimestamp(os.stat(p).st_mtime).strftime("%Y-%m-%dT%H:%M:%S")

say("PART 24 item 2 independent verifier (pass 2)", datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
say("env python", platform.python_version(), "pymupdf", pymupdf.VersionBind, "numpy", np.__version__,
    "pandas", pd.__version__, "sklearn", sklearn.__version__)

# ---------------- CHECK 1 + 2: font sizes and font names ----------------
res = {}
for tag, p in PDFS.items():
    doc = pymupdf.open(p)
    sizes, spans, fonts_used, n_chars = [], [], set(), 0
    for pg in doc:
        for b in pg.get_text("rawdict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    txt = "".join(c["c"] for c in s["chars"])
                    if not txt.strip(): continue
                    # size from the span and, independently, from the glyph bbox height via the font's ascender/descender
                    sizes.append(s["size"]); spans.append((round(s["size"], 4), txt, s["font"]))
                    fonts_used.add(s["font"]); n_chars += len(txt.replace(" ", ""))
    # raw Tf operands and text matrices in the content stream (independent of PyMuPDF's text layer)
    raw_tf = []
    for pg in doc:
        cs = b"".join(doc.xref_stream(x) for x in pg.get_contents()).decode("latin-1")
        # matplotlib writes "a b c d e f cm BT /Fn size Tf": effective size = size * sqrt(|a d - b c|)
        for m in re.finditer(r"([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) cm\s*BT\s*/(\w+) ([\d.]+) Tf", cs):
            a, b_, c, d = (float(m.group(i)) for i in range(1, 5))
            raw_tf.append(float(m.group(8)) * abs(a * d - b_ * c) ** 0.5)
        n_bt = len(re.findall(r"\bBT\b", cs))
    # every font object in the file, not only those on the page text layer
    allfonts = []
    for x in range(1, doc.xref_length()):
        try: t = doc.xref_get_key(x, "Type")
        except Exception: continue
        if t == ("name", "/Font"):
            sub = doc.xref_get_key(x, "Subtype")[1]; base = doc.xref_get_key(x, "BaseFont")[1]
            allfonts.append((x, sub, base))
    # embedded font programs
    emb = [f for f in doc[0].get_fonts(full=True)]
    ffile = {}
    for x in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(x, "Type") == ("name", "/FontDescriptor"):
                fn = doc.xref_get_key(x, "FontName")[1]
                ffile[fn] = [k for k in ("FontFile", "FontFile2", "FontFile3") if doc.xref_get_key(x, k)[0] != "null"]
        except Exception: pass
    # filled black curve paths could be text drawn as outlines; count them
    outline_like = 0
    for g in doc[0].get_drawings():
        if g.get("fill") == (0.0, 0.0, 0.0) and any(it[0] == "c" for it in g["items"]) and len(g["items"]) > 3:
            outline_like += 1
    res[tag] = dict(min=min(sizes), max=max(sizes), n_spans=len(spans), raw_min=min(raw_tf) if raw_tf else None,
                    raw_max=max(raw_tf) if raw_tf else None, n_raw=len(raw_tf), n_bt=n_bt)
    say(f"[{tag}] {os.path.basename(p)} page {doc[0].rect.width:.2f} x {doc[0].rect.height:.2f} pt")
    say(f"[{tag}]   PyMuPDF span size min {min(sizes):.4f} max {max(sizes):.4f} over {len(spans)} spans, {n_chars} glyphs")
    say(f"[{tag}]   raw content stream Tf x cm scale: min {res[tag]['raw_min']:.4f} max {res[tag]['raw_max']:.4f} over {len(raw_tf)} text objects (BT count {n_bt})")
    small = [s for s in spans if s[0] < 9.0]
    say(f"[{tag}]   spans under 9 pt: {small if small else 'none'}")
    say(f"[{tag}]   span fonts: {sorted(fonts_used)}")
    say(f"[{tag}]   font objects in file: {allfonts}")
    say(f"[{tag}]   font programs: {ffile}")
    say(f"[{tag}]   filled black outline paths (text drawn as curves): {outline_like}")
    res[tag]["fonts"] = sorted(fonts_used); res[tag]["allfonts"] = allfonts; res[tag]["ffile"] = ffile
    res[tag]["spans"] = spans

# column width in the paper (spconf.sty) vs figure width: effective size once \includegraphics[width=\columnwidth] scales it
sty = open(f"{SUB}/zip/spconf.sty", encoding="latin-1").read()
tw = float(re.search(r"\\textwidth\s*([\d.]+)truemm", sty).group(1)); cs_ = float(re.search(r"\\columnsep\s*([\d.]+)truemm", sty).group(1))
colw_pt = (tw - cs_) / 2 / 25.4 * 72
for tag in ("whole", "current_overleaf"):
    w = pymupdf.open(PDFS[tag])[0].rect.width
    say(f"[{tag}] columnwidth {(tw-cs_)/2:.1f} mm = {colw_pt:.3f} pt; figure width {w:.3f} pt; scale {colw_pt/w:.5f}; "
        f"smallest text in the compiled paper {res[tag]['min']*colw_pt/w:.3f} pt")

# ---------------- CHECK 3: E-DAIC curves in the whole csvs vs the pod files ----------------
pod = {}
for stream in ("encoder", "llm"):
    copies = {}
    for sub in ("out", "probe"):
        copies[f"disk/{sub}"] = rb(f"{POD}/{sub}/o25_whole_{stream}_perlayer.csv")
    with tarfile.open(f"{P23}/pull/p23-a-probes.tar") as tf:
        for sub in ("out", "probe"):
            copies[f"tar/{sub}"] = tf.extractfile(f"p23a/{sub}/o25_whole_{stream}_perlayer.csv").read()
    hs = {k: sha(v) for k, v in copies.items()}
    say(f"[pod {stream}] sha256 of the 4 copies (disk out, disk probe, tar out, tar probe) all equal: {len(set(hs.values()))==1} {hs['disk/out'][:16]}")
    side = json.load(open(f"{FIG}/fig1_gap_whole.sidecar.json"))["sources"]["E-DAIC"]
    say(f"[pod {stream}] figure sidecar sha256 matches pod file: {side['encoder_sha256' if stream=='encoder' else 'llm_sha256']==hs['disk/out']}")
    df = pd.read_csv(io.BytesIO(copies["tar/out"]))
    key = df.columns[0]
    pod[stream] = df.set_index(key)["auc_oof"].astype(float)
    say(f"[pod {stream}] n {len(df)} key {key} auc_oof min {pod[stream].min():.4f} max {pod[stream].max():.4f} peak at {int(pod[stream].idxmax())}")

check3 = {}
for name in ("fig1_gap_whole_plotted.csv", "fig1_gap_whole_v4layout_plotted.csv"):
    P = pd.read_csv(f"{FIG}/{name}")
    E = P[P.dataset == "E-DAIC"]
    for stream in ("encoder", "llm"):
        e = E[E.stream == stream].set_index("layer")["auc_plotted"].astype(float)
        ref = pod[stream]
        same_index = list(e.index) == list(ref.index)
        d = (e - ref.reindex(e.index)).abs()
        check3[(name, stream)] = d.max()
        say(f"[csv {name}] E-DAIC {stream}: n {len(e)} (pod n {len(ref)}), same layers {same_index}, "
            f"max |plotted - pod auc_oof| = {d.max():.3g} at layer {int(d.idxmax())}; "
            f"vs pod auc_mean max diff {(e - pd.read_csv(f'{POD}/out/o25_whole_{stream}_perlayer.csv').iloc[:,1].values).abs().max():.4f} (sanity: the column choice matters)")

# ---------------- CHECK 3b: curves read back out of the PDF content stream ----------------
def parse_paths(pdfpath):
    doc = pymupdf.open(pdfpath)
    cs = b"".join(doc.xref_stream(x) for x in doc[0].get_contents()).decode("latin-1")
    toks = re.findall(r"\[(?:\((?:\\.|[^\\)])*\)|[^\]\(])*\]|\((?:\\.|[^\\)])*\)|/\w+|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[A-Za-z*'\"]+", cs, flags=re.S)
    st = {"RG": None, "d": "[]", "clip": None}; stack = []; ops = []; cur = []; paths = []; texts = []; cm = None; lastre = None
    for t in toks:
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t) or t.startswith("[") or t.startswith("(") or t.startswith("/"):
            ops.append(t); continue
        if t == "q": stack.append(dict(st))
        elif t == "Q": st = stack.pop() if stack else st
        elif t == "RG": st["RG"] = tuple(round(float(v), 4) for v in ops[-3:])
        elif t == "d": st["d"] = ops[-2] if len(ops) >= 2 else "[]"
        elif t == "re": lastre = tuple(float(v) for v in ops[-4:])
        elif t == "W": st["clip"] = lastre
        elif t == "m": cur = [(float(ops[-2]), float(ops[-1]))]
        elif t == "l": cur.append((float(ops[-2]), float(ops[-1])))
        elif t in ("S", "B", "f", "b", "F", "f*", "B*"):
            if cur: paths.append(dict(op=t, pts=cur, color=st["RG"], dash=st["d"], clip=st["clip"]))
            cur = []
        elif t == "n": cur = []
        elif t == "cm": cm = tuple(float(v) for v in ops[-6:])
        elif t == "TJ" or t == "Tj":
            raw = "".join(re.findall(r"\(((?:\\.|[^\\)])*)\)", " ".join(ops), flags=re.S))
            raw = re.sub(r"\\([0-7]{1,3})", lambda m: chr(int(m.group(1), 8)), raw)
            raw = re.sub(r"\\(.)", r"\1", raw, flags=re.S)
            s = raw.encode("latin-1").decode("utf-16-be", errors="replace")
            texts.append((s.replace(" ", ""), cm))
        ops = [] if t not in ("BT",) else ops
        if t not in ("BT",): ops = []
    return paths, texts

def read_curves(pdfpath):
    paths, texts = parse_paths(pdfpath)
    # y ticks: stroked+filled ('B') horizontal 2.5 pt segments left of an axes; x ticks: vertical 2.5 pt segments below
    yt = [p for p in paths if p["op"] == "B" and len(p["pts"]) == 2 and abs(p["pts"][0][1]-p["pts"][1][1]) < 1e-9 and abs(abs(p["pts"][0][0]-p["pts"][1][0])-2.5) < 1e-6]
    xt = [p for p in paths if p["op"] == "B" and len(p["pts"]) == 2 and abs(p["pts"][0][0]-p["pts"][1][0]) < 1e-9 and abs(abs(p["pts"][0][1]-p["pts"][1][1])-2.5) < 1e-6]
    # tick label texts with their placement matrices
    num = [(t, cm) for t, cm in texts if re.fullmatch(r"\d+(\.\d+)?", t) and cm is not None]
    ylab = [(float(t), cm[5]) for t, cm in num if "." in t and len(t) == 3]
    xlab = [(int(t), cm[4]) for t, cm in num if "." not in t]
    return paths, yt, xt, ylab, xlab, texts

def yscale_for(clip, yt, ylab):
    y0, y1 = clip[1], clip[1] + clip[3]
    ys = sorted(p["pts"][0][1] for p in yt if y0 - 1e-6 <= p["pts"][0][1] <= y1 + 1e-6)
    labs = sorted([(v, yb) for v, yb in ylab if y0 - 12 <= yb <= y1 + 12], key=lambda r: r[1])
    # pair each tick with the label whose baseline sits a constant offset below it
    pairs = []
    for y in ys:
        v, yb = min(labs, key=lambda r: abs((y - r[1]) - (ys[0] - labs[0][1])))
        pairs.append((v, y, y - yb))
    offs = [o for _, _, o in pairs]
    vals = np.array([v for v, _, _ in pairs]); yy = np.array([y for _, y, _ in pairs])
    b, a = np.polyfit(yy, vals, 1)
    resid = np.abs(a + b * yy - vals).max()
    return a, b, pairs, max(offs) - min(offs), resid

def check_pdf(tag, pdfpath, refs, answers):
    paths, yt, xt, ylab, xlab, texts = read_curves(pdfpath)
    clips = sorted({p["clip"] for p in paths if p["clip"] is not None}, key=lambda c: -c[1])
    say(f"[{tag} pdf] axes clip boxes (top to bottom): {len(clips)}; y ticks {len(yt)}; x ticks {len(xt)}")
    # panel label -> clip box
    panel_names = {}
    for t, cm in texts:
        for ds, key in (("PC-GITA", "PC-GITA(PD)"), ("Pitt", "Pitt(AD)"), ("E-DAIC", "E-DAIC(MDD)")):
            if t == key and cm is not None:
                c = min(clips, key=lambda c: 0 if c[1] <= cm[5] <= c[1] + c[3] else min(abs(cm[5]-c[1]), abs(cm[5]-c[1]-c[3])))
                panel_names[ds] = c
    say(f"[{tag} pdf] panel labels found: {sorted(panel_names)}")
    # x scale per stream from x ticks under the bottom panel plus their labels
    bottom = clips[-1]
    xs = sorted({round(p["pts"][0][0], 6) for p in xt if abs(min(p["pts"][0][1], p["pts"][1][1]) - (bottom[1] - 2.5)) < 1e-6})
    xl = sorted(xlab, key=lambda r: r[1])
    say(f"[{tag} pdf] x tick positions {xs}; x tick labels in order {[v for v, _ in xl]}")
    # split ticks into encoder (first group) and llm (second group) at the projector break
    vals = [v for v, _ in xl]
    split = next(i for i in range(1, len(vals)) if vals[i] < vals[i-1])
    enc = np.polyfit(xs[:split], vals[:split], 1); llm = np.polyfit(xs[split:], vals[split:], 1)
    out = {}
    for ds, clip in panel_names.items():
        a, b, pairs, offspread, resid = yscale_for(clip, yt, ylab)
        say(f"[{tag} pdf] {ds}: y ticks {[(v, round(y, 4)) for v, y, _ in pairs]}, label offset spread {offspread:.2e}, linear fit residual {resid:.2e}")
        curves = [p for p in paths if p["clip"] == clip and p["op"] == "S" and len(p["pts"]) >= 20]
        for p in curves:
            x = np.array([q[0] for q in p["pts"]]); y = np.array([q[1] for q in p["pts"]])
            stream = "encoder" if x[0] < xs[split - 1] + 1 else "llm"
            f = enc if stream == "encoder" else llm
            layer = f[0] * x + f[1]
            v = a + b * y
            ref = refs[ds][stream]
            li = np.round(layer).astype(int)
            d = np.abs(v - ref.reindex(li).values)
            out[(ds, stream)] = d.max()
            say(f"[{tag} pdf] {ds} {stream}: n {len(x)} (ref n {len(ref)}), layers {li.min()}..{li.max()} max |layer - int| {np.abs(layer-li).max():.1e}, "
                f"max |pdf - source| = {d.max():.2e} at layer {li[d.argmax()]}")
        # answer line: dotted 2 point stroke spanning the axes, same colour as the curves
        dots = [p for p in paths if p["clip"] == clip and p["op"] == "S" and len(p["pts"]) == 2
                and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-9 and p["color"] == (curves[0]["color"] if curves else None)]
        for p in dots:
            v = a + b * p["pts"][0][1]
            say(f"[{tag} pdf] {ds} answer line dash {p['dash']} at AUC {v:.6f} (expected {answers[ds]}) diff {abs(v-answers[ds]):.1e}")
    return out

def mirror(ds):
    s = {"PC-GITA": "pcgita", "Pitt": "pitt"}[ds]
    o = {}
    for st, fn in (("encoder", "encoder"), ("llm", "llm")):
        df = pd.read_csv(f"{MIRROR}/{s}_{fn}.csv"); o[st] = df.set_index(df.columns[0])["auc"].astype(float)
    return o
refs = {"PC-GITA": mirror("PC-GITA"), "Pitt": mirror("Pitt"), "E-DAIC": pod}
side = json.load(open(f"{FIG}/fig1_gap_whole.sidecar.json"))
for ds in ("PC-GITA", "Pitt"):
    for st in ("encoder", "llm"):
        pth = side["sources"][ds][st]
        say(f"[mirror {ds} {st}] sha256 matches figure sidecar: {sha(rb(pth)) == side['sources'][ds][st + '_sha256']}")
# E-DAIC answer: sklearn AUC on the part23 per-clip zero shot file
Z = pd.read_csv(f"{POD}/out/A2_edaic_whole_zeroshot_perclip.csv")
pz = Z.p_yes_whole.astype(str).str.replace(r"np\.float64\((.*)\)", r"\1", regex=True).astype(float)
auc_zs = roc_auc_score(Z.label.astype(int), pz)
say(f"[answer] E-DAIC whole zero shot roc_auc_score(label, p_yes_whole) = {auc_zs:.6f} -> {auc_zs:.4f}; n {len(Z)}, speakers {Z.speaker.nunique()}, positives {int(Z.label.sum())}")
answers = {"PC-GITA": side["answer_lines"]["PC-GITA"], "Pitt": side["answer_lines"]["Pitt"], "E-DAIC": round(auc_zs, 4)}
pdfdiff = check_pdf("whole", PDFS["whole"], refs, answers)

# ---------------- CHECK 4: pod curves vs part23 Linux recompute ----------------
L = pd.read_csv(f"{P23}/verify/curves_linux191_pid.csv")
say(f"[linux191] file sha256 {sha(rb(f'{P23}/verify/curves_linux191_pid.csv'))[:16]}, rows {len(L)}, env {open(f'{P23}/verify/curves_linux191_env.txt').read().strip()}")
check4 = {}
for stream in ("encoder", "llm"):
    s = L[L.stream == stream].set_index("stage")
    ref = pod[stream]
    d = (s.recomputed - ref.reindex(s.index)).abs()
    dpodcol = (s.pod - ref.reindex(s.index)).abs().max()
    eq4 = int((s.recomputed.round(4) == ref.reindex(s.index).round(4)).sum())
    i = int(d.idxmax()); check4[stream] = d.max()
    say(f"[linux191] {stream}: n {len(s)} (pod n {len(ref)}), max |recomputed - pod file| = {d.max():.6f} at stage {i} "
        f"(pod {ref[i]:.4f}, recompute {s.recomputed[i]:.4f}); equal at 4 dp {eq4}/{len(s)}; mean |diff| {d.mean():.6f}; "
        f"file's own pod column vs pod file max {dpodcol:.1e}; own absdiff column consistent {np.allclose(s.absdiff, (s.recomputed - s.pod).abs())}")
    # does any recompute difference move the figure visibly?  3rd panel spans 44.46 pt for 0.48 AUC
    say(f"[linux191] {stream}: that largest gap is {d.max() * 44.46 / 0.48:.3f} pt on the printed panel")

# ---------------- CHECK 5: current Figure 1 file not overwritten ----------------
cur = f"{SUB}/zip/figures/fig1_gap.pdf"
b = rb(cur)
say(f"[fig1_gap.pdf] zip/figures: md5 {md5(b)} sha256 {sha(b)[:16]} size {len(b)} mtime {mtime(cur)}")
with zipfile.ZipFile(f"{SUB}/overleaf.zip") as z:
    names = [n for n in z.namelist() if n.endswith("fig1_gap.pdf")]
    for n in names:
        zb = z.read(n); zi = z.getinfo(n)
        say(f"[fig1_gap.pdf] overleaf.zip entry {n}: md5 {md5(zb)} size {len(zb)} zip date {zi.date_time}; identical to zip/figures file {zb == b}")
    texn = [n for n in z.namelist() if n.endswith("AudioLLMHealthUtilityGap.tex")]
    for n in texn:
        tb = z.read(n)
        say(f"[tex] overleaf.zip {n} identical to zip/ tex {tb == rb(f'{SUB}/zip/AudioLLMHealthUtilityGap.tex')}, "
            f"to Desktop tex {tb == rb('<local data dir>/paper1_final/AudioLLMHealthUtilityGap.tex')}; "
            f"includegraphics lines: {re.findall(rb'includegraphics[^}]*fig1[^}]*}', tb)}")
say(f"[fig1_gap.pdf] zip/ tex mtime {mtime(f'{SUB}/zip/AudioLLMHealthUtilityGap.tex')}, Desktop tex mtime {mtime('<local data dir>/paper1_final/AudioLLMHealthUtilityGap.tex')}")
fig_start = min(os.stat(f"{FIG}/{f}").st_mtime for f in os.listdir(FIG) if not f.startswith("._"))
say(f"[fig1_gap.pdf] earliest file in part24/fig {datetime.datetime.fromtimestamp(fig_start):%Y-%m-%dT%H:%M:%S}; "
    f"fig1_gap.pdf mtime is before it: {os.stat(cur).st_mtime < fig_start}")
dk = "<local data dir>/paper1_final/figures/fig1_gap.pdf"
say(f"[fig1_gap.pdf] Desktop paper1_final copy: md5 {md5(rb(dk))} mtime {mtime(dk)} (older v4 output; not the Overleaf file)")
new = rb(f"{SUB}/zip/figures/fig1_gap_whole.pdf")
say(f"[fig1_gap_whole.pdf] zip/figures copy md5 {md5(new)} identical to part24/fig copy {new == rb(PDFS['whole'])}; differs from fig1_gap.pdf {new != b}")

json.dump(dict(written_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
               command=" ".join([sys.executable] + sys.argv), script_sha256=sha(rb(os.path.abspath(__file__))),
               seed=None, gpu=None, bootstrap_rule="none (figure checks)", n=275, n_speakers=275,
               library_versions=dict(python=platform.python_version(), pymupdf=pymupdf.VersionBind, numpy=np.__version__,
                                     pandas=pd.__version__, sklearn=sklearn.__version__, platform=platform.platform())),
          open(f"{HERE}/verify_item2_indep.sidecar.json", "w"), indent=1)
open(f"{HERE}/verify_item2_indep.out", "w").write("\n".join(out_lines) + "\n")
