"""Verifier for checks/final_2325Z/figure (item 3: Figure 1 from make_fig1_v4.py,
whole interview E-DAIC curves, dotted answer line at 0.8366, text >= 9 pt).

Runs in a uv venv built offline from the local uv cache with the PART 24 venv versions
(python 3.14.6, pymupdf 1.28.2, numpy 2.5.3, pandas 3.0.6, sklearn 1.9.1), not in
/usr/local/bin/python3 that drew the figure, and never imports the figure script.
Reads only; writes verify_fig1_v4_whole.out / _perlayer.csv / .sidecar.json here.

usage: <venv>/bin/python verify_fig1_v4_whole.py NEW.pdf [NEW2.pdf ...]
"""
import csv
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pymupdf
import sklearn
from sklearn.metrics import roc_auc_score

HERE = "checks/final_2325Z/figure"
OVL = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z"
P23 = "scores/part23"
OUT = f"{P23}/pull/p23-a-probes/p23a/out"
LINUX = f"{P23}/verify/curves_linux191_pid.csv"
A2 = f"{OUT}/A2_edaic_whole_zeroshot_perclip.csv"
MIRROR = "<local data dir>/release/overnight/curves/omni"
VARIANTS = "<local data dir>/release/edaic_rerun/variants"
COMPILED_OLD = ("scores/leftovers_23sep/tex/"
                "original_AudioLLMHealthUtilityGap.pdf")
POD = {"encoder": f"{OUT}/o25_whole_encoder_perlayer.csv",
       "llm": f"{OUT}/o25_whole_llm_perlayer.csv"}
MIR = {("PC-GITA", "encoder"): f"{MIRROR}/pcgita_encoder.csv",
       ("PC-GITA", "llm"): f"{MIRROR}/pcgita_llm.csv",
       ("Pitt", "encoder"): f"{MIRROR}/pitt_encoder.csv",
       ("Pitt", "llm"): f"{MIRROR}/pitt_llm.csv"}
V4_COL = {"PC-GITA": "#1f4e9a", "Pitt": "#d9531e", "E-DAIC": "#2e7d4f"}
C_COL = {"PC-GITA": "#1f4e9c", "Pitt": "#b03a2e", "E-DAIC": "#c98a00"}
V4_ANS = {"PC-GITA": 0.5635, "Pitt": 0.6578, "E-DAIC": 0.8366}
ORDER = ("PC-GITA", "Pitt", "E-DAIC")
lines, rows = [], []


def say(s=""):
    print(s)
    lines.append(s)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def csv_curve(path, col=None, sort=True):
    d = pd.read_csv(path)
    key = "layer" if "layer" in d.columns else "stage"
    col = col or ("auc_oof" if "auc_oof" in d.columns else "auc")
    if sort:
        d = d.sort_values(key)
    return d[col].to_numpy(float), d[key].to_numpy(int)


def truth(k, stream):
    return csv_curve(POD[stream])[0] if k == "E-DAIC" else csv_curve(MIR[(k, stream)])[0]


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def same(a, b, tol=0.006):
    return a is not None and all(abs(x - y) < tol for x, y in zip(a, b))


def polylines(page):
    out = []
    for d in page.get_drawings():
        segs = [it for it in d["items"] if it[0] == "l"]
        if not segs or d.get("color") is None or len(segs) != len(d["items"]):
            continue
        pts = [(segs[0][1].x, segs[0][1].y)] + [(s[2].x, s[2].y) for s in segs]
        out.append(dict(color=tuple(d["color"]), width=d.get("width"), dashes=d.get("dashes"),
                        opacity=d.get("stroke_opacity"), pts=pts))
    return out


def spans(pdf):
    page = pymupdf.open(pdf)[0]
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                txt = "".join(ch["c"] for ch in sp["chars"])
                if txt.strip():
                    out.append((round(sp["size"], 4), sp["font"], txt, [round(v, 2) for v in sp["bbox"]]))
    return out, page.rect


def pdffonts(pdf):
    r = subprocess.run(["/opt/homebrew/bin/pdffonts", pdf], capture_output=True, text=True)
    return r.stdout.rstrip().splitlines()


def png_info(p):
    b = open(p, "rb").read()
    w, h = struct.unpack(">II", b[16:24])
    i = b.find(b"pHYs")
    dpi = None
    if i > 0:
        px, py, unit = struct.unpack(">IIB", b[i + 4:i + 13])
        dpi = (round(px * 0.0254, 2), round(py * 0.0254, 2)) if unit == 1 else (px, py, unit)
    return w, h, dpi


def v4_read(pdf):
    """One axes, ylim (0.45, 1.00), xlim (-1, 61): curves and answer lines from the vertices."""
    page = pymupdf.open(pdf)[0]
    P = polylines(page)
    black = [p for p in P if same(p["color"], (0, 0, 0)) and len(p["pts"]) == 2]
    vs = [p for p in black if abs(p["pts"][0][0] - p["pts"][1][0]) < 1e-6
          and abs(p["pts"][0][1] - p["pts"][1][1]) > 30]
    hs = [p for p in black if abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-6
          and abs(p["pts"][0][0] - p["pts"][1][0]) > 30]
    assert len(vs) == 1 and len(hs) == 1, (len(vs), len(hs))
    ytop, ybot = sorted(p[1] for p in vs[0]["pts"])
    x0, x1 = sorted(p[0] for p in hs[0]["pts"])
    spine_y = lambda y: 0.45 + (ybot - y) / (ybot - ytop) * 0.55
    to_x = lambda x: -1 + (x - x0) / (x1 - x0) * 62.0
    grid = sorted({round(p["pts"][0][1], 6) for p in P
                   if same(p["color"], hexrgb("#ececec")) and len(p["pts"]) == 2
                   and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-6}, reverse=True)
    assert len(grid) == 6, grid
    a, b = np.polyfit(grid, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 1)
    grid_y = lambda y: a * y + b
    got = {}
    for k in ORDER:
        col = hexrgb(V4_COL[k])
        cur = [p for p in P if same(p["color"], col) and len(p["pts"]) == 61]
        ans = [p for p in P if same(p["color"], col) and len(p["pts"]) == 2
               and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-6]
        assert len(cur) == 1 and len(ans) == 1, (k, len(cur), len(ans))
        ys = np.array([q[1] for q in cur[0]["pts"]])
        xs = np.array([to_x(q[0]) for q in cur[0]["pts"]])
        got[k] = dict(spine=np.array([spine_y(y) for y in ys]), grid=np.array([grid_y(y) for y in ys]),
                      x=xs, ans_spine=spine_y(ans[0]["pts"][0][1]), ans_grid=grid_y(ans[0]["pts"][0][1]),
                      ans_dashes=ans[0]["dashes"], ans_width=ans[0]["width"], ans_opacity=ans[0]["opacity"],
                      ans_x=[round(to_x(q[0]), 4) for q in ans[0]["pts"]],
                      cur_dashes=cur[0]["dashes"], cur_width=cur[0]["width"])
    return got


def c_read(pdf):
    """fig_options option C (3 panels, ylim 0.48-0.96): E-DAIC panel curves and answer line."""
    page = pymupdf.open(pdf)[0]
    P = polylines(page)
    vs = sorted((min(q[1] for q in p["pts"]), max(q[1] for q in p["pts"]))
                for p in P if same(p["color"], (0, 0, 0)) and len(p["pts"]) == 2
                and abs(p["pts"][0][0] - p["pts"][1][0]) < 1e-6
                and abs(p["pts"][0][1] - p["pts"][1][1]) > 30)
    assert len(vs) == 3, vs
    got = {}
    for pi, k in enumerate(ORDER):
        top, bot = vs[pi]
        f = lambda y: 0.48 + (bot - y) / (bot - top) * 0.48
        col = hexrgb(C_COL[k])
        enc = [p for p in P if same(p["color"], col) and len(p["pts"]) == 32]
        lm = [p for p in P if same(p["color"], col) and len(p["pts"]) == 29]
        ans = [p for p in P if same(p["color"], col) and len(p["pts"]) == 2
               and abs(p["pts"][0][1] - p["pts"][1][1]) < 1e-6]
        assert len(enc) == 1 and len(lm) == 1 and len(ans) == 1, (k, len(enc), len(lm), len(ans))
        got[k] = dict(encoder=np.array([f(q[1]) for q in enc[0]["pts"]]),
                      llm=np.array([f(q[1]) for q in lm[0]["pts"]]),
                      ans=f(ans[0]["pts"][0][1]), ans_dashes=ans[0]["dashes"])
    return got


def columnwidth_bp():
    sty = open(f"{OVL}/spconf.sty").read()
    tw = float(re.search(r"^\\textwidth\s+([\d.]+)truemm", sty, re.M).group(1))
    cs = float(re.search(r"^\\columnsep\s+([\d.]+)truemm", sty, re.M).group(1))
    return tw, cs, (tw - cs) / 2.0 / 25.4 * 72.0


def main():
    news = sys.argv[1:] or [f"{HERE}/fig1_gap_whole.pdf"]
    say("verify_fig1_v4_whole.py  " + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    say(f"env python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__} "
        f"sklearn {sklearn.__version__} pymupdf {pymupdf.__version__}")

    # answer line value from the per clip zero shot csv
    a2 = pd.read_csv(A2)
    num = lambda s: float(re.sub(r"^np\.float64\((.*)\)$", r"\1", str(s)))
    auc = roc_auc_score(a2["label"].astype(int), a2["p_yes_whole"].map(num))
    say(f"[answer] roc_auc_score(label, p_yes_whole) on {os.path.basename(A2)}: {auc:.6f} -> {auc:.4f} "
        f"(n {len(a2)}, speakers {a2['speaker'].nunique()}, positives {int(a2['label'].sum())}, "
        f"capped at 900 s {int(a2['capped'].sum())})")

    # pod csv row order (make_fig1_v4 reads rows in file order, it does not sort)
    for s in ("encoder", "llm"):
        _, idx = csv_curve(POD[s], sort=False)
        say(f"[pod order] {os.path.basename(POD[s])}: rows {len(idx)}, file order is 0..{len(idx) - 1}: "
            f"{bool((idx == np.arange(len(idx))).all())}")

    tw, cs, colw = columnwidth_bp()
    say(f"[column] spconf.sty \\textwidth {tw} truemm, \\columnsep {cs} truemm -> \\columnwidth "
        f"{(tw - cs) / 2} mm = {colw:.4f} bp = {colw / 72:.6f} in")

    for pdf in news:
        tag = os.path.basename(pdf)[:-4]
        say()
        say(f"===== {pdf}")
        say(f"md5 {md5(pdf)}")
        sp, rect = spans(pdf)
        sizes = [s[0] for s in sp]
        scale = colw / rect.width
        say(f"[size] page {rect.width:.2f} x {rect.height:.2f} bp ({rect.width / 72:.4f} x {rect.height / 72:.4f} in)")
        say(f"[fonts] PyMuPDF spans {len(sp)}: min {min(sizes)} pt, max {max(sizes)} pt; "
            f"span fonts {sorted({s[1] for s in sp})}")
        for s in sorted(sp):
            say(f"    {s[0]:>7} pt  {s[1]:<22} {s[2]!r}")
        say(f"[fonts] placed with width=\\columnwidth the scale is {colw:.4f}/{rect.width:.4f} = {scale:.6f}: "
            f"smallest {min(sizes)} pt prints at {min(sizes) * scale:.4f} pt, largest {max(sizes) * scale:.4f} pt")
        for ln in pdffonts(pdf):
            say(f"    pdffonts | {ln}")
        png = pdf[:-4] + ".png"
        if os.path.exists(png):
            w, h, dpi = png_info(png)
            say(f"[png] {os.path.basename(png)} {w} x {h} px, pHYs dpi {dpi}, md5 {md5(png)}")

        # Line2D arrays dumped by the figure script vs pod / mirror csvs
        plotted = pdf[:-4] + "_plotted.csv"
        d = pd.read_csv(plotted)
        for k in ORDER:
            for s in ("encoder", "llm"):
                y = d[(d.dataset == k) & (d.stream == s)].sort_values("index")["auc_plotted"].to_numpy(float)
                ref = truth(k, s)
                diff = np.abs(y - ref)
                for i in range(len(y)):
                    rows.append(dict(figure=tag, source="Line2D arrays", dataset=k, stream=s, layer=i,
                                     value=y[i], reference=ref[i], absdiff=diff[i]))
                say(f"[arrays] Line2D vs {'pod' if k == 'E-DAIC' else 'mirror'} csv  {k:<8}{s:<8} n {len(y)}  "
                    f"max |diff| {diff.max():.3g}  exactly equal {int((y == ref).sum())}/{len(y)}")
            a = d[(d.dataset == k) & (d.stream == "answer_line")]["auc_plotted"].to_numpy(float)
            say(f"[arrays] {k} answer line Line2D y {a.tolist()}")

        # the same, read back from the PDF vertices
        g = v4_read(pdf)
        for k in ORDER:
            xs = g[k]["x"]
            say(f"[pdf] {k:<8} curve x read back {xs.min():.4f}..{xs.max():.4f}, max |x - round(x)| "
                f"{np.abs(xs - np.round(xs)).max():.2g}, dashes {g[k]['cur_dashes']}, width {g[k]['cur_width']}")
            for s, sl in (("encoder", slice(0, 32)), ("llm", slice(32, 61))):
                ref = truth(k, s)
                for cal in ("spine", "grid"):
                    y = g[k][cal][sl]
                    diff = np.abs(y - ref)
                    for i in range(len(y)):
                        rows.append(dict(figure=tag, source=f"PDF vertices ({cal} calibration)", dataset=k,
                                         stream=s, layer=i, value=y[i], reference=ref[i], absdiff=diff[i]))
                    say(f"[pdf] PDF vertices ({cal:<5} cal.) vs {'pod' if k == 'E-DAIC' else 'mirror'} csv  "
                        f"{k:<8}{s:<8} max |diff| {diff.max():.2g}")
            say(f"[answer] {k:<8} line y read back: spine cal. {g[k]['ans_spine']:.6f}, grid cal. "
                f"{g[k]['ans_grid']:.6f} (script value {V4_ANS[k]}); x span {g[k]['ans_x']}; dashes "
                f"{g[k]['ans_dashes']}, width {g[k]['ans_width']}, opacity {g[k]['ans_opacity']}")

    # pod curves vs the independent Linux recompute
    say()
    L = pd.read_csv(LINUX)
    for s in ("encoder", "llm"):
        pod = truth("E-DAIC", s)
        t = L[L.stream == s].sort_values("stage")
        rec = t["recomputed"].to_numpy(float)
        diff = np.abs(rec - pod)
        i = int(diff.argmax())
        for j in range(len(pod)):
            rows.append(dict(figure="pod", source="Linux recompute sklearn 1.9.1", dataset="E-DAIC", stream=s,
                             layer=j, value=pod[j], reference=rec[j], absdiff=diff[j]))
        say(f"[linux] pod vs {os.path.basename(LINUX)} recomputed  {s:<8} n {len(pod)}  max |diff| "
            f"{diff.max():.6f} at stage {i} (pod {pod[i]:.6f}, recompute {rec[i]:.6f}); mean |diff| "
            f"{diff.mean():.6f}; equal at 4 dp {int((np.round(rec, 4) == np.round(pod, 4)).sum())}/{len(pod)}; "
            f"file's own pod column vs pod csv max |diff| {np.abs(t['pod'].to_numpy(float) - pod).max():.3g}")

    # the figure the final tex includes
    say()
    inc = f"{OVL}/figures/fig1_gap.pdf"
    say(f"===== final tex figure {inc} (read only), md5 {md5(inc)}")
    sp, rect = spans(inc)
    sizes = [s[0] for s in sp]
    say(f"[size] page {rect.width:.2f} x {rect.height:.2f} bp; PyMuPDF spans {len(sp)} min {min(sizes)} max "
        f"{max(sizes)}; fonts {sorted({s[1] for s in sp})}")
    say(f"[fonts] spans under 9 pt: {sorted({(s[0], s[2]) for s in sp if s[0] < 9})}")
    c = c_read(inc)
    cands = {"pod whole (o25_whole)": POD}
    for v in ("full", "mid300", "mid30"):
        cands[f"variants o25_{v}"] = {s: f"{VARIANTS}/o25_{v}_{s}_perlayer.csv" for s in ("encoder", "llm")}
    for name, paths in cands.items():
        m = max(np.abs(c["E-DAIC"][s] - csv_curve(paths[s])[0]).max() for s in ("encoder", "llm"))
        say(f"[final fig] E-DAIC panel vertices vs {name:<24} max |diff| {m:.3g}")
    for k in ORDER:
        say(f"[final fig] {k:<8} answer line read back {c[k]['ans']:.6f}, dashes {c[k]['ans_dashes']}")
        if k != "E-DAIC":
            m = max(np.abs(c[k][s] - truth(k, s)).max() for s in ("encoder", "llm"))
            say(f"[final fig] {k:<8} vertices vs mirror csv max |diff| {m:.3g}")

    # an earlier compile of the paper, to see the real include scale
    if os.path.exists(COMPILED_OLD):
        sp, _ = spans(COMPILED_OLD)
        st = sorted({(s[0], s[2]) for s in sp if "STIX" in s[1]})
        say(f"[compiled] {COMPILED_OLD}: STIX spans (figure text) sizes {sorted({s[0] for s in st})}; "
            f"9 pt source text measured at {[s[0] for s in st if s[1] in ('AUC', 'linear probe')][:1]}")

    pd.DataFrame(rows).to_csv(f"{HERE}/verify_fig1_v4_whole_perlayer.csv", index=False)
    with open(f"{HERE}/verify_fig1_v4_whole.out", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    side = {"result": "item 3 figure verifier (checks/final_2325Z/figure)",
            "command": f"{sys.executable} {os.path.abspath(__file__)} " + " ".join(news),
            "script_sha256": sha256(os.path.abspath(__file__)),
            "seed": None, "gpu": None, "bootstrap_rule": "none (figure checks)",
            "n": int(len(a2)), "n_speakers": int(a2["speaker"].nunique()),
            "model_id": "Qwen/Qwen2.5-Omni-7B (curves only)",
            "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                 "pymupdf": pymupdf.__version__, "platform": platform.platform()},
            "inputs": {p: sha256(p) for p in [POD["encoder"], POD["llm"], LINUX, A2, *MIR.values(), inc,
                                             f"{OVL}/spconf.sty", *news]},
            "outputs": [f"{HERE}/verify_fig1_v4_whole_perlayer.csv", f"{HERE}/verify_fig1_v4_whole.out"]}
    with open(f"{HERE}/verify_fig1_v4_whole.sidecar.json", "w") as fh:
        json.dump(side, fh, indent=1)


main()
