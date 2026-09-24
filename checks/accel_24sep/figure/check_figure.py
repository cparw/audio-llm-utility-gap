"""Independent check of Figure 1 (whole interview E-DAIC variant).
Independent code. Reads only. Writes only to this folder.
Run with the uv venv in this folder (PyMuPDF) for the PDF parts.
"""
import hashlib
import json
import os
import sys

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = "checks/final_2325Z/figure"
POD = ("scores/part23/pull/"
       "p23-a-probes/p23a/out")
OVL = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z"

PDFS = [f"{FIGDIR}/fig1_gap_whole.pdf", f"{FIGDIR}/fig1_gap_whole_86mm.pdf",
        f"{FIGDIR}/for_overleaf/fig1_gap_whole.pdf", f"{OVL}/figures/fig1_gap.pdf"]
COLUMNWIDTH_BP = 86.0 / 25.4 * 72.0  # spconf: (178 - 6) / 2 = 86 truemm

GREEN = (0x2e / 255, 0x7d / 255, 0x4f / 255)
GRID = (0xec / 255, 0xec / 255, 0xec / 255)

out = {}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def close(c, ref, tol=0.01):
    return c is not None and len(c) >= 3 and all(abs(a - b) < tol for a, b in zip(c[:3], ref))


def pdf_check(path):
    r = {"md5": md5(path)}
    doc = fitz.open(path)
    page = doc[0]
    r["page_w_bp"], r["page_h_bp"] = page.rect.width, page.rect.height
    scale = COLUMNWIDTH_BP / page.rect.width
    r["include_scale_at_columnwidth"] = scale
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    spans.append((round(s["size"], 4), s["font"], s["text"]))
    sizes = [s[0] for s in spans]
    r["n_spans"] = len(spans)
    r["span_min_pt"], r["span_max_pt"] = min(sizes), max(sizes)
    r["span_min_pt_as_printed"] = min(sizes) * scale
    r["span_max_pt_as_printed"] = max(sizes) * scale
    r["spans"] = spans
    r["fonts_pymupdf"] = sorted(set(s[1] for s in spans))
    r["fonts_embedded_list"] = [list(f) for f in page.get_fonts(full=True)]

    # drawings: calibrate y with the light grey horizontal gridlines at 0.5..1.0
    dr = page.get_drawings()
    grid_y = []
    for d in dr:
        if close(d.get("color"), GRID) and len(d["items"]) == 1 and d["items"][0][0] == "l":
            p1, p2 = d["items"][0][1], d["items"][0][2]
            if abs(p1.y - p2.y) < 1e-6 and abs(p1.x - p2.x) > 50:
                grid_y.append(p1.y)
    grid_y = sorted(set(round(y, 6) for y in grid_y))
    r["gridlines_found"] = len(grid_y)
    if len(grid_y) < 6:
        return r
    # fitz y grows downward; top gridline is 1.0
    vals = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    b, a = np.polyfit(grid_y, vals, 1)
    r["grid_fit_resid_max"] = float(np.max(np.abs(np.polyval([b, a], grid_y) - vals)))
    to_auc = lambda y: a + b * y

    greens = [d for d in dr if close(d.get("color"), GREEN) and d.get("type") == "s"]
    ans, curve = [], []
    for d in greens:
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                if not pts:
                    pts.append(it[1])
                pts.append(it[2])
        if len(pts) == 2 and abs(pts[0].y - pts[1].y) < 1e-6 and abs(pts[0].x - pts[1].x) > 50:
            ans.append(dict(y_auc=to_auc(pts[0].y), width=d.get("width"),
                            dashes=d.get("dashes"), x0=pts[0].x, x1=pts[1].x))
        elif len(pts) > 20:
            curve.append(dict(n=len(pts), width=d.get("width"), dashes=d.get("dashes"),
                              y=[to_auc(p.y) for p in pts], x=[p.x for p in pts]))
    r["edaic_answer_lines"] = ans
    r["edaic_curves"] = [{k: v for k, v in c.items() if k not in ("x", "y")} for c in curve]
    if len(curve) == 1:
        c = curve[0]
        enc = pd.read_csv(f"{POD}/o25_whole_encoder_perlayer.csv")
        llm = pd.read_csv(f"{POD}/o25_whole_llm_perlayer.csv")
        ref = np.concatenate([enc["auc_oof"].values, llm["auc_oof"].values])
        y = np.array(c["y"])
        dx = np.diff(c["x"])
        r["edaic_curve_x_step_spread"] = float(dx.max() - dx.min())
        if len(y) == len(ref):
            r["pdf_curve_vs_pod_auc_oof_maxabs"] = float(np.max(np.abs(y - ref)))
            refm = np.concatenate([enc["auc_mean"].values, llm["auc_mean"].values])
            r["pdf_curve_vs_pod_auc_mean_maxabs"] = float(np.max(np.abs(y - refm)))
    return r


for p in PDFS:
    if os.path.exists(p):
        out[p] = pdf_check(p)

# plotted csv vs pod csv
for tag in ("fig1_gap_whole_plotted.csv", "fig1_gap_whole_86mm_plotted.csv"):
    p = f"{FIGDIR}/{tag}"
    if not os.path.exists(p):
        continue
    pl = pd.read_csv(p)
    ed = pl[pl.dataset == "E-DAIC"]
    enc = pd.read_csv(f"{POD}/o25_whole_encoder_perlayer.csv")
    llm = pd.read_csv(f"{POD}/o25_whole_llm_perlayer.csv")
    pe = ed[ed.stream == "encoder"].sort_values("index")
    pm = ed[ed.stream == "llm"].sort_values("index")
    r = {"md5": md5(p), "n_enc_plotted": len(pe), "n_llm_plotted": len(pm),
         "n_enc_pod": len(enc), "n_llm_pod": len(llm)}
    assert list(pe["index"].astype(int)) == list(enc["layer"])
    assert list(pm["index"].astype(int)) == list(llm["stage"])
    for col in ("auc_oof", "auc_mean"):
        r[f"enc_maxabs_vs_{col}"] = float(np.max(np.abs(pe.auc_plotted.values - enc[col].values)))
        r[f"llm_maxabs_vs_{col}"] = float(np.max(np.abs(pm.auc_plotted.values - llm[col].values)))
    r["answer_line_edaic"] = ed[ed.stream == "answer_line"].auc_plotted.tolist()
    r["x_plot_is_0_to_60"] = (ed[ed.stream != "answer_line"].x_plot.tolist()
                              == [float(i) for i in range(61)])
    out[p] = r

json.dump(out, open(f"{HERE}/check_figure.json", "w"), indent=1, default=str)
for k, v in out.items():
    print("==", k)
    for kk, vv in v.items():
        if kk in ("spans", "fonts_embedded_list"):
            continue
        print(f"  {kk}: {vv}")
