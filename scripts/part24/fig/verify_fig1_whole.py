"""Independent verifier for PART 24 item 2 (Figure 1, whole interview).

Runs in its own uv venv (python 3.14, numpy/pandas/sklearn/pymupdf from PyPI),
not in the /usr/local/bin/python3 environment that drew the figure. It never
imports make_fig1_whole.py. It reads:
  - the PDFs themselves (PyMuPDF): character sizes, and the curve vertices,
    converted back to AUC through each panel's own drawn left spine
  - the Line2D arrays the figure script dumped (<fig>_plotted.csv)
  - the pod curve csvs, the PC-GITA / Pitt mirror csvs
  - the independent Linux recompute curves_linux191_pid.csv
  - the per clip zero shot csv, to recompute the 0.8366 answer line
usage: venv/bin/python verify_fig1_whole.py
"""
import csv
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pymupdf
import sklearn
from sklearn.metrics import roc_auc_score

HERE = "<local data dir>/part24/fig"
P23 = "<local data dir>/part23"
OUT = f"{P23}/pull/p23-a-probes/p23a/out"
MIRROR = "<local data dir>/release/overnight/curves/omni"
ZIPFIG = "<local data dir>/paper1_submission_23sep/zip/figures"
OVERLEAF_PDF = f"{ZIPFIG}/fig1_gap.pdf"
LINUX = f"{P23}/verify/curves_linux191_pid.csv"

POD = {"encoder": f"{OUT}/o25_whole_encoder_perlayer.csv",
       "llm": f"{OUT}/o25_whole_llm_perlayer.csv"}
MIR = {("PC-GITA", "encoder"): f"{MIRROR}/pcgita_encoder.csv",
       ("PC-GITA", "llm"): f"{MIRROR}/pcgita_llm.csv",
       ("Pitt", "encoder"): f"{MIRROR}/pitt_encoder.csv",
       ("Pitt", "llm"): f"{MIRROR}/pitt_llm.csv"}
lines = []


def say(s):
    print(s)
    lines.append(s)


def pod_curve(stream):
    d = pd.read_csv(POD[stream])
    d = d.sort_values("layer" if "layer" in d.columns else "stage")
    return d["auc_oof"].to_numpy(float)


def mirror_curve(k, stream):
    d = pd.read_csv(MIR[(k, stream)])
    key = "layer" if "layer" in d.columns else "stage"
    col = "auc_oof" if "auc_oof" in d.columns else "auc"
    return d.sort_values(key)[col].to_numpy(float)


def truth(k, stream):
    return pod_curve(stream) if k == "E-DAIC" else mirror_curve(k, stream)


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def same_rgb(a, b):
    return a is not None and all(abs(x - y) < 0.01 for x, y in zip(a, b))


def polylines(page):
    """Every stroked path as (stroke rgb, width, list of points)."""
    out = []
    for d in page.get_drawings():
        segs = [it for it in d["items"] if it[0] == "l"]
        if not segs or d.get("color") is None:
            continue
        pts = [segs[0][1]] + [s[2] for s in segs]
        out.append((tuple(d["color"]), d.get("width"), pts))
    return out


def spines(paths):
    """Left spines: black vertical single segments longer than 30 pt, top first."""
    v = []
    for col, w, pts in paths:
        if len(pts) == 2 and same_rgb(col, (0, 0, 0)):
            (x0, y0), (x1, y1) = (pts[0].x, pts[0].y), (pts[1].x, pts[1].y)
            if abs(x0 - x1) < 1e-6 and abs(y0 - y1) > 30:
                v.append((min(y0, y1), max(y0, y1), x0))
    return sorted(v)


def pdf_curves(pdf, colors, ylim, n_panels, seglens):
    """Curve values read back from the PDF drawing, per dataset and stream."""
    page = pymupdf.open(pdf)[0]
    paths = polylines(page)
    sp = spines(paths)
    assert len(sp) == n_panels, (pdf, sp)
    got = {}
    for pi, (k, hexc) in enumerate(colors.items()):
        top, bot, _ = sp[pi if n_panels > 1 else 0]
        cands = [pts for col, w, pts in paths
                 if same_rgb(col, hexrgb(hexc)) and len(pts) - 1 in seglens]
        ys = [np.array([ylim[0] + (bot - p.y) / (bot - top) * (ylim[1] - ylim[0]) for p in pts])
              for pts in cands]
        xs = [np.array([p.x for p in pts]) for pts in cands]
        order = np.argsort([x[0] for x in xs])
        ys = [ys[i] for i in order]
        if len(ys) == 1:                       # v4: one 61 point trace
            ys = [ys[0][:32], ys[0][32:]]
        assert len(ys) == 2 and len(ys[0]) == 32 and len(ys[1]) == 29, (pdf, k, [len(y) for y in ys])
        got[k] = {"encoder": ys[0], "llm": ys[1]}
    return got


def char_sizes(pdf):
    page = pymupdf.open(pdf)[0]
    spans = []
    for b in page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                txt = "".join(ch["c"] for ch in sp["chars"])
                if txt.strip():
                    spans.append((round(sp["size"], 4), sp["font"], txt))
    sizes = [s[0] for s in spans]
    return min(sizes), max(sizes), spans, page.rect


def pdffonts(pdf):
    r = subprocess.run(["/opt/homebrew/bin/pdffonts", pdf], capture_output=True, text=True)
    return r.stdout.rstrip().splitlines()


def main():
    rows = []
    say("PART 24 item 2 verifier, " + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    say(f"env python {platform.python_version()} numpy {np.__version__} pandas {pd.__version__} "
        f"sklearn {sklearn.__version__} pymupdf {pymupdf.__version__}")

    # 0. the answer line value, recomputed from the per clip zero shot csv
    a2 = pd.read_csv(f"{OUT}/A2_edaic_whole_zeroshot_perclip.csv")
    num = lambda s: float(re.sub(r"^np\.float64\((.*)\)$", r"\1", str(s)))
    auc_ans = roc_auc_score(a2["label"].astype(int), a2["p_yes_whole"].map(num))
    say(f"answer line: E-DAIC whole zero shot AUC recomputed {auc_ans:.6f} -> {auc_ans:.4f} "
        f"(n {len(a2)}, speakers {a2['speaker'].nunique()}, positives {int(a2['label'].sum())}); "
        f"figure uses 0.8366")
    assert round(auc_ans, 4) == 0.8366

    targets = {
        "fig1_gap_whole.pdf": dict(pdf=f"{HERE}/fig1_gap_whole.pdf",
                                   colors={"PC-GITA": "#1f4e9c", "Pitt": "#b03a2e", "E-DAIC": "#c98a00"},
                                   ylim=(0.48, 0.96), n_panels=3, seglens=(31, 28)),
        "fig1_gap_whole_v4layout.pdf": dict(pdf=f"{HERE}/fig1_gap_whole_v4layout.pdf",
                                            colors={"PC-GITA": "#1f4e9a", "Pitt": "#d9531e", "E-DAIC": "#2e7d4f"},
                                            ylim=(0.45, 1.00), n_panels=1, seglens=(60,)),
        "OVERLEAF fig1_gap.pdf (current, for comparison)": dict(
            pdf=OVERLEAF_PDF, colors={"PC-GITA": "#1f4e9c", "Pitt": "#b03a2e", "E-DAIC": "#c98a00"},
            ylim=(0.48, 0.96), n_panels=3, seglens=(31, 28)),
    }

    # 1. copy beside the current Figure 1 is byte identical
    import hashlib
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    for ext in ("pdf", "png"):
        a, b = f"{HERE}/fig1_gap_whole.{ext}", f"{ZIPFIG}/fig1_gap_whole.{ext}"
        say(f"copy check {ext}: part24 {md5(a)}  zip/figures {md5(b)}  identical {md5(a) == md5(b)}")
    say(f"current zip/figures/fig1_gap.pdf md5 {md5(OVERLEAF_PDF)} (unchanged if bbefb559a5768dcb07d11ed5d440da18)")

    pdf_vals = {}
    for name, t in targets.items():
        fmin, fmax, spans, rect = char_sizes(t["pdf"])
        say(f"[{name}] page {rect.width:.2f} x {rect.height:.2f} pt; PyMuPDF char size min {fmin} max {fmax} "
            f"over {len(spans)} spans")
        small = sorted({(s, txt) for s, f, txt in spans if s < 9.0})
        if small:
            say(f"[{name}] spans under 9 pt: {small}")
        say(f"[{name}] span fonts: {sorted({f for s, f, txt in spans})}")
        for ln in pdffonts(t["pdf"]):
            say(f"[{name}] pdffonts | {ln}")
        pdf_vals[name] = pdf_curves(t["pdf"], t["colors"], t["ylim"], t["n_panels"], t["seglens"])

    # 2. arrays the figure plots (Line2D dump) vs the pod / mirror csvs
    for tag in ("fig1_gap_whole", "fig1_gap_whole_v4layout"):
        d = pd.read_csv(f"{HERE}/{tag}_plotted.csv")
        worst = {}
        for k in ("PC-GITA", "Pitt", "E-DAIC"):
            for stream in ("encoder", "llm"):
                y = d[(d.dataset == k) & (d.stream == stream)].sort_values("layer")["auc_plotted"].to_numpy(float)
                ref = truth(k, stream)
                diff = np.abs(y - ref)
                worst[(k, stream)] = (float(diff.max()), int(diff.argmax()), len(y))
                for i in range(len(y)):
                    rows.append(dict(figure=tag, source="Line2D arrays", dataset=k, stream=stream, layer=i,
                                     plotted=y[i], reference=ref[i], absdiff=abs(y[i] - ref[i])))
            ans = float(d[(d.dataset == k) & (d.stream == "answer_line")]["auc_plotted"].iloc[0])
            say(f"[{tag}] {k} answer line plotted {ans}")
        for (k, stream), (m, i, n) in worst.items():
            ref = "pod" if k == "E-DAIC" else "mirror"
            say(f"[{tag}] Line2D vs {ref} csv  {k:<8}{stream:<8} n {n}  max |diff| {m:.3g} (layer {i})")
        ed = max(worst[("E-DAIC", s)][0] for s in ("encoder", "llm"))
        say(f"[{tag}] LARGEST per layer |figure arrays - pod curves| E-DAIC = {ed:.3g}")

    # 3. the same, read back out of the PDF drawings themselves
    for name in ("fig1_gap_whole.pdf", "fig1_gap_whole_v4layout.pdf"):
        for k in ("PC-GITA", "Pitt", "E-DAIC"):
            for stream in ("encoder", "llm"):
                y = pdf_vals[name][k][stream]
                ref = truth(k, stream)
                diff = np.abs(y - ref)
                for i in range(len(y)):
                    rows.append(dict(figure=name, source="PDF vertices", dataset=k, stream=stream, layer=i,
                                     plotted=y[i], reference=ref[i], absdiff=diff[i]))
                say(f"[{name}] PDF vertices vs {'pod' if k == 'E-DAIC' else 'mirror'} csv  {k:<8}{stream:<8} "
                    f"max |diff| {diff.max():.2g}")

    # 4. PC-GITA and Pitt unchanged against the figure the tex includes now
    ov = pdf_vals["OVERLEAF fig1_gap.pdf (current, for comparison)"]
    for k in ("PC-GITA", "Pitt", "E-DAIC"):
        for stream in ("encoder", "llm"):
            a, b = pdf_vals["fig1_gap_whole.pdf"][k][stream], ov[k][stream]
            say(f"new vs current Overleaf figure, PDF vertices  {k:<8}{stream:<8} max |diff| {np.abs(a - b).max():.3g}")

    # 5. pod curves vs the independent Linux recompute
    L = pd.read_csv(LINUX)
    for stream in ("encoder", "llm"):
        pod = pod_curve(stream)
        s = L[L.stream == stream].sort_values("stage")
        rec = s["recomputed"].to_numpy(float)
        podcol_ok = np.abs(s["pod"].to_numpy(float) - pod).max()
        diff = np.abs(rec - pod)
        for i in range(len(pod)):
            rows.append(dict(figure="pod", source="Linux recompute sklearn 1.9.1", dataset="E-DAIC",
                             stream=stream, layer=i, plotted=pod[i], reference=rec[i], absdiff=diff[i]))
        say(f"pod vs Linux recompute  {stream:<8} n {len(pod)}  max |diff| {diff.max():.4f} at stage "
            f"{int(diff.argmax())} (pod {pod[diff.argmax()]:.4f}, recompute {rec[diff.argmax()]:.4f}); "
            f"stages equal at 4 dp {int((np.round(rec, 4) == np.round(pod, 4)).sum())}/{len(pod)}; "
            f"recompute file's own pod column vs pod csv {podcol_ok:.3g}")

    pd.DataFrame(rows).to_csv(f"{HERE}/verify_fig1_whole_perlayer.csv", index=False)
    with open(f"{HERE}/verify_fig1_whole.out", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    side = {"result": "PART 24 item 2 verifier", "command": f"{sys.executable} {os.path.abspath(__file__)}",
            "seed": None, "gpu": None, "bootstrap_rule": "none (figure checks)",
            "n": int(len(a2)), "n_speakers": int(a2["speaker"].nunique()),
            "model_id": "Qwen/Qwen2.5-Omni-7B (curves only)",
            "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__,
                                 "pymupdf": pymupdf.__version__, "platform": platform.platform()},
            "inputs": [POD["encoder"], POD["llm"], LINUX, f"{OUT}/A2_edaic_whole_zeroshot_perclip.csv",
                       *MIR.values(), OVERLEAF_PDF],
            "outputs": [f"{HERE}/verify_fig1_whole_perlayer.csv", f"{HERE}/verify_fig1_whole.out"]}
    with open(f"{HERE}/verify_fig1_whole.sidecar.json", "w") as fh:
        json.dump(side, fh, indent=1)


main()
