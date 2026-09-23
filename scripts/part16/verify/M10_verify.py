#!/usr/bin/env python3
"""
M10 INDEPENDENT VERIFICATION.
Written from scratch. Does NOT read, import or exec M10_kcl_transcript.py,
and does NOT read M10_kcl_transcript.csv for any computed value.
Goes back to the ORIGINAL sources named in the report.

AUC route: explicit Mann-Whitney U concordant-pair count, ties = 0.5.
Cross-checked against a trapezoid integration over the full ROC.
Deliberately NOT a rank-sum one-liner.
"""
import csv, json, itertools
import numpy as np

SRC_TEXT  = "<local data dir>/Desktop/release/overnight2/text_new/o25_kcl_text.csv"
SRC_AUDIO = "<local data dir>/Desktop/release/omni_final/omni_kcl_zeroshot_scores.csv"
SRC_ENC   = "<local data dir>/Desktop/release/omni_final/omni_kcl_enc_nested_oof.csv"

# ---------- load originals ----------
def load(path, idcol, spkcol, scorecol):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row[idcol]] = (row[spkcol], int(row["label"]), float(row[scorecol]))
    return out

T = load(SRC_TEXT,  "id",   "speaker_id", "p_yes")
A = load(SRC_AUDIO, "clip", "speaker",    "p_yes")
E = load(SRC_ENC,   "clip", "speaker",    "p_probe")

print("=== SOURCE ROW COUNTS ===")
for nm, d, p in (("transcript", T, SRC_TEXT), ("audio", A, SRC_AUDIO), ("encoder", E, SRC_ENC)):
    print(f"  {nm:11s} n={len(d):3d}  {p}")

clips = sorted(set(T) & set(A) & set(E))
assert len(clips) == len(T) == len(A) == len(E), "clip sets differ across sources"
for c in clips:
    assert T[c][1] == A[c][1] == E[c][1], f"label disagreement on {c}"
    assert T[c][0] == A[c][0] == E[c][0], f"speaker disagreement on {c}"

speakers = [T[c][0] for c in clips]
y   = np.array([T[c][1] for c in clips], dtype=int)
s_t = np.array([T[c][2] for c in clips], dtype=float)
s_a = np.array([A[c][2] for c in clips], dtype=float)
s_e = np.array([E[c][2] for c in clips], dtype=float)

n = len(clips)
uspk = sorted(set(speakers))
print(f"\n=== JOINED ===\n  n_clips={n}  n_speakers={len(uspk)}  "
      f"pos={int(y.sum())}  neg={int((y==0).sum())}  "
      f"max_clips_per_speaker={max(speakers.count(s) for s in uspk)}")

# ---------- AUC route 1: explicit U count, ties 0.5 ----------
def auc_u(labels, scores):
    pos = [sc for lb, sc in zip(labels, scores) if lb == 1]
    neg = [sc for lb, sc in zip(labels, scores) if lb == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p, q in itertools.product(pos, neg):
        if   p > q: wins += 1.0
        elif p == q: wins += 0.5
    return wins / (len(pos) * len(neg))

# ---------- AUC route 2: trapezoid over the full ROC ----------
def auc_trap(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=float)
    P = int((labels == 1).sum()); N = int((labels == 0).sum())
    if P == 0 or N == 0:
        return None
    pts = [(1.0, 1.0)]
    for thr in np.sort(np.unique(scores))[::-1]:
        pred = scores >= thr
        tpr = float(((pred) & (labels == 1)).sum()) / P
        fpr = float(((pred) & (labels == 0)).sum()) / N
        pts.append((fpr, tpr))
    pts.append((0.0, 0.0))
    pts = sorted(set(pts))
    x = np.array([p[0] for p in pts]); yy = np.array([p[1] for p in pts])
    return float(np.trapezoid(yy, x))

# ---------- tie diagnosis, transcript ----------
pos_t = [sc for lb, sc in zip(y, s_t) if lb == 1]
neg_t = [sc for lb, sc in zip(y, s_t) if lb == 0]
strict = sum(1 for p, q in itertools.product(pos_t, neg_t) if p > q)
tied   = sum(1 for p, q in itertools.product(pos_t, neg_t) if p == q)
npairs = len(pos_t) * len(neg_t)
print("\n=== TIE DIAGNOSIS, transcript ===")
print(f"  pairs={npairs}  strict_wins={strict}  tied_pairs={tied}")
print(f"  ties-as-loss   strict/pairs        = {strict/npairs:.6f}")
print(f"  ties half credit (strict+0.5*t)/p  = {(strict+0.5*tied)/npairs:.6f}")
if tied:
    vals = sorted(set(pos_t) & set(neg_t))
    for v in vals:
        print(f"  tied value {v!r}: pd_clips={[c for c in clips if T[c][1]==1 and T[c][2]==v]} "
              f"hc_clips={[c for c in clips if T[c][1]==0 and T[c][2]==v]}")

# ---------- point estimates ----------
pt = {}
print("\n=== POINT ESTIMATES (two independent routes) ===")
for nm, sc in (("transcript", s_t), ("audio", s_a), ("encoder", s_e)):
    u  = auc_u(y, sc)
    tr = auc_trap(y, sc)
    pt[nm] = u
    flag = "OK" if abs(u - tr) < 1e-9 else "ROUTE MISMATCH"
    print(f"  {nm:11s} U-count={u:.6f}  trapezoid={tr:.6f}  [{flag}]")
pt["t_minus_a"] = pt["transcript"] - pt["audio"]
pt["t_minus_e"] = pt["transcript"] - pt["encoder"]
print(f"  transcript-audio   = {pt['t_minus_a']:+.6f}")
print(f"  transcript-encoder = {pt['t_minus_e']:+.6f}")

# speaker-level check: one row per speaker so they must coincide
spk_idx = {s: i for i, s in enumerate(uspk)}
print(f"\n  speaker-level transcript AUC = {auc_u(y, s_t):.6f} (identical, 1 clip/speaker)")

# ---------- bootstrap ----------
# fast vectorised U with ties 0.5, used only inside the bootstrap loop.
# verified below to agree with the explicit pair count on the full sample.
def auc_fast(labels, scores):
    P = labels == 1; N = ~P
    if not P.any() or not N.any():
        return None
    a = scores[P][:, None]; b = scores[N][None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (P.sum() * N.sum()))

for nm, sc in (("transcript", s_t), ("audio", s_a), ("encoder", s_e)):
    assert abs(auc_fast(y, sc) - pt[nm]) < 1e-12, f"fast/explicit mismatch {nm}"

def run_boot(draw_kind):
    rng = np.random.default_rng(0)
    keep = {k: [] for k in ("transcript", "audio", "encoder", "t_minus_a", "t_minus_e")}
    usable = 0
    for _ in range(2000):
        if draw_kind == "integers":
            pick = rng.integers(0, len(uspk), len(uspk))
        else:
            pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        rows = np.array([spk_idx[speakers[i]] for i in range(n)])
        sel = np.concatenate([np.where(rows == p)[0] for p in pick])
        yy = y[sel]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        at = auc_fast(yy, s_t[sel]); aa = auc_fast(yy, s_a[sel]); ae = auc_fast(yy, s_e[sel])
        keep["transcript"].append(at); keep["audio"].append(aa); keep["encoder"].append(ae)
        keep["t_minus_a"].append(at - aa); keep["t_minus_e"].append(at - ae)
        usable += 1
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in keep.items()}
    pgt = float(np.mean(np.array(keep["transcript"]) > np.array(keep["audio"])))
    return ci, usable, pgt

results = {}
for kind in ("integers", "choice"):
    ci, usable, pgt = run_boot(kind)
    results[kind] = (ci, usable, pgt)
    print(f"\n=== BOOTSTRAP, default_rng(0), 2000 draws, speakers, rng.{kind} ===")
    print(f"  usable draws: {usable}/2000   P(boot transcript > audio) = {pgt:.4f}")
    for k in ("transcript", "audio", "encoder", "t_minus_a", "t_minus_e"):
        lo, hi = ci[k]
        print(f"  {k:11s} [{lo:.4f}, {hi:.4f}]")

# ---------- compare to their report ----------
theirs = {
    "transcript": (0.7634, 0.5936, 0.9118),
    "audio":      (0.7143, 0.5210, 0.8810),
    "encoder":    (0.7768, 0.5994, 0.9265),
    "t_minus_a":  (0.0491, -0.1827, 0.2818),
    "t_minus_e":  (-0.0134, -0.2383, 0.2381),
}
print("\n" + "=" * 78)
print("COMPARISON vs REPORTED (4 decimals)")
print("=" * 78)
print(f"{'quantity':12s} {'theirs':>9s} {'mine':>9s} {'pt':>4s}  {'their CI':>20s} {'my CI(int)':>20s} {'ci':>4s}")
verdict = {}
for k, (tv, tlo, thi) in theirs.items():
    mv = pt[k] if k in pt else None
    ci_i = results["integers"][0][k]
    ok_pt = f"{mv:.4f}" == f"{tv:.4f}"
    ok_ci = (f"{ci_i[0]:.4f}" == f"{tlo:.4f}") and (f"{ci_i[1]:.4f}" == f"{thi:.4f}")
    verdict[k] = (mv, ci_i, ok_pt, ok_ci)
    print(f"{k:12s} {tv:9.4f} {mv:9.4f} {'OK' if ok_pt else 'DIFF':>4s}  "
          f"[{tlo:8.4f},{thi:8.4f}] [{ci_i[0]:8.4f},{ci_i[1]:8.4f}] {'OK' if ok_ci else 'DIFF':>4s}")

json.dump(
    {"point": {k: round(v, 6) for k, v in pt.items()},
     "ci_integers": {k: [round(a, 6), round(b, 6)] for k, (a, b) in results["integers"][0].items()},
     "ci_choice": {k: [round(a, 6), round(b, 6)] for k, (a, b) in results["choice"][0].items()},
     "usable": results["integers"][1],
     "p_gt": results["integers"][2],
     "n": n, "n_speakers": len(uspk), "ties_transcript": tied},
    open("reports/part16/verify/M10_verify.json", "w"), indent=1)
print("\nwrote M10_verify.json")
