# Independent verification of PART 16 / M3.
# Written from scratch. Does NOT read, import or exec the primary job's script.
# Two independent AUC routes, neither of them a rank-sum one-liner:
#   route A: exact Mann-Whitney U, counting concordant pairs, ties weighted 0.5
#   route B: trapezoid over a hand-built ROC curve (own threshold sweep)
# sklearn is used ONLY as a third cross-check at the very end.

import csv, json, os, hashlib
import numpy as np

SRC = {
    "pitt":     "<local data dir>/Desktop/release/omni_final/omni_pitt_zeroshot_scores.csv",
    "pcgita":   "<local data dir>/Desktop/release/omni_final/omni_pcgita_zeroshot_scores.csv",
    "neurovoz": "<local data dir>/Desktop/release/omni_final/omni_neurovoz_zeroshot_scores.csv",
}
MIRROR = {
    k: v.replace("<local data dir>/Desktop/release/omni_final",
                 "<local data dir>/paper1_local_runs/omni_final")
    for k, v in SRC.items()
}
ORDER = ["pitt", "pcgita", "neurovoz"]


# ---------- AUC route A: exact concordant-pair count, ties = 0.5 ----------
def auc_pairs(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # chunk so the outer product never blows memory
    wins = 0.0
    ties = 0.0
    step = max(1, int(2e7 // max(1, neg.size)))
    for i in range(0, pos.size, step):
        blk = pos[i:i + step][:, None]          # (b,1)
        d = blk - neg[None, :]                  # (b, n_neg)
        wins += float((d > 0).sum())
        ties += float((d == 0).sum())
    return (wins + 0.5 * ties) / (pos.size * neg.size)


# ---------- AUC route B: trapezoid over a hand-built ROC ----------
def auc_trapz(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ys = y[o]; ss = s[o]
    tp = 0; fp = 0
    xs = [0.0]; ysr = [0.0]
    i = 0; n = ss.size
    while i < n:                                 # advance one whole tie-group at a time
        j = i
        while j < n and ss[j] == ss[i]:
            j += 1
        tp += int((ys[i:j] == 1).sum())
        fp += int((ys[i:j] == 0).sum())
        xs.append(fp / N); ysr.append(tp / P)
        i = j
    xs = np.asarray(xs); ysr = np.asarray(ysr)
    return float(np.sum((xs[1:] - xs[:-1]) * (ysr[1:] + ysr[:-1]) / 2.0))


def load(path):
    spk = []; lab = []; p = []; clip = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            clip.append(row["clip"]); spk.append(row["speaker"])
            lab.append(int(row["label"])); p.append(float(row["p_yes"]))
    return np.array(clip), np.array(spk), np.array(lab), np.array(p, dtype=float)


def sha_first_mb(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(1024 * 1024)).hexdigest()


def f4(x):
    return f"{x:.4f}"


out = {}
print("=" * 78)
for ds in ORDER:
    path = SRC[ds]
    clip, spk, lab, p = load(path)
    n = len(clip)

    # speaker aggregation: mean p_yes, label must be constant within speaker
    order = []
    seen = {}
    for s in spk:
        if s not in seen:
            seen[s] = len(order); order.append(s)
    bad = []
    s_lab = np.zeros(len(order), dtype=int)
    s_mean = np.zeros(len(order), dtype=float)
    s_n = np.zeros(len(order), dtype=int)
    idx_of = {s: i for i, s in enumerate(order)}
    groups = [[] for _ in order]
    for i, s in enumerate(spk):
        groups[idx_of[s]].append(i)
    for gi, g in enumerate(groups):
        ls = set(lab[g].tolist())
        if len(ls) != 1:
            bad.append(order[gi])
        s_lab[gi] = lab[g][0]
        s_mean[gi] = float(np.mean(p[g]))
        s_n[gi] = len(g)

    clip_A = auc_pairs(lab, p);        clip_B = auc_trapz(lab, p)
    spk_A  = auc_pairs(s_lab, s_mean); spk_B  = auc_trapz(s_lab, s_mean)

    # mirror identity
    mp = MIRROR[ds]
    mirror = {"exists": os.path.exists(mp)}
    if mirror["exists"]:
        _, _, ml, mpv = load(mp)
        mirror["n"] = len(mpv)
        mirror["max_abs_diff_p_yes"] = float(np.max(np.abs(mpv - p))) if len(mpv) == n else None
        mirror["sha_match"] = (sha_first_mb(mp) == sha_first_mb(path))

    # ---------------- paired bootstrap over speakers ----------------
    # One speaker draw per replicate; clip AUC and speaker AUC both recomputed
    # inside that draw, so the difference is paired.
    def boot(route, draw):
        rng = np.random.default_rng(0)
        K = len(order)
        cl, sp, df = [], [], []
        usable = 0
        for _ in range(2000):
            if draw == "integers":
                pick = rng.integers(0, K, size=K)
            else:
                pick = rng.choice(K, size=K, replace=True)
            bl = s_lab[pick]
            if bl.min() == bl.max():
                continue
            ci = np.concatenate([groups[k] for k in pick])
            a_clip = route(lab[ci], p[ci])
            a_spk = route(bl, s_mean[pick])
            if np.isnan(a_clip) or np.isnan(a_spk):
                continue
            usable += 1
            cl.append(a_clip); sp.append(a_spk); df.append(a_clip - a_spk)
        q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        return q(cl), q(sp), q(df), usable

    res = {}
    for draw in ("integers", "choice"):
        res[draw] = boot(auc_pairs, draw)

    out[ds] = dict(n=n, n_spk=len(order), bad=bad,
                   clip_A=clip_A, clip_B=clip_B, spk_A=spk_A, spk_B=spk_B,
                   diff=clip_A - spk_A, mirror=mirror, boot=res,
                   pos_spk=int((s_lab == 1).sum()), pos_clip=int((lab == 1).sum()),
                   s_lab=s_lab, s_mean=s_mean, s_n=s_n, order=order)

    print(f"[{ds}] n_clips={n} n_speakers={len(order)} "
          f"pos_clips={int((lab==1).sum())} pos_speakers={int((s_lab==1).sum())} "
          f"inconsistent_label_speakers={bad}")
    print(f"   CLIP    AUC  pairs={clip_A:.10f}  trapz={clip_B:.10f}  -> {f4(clip_A)}")
    print(f"   SPEAKER AUC  pairs={spk_A:.10f}  trapz={spk_B:.10f}  -> {f4(spk_A)}")
    print(f"   DIFF   clip-speaker = {clip_A-spk_A:.10f} -> {f4(clip_A-spk_A)}")
    for draw in ("integers", "choice"):
        c, s_, d, u = res[draw]
        print(f"   boot[{draw:8s}] usable={u}/2000  clipCI=[{f4(c[0])},{f4(c[1])}]  "
              f"spkCI=[{f4(s_[0])},{f4(s_[1])}]  diffCI=[{f4(d[0])},{f4(d[1])}]")
    print(f"   mirror: {mirror}")
    print("-" * 78)

# sklearn third cross-check
try:
    from sklearn.metrics import roc_auc_score
    print("sklearn cross-check (3rd route):")
    for ds in ORDER:
        _, _, lab, p = load(SRC[ds])
        o = out[ds]
        print(f"   {ds:9s} clip={roc_auc_score(lab,p):.10f}  "
              f"speaker={roc_auc_score(o['s_lab'],o['s_mean']):.10f}")
except Exception as e:
    print("sklearn unavailable:", e)

# check their speaker csv against mine
print("=" * 78)
theirs = {}
with open("scores/part16/M/M3_speaker_level.csv",
          newline="") as fh:
    for r in csv.DictReader(fh):
        theirs[(r["dataset"], r["spk"])] = (int(r["label"]), int(r["n_clips"]),
                                            float(r["mean_p_yes"]))
print(f"their speaker csv rows = {len(theirs)}")
for ds in ORDER:
    o = out[ds]
    mism = 0; missing = 0; maxd = 0.0
    for i, s in enumerate(o["order"]):
        k = (ds, s)
        if k not in theirs:
            missing += 1; continue
        tl, tn, tm = theirs[k]
        if tl != o["s_lab"][i] or tn != o["s_n"][i]:
            mism += 1
        maxd = max(maxd, abs(tm - o["s_mean"][i]))
    print(f"   {ds:9s} mine={len(o['order'])} missing_from_theirs={missing} "
          f"label/n_clips_mismatch={mism} max_abs_diff_mean_p_yes={maxd:.3e} "
          f"pos_speakers={o['pos_spk']}")
