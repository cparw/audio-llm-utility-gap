"""PART20 POD6 shared helpers: rank AUC, speaker bootstrap, sha256 of first 1 MB, fold rule, paste lines, rows."""
import hashlib, json, os, numpy as np
from scipy.stats import rankdata

MAC_OUT = "scores/part20/POD6"
ROWS = "reports/part20/rows/POD6_provisional.tsv"

def auc_rank(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    r = rankdata(s)  # ties averaged
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def spk_index(spk):
    u = np.unique(spk); return u, {p: np.where(spk == p)[0] for p in u}

def boot_auc(y, s, spk, nb=2000):
    """fresh default_rng(0) per cell; speakers with replacement; draws with one class are skipped and counted."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); spk = np.asarray(spk).astype(str)
    rng = np.random.default_rng(0); u, idx = spk_index(spk); v = []
    for _ in range(nb):
        ii = np.concatenate([idx[p] for p in rng.choice(u, size=len(u), replace=True)])
        if len(np.unique(y[ii])) < 2: continue
        v.append(auc_rank(y[ii], s[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))

def sha1mb(p):
    h = hashlib.sha256()
    with open(p, "rb") as f: h.update(f.read(1 << 20))
    return h.hexdigest()

def pod_groupkfold(groups, n_splits=5):
    """sklearn GroupKFold(5) (no shuffle) with numpy.argsort(counts, kind='stable') reversed: the POD tie rule."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(counts, kind="stable")[::-1]
    fold_of_group = np.zeros(len(uniq), dtype=int); load = np.zeros(n_splits)
    for gi in order:
        f = int(np.argmin(load)); fold_of_group[gi] = f; load[f] += counts[gi]
    fold = fold_of_group[inv]
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)], fold

def vs_half(lo, hi):
    return "above 0.5" if lo > 0.5 else ("below 0.5" if hi < 0.5 else "interval includes 0.5")

def paste(rid, what, v, lo, hi, n, nspk, f, diff=False):
    if diff:
        tag = "interval excludes zero" if (lo > 0 or hi < 0) else "interval includes zero"
    else:
        tag = vs_half(lo, hi)
    line = (f"PASTE {rid} | {what} | {v:.4f} [{lo:.4f}, {hi:.4f}] | n={n} n_speakers={nspk} | {f} | "
            f"two-dp {v:.2f} [{lo:.2f}, {hi:.2f}] {tag}")
    print(line, flush=True)
    new = not os.path.exists(ROWS)
    with open(ROWS, "a") as fh:
        if new: fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
        fh.write(f"{rid}\t{what}\t{v:.4f}\t{lo:.4f}\t{hi:.4f}\t{n}\t{nspk}\t{f}\t{v:.2f} [{lo:.2f}, {hi:.2f}]\t{tag}\n")
    return line

def sidecar(path, **kw):
    srcs = kw.get("sources", [])
    kw["sources_sha256_first_1MB"] = {s: sha1mb(s) for s in srcs if os.path.isfile(s)}
    json.dump(kw, open(path, "w"), indent=1, default=str)
