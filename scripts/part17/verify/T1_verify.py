#!/usr/local/bin/python3
"""PART 17 T1 independent verifier.

Written from scratch. Reads only the ORIGINAL source files (manifest, score
files, Pitt manifest). Does not import, exec or read the main job's script
or its intermediate csv/json outputs.

AUC: explicit Mann-Whitney U over every positive x negative pair (ties 0.5),
cross-checked by trapezoid integration of the full empirical ROC.
Bootstrap: 2000 draws, fresh numpy.random.default_rng(0) per cell, speakers
drawn with replacement from the sorted unique speaker list, percentile
2.5/97.5. A drawn speaker contributes all of its (cell) items once per draw.
Paired cells use one speaker draw per replicate for both arms.
"""
import csv, os, sys, json, hashlib, datetime
from collections import defaultdict, Counter
import numpy as np

BASE = "<local data dir>/release/edaic_rerun"
OUT = os.path.join(BASE, "part17", "verify")
MANIFEST = os.path.join(BASE, "part14_manifest_new.csv")
PITT = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
STREAMS = [
    # key, model, stream, path, id-extractor kind
    ("q2a", "Qwen2-Audio", "audio", os.path.join(BASE, "part14/p14_q2a_zeroshot_scores.csv"), "clip"),
    ("o25", "Qwen2.5-Omni", "audio", os.path.join(BASE, "part14/p14_o25_zeroshot_scores.csv"), "clip"),
    ("q3o", "Qwen3-Omni-30B-A3B", "audio", os.path.join(BASE, "part14/p14_q3o_zeroshot_scores.csv"), "clip"),
    ("af2", "Audio Flamingo 2", "audio", os.path.join(BASE, "part14/p14_af2.csv"), "clip_path"),
    ("af3", "Audio Flamingo 3", "audio", os.path.join(BASE, "part14/p14_af3.csv"), "clip_path"),
    ("kimi", "Kimi-Audio", "audio", os.path.join(BASE, "part14/p14_kimi_zeroshot_scores.csv"), "clip"),
    ("o25t", "Qwen2.5-Omni", "transcript", os.path.join(BASE, "part16/pod_sync/p14_o25_text.csv"), "id"),
    ("q2at", "Qwen2-Audio", "transcript", os.path.join(BASE, "part16/pod_sync/p14_q2a_text.csv"), "id"),
]
POD1 = {
    "o25": os.path.join(BASE, "part16/POD1/p14_o25_1a_joined.csv"),
    "q2a": os.path.join(BASE, "part16/POD1/p14_q2a_1a_joined.csv"),
}
# claimed by the main job (to compare against), all 4 dp
CLAIM = {
    "q2a": dict(old=(0.9478, 0.9128, 0.9762), new=(0.9608, 0.9351, 0.9814), conf=(0.2280, 0.1816, 0.2782), pair=(-0.7328, -0.7780, -0.6837), paper="0.95"),
    "o25": dict(old=(0.9482, 0.9039, 0.9828), new=(0.9647, 0.9376, 0.9857), conf=(0.2218, 0.1829, 0.2632), pair=(-0.7429, -0.7869, -0.6944), paper="0.95"),
    "q3o": dict(old=(0.9638, 0.9336, 0.9864), new=(0.9727, 0.9507, 0.9904), conf=(0.3008, 0.2454, 0.3641), pair=(-0.6719, -0.7266, -0.6105), paper="0.96"),
    "af2": dict(old=(0.6083, 0.5126, 0.6997), new=(0.6297, 0.5451, 0.7087), conf=(0.5286, 0.4512, 0.6015), pair=(-0.1011, -0.1647, -0.0384), paper="0.61"),
    "af3": dict(old=(0.8095, 0.7447, 0.8696), new=(0.8425, 0.7963, 0.8854), conf=(0.4235, 0.3606, 0.4880), pair=(-0.4190, -0.4878, -0.3461), paper="0.81"),
    "kimi": dict(old=(0.8744, 0.8172, 0.9223), new=(0.8936, 0.8479, 0.9329), conf=(0.5258, 0.4501, 0.5977), pair=(-0.3678, -0.4394, -0.3020), paper="0.87"),
    "o25t": dict(old=(0.9596, 0.9306, 0.9823), new=(0.9620, 0.9345, 0.9822), conf=(0.1498, 0.1131, 0.1887), pair=(-0.8123, -0.8536, -0.7692), paper=None),
    "q2at": dict(old=(0.9081, 0.8515, 0.9547), new=(0.9199, 0.8791, 0.9560), conf=(0.2359, 0.1851, 0.2886), pair=(-0.6840, -0.7516, -0.6161), paper=None),
}
NDRAW = 2000
SEED = 0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


# ---------------- AUC implementations ----------------
def pair_matrix(pos, neg):
    """C[i,j] = 1 if pos_i > neg_j, 0.5 if equal, 0 otherwise."""
    P = np.asarray(pos, float)[:, None]
    N = np.asarray(neg, float)[None, :]
    return (P > N).astype(float) + 0.5 * (P == N).astype(float)


def auc_mwu(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    C = pair_matrix(s[y == 1], s[y == 0])
    return C.sum() / C.size


def auc_trapz(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    npos = (y == 1).sum(); nneg = (y == 0).sum()
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append(((s >= t) & (y == 1)).sum() / npos)
        fpr.append(((s >= t) & (y == 0)).sum() / nneg)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


class Cell:
    """Precomputed pair matrix and speaker index for weighted-bootstrap MWU."""

    def __init__(self, scores, labels, spk, spk_order):
        s = np.asarray(scores, float); y = np.asarray(labels, int); sp = np.asarray(spk)
        self.n = len(s); self.npos = int((y == 1).sum()); self.nneg = int((y == 0).sum())
        self.nspk = len(set(sp.tolist()))
        pos = y == 1; neg = y == 0
        self.C = pair_matrix(s[pos], s[neg])
        idx = {k: i for i, k in enumerate(spk_order)}
        self.pos_spk = np.array([idx[k] for k in sp[pos]])
        self.neg_spk = np.array([idx[k] for k in sp[neg]])
        self.point = self.C.sum() / self.C.size
        self.point_trapz = auc_trapz(s, y)

    def weighted(self, counts):
        wp = counts[self.pos_spk].astype(float); wn = counts[self.neg_spk].astype(float)
        Wp = wp.sum(); Wn = wn.sum()
        if Wp == 0 or Wn == 0:
            return np.nan
        # elementwise (no BLAS) to avoid spurious Accelerate matmul FP warnings
        return float((wp[:, None] * self.C * wn[None, :]).sum()) / (Wp * Wn)


def draws(K, mode):
    rng = np.random.default_rng(SEED)
    if mode == "loop_integers":
        for _ in range(NDRAW):
            yield rng.integers(0, K, K)
    elif mode == "loop_choice":
        for _ in range(NDRAW):
            yield rng.choice(K, K, replace=True)
    elif mode == "block_integers":
        M = rng.integers(0, K, (NDRAW, K))
        for row in M:
            yield row


def boot_single(cell, spk_order, mode="loop_integers"):
    K = len(spk_order); vals = []
    for d in draws(K, mode):
        v = cell.weighted(np.bincount(d, minlength=K))
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def boot_paired(cellA, cellB, spk_order, mode="loop_integers"):
    """A minus B with one speaker draw per replicate."""
    K = len(spk_order); vals = []
    for d in draws(K, mode):
        c = np.bincount(d, minlength=K)
        a = cellA.weighted(c); b = cellB.weighted(c)
        if np.isnan(a) or np.isnan(b):
            continue
        vals.append(a - b)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def f4(x):
    return f"{x:.4f}"


def main():
    log = []
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)

    # ---------------- manifest ----------------
    mcols, man = read_csv(MANIFEST)
    P("manifest rows", len(man), "cols", len(mcols))
    arms = defaultdict(list)
    for i, r in enumerate(man):
        arms[r["set"]].append(i)
    for a in arms:
        rows = [man[i] for i in arms[a]]
        P(f"arm {a}: rows {len(rows)} distinct seg_uid {len(set(r['seg_uid'] for r in rows))} "
          f"distinct (spk,start,end) {len(set((r['speaker_id'], r['start'], r['end']) for r in rows))} "
          f"speakers {len(set(r['speaker_id'] for r in rows))} pos_rows {sum(r['label']=='1' for r in rows)}")
    agr = [man[i] for i in arms["agreement"]]
    seg_count = Counter(r["seg_uid"] for r in agr)
    rep_dist = Counter(seg_count.values())
    P("agreement repeat distribution (times seen -> n segments)", dict(sorted(rep_dist.items())),
      "extra rows", sum((k - 1) * v for k, v in rep_dist.items()))
    # attribute constancy within seg_uid
    attrs = ["speaker_id", "start", "end", "label", "text", "set"]
    byseg = defaultdict(set)
    for r in man:
        byseg[r["seg_uid"]].add(tuple(r[a] for a in attrs))
    P("max attribute variants within one seg_uid (all arms)", max(len(v) for v in byseg.values()))
    # does any seg_uid cross arms?
    seg_arm = defaultdict(set)
    for r in man:
        seg_arm[r["seg_uid"]].add(r["set"])
    P("seg_uids appearing in both arms", sum(len(v) > 1 for v in seg_arm.values()))
    # speaker label constancy
    spk_lab = defaultdict(set)
    for r in man:
        spk_lab[r["speaker_id"]].add(r["label"])
    P("speakers with >1 label", sum(len(v) > 1 for v in spk_lab.values()))
    spk_conf = set(man[i]["speaker_id"] for i in arms["conflict"])
    spk_agr = set(man[i]["speaker_id"] for i in arms["agreement"])
    P("speaker sets equal across arms", spk_conf == spk_agr, len(spk_conf | spk_agr))
    # distinct agreement set: first occurrence in manifest order
    first_idx = []
    seen = set()
    for i in arms["agreement"]:
        u = man[i]["seg_uid"]
        if u not in seen:
            seen.add(u); first_idx.append(i)
    dpos = sum(man[i]["label"] == "1" for i in first_idx)
    P("distinct agreement segments", len(first_idx), "pos", dpos, "neg", len(first_idx) - dpos,
      "speakers", len(set(man[i]["speaker_id"] for i in first_idx)))

    # ---------------- Pitt ----------------
    pcols, pitt = read_csv(PITT)
    for a in sorted(set(r["set"] for r in pitt)):
        rr = [r for r in pitt if r["set"] == a]
        P(f"Pitt arm {a}: rows {len(rr)} distinct segment_path {len(set(r['segment_path'] for r in rr))} "
          f"distinct (spk,start_ms,end_ms) {len(set((r['spk'], r['start_ms'], r['end_ms']) for r in rr))} speakers {len(set(r['spk'] for r in rr))}")

    # ---------------- score files ----------------
    exp_id = [f"{r['set']}_{r['speaker_id']}_{r['seg_uid']}" for r in man]
    exp_lab = [int(r["label"]) for r in man]
    exp_spk = [str(int(r["speaker_id"])) for r in man]
    pyes = {}
    sources = [{"path": MANIFEST, "sha256": sha(MANIFEST), "rows": len(man)},
               {"path": PITT, "sha256": sha(PITT), "rows": len(pitt)}]
    for key, model, stream, path, kind in STREAMS:
        cols, rows = read_csv(path)
        if kind == "clip":
            ids = [r["clip"][:-4] if r["clip"].endswith(".wav") else r["clip"] for r in rows]
            spk = [str(int(r["speaker"])) for r in rows]
        elif kind == "clip_path":
            ids = [os.path.basename(r["clip_path"])[:-4] for r in rows]
            spk = [str(int(r["speaker_id"])) for r in rows]
        else:
            ids = [r["id"] for r in rows]
            spk = [str(int(r["speaker_id"])) for r in rows]
        lab = [int(float(r["label"])) for r in rows]
        p = np.array([float(r["p_yes"]) for r in rows])
        n = len(rows)
        idm = sum(a == b for a, b in zip(ids, exp_id)) if n == len(man) else -1
        lm = sum(a == b for a, b in zip(lab, exp_lab)) if n == len(man) else -1
        sm = sum(a == b for a, b in zip(spk, exp_spk)) if n == len(man) else -1
        nnan = int(np.isnan(p).sum())
        # repeat consistency within agreement arm
        grp = defaultdict(list)
        for i in arms["agreement"]:
            grp[man[i]["seg_uid"]].append(p[i])
        rep_spread = max(max(v) - min(v) for v in grp.values())
        # id-keyed cross-check (unique ids: conflict unique, agreement repeats share p)
        P(f"[{key}] {path}: rows {n} id_match {idm} label_match {lm} speaker_match {sm} nan {nnan} "
          f"max p_yes spread among agreement repeats {rep_spread:.3g} range [{p.min():.4f},{p.max():.4f}]")
        if not (idm == lm == sm == len(man) == n) or nnan:
            P(f"   !!! ALIGNMENT FAILURE for {key}")
        pyes[key] = p
        sources.append({"path": path, "sha256": sha(path), "rows": n})
    # POD1 cross-check
    for key, path in POD1.items():
        cols, rows = read_csv(path)
        tkey = key + "t"
        km = sum(r["key"] == e for r, e in zip(rows, exp_id))
        pt = np.array([float(r["p_yes_text"]) for r in rows]); pa = np.array([float(r["p_yes_audio"]) for r in rows])
        P(f"[POD1 {key}] rows {len(rows)} key_match {km} max|text-pod_sync| {np.abs(pt-pyes[tkey]).max():.3g} "
          f"max|audio-part14| {np.abs(pa-pyes[key]).max():.3g}")
        sources.append({"path": path, "sha256": sha(path), "rows": len(rows)})

    # ---------------- cells ----------------
    conf_idx = arms["conflict"]; agr_idx = arms["agreement"]
    spk_order = sorted(set(exp_spk), key=lambda s: int(s))
    P("speaker order: K", len(spk_order), "first", spk_order[:3], "last", spk_order[-3:],
      "(numeric sort == string sort:", spk_order == sorted(spk_order), ")")
    results = []; perclip = []
    for key, model, stream, path, kind in STREAMS:
        p = pyes[key]
        def mk(idx):
            return Cell(p[idx], [exp_lab[i] for i in idx], [exp_spk[i] for i in idx], spk_order)
        cC = mk(conf_idx); cO = mk(agr_idx); cN = mk(first_idx)
        # independent point-estimate checks
        for nm, c, idx in (("conf", cC, conf_idx), ("old", cO, agr_idx), ("new", cN, first_idx)):
            y = [exp_lab[i] for i in idx]
            a1 = auc_mwu(p[idx], y); a2 = auc_trapz(p[idx], y)
            assert abs(a1 - a2) < 1e-12 and abs(a1 - c.point) < 1e-12, (key, nm, a1, a2)
        out = {"key": key, "model": model, "stream": stream, "source": path}
        for nm, c in (("conflict", cC), ("agreement_old_483", cO), ("agreement_new_distinct", cN)):
            lo, hi, u = boot_single(c, spk_order)
            out[nm] = dict(value=c.point, trapz=c.point_trapz, lo=lo, hi=hi, usable=u, n=c.n, n_pos=c.npos, n_neg=c.nneg, n_spk=c.nspk)
        lo, hi, u = boot_paired(cC, cN, spk_order)
        out["paired_new"] = dict(value=cC.point - cN.point, lo=lo, hi=hi, usable=u, n=cC.n + cN.n)
        lo, hi, u = boot_paired(cC, cO, spk_order)
        out["paired_old"] = dict(value=cC.point - cO.point, lo=lo, hi=hi, usable=u, n=cC.n + cO.n)
        # brute force: physically replicate rows for the first 100 draws of the NEW cell,
        # recompute AUC by trapezoid ROC, compare to the weighted MWU value
        rng = np.random.default_rng(SEED); K = len(spk_order); maxdiff = 0.0
        sp_new = np.array([exp_spk[i] for i in first_idx]); y_new = np.array([exp_lab[i] for i in first_idx]); s_new = p[first_idx]
        for _ in range(100):
            d = rng.integers(0, K, K)
            ss = []; yy = []
            for k in d:
                m = sp_new == spk_order[k]
                ss.extend(s_new[m].tolist()); yy.extend(y_new[m].tolist())
            ss = np.array(ss); yy = np.array(yy)
            if yy.min() == yy.max():
                continue
            maxdiff = max(maxdiff, abs(auc_trapz(ss, yy) - cN.weighted(np.bincount(d, minlength=K))))
        P(f"    [{key}] brute-force replicated-row trapezoid vs weighted MWU, first 100 NEW draws: max |diff| {maxdiff:.2e}")
        assert maxdiff < 1e-12
        # stream-convention sensitivity for the NEW cell only
        alt = {}
        for mode in ("loop_choice", "block_integers"):
            lo, hi, u = boot_single(cN, spk_order, mode)
            alt[mode] = (lo, hi)
        out["new_alt_conventions"] = alt
        results.append(out)
        for i in first_idx:
            perclip.append(dict(stream_key=key, model=model, stream=stream, seg_uid=man[i]["seg_uid"],
                                clip_id=exp_id[i], speaker_id=exp_spk[i], label=exp_lab[i],
                                n_repeats_in_agreement_arm=seg_count[man[i]["seg_uid"]], p_yes=repr(float(p[i]))))
        n_ = out["agreement_new_distinct"]; o_ = out["agreement_old_483"]; c_ = out["conflict"]; pn = out["paired_new"]
        P(f"{model} | {stream} | OLD {f4(o_['value'])} [{f4(o_['lo'])}, {f4(o_['hi'])}] n={o_['n']} | "
          f"NEW {f4(n_['value'])} [{f4(n_['lo'])}, {f4(n_['hi'])}] n={n_['n']} n_spk={n_['n_spk']} pos={n_['n_pos']} neg={n_['n_neg']} usable={n_['usable']} | "
          f"conflict {f4(c_['value'])} [{f4(c_['lo'])}, {f4(c_['hi'])}] | paired NEW {f4(pn['value'])} [{f4(pn['lo'])}, {f4(pn['hi'])}] {pn['usable']}/2000 | "
          f"paired OLD {f4(out['paired_old']['value'])} [{f4(out['paired_old']['lo'])}, {f4(out['paired_old']['hi'])}]")
        P(f"    alt conventions NEW interval: " + "; ".join(f"{k} [{f4(v[0])}, {f4(v[1])}]" for k, v in alt.items()))

    # ---------------- compare with claims ----------------
    P("\nCOMPARISON with main job claims (4 dp)")
    n_agree = n_dis = 0; cmp_rows = []
    for out in results:
        cl = CLAIM[out["key"]]
        for nm, ck in (("agreement_old_483", "old"), ("agreement_new_distinct", "new"), ("conflict", "conf"), ("paired_new", "pair")):
            v = out[nm]; mine = (f4(v["value"]), f4(v["lo"]), f4(v["hi"])); theirs = tuple(f4(x) for x in cl[ck])
            ok = mine == theirs
            n_agree += ok; n_dis += (not ok)
            cmp_rows.append(dict(model=out["model"], stream=out["stream"], quantity=nm, verifier=" ".join(mine), task=" ".join(theirs), agree=ok))
            P(f"  {out['model']:<20} {out['stream']:<10} {nm:<24} verifier {mine} task {theirs} {'AGREE' if ok else 'DISAGREE'}")
        # 2dp change
        o2 = f"{out['agreement_old_483']['value']:.2f}"; n2 = f"{out['agreement_new_distinct']['value']:.2f}"
        paper = cl["paper"]
        P(f"  -> {out['model']} {out['stream']}: paper {paper} old2dp {o2} new2dp {n2} changes {'yes' if n2 != (paper or o2) else 'no'}")
    P(f"\nagree {n_agree} disagree {n_dis}")
    # paper text claims
    above = [(o["model"], f4(o["agreement_new_distinct"]["value"])) for o in results if o["stream"] == "audio"]
    P("paper text check: audio agreement NEW > 0.80 except AF2:", above)
    P("paired NEW intervals exclude zero:", [(o["model"], o["stream"], o["paired_new"]["hi"] < 0) for o in results])

    # ---------------- write outputs (verify dir only) ----------------
    with open(os.path.join(OUT, "T1_verify_perclip_distinct.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perclip[0].keys())); w.writeheader(); w.writerows(perclip)
    with open(os.path.join(OUT, "T1_verify_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys())); w.writeheader(); w.writerows(cmp_rows)
    side = dict(task="PART17 T1 independent verification", created=datetime.datetime.now().isoformat(timespec="seconds"),
                command="/usr/local/bin/python3 " + os.path.abspath(__file__), seed=SEED, draws=NDRAW,
                n=dict(conflict=len(conf_idx), agreement_rows=len(agr_idx), agreement_distinct=len(first_idx)),
                n_speakers=len(spk_order), sources=sources,
                auc="explicit pairwise Mann-Whitney (ties 0.5) + trapezoid ROC cross-check",
                bootstrap="speaker cluster bootstrap, default_rng(0) fresh per cell, rng.integers(0,K,K) per draw over numerically sorted unique speakers, percentile 2.5/97.5",
                results=results, n_agree=n_agree, n_disagree=n_dis)
    with open(os.path.join(OUT, "T1_verify.json"), "w") as f:
        json.dump(side, f, indent=1, default=float)
    with open(os.path.join(OUT, "T1_verify.log"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
