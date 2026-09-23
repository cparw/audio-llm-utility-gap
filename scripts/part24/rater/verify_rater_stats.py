#!/usr/local/bin/python3
"""
PART 24 item 3, independent verifier for rater_stats.py.

Recomputes every row of rater_stats.csv from the raw sheet, key and sentiment
csv with its own code path: csv module parsing (no pandas), its own cell
mapping, Cohen's kappa by the textbook formula (po - pe) / (1 - pe) (no
sklearn), its own bootstrap loop under the same rule (2000 draws, fresh numpy
default_rng(0) per cell, rng.choice(N, size=N, replace=True), 2.5/97.5
percentiles; speakers = sorted unique pid). The scorer direction is rebuilt
from p_pos_heard > p_neg_heard in part9, not read from the key.

Usage
  verify_rater_stats.py --stats rater_stats.csv [--sidecar rater_stats.json]
Exit 0 only when every value matches within 1e-6.
"""
import argparse
import csv
import json
import sys

import numpy as np

TOL = 1e-6


def norm(c):
    return c.strip().lower().replace(" ", "").replace("_", "")


def cell(v):
    s = (v or "").strip().lower().rstrip(".!")
    table = {"yes": "Y", "y": "Y", "1": "Y", "true": "Y", "t": "Y",
             "no": "N", "n": "N", "0": "N", "false": "N", "f": "N",
             "positive": "+", "pos": "+", "negative": "-", "neg": "-",
             "conflict": "C", "away": "C", "agreement": "A", "toward": "A", "towards": "A"}
    if s in ("", "nan", "none"):
        return None
    return table.get(s, "?")


def text_dir(tok, label, reading):
    if tok == "+":
        return "positive"
    if tok == "-":
        return "negative"
    if tok == "C":
        tok, reading = "Y", "away"
    if tok == "A":
        tok, reading = "N", "away"
    if tok not in ("Y", "N"):
        return None
    if reading == "depressed":
        return {"Y": "negative", "N": "positive"}[tok]
    goes_with = "negative" if label == 1 else "positive"
    points_away = "positive" if label == 1 else "negative"
    return points_away if tok == "Y" else goes_with


def kap(a, b):
    a = list(a)
    b = list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if abs(1 - pe) < 1e-15:
        return float("nan")
    return (po - pe) / (1 - pe)


def boot(fn, n, pids=None):
    rng = np.random.default_rng(0)
    out, bad = [], 0
    if pids is not None:
        sp = sorted(set(pids))
        where = {p: [i for i, q in enumerate(pids) if q == p] for p in sp}
    for _ in range(2000):
        if pids is None:
            idx = rng.choice(n, size=n, replace=True)
        else:
            pick = rng.choice(len(sp), size=len(sp), replace=True)
            idx = np.array([i for j in pick for i in where[sp[j]]])
        v = fn(idx)
        if v != v:
            bad += 1
        else:
            out.append(v)
    if not out:
        return float("nan"), float("nan"), 0, bad
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--sidecar")
    a = ap.parse_args()
    side_path = a.sidecar or a.stats[:-4] + ".json"
    side = json.load(open(side_path))
    paths = list(side["inputs"].keys())
    sheet_p, key_p, sent_p = paths[0], paths[1], paths[2]
    reading = side["reading"]

    with open(sent_p, newline="") as fh:
        sent = {}
        for r in csv.DictReader(fh):
            sent.setdefault(r["seg_uid"], r)
    with open(key_p, newline="") as fh:
        key = {r["id"]: r for r in csv.DictReader(fh)}
    with open(sheet_p, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        cols = {norm(c): i for i, c in enumerate(head)}
        rows = list(rd)
    layout = "local" if "label" in cols else "live"
    tcol = cols.get("transcript", cols.get("text"))

    items = []
    for r in rows:
        k = key[r[cols["id"]].strip()]
        s = sent[k["source_id"]]
        assert r[tcol].strip() == s["heard_text"].strip(), "text mismatch " + k["id"]
        lab = int(k["label"])
        pneg, ppos = float(s["p_neg_heard"]), float(s["p_pos_heard"])
        items.append({"id": k["id"], "pid": k["pid"], "arm": k["automatic_decision"], "label": lab,
                      "roberta_heard": "positive" if ppos > pneg else "negative",
                      "lexicon_arm": s["lex_direction"],
                      "t1": cell(r[cols["rater1"]]), "t2": cell(r[cols["rater2"]])})
    items.sort(key=lambda d: d["id"])
    # arm must follow the lexicon rule
    for d in items:
        conflict = (d["label"] == 1 and d["lexicon_arm"] == "positive") or (d["label"] == 0 and d["lexicon_arm"] == "negative")
        assert (d["arm"] == "conflict") == conflict, "arm rule broken " + d["id"]

    # independent reading check for auto
    toks = {d["t1"] for d in items} | {d["t2"] for d in items}
    toks.discard(None)
    auto = "valence" if toks and toks <= {"+", "-"} else ("away" if layout == "local" else "depressed")
    note = "" if (side["reading_reason"].startswith("set by") or auto == reading) else " (AUTO WOULD GIVE %s)" % auto
    rmode = "depressed" if reading == "valence" else reading
    for d in items:
        for t in ("1", "2"):
            d["d" + t] = text_dir(d["t" + t], d["label"], rmode)
            g = "negative" if d["label"] == 1 else "positive"
            d["w" + t] = None if d["d" + t] is None else ("N" if d["d" + t] == g else "Y")
    inter = "w" if reading == "away" else "d"

    want = {}
    for sub in ("all80", "conflict40", "agreement40"):
        S = [d for d in items if sub == "all80" or d["arm"] == sub[:-2]]
        for t in ("1", "2"):
            T = [d for d in S if d["d" + t] is not None]
            pids = [d["pid"] for d in T]
            for sc in ("roberta_heard", "lexicon_arm"):
                m = np.array([1.0 if d["d" + t] == d[sc] else 0.0 for d in T])
                x = [d["d" + t] for d in T]
                y = [d[sc] for d in T]
                want[(sub, sc, "rater%s_agree_scorer" % t)] = (
                    m.mean(), int(m.sum()), len(T), boot(lambda i, m=m: m[i].mean(), len(T)),
                    boot(lambda i, m=m: m[i].mean(), len(T), pids))
                want[(sub, sc, "rater%s_kappa_scorer" % t)] = (
                    kap(x, y), None, len(T),
                    boot(lambda i, x=x, y=y: kap([x[j] for j in i], [y[j] for j in i]), len(T)),
                    boot(lambda i, x=x, y=y: kap([x[j] for j in i], [y[j] for j in i]), len(T), pids))
        B = [d for d in S if d["d1"] is not None and d["d2"] is not None]
        pids = [d["pid"] for d in B]
        x = [d[inter + "1"] for d in B]
        y = [d[inter + "2"] for d in B]
        eq = np.array([1.0 if p == q else 0.0 for p, q in zip(x, y)])
        want[(sub, "none", "interrater_agree")] = (eq.mean(), int(eq.sum()), len(B),
                                                   boot(lambda i, e=eq: e[i].mean(), len(B)),
                                                   boot(lambda i, e=eq: e[i].mean(), len(B), pids))
        want[(sub, "none", "interrater_kappa")] = (kap(x, y), None, len(B),
                                                   boot(lambda i: kap([x[j] for j in i], [y[j] for j in i]), len(B)),
                                                   boot(lambda i: kap([x[j] for j in i], [y[j] for j in i]), len(B), pids))

    def same(u, v):
        if u != u and v != v:
            return True
        return abs(u - v) <= TOL

    with open(a.stats, newline="") as fh:
        got = list(csv.DictReader(fh))
    fails, n = 0, 0
    print("VERIFY %s  reading %s%s" % (a.stats, reading, note))
    for r in got:
        kk = (r["subset"], r["scorer"], r["metric"])
        v, k, nn, bi, bs = want.pop(kk)
        checks = [("value", float(r["value"] or "nan"), v),
                  ("ci_item_lo", float(r["ci_item_lo"] or "nan"), bi[0]), ("ci_item_hi", float(r["ci_item_hi"] or "nan"), bi[1]),
                  ("ci_spk_lo", float(r["ci_spk_lo"] or "nan"), bs[0]), ("ci_spk_hi", float(r["ci_spk_hi"] or "nan"), bs[1])]
        ok = all(same(p, q) for _, p, q in checks) and int(r["n"]) == nn and \
            int(r["boot_item_used"]) == bi[2] and int(r["boot_item_undefined"]) == bi[3] and \
            int(r["boot_spk_used"]) == bs[2] and int(r["boot_spk_undefined"]) == bs[3] and \
            (k is None or int(float(r["k"])) == k)
        n += 1
        if not ok:
            fails += 1
            print("  MISMATCH %s: file %s  recomputed value %.6f k %s n %d item %s spk %s"
                  % (kk, {c: r[c] for c in ("value", "k", "n", "ci_item_lo", "ci_item_hi", "ci_spk_lo", "ci_spk_hi")},
                     v, k, nn, bi, bs))
    if want:
        fails += len(want)
        print("  rows missing from the file: %s" % list(want))
    if note:
        fails += 1
        print("  reading disagrees with the verifier's own auto rule")
    print("VERIFY %s: %d rows checked, %d failed" % ("PASS" if fails == 0 else "FAIL", n, fails))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
