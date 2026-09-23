#!/usr/bin/env python3
"""Append FINAL rows to rows/POD4.tsv only when compute and verified agree to 4 dp
(value, lo, hi). Input: a JSON list of dicts with keys
id, what, comp (value, lo, hi), ver (value, lo, hi), n, n_spk, file, kind ('auc'|'diff')."""
import json, sys, csv, os
T = "reports/part20/rows/POD4.tsv"
def r4(x): return f"{float(x):.4f}"
def vs(kind, v, lo, hi):
    ref = 0.5 if kind == "auc" else 0.0
    tag = "0.5" if kind == "auc" else "0 (diff)"
    if lo > ref: return f"above {tag}"
    if hi < ref: return f"below {tag}"
    return f"interval includes {tag}"
rows = json.load(open(sys.argv[1]))
done = set()
if os.path.exists(T):
    done = {l.split("\t")[0] for l in open(T).read().splitlines()[1:]}
for r in rows:
    c, v = r["comp"], r["ver"]
    ok = all(r4(c[k]) == r4(v[k]) for k in ("value", "lo", "hi")) and not r.get("checks")
    line = [r["id"], r["what"], r4(c["value"]), r4(v["value"]), r4(v["lo"]), r4(v["hi"]),
            str(r["n"]), str(r["n_spk"]), r["file"], f"{v['value']:.2f} ({v['lo']:.2f} to {v['hi']:.2f})",
            vs(r["kind"], v["value"], v["lo"], v["hi"]), "YES" if ok else "NO"]
    if r.get("checks"): print("CHECK FAILED !!! | " + r["id"] + " | " + "; ".join(r["checks"]))
    if ok:
        if r["id"] in done:
            print("already final:", r["id"]); continue
        open(T, "a").write("\t".join(line) + "\n"); done.add(r["id"])
        print("FINAL | " + " | ".join(line))
    else:
        print("DISAGREE !!! | " + r["id"] + f" | computed {r4(c['value'])} [{r4(c['lo'])}, {r4(c['hi'])}]"
              + f" vs verified {r4(v['value'])} [{r4(v['lo'])}, {r4(v['hi'])}]")
