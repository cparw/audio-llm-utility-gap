"""PART24 item 1 independent verifier. Written separately from build_ft24.py and imports none of it.

Starts from the raw pulled pod files (part24/pull/<pod>/out/FT_whole_s{s}_fold{k}_oof.csv), not from the built csvs.
  1. Rebuilds each set from the raw pod rows and checks the built per-clip csv matches it cell for cell.
  2. Recomputes the mass rule from the raw seed 0 rows.
  3. AUC by explicit Mann-Whitney over every positive x negative pair (ties 0.5), cross-checked with sklearn roc_auc_score.
  4. Bootstrap re-implemented: speakers in sorted order, fresh default_rng(0) per cell, rng.choice(N, size=N, replace=True),
     single-class draws skipped, 2.5 / 97.5 percentiles; paired uses one draw for both terms.
  5. Compares every row of FT24_values.tsv (value, lo, hi, n) at 4 decimals and prints VERIFIED / MISMATCH per row.
usage: /usr/local/bin/python3 verify_ft24.py
"""
import os, csv, json, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score

R = os.environ.get("P24ROOT", "<local data dir>/part24")
ZS = "<local data dir>/part23/pull/p23-a-probes/p23a/out/A2_edaic_whole_zeroshot_perclip.csv"


def table(p, sep=","):
    with open(p, newline="") as fh: return list(csv.DictReader(fh, delimiter=sep))


def mw(y, s):
    y = np.asarray(y, int); s = np.asarray(s, np.float64)
    P, N = s[y == 1], s[y == 0]
    return float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))


def ci(y, a, b=None):
    y = np.asarray(y, int); a = np.asarray(a, np.float64); b = None if b is None else np.asarray(b, np.float64)
    g = np.random.default_rng(0); keep = []
    for _ in range(2000):
        d = g.choice(len(y), size=len(y), replace=True)
        if y[d].sum() in (0, len(d)): continue
        keep.append(mw(y[d], a[d]) - (mw(y[d], b[d]) if b is not None else 0.0))
    return np.percentile(keep, 2.5), np.percentile(keep, 97.5)


def raw(pod, k, s):
    p = f"{R}/pull/{pod}/out/FT_whole_s{s}_fold{k}_oof.csv"
    return table(p) if os.path.exists(p) else None


zs = {}
for r in table(ZS):
    zs[str(r["pid"])] = (int(r["label"]), float(r["p_yes_whole"].replace("np.float64(", "").rstrip(")")))
claimed = {r["id"]: r for r in table(f"{R}/FT/FT24_values.tsv", "\t")}
mine = {}

S0 = {k: raw(f"p24-ft{k}", k, 0) for k in range(5)}
assert all(S0[k] is not None for k in range(5))
med0 = {k: float(np.median(np.array([float(r["answer_mass"]) for r in S0[k]]))) for k in range(5)}
trig = [k for k in range(5) if med0[k] < 0.5]
print("seed 0 median answer_mass by fold:", {k: round(v, 4) for k, v in med0.items()}, "-> mass rule folds:", trig)

sets = {"seed0": ("FT24_whole_seed0_perclip.csv", {k: S0[k] for k in range(5)})}
if trig:
    rule = {}
    for k in range(5):
        if k in trig:
            rule[k] = raw(f"p24-ft{k}", k, 1) or raw(f"p24s1-ft{k}", k, 1)
        else:
            rule[k] = S0[k]
    sets["seedrule"] = ("FT24_whole_seedrule_perclip.csv", rule)
pre = {k: raw(f"p24s1-ft{k}", k, 1) for k in range(5)}
if all(pre[k] is not None for k in range(5)):
    sets["extra_seed1all"] = ("FT24_whole_seed1all_perclip.csv", pre)

bad = 0
for tag, (fn, parts) in sets.items():
    rows = [r for k in range(5) for r in parts[k]]
    rows.sort(key=lambda r: str(r["speaker"]))
    built = table(f"{R}/FT/{fn}")
    side = json.load(open(f"{R}/FT/{fn[:-4]}.sidecar.json"))
    cols = ["pid", "label", "fold", "seed", "p_yes", "answer_mass", "audio_tok"]
    same = len(built) == len(rows) and all(b[c] == r[c] for b, r in zip(built, rows) for c in cols)
    ok_side = side["n"] == 275 and side["n_speakers"] == 275
    print(f"{fn}: rows {len(built)} match raw pod rows cell for cell: {same}; sidecar n/n_speakers ok: {ok_side}")
    bad += (not same) + (not ok_side)
    y = np.array([int(r["label"]) for r in rows]); p = np.array([float(r["p_yes"]) for r in rows])
    assert all(zs[r["pid"]][0] == int(r["label"]) for r in rows)
    z = np.array([zs[r["pid"]][1] for r in rows])
    a = mw(y, p); assert abs(a - roc_auc_score(y, p)) < 1e-12
    mine[f"{tag}_pooled_oof_auc"] = (a, *ci(y, p), len(y))
    mine[f"{tag}_ft_minus_zs_whole_paired"] = (a - mw(y, z), *ci(y, p, z), len(y))
    if tag == "seed0": mine["ref_zs_whole_auc"] = (mw(y, z), *ci(y, z), len(y))
    for k in range(5):
        fr = [r for r in rows if int(r["fold"]) == k]
        yk = np.array([int(r["label"]) for r in fr]); pk = np.array([float(r["p_yes"]) for r in fr])
        mine[f"{tag}_fold{k}_auc"] = (mw(yk, pk), *ci(yk, pk), len(yk))
        mine[f"{tag}_fold{k}_median_answer_mass"] = (float(np.median([float(r["answer_mass"]) for r in fr])), None, None, len(yk))

r4 = lambda x: None if x in (None, "") else round(float(x), 4)
for i, c in claimed.items():
    if i not in mine:
        print(f"UNCHECKED {i}"); bad += 1; continue
    v, lo, hi, n = mine[i]
    ok = r4(c["value"]) == r4(v) and r4(c["lo"]) == r4(lo) and r4(c["hi"]) == r4(hi) and int(c["n"]) == n
    bad += not ok
    print(f"{'VERIFIED' if ok else 'MISMATCH'} {i}: claimed {r4(c['value'])} [{r4(c['lo'])}, {r4(c['hi'])}] n={c['n']} | recomputed {r4(v)} [{r4(lo)}, {r4(hi)}] n={n}")
for i in mine:
    if i not in claimed: print(f"MISSING_FROM_TABLE {i}"); bad += 1
print("RESULT", "ALL VERIFIED" if bad == 0 else f"{bad} problems")
sys.exit(1 if bad else 0)
