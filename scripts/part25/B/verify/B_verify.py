#!/usr/local/bin/python3
"""PART 25 B independent check. Separate code from scripts/B_holm_r2.py (nothing imported from it).

1. Rebuilds every per-clip frame from the ORIGINAL source files with its own joins (dicts keyed on clip
   basename) and compares clip set, speaker, label and every score column to the B per-clip csv.
2. Mann-Whitney rank AUC (scipy.stats.rankdata) for every point estimate.
3. Own speaker indexing (Python sorted() of the distinct string ids, dict of row lists), fresh
   default_rng(0), rng.choice(n, size=n, replace=True); rebuilds all 2000 draws and compares to the saved npz.
4. Own p-value, percentile and Holm (sorted order + np.maximum.accumulate) and compares every TSV cell.
5. Scans the tex (comments stripped) for significance wording and checks each claim's line and quote.
6. Checks that nothing under omni_final changed after the B start time and that the tex is unchanged.
"""
import os, re, sys, json, hashlib, datetime
import numpy as np, pandas as pd
from scipy.stats import rankdata

B = "<local data dir>/release_from_mac/scores/part25/B"
A = "<local data dir>/release_from_mac/scores/part25/A"
REL = "<local data dir>/release"
LR = "<local data dir>/paper1_local_runs"
SC = "<local data dir>/release_from_mac/scores"
TEX = "<local data dir>/paper1_submission_23sep/final_overleaf_2325Z/AudioLLMHealthUtilityGap.tex"
TEX_SHA = "6c886f3513e93de7f78a7324f608091e7cdc24f257089bb5bd0a5c40638e4a20"
MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
NDRAW = 2000
fails = []
report = {}


def FAIL(msg):
    fails.append(msg)
    print("FAIL", msg, flush=True)


def base(x):
    s = str(x)
    return s.rsplit("/", 1)[-1]


def num(x):
    s = str(x).strip()
    m = re.fullmatch(r"np\.float64\((.*)\)", s)
    return float(m.group(1)) if m else float(s)


def mw_auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def table(path, key, spk, lab, cols):
    """dict clip -> (speaker, label, [scores]) ; asserts unique keys"""
    D = pd.read_csv(path, dtype=str)
    out = {}
    for _, r in D.iterrows():
        k = base(r[key])
        if k in out:
            raise AssertionError(f"duplicate {k} in {path}")
        out[k] = (str(r[spk]), int(float(r[lab])), [num(r[c]) for c in cols])
    return out


# ------------------------------------------------------------------ independent frame builders
def pair(fa, ka, sa, la, ca, fb, kb, sb, lb, cb):
    X = table(fa, ka, sa, la, [ca])
    Y = table(fb, kb, sb, lb, [cb])
    if set(X) != set(Y):
        raise AssertionError(f"clip sets differ {fa} {fb}")
    rows = []
    for k in sorted(X):
        if X[k][0] != Y[k][0] or X[k][1] != Y[k][1]:
            raise AssertionError(f"speaker/label differ {k}")
        rows.append((k, X[k][0], X[k][1], X[k][2][0], Y[k][2][0]))
    return rows


def build(i):
    """returns (kind, rows) ; rows: list of (clip, speaker, label, *scores) ; arms rows carry arm as last element"""
    if i in ("G_pcgita", "G_neurovoz", "G_adresso", "G_adress2020"):
        ds = i[2:]
        T = table(f"{A}/per_clip/A_{ds}_perclip.csv", "clip", "speaker", "label", [f"p_probe_seed{r}" for r in range(5)] + ["p_yes_zeroshot_saved"])
        Zs = table(f"{REL}/omni_final/omni_{ds}_zeroshot_scores.csv", "clip", "speaker", "label", ["p_yes"])
        assert set(T) == set(Zs)
        for k in T:
            assert T[k][0] == Zs[k][0] and T[k][1] == Zs[k][1] and abs(T[k][2][5] - Zs[k][2][0]) < 1e-15, k
        return "mean5", [(k, T[k][0], T[k][1], *T[k][2]) for k in sorted(T)]
    if i == "G_pitt":
        Z = np.load(f"{REL}/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
        Zs = table(f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv", "clip", "speaker", "label", ["p_yes"])
        rows = []
        for j, nm in enumerate(Z["name"]):
            k = base(nm)
            assert Zs[k][0] == str(Z["spk"][j]) and Zs[k][1] == int(Z["label"][j]), k
            rows.append((k, Zs[k][0], Zs[k][1], *[float(Z["oof"][r][j]) for r in range(5)], Zs[k][2][0]))
        assert len(rows) == 468 == len(Zs)
        return "mean5", sorted(rows)
    if i in ("G_edaic_llm", "G_edaic_enc", "Gs_edaic_ansstate"):
        st = {"G_edaic_llm": "llm", "G_edaic_enc": "enc", "Gs_edaic_ansstate": "ans"}[i]
        T = table(f"{SC}/part23/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv", "pid", "speaker", "label",
                  [f"oof_{st}_r{r}" for r in range(5)] + ["p_yes_whole"])
        Z2 = table(f"{SC}/part23/A/A2_edaic_whole_zeroshot_perclip.csv", "pid", "speaker", "label", ["p_yes_whole"])
        assert set(T) == set(Z2) and len(T) == 275
        for k in T:
            assert T[k][1] == Z2[k][1] and abs(T[k][2][5] - Z2[k][2][0]) < 1e-15, k
        return "mean5", [(k, T[k][0], T[k][1], *T[k][2]) for k in sorted(T)]
    if i == "Gs_kcl_single":
        return "two", pair(f"{REL}/omni_final/omni_kcl_enc_nested_oof.csv", "clip", "speaker", "label", "p_probe",
                           f"{REL}/omni_final/omni_kcl_zeroshot_scores.csv", "clip", "speaker", "label", "p_yes")
    if i.startswith("R_edaic_") or i.startswith("X_edaic_conf_"):
        m = i.split("_")[-1]
        T = pd.read_csv(f"{REL}/edaic_rerun/part17/T1_perclip_all966.csv", dtype=str)
        # cross-check T1 against the original part14 per-model score files
        srcs = {"o25": ("p14_o25_zeroshot_scores.csv", "clip"), "q2a": ("p14_q2a_zeroshot_scores.csv", "clip"),
                "q3o": ("p14_q3o_zeroshot_scores.csv", "clip"), "af2": ("p14_af2.csv", "clip_path"),
                "af3": ("p14_af3.csv", "clip_path"), "kimi": ("p14_kimi_zeroshot_scores.csv", "clip")}
        fn, kc = srcs[m]
        P = pd.read_csv(f"{REL}/edaic_rerun/part14/{fn}", dtype=str)
        pm = {}
        for _, r in P.iterrows():
            k = base(r[kc]).replace(".wav", "")
            v = num(r["p_yes"])
            if k in pm:
                assert abs(pm[k] - v) < 1e-12, ("repeat differs", k)
            pm[k] = v
        rows = []
        seen = set()
        for _, r in T.iterrows():
            k = r["clip_id"]
            assert abs(num(r[f"p_{m}"]) - pm[k]) < 1e-12, ("T1 vs part14", k)
            if r["set"] == "conflict" or (r["set"] == "agreement" and r["in_distinct_set"] == "1" and not i.startswith("X_")):
                if r["set"] == "agreement":
                    assert r["seg_uid"] not in seen
                    seen.add(r["seg_uid"])
                rows.append((k, r["speaker_id"], int(r["label"]), num(r[f"p_{m}"]), r["set"]))
        n_c = sum(1 for x in rows if x[4] == "conflict")
        n_a = sum(1 for x in rows if x[4] == "agreement")
        assert n_c == 483 and n_a == (0 if i.startswith("X_") else 311), (n_c, n_a)
        if i.startswith("X_"):
            return "one", sorted([x[:4] for x in rows])
        return "arms", sorted(rows)
    if i.startswith("R_pitt_") or i.startswith("Rs_pitt_") or i.startswith("X_pitt_conf_"):
        m = i.split("_")[-1]
        M = pd.read_csv(MAN, dtype=str)
        arm = {base(r.segment_path): (r.grp + r.spk, int(r.label), r.set) for r in M.itertuples()}
        assert len(arm) == 468
        if m == "af2":
            T1 = table(f"{SC}/part20/POD4c/af2_orig30_conflict.csv", "orig_clip_id", "speaker_id", "label", ["p_yes"])
            T2 = table(f"{SC}/part20/POD4c/af2_orig30_agreement.csv", "orig_clip_id", "speaker_id", "label", ["p_yes"])
            assert not (set(T1) & set(T2))
            T = {**T1, **T2}
        else:
            fn = {"q2a": (f"{LR}/probe2/pitt_zeroshot_scores.csv", "clip", "speaker"),
                  "o25": (f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv", "clip", "speaker"),
                  "q3o": (f"{REL}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv", "clip", "speaker"),
                  "af3": (f"{REL}/overnight2/part10/af3_pitt.csv", "clip_path", "speaker_id"),
                  "kimi": (f"{REL}/overnight2/part3/kimi_pitt_zeroshot_scores.csv", "clip", "speaker")}[m]
            T = table(fn[0], fn[1], fn[2], "label", ["p_yes"])
        assert set(T) == set(arm), m
        rows = []
        for k in sorted(T):
            assert T[k][0] == arm[k][0] and T[k][1] == arm[k][1], k
            if i.startswith("X_"):
                if arm[k][2] == "conflict":
                    rows.append((k, T[k][0], T[k][1], T[k][2][0]))
            else:
                rows.append((k, T[k][0], T[k][1], T[k][2][0], arm[k][2]))
        if i.startswith("X_"):
            assert len(rows) == 146
            return "one", rows
        return "arms", rows
    if i in ("F_pcgita", "F_neurovoz", "F_pitt", "F_adresso", "F_adress2020", "Fs_kcl"):
        ds = i.split("_", 1)[1]
        return "two", pair(f"{REL}/omni_final/omnisft_{ds}_oof.csv", "path", "speaker", "label", "p_yes",
                           f"{REL}/omni_final/omni_{ds}_zeroshot_scores.csv", "clip", "speaker", "label", "p_yes")
    if i == "Fs_edaic_seed0":
        F = table(f"{SC}/part24/FT/FT24_whole_seed0_perclip.csv", "pid", "pid", "label", ["p_yes"])
        Z2 = table(f"{SC}/part23/A/A2_edaic_whole_zeroshot_perclip.csv", "pid", "speaker", "label", ["p_yes_whole"])
        assert set(F) == set(Z2) and len(F) == 275
        return "two", [(k, Z2[k][0], Z2[k][1], F[k][2][0], Z2[k][2][0]) for k in sorted(F) if F[k][1] == Z2[k][1] or FAIL(k)]
    if i == "F_lora_minus_projector":
        return "two", pair(f"{REL}/edaic_rerun/part16/POD2/lora_pitt_oof.csv", "name", "spk", "label", "p_yes",
                           f"{REL}/omni_final/omnisft_pitt_oof.csv", "path", "speaker", "label", "p_yes")
    if i in ("C_pg_probe_drop", "C_nv_probe_drop"):
        d = i.split("_")[1]
        P4 = f"{REL}/edaic_rerun/part16/POD4"
        return "two", pair(f"{P4}/p16_{d}_after_enc_nested_oof.csv", "clip", "speaker", "label", "p_probe",
                           f"{P4}/p16_{d}_before_enc_nested_oof.csv", "clip", "speaker", "label", "p_probe")
    if i == "Cs_pg_answer":
        P4 = f"{REL}/edaic_rerun/part16/POD4"
        return "two", pair(f"{P4}/p16_pg_after_zeroshot_scores.csv", "clip", "speaker", "label", "p_yes",
                           f"{P4}/p16_pg_before_zeroshot_scores.csv", "clip", "speaker", "label", "p_yes")
    raise KeyError(i)


# ------------------------------------------------------------------ statistic + draws
def stat_fn(kind, y, S, arm):
    if kind == "mean5":
        def g(ix):
            per = [mw_auc(y[ix], S[ix, r]) for r in range(5)]
            return np.mean(per) - mw_auc(y[ix], S[ix, 5]), per + [mw_auc(y[ix], S[ix, 5])]
    elif kind == "two":
        def g(ix):
            a, b = mw_auc(y[ix], S[ix, 0]), mw_auc(y[ix], S[ix, 1])
            return a - b, [a, b]
    elif kind == "one":
        def g(ix):
            a = mw_auc(y[ix], S[ix, 0])
            return a - 0.5, [a]
    elif kind == "arms":
        isc = arm == "conflict"
        isa = arm == "agreement"

        def g(ix):
            c = ix[isc[ix]]
            a = ix[isa[ix]]
            u, v = mw_auc(y[c], S[c, 0]), mw_auc(y[a], S[a, 0])
            return u - v, [u, v]
    return g


def draws(spk, g):
    order = sorted(set(spk))
    where = {}
    for j, s in enumerate(spk):
        where.setdefault(s, []).append(j)
    groups = [np.array(where[s]) for s in order]
    rng = np.random.default_rng(0)
    out = np.empty(NDRAW)
    for b in range(NDRAW):
        pick = rng.choice(len(order), size=len(order), replace=True)
        ix = np.concatenate([groups[q] for q in pick])
        v, t = g(ix)
        out[b] = np.nan if any(np.isnan(t)) else v
    return out, order


def p_two(d):
    d = d[np.isfinite(d)]
    d = np.round(d, 12)  # exact ties at zero: rank and sklearn AUC differ by ~1e-16 there
    n = d.size
    return min(1.0, 2.0 * min((1 + np.sum(d <= 0)) / (n + 1), (1 + np.sum(d >= 0)) / (n + 1)))


def holm_v(p):
    p = np.asarray(p, float)
    m = p.size
    o = np.argsort(p, kind="stable")
    step = np.minimum(1.0, (m - np.arange(m)) * p[o])
    adj = np.maximum.accumulate(step)
    res = np.empty(m)
    res[o] = adj
    return res


# ------------------------------------------------------------------ main
T = pd.read_csv(f"{B}/B_holm.tsv", sep="\t", dtype=str, keep_default_na=False)
print("rows in TSV:", len(T))
recomputed = {}
for r in T.itertuples():
    i = r.id
    try:
        kind, rows = build(i)
    except AssertionError as ex:
        FAIL(f"{i}: source rebuild assertion {ex}")
        continue
    Pc = pd.read_csv(f"{B}/per_clip/B_{i}_perclip.csv", dtype=str, keep_default_na=False)
    Pc = Pc.sort_values("clip").reset_index(drop=True)
    if [x[0] for x in rows] != list(Pc["clip"]):
        FAIL(f"{i}: clip list differs from B per-clip")
        continue
    if [x[1] for x in rows] != list(Pc["speaker"]) or [x[2] for x in rows] != [int(v) for v in Pc["label"]]:
        FAIL(f"{i}: speaker or label differs from B per-clip")
    if kind == "mean5":
        bc = [f"probe_r{q}" for q in range(5)] + ["zeroshot"]
    elif kind in ("arms", "one"):
        bc = ["score"]
    else:
        bc = [c for c in Pc.columns if c not in ("clip", "speaker", "label")]
        assert len(bc) == 2, (i, bc)
    S = np.array([[float(v) for v in x[3:3 + len(bc)]] for x in rows])
    SB = Pc[bc].astype(float).values
    md = float(np.max(np.abs(S - SB)))
    if md > 1e-12:
        FAIL(f"{i}: score max abs diff vs B per-clip {md}")
    arm = np.array([x[-1] for x in rows]) if kind == "arms" else None
    if kind == "arms" and list(arm) != list(Pc["arm"]):
        FAIL(f"{i}: arm labels differ")
    y = np.array([x[2] for x in rows], int)
    spk = [x[1] for x in rows]
    g = stat_fn(kind, y, S, arm)
    pt, terms = g(np.arange(len(y)))
    d, order = draws(spk, g)
    Z = np.load(f"{B}/draws/B_{i}_draws.npz", allow_pickle=True)
    if [str(s) for s in Z["speakers"]] != order:
        FAIL(f"{i}: speaker order differs from npz")
    dz = Z["diff"].astype(float)
    same_nan = bool(np.array_equal(np.isnan(d), np.isnan(dz)))
    mx = float(np.nanmax(np.abs(d - dz)))
    if not same_nan or mx > 1e-9:
        FAIL(f"{i}: draws differ from npz (nan pattern same {same_nan}, max abs {mx})")
    if abs(float(Z["point_diff"]) - pt) > 1e-9:
        FAIL(f"{i}: point diff npz {float(Z['point_diff'])} vs {pt}")
    if i in ("G_pcgita", "G_neurovoz", "G_adresso", "G_adress2020", "G_pitt"):
        ZA = np.load(f"{A}/draws/A_{i[2:]}_draws.npz", allow_pickle=True)
        if not np.array_equal(ZA["diff"], Z["diff"]):
            FAIL(f"{i}: B draws not byte-equal to Track A draws")
    p = p_two(d)
    lo, hi = np.percentile(d[np.isfinite(d)], [2.5, 97.5])
    recomputed[i] = dict(kind=kind, value=pt, lo=float(lo), hi=float(hi), p=float(p), usable=int(np.isfinite(d).sum()),
                         n_clips=len(y), n_spk=len(order), max_draw_diff=mx, max_score_diff=md)
    for col, v in (("value", pt), ("lo", lo), ("hi", hi)):
        if getattr(r, col) != f"{v:+.4f}":
            FAIL(f"{i}: TSV {col} {getattr(r, col)} vs {v:+.4f}")
    if r.p != f"{p:.4f}":
        FAIL(f"{i}: TSV p {r.p} vs {p:.4f}")
    if int(r.usable_draws) != recomputed[i]["usable"] or int(r.n_clips) != len(y) or int(r.n_spk) != len(order):
        FAIL(f"{i}: counts differ")
    print(f"ok {i:26s} {kind:5s} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} maxdraw={mx:.1e}", flush=True)

# Holm recomputation
for fam in T.family.unique():
    for scope, roles, col_p, col_s, col_m in (("primary", ["claimed"], "holm_p", "survives", "m_family"),
                                             ("full", ["claimed", "sensitivity only"], "holm_p_full", "survives_full", "m_family_full")):
        sub = T[(T.family == fam) & T.role.isin(roles)]
        ids = list(sub.id)
        if not ids or not all(k in recomputed for k in ids):
            if ids:
                FAIL(f"holm {fam} {scope}: missing recomputed ids")
            continue
        adj = holm_v([recomputed[k]["p"] for k in ids])
        for k, a, (_, rr) in zip(ids, adj, sub.iterrows()):
            if rr[col_p] != f"{a:.4f}":
                FAIL(f"holm {fam} {scope} {k}: TSV {rr[col_p]} vs {a:.4f}")
            if rr[col_s] != ("yes" if a <= 0.05 else "no"):
                FAIL(f"holm {fam} {scope} {k}: survives flag")
            if int(rr[col_m]) != len(ids):
                FAIL(f"holm {fam} {scope} {k}: m")
            recomputed[k][f"holm_{scope}"] = float(a)

# paper line
MAIN = ["GAPS", "ARMS", "FINE TUNE", "CONTROLS"]
cl = T[(T.role == "claimed") & T.family.isin(MAIN)]
n_s = int(sum(recomputed[k]["holm_primary"] <= 0.05 for k in cl.id))
side = json.load(open(f"{B}/B_holm.sidecar.json"))
exp_start = f"{n_s} of {len(cl)} differences survive Holm correction within their family"
if not side["paper_line"].startswith(exp_start):
    FAIL(f"paper line {side['paper_line']} vs {exp_start}")
fam_counts = {f: int(((T.family == f) & (T.role == "claimed")).sum()) for f in MAIN}
if fam_counts != {"GAPS": 7, "ARMS": 9, "FINE TUNE": 6, "CONTROLS": 2}:
    FAIL(f"family counts {fam_counts}")

# ------------------------------------------------------------------ tex scan
raw = open(TEX, encoding="utf-8").read()
if hashlib.sha256(raw.encode()).hexdigest() != TEX_SHA:
    FAIL("tex sha changed")
lines = raw.split("\n")
live = []
in_comment = False
for n, L in enumerate(lines, 1):
    if "\\begin{comment}" in L:
        in_comment = True
    if in_comment:
        if "\\end{comment}" in L:
            in_comment = False
        live.append("")
        continue
    live.append(re.sub(r"(?<!\\)%.*$", "", L))
pat = re.compile(r"significan|exclude|excluding|below chance|beats|above the projector|improves|reverses|\[[^\]]*\d\.\d+[^\]]*\]")
scan = []
for n, L in enumerate(live, 1):
    for sent in re.split(r"(?<=[.;])\s+", L):
        if pat.search(sent):
            scan.append(dict(line=n, text=sent.strip()[:300]))
report["tex_scan"] = scan
for r in T.itertuples():
    ln = int(r.tex_line)
    sc = json.load(open(f"{B}/per_clip/B_{r.id}_perclip.sidecar.json"))
    if sc["tex_quote"][:40] not in live[ln - 1]:
        FAIL(f"{r.id}: quote not on live tex line {ln}")
    if sc.get("tex_check"):
        for k, v in sc["tex_check"].items():
            if not v["match"]:
                FAIL(f"{r.id}: tex value {k} mismatch {v}")

# ------------------------------------------------------------------ omni_final untouched
start = datetime.datetime.strptime(open(f"{B}/START_UTC.txt").read().strip(), "%a %b %d %H:%M:%S UTC %Y").replace(tzinfo=datetime.timezone.utc).timestamp()
newer = []
for root, _, files in os.walk(f"{REL}/omni_final"):
    for fn in files:
        pth = os.path.join(root, fn)
        if os.path.getmtime(pth) >= start:
            newer.append(pth)
if newer:
    FAIL(f"files under omni_final modified after B start: {newer[:5]}")
report["omni_final_files_newer_than_B_start"] = len(newer)

report.update(dict(ALL_PASS=len(fails) == 0, fails=fails, n_rows=len(T), recomputed=recomputed,
                   written_utc=datetime.datetime.utcnow().isoformat() + "Z",
                   method="independent rebuild from original sources; Mann-Whitney rank AUC; sorted() speaker order; own p, percentile and Holm",
                   versions=dict(numpy=np.__version__, pandas=pd.__version__)))
json.dump(report, open(f"{B}/verify/B_verify.json", "w"), indent=1, default=float)
print("ALL_PASS", len(fails) == 0, "fails", len(fails))
