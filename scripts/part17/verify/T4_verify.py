"""Independent verifier for PART 17 T4 (master_lookup.csv edits).

Written from scratch. Reads only original per-clip sources, the manifest,
the backup and the edited lookup. Writes only T4_verify.json / .log next to
this file. Nothing under omni_final is opened for writing.
"""
import csv, json, os, sys, hashlib
import numpy as np
import pandas as pd

R = "<local data dir>/release"
GD = "<local data dir>/paper work"
MAN = GD + "/Hard drive data for paper/DementiaBank/pitt_conflict_manifest.csv"
LK = R + "/master/master_lookup.csv"
BK = R + "/master/master_lookup.csv.bak_pre_part17"
SUP = R + "/master/SUPERSEDED.csv"
OUT = os.path.dirname(os.path.abspath(__file__))

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)


# ---------------------------------------------------------------- AUC
def auc_mw(y, s):
    """Mann-Whitney over every positive/negative pair, ties count 0.5."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(p) * len(n)))


def auc_trap(y, s):
    """Trapezoid area under the empirical ROC, one point per distinct threshold."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        sel = s >= t
        tpr.append((sel & (y == 1)).sum() / P)
        fpr.append((sel & (y == 0)).sum() / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def both(y, s):
    a = auc_mw(y, s); b = auc_trap(y, s)
    return a, b, round(a, 4) == round(b, 4)


def n_ties(s):
    s = np.asarray(s, float)
    u, c = np.unique(s, return_counts=True)
    return int(c[c > 1].sum()), int((c > 1).sum())


# ------------------------------------------------------------ bootstrap
def boot(y, spk, scores, B=2000, diff=None):
    """Speaker bootstrap. scores: list of 1d arrays (or 2d arrays (R, n) for
    mean-of-repeats). One speaker draw per replicate shared by all scores.
    diff=(i,j) also returns the paired difference i-j."""
    rng = np.random.default_rng(0)
    y = np.asarray(y).astype(int); spk = np.asarray(spk)
    uniq = np.unique(spk)
    idx = [np.where(spk == u)[0] for u in uniq]
    res = [[] for _ in scores]; dres = []
    for _ in range(B):
        d = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx[k] for k in d])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        vals = []
        for k, s in enumerate(scores):
            s = np.asarray(s, float)
            if s.ndim == 1:
                v = auc_mw(yy, s[ii])
            else:
                v = float(np.mean([auc_mw(yy, r[ii]) for r in s]))
            vals.append(v); res[k].append(v)
        if diff is not None:
            dres.append(vals[diff[0]] - vals[diff[1]])
    ci = [tuple(np.round(np.percentile(r, [2.5, 97.5]), 4)) for r in res]
    dci = tuple(np.round(np.percentile(dres, [2.5, 97.5]), 4)) if diff else None
    return ci, dci, len(res[0])


# ------------------------------------------------------------- manifest
man = pd.read_csv(MAN, dtype={"spk": str})
man["base"] = man.segment_path.map(os.path.basename)
man["spkkey"] = man.grp + man.spk.astype(int).astype(str)   # Control15 style
MB = man.set_index("base")
log("manifest rows", len(man), "arms", man.set.value_counts().to_dict(),
    "speakers grp+spk", man.spkkey.nunique())
for a, g in man.groupby("set"):
    log("  arm", a, "clips", len(g), "speakers", g.spkkey.nunique())


def attach(df, basecol):
    b = df[basecol].map(os.path.basename)
    assert b.is_unique, "duplicate clip names"
    miss = set(b) - set(MB.index)
    assert not miss, ("clips not in manifest", list(miss)[:3])
    df = df.copy()
    df["base"] = b.values
    df["arm"] = MB.loc[b, "set"].values
    df["mspk"] = MB.loc[b, "spkkey"].values
    df["mlabel"] = MB.loc[b, "label"].values
    return df


checks = []  # every value recomputed
def rec(what, claimed, mw, tr, note=""):
    agree_sources = round(mw, 4) == round(tr, 4)
    agree = agree_sources and (claimed is None or abs(round(mw, 4) - claimed) < 1e-9)
    checks.append(dict(what=what, claimed=claimed, mw=round(mw, 6), trap=round(tr, 6),
                       mw_trap_agree=agree_sources, agrees=agree, note=note))
    log(f"[{'OK ' if agree else 'BAD'}] {what}: claimed {claimed} | MW {mw:.6f} | trap {tr:.6f} {note}")


def rec_arith(what, claimed, val, note=""):
    agree = abs(round(val, 4) - claimed) < 1e-9
    checks.append(dict(what=what, claimed=claimed, mw=round(val, 6), trap=None,
                       mw_trap_agree=None, agrees=agree, note=note))
    log(f"[{'OK ' if agree else 'BAD'}] {what}: claimed {claimed} | computed {val:.6f} {note}")


def mean5(y, M):
    per = [auc_mw(y, r) for r in M]; pert = [auc_trap(y, r) for r in M]
    return per, pert, float(np.mean(per)), float(np.mean(pert))


# =============== 1. Qwen2.5-Omni Pitt encoder, part10 npz
d = np.load(R + "/overnight2/part10/pitt_enc_nested5_oof.npz", allow_pickle=True)
y = d["label"]; M = d["oof"]
nm = pd.DataFrame({"name": d["name"], "label": y})
nm = attach(nm, "name")
assert (nm.label.values == nm.mlabel.values).all()
per, pert, m, mt = mean5(y, M)
log("part10 per-repeat", np.round(per, 4))
rec("Q25O pitt encoder mean-of-5 (row 1)", 0.7706, m, mt, f"n={len(y)}")
a, b, _ = both(y, M.mean(0))
rec("Q25O pitt encoder averaged-prob (not used)", 0.7917, a, b)
old = pd.read_csv(R + "/omni_final/omni_pitt_enc_nested_oof.csv")
a, b, _ = both(old.label, old.p_probe)
rec("Q25O pitt encoder old single-split (SUPERSEDED)", 0.7969, a, b)

# =============== 2. E-DAIC full, POD2 rebuild
ed = pd.read_csv(R + "/edaic_rerun/part16/EDAICFULL/edaic_full_perclip.csv")
assert ed.pid.is_unique and len(ed) == 275
ey = ed.label.values
for col, cl, tag in [("oof_llm", 0.7088, "row 2 LM"), ("oof_enc", 0.5916, "row 3 encoder"),
                     ("oof_ans", 0.7748, "json ans auc_of_mean_oof"), ("oof_proj", 0.5631, "json proj auc_of_mean_oof")]:
    a, b, _ = both(ey, ed[col])
    rec(f"E-DAIC full {tag} ({col})", cl, a, b, f"n={len(ed)} pos={int(ey.sum())}")
a, b, _ = both(ey, ed.p_yes_answer)
rec("E-DAIC full zero-shot answer p_yes_answer", None, a, b)
ej = json.load(open(R + "/edaic_rerun/part16/EDAICFULL/edaic_full_nested_repeats.json"))
rec_arith("E-DAIC LM mean of json per_repeat (row 4)", 0.7001, float(np.mean(ej["llm"]["per_repeat"])))
rec_arith("E-DAIC enc mean of json per_repeat (row 5)", 0.5884, float(np.mean(ej["enc"]["per_repeat"])))
log("json llm per_repeat", ej["llm"]["per_repeat"], "enc per_repeat", ej["enc"]["per_repeat"])
ci, dci, nb = boot(ey, ed.pid.values, [ed.oof_llm.values, ed.p_yes_answer.values], diff=(0, 1))
dpt = auc_mw(ey, ed.oof_llm) - auc_mw(ey, ed.p_yes_answer)
log(f"E-DAIC LM minus answer {dpt:.4f} CI {dci} (claimed -0.1190 [-0.1995, -0.0478]) B={nb}")
checks.append(dict(what="E-DAIC LM minus answer paired CI", claimed="-0.1190 [-0.1995, -0.0478]",
                   mw=f"{dpt:.4f} [{dci[0]:.4f}, {dci[1]:.4f}]", trap=None, mw_trap_agree=None,
                   agrees=(round(dpt, 4) == -0.1190 and dci == (-0.1995, -0.0478))))
ci, dci, nb = boot(ey, ed.pid.values, [ed.oof_enc.values, ed.p_yes_answer.values], diff=(0, 1))
dpt = auc_mw(ey, ed.oof_enc) - auc_mw(ey, ed.p_yes_answer)
log(f"E-DAIC enc minus answer {dpt:.4f} CI {dci} (claimed -0.2361 [-0.3272, -0.1394]) B={nb}")
checks.append(dict(what="E-DAIC encoder minus answer paired CI", claimed="-0.2361 [-0.3272, -0.1394]",
                   mw=f"{dpt:.4f} [{dci[0]:.4f}, {dci[1]:.4f}]", trap=None, mw_trap_agree=None,
                   agrees=(round(dpt, 4) == -0.2361 and dci == (-0.3272, -0.1394))))
# old values (SUPERSEDED json)
oj = json.load(open(R + "/edaic_rerun/variants/o25_full_nested_repeats.json"))
rec_arith("old E-DAIC LM mean of per_repeat", 0.6853, float(np.mean(oj["llm"]["per_repeat"])))
rec_arith("old E-DAIC enc mean of per_repeat", 0.5930, float(np.mean(oj["enc"]["per_repeat"])))
log("old json ans", oj["ans"]["mean"], "proj", oj["proj"]["mean"])
# 300 s truncation count
br = pd.read_csv(R + "/edaic_rerun/part16/EDAICFULL/build_report.csv")
ntr = int((br.win_dur_s > 300.0 + 1e-6).sum())
log("E-DAIC windows longer than 300 s:", ntr, "of", len(br), "(claimed 217 of 275)")
checks.append(dict(what="E-DAIC windows > 300 s", claimed=217, mw=ntr, trap=None,
                   mw_trap_agree=None, agrees=(ntr == 217 and len(br) == 275)))

# =============== 3. Qwen3-Omni POD3 encoder, Pitt and PC-GITA
for ds, f, cl_m, cl_avg, row in [
        ("pitt", "q3o_pitt_encoder_nested_oof.csv", 0.8122, 0.8354, "row 7"),
        ("pcgita", "q3o_pcgita_encoder_nested_oof.csv", 0.8969, 0.9109, "row 6")]:
    q = pd.read_csv(R + "/edaic_rerun/part16/POD3/" + f)
    Mq = q[[f"p_probe_rep{i}" for i in range(5)]].values.T
    per, pert, m, mt = mean5(q.label.values, Mq)
    log(f"q3o {ds} enc per-repeat", np.round(per, 4))
    rec(f"Q3O {ds} encoder mean-of-5 ({row})", cl_m, m, mt, f"n={len(q)}")
    a, b, _ = both(q.label, Mq.mean(0))
    rec(f"Q3O {ds} encoder averaged-prob (not used)", cl_avg, a, b)
    if ds == "pitt":
        qa = attach(q, "clip")
        assert (qa.label.values == qa.mlabel.values).all()
        pj = json.load(open(R + "/edaic_rerun/part16/pod_sync/q3o_pitt_nested_repeats.json"))
        log("pod_sync json enc per_repeat", pj["enc"]["per_repeat"], "vs csv", list(np.round(per, 4)))
        checks.append(dict(what="Q3O pitt enc per-repeat csv == pod_sync json", claimed=pj["enc"]["per_repeat"],
                           mw=[round(x, 4) for x in per], trap=None, mw_trap_agree=None,
                           agrees=[round(x, 4) for x in per] == pj["enc"]["per_repeat"]))
        # other POD3 streams (side note only)
        for st, cl in [("llm", 0.8445), ("ans", 0.8237), ("proj", 0.7820)]:
            p3 = pd.read_csv(R + f"/edaic_rerun/part16/POD3/q3o_pitt_{st}_nested_oof.csv")
            cols = [c for c in p3.columns if c.startswith("p_probe_rep")]
            M3 = p3[cols].values.T
            per3, pert3, m3, mt3 = mean5(p3.label.values, M3)
            rec(f"Q3O pitt POD3 {st} mean over {len(cols)} repeat(s) (side note)", cl, m3, mt3)
        oj3 = json.load(open(R + "/overnight2/final/q3o_pitt_nested_repeats.json"))
        rec_arith("old Q3O pitt enc json mean (SUPERSEDED)", 0.8289, float(np.mean(oj3["enc"]["per_repeat"])))
        log("old json llm/ans/proj means", oj3["llm"]["mean"], oj3["ans"]["mean"], oj3["proj"]["mean"])
    else:
        oj3 = json.load(open(R + "/overnight2/q3o_new/q3o_pcgita_nested_repeats.json"))
        log("old pcgita json per_repeat", oj3["enc"]["per_repeat"], "mean", np.mean(oj3["enc"]["per_repeat"]))
        rec_arith("old Q3O pcgita enc json mean", 0.8969, float(np.mean(oj3["enc"]["per_repeat"])))

# =============== 4. Zero-shot Kimi / AF2 (pooled and arms)
def zs_block(tag, df, basecol, score, claims, lcol="label"):
    df = attach(df, basecol)
    lab_ok = bool((df[lcol].values == df.mlabel.values).all())
    log(f"{tag}: n={len(df)} labels match manifest: {lab_ok}; ties {n_ties(df[score])}")
    out = {}
    for arm, cl in claims.items():
        sub = df if arm == "pooled" else df[df.arm == arm]
        a, b, _ = both(sub[lcol], sub[score])
        rec(f"{tag} {arm}", cl, a, b, f"n={len(sub)} spk={sub.mspk.nunique()}")
        out[arm] = (sub, a)
    return df, out

kim = pd.read_csv(R + "/overnight2/part3/kimi_pitt_zeroshot_scores.csv")
zs_block("Kimi pitt canonical (rows 8-10)", kim, "clip", "p_yes",
         {"pooled": 0.7180, "conflict": 0.7197, "agreement": 0.7227})
kold = pd.read_csv(R + "/overnight/kimi/kimi_pitt_zeroshot_scores.csv")
zs_block("Kimi pitt OLD (SUPERSEDED)", kold, "clip", "p_yes",
         {"pooled": 0.6964, "conflict": 0.7078, "agreement": 0.7014})
af = pd.read_csv(GD + "/paper1_local_runs/af2_results/af2_pitt468.csv")
afa, _ = zs_block("AF2 pitt canonical (rows 11-13)", af, "clip_path", "p_yes",
                  {"pooled": 0.5724, "conflict": 0.6310, "agreement": 0.5544})
afo = pd.read_csv(GD + "/paper1_local_runs/af2_pitt_audio.csv")
zs_block("AF2 pitt OLD af2_pitt_audio.csv (SUPERSEDED)", afo, "path", "p_yes",
         {"pooled": 0.5273, "conflict": 0.6691, "agreement": 0.4895})
# is 0.5536 reachable from AF2 agreement with any tie rule? count ties within arm
ag = afa[afa.arm == "agreement"]
log("AF2 agreement arm ties:", n_ties(ag.p_yes), "; old file agreement ties:",
    n_ties(attach(afo, "path").query("arm=='agreement'").p_yes))

# =============== 5. Qwen3-Omni answer-state probe by arm (full-fit restricted)
pp = np.load(R + "/edaic_rerun/part16/pod_sync/q3o_perp_cols.npz", allow_pickle=True)
pdf = pd.DataFrame({"name": pp["name"], "spk": pp["spk"], "label": pp["label"], "oof": pp["oof"]})
pdf = attach(pdf, "name")
assert (pdf.label.values == pdf.mlabel.values).all()
log("perp npz spk == manifest grp+spk:", bool((pdf.spk.values == pdf.mspk.values).all()))
a, b, _ = both(pdf.label, pdf.oof)
rec("Q3O pitt answer-state full-fit pooled (vs readout csv 'all' 0.8307)", 0.8307, a, b)
for arm, cl, row in [("conflict", 0.6394, "row 14"), ("agreement", 0.9032, "row 15")]:
    sub = pdf[pdf.arm == arm]
    a, b, _ = both(sub.label, sub.oof)
    rec(f"Q3O pitt answer-state {arm} full-fit restricted ({row})", cl, a, b,
        f"n={len(sub)} spk={sub.mspk.nunique()}")
    ci1, _, nb1 = boot(sub.label.values, sub.spk.values, [sub.oof.values])
    ci2, _, nb2 = boot(sub.label.values, sub.mspk.values, [sub.oof.values])
    mid = np.array([g + k for g, k in zip(MB.loc[sub.base, "grp"].values, MB.loc[sub.base, "spk"].values)])
    ci3, _, nb3 = boot(sub.label.values, mid, [sub.oof.values])
    log(f"   CI npz IDs {ci1[0]} | grp+int IDs {ci2[0]} | grp+zero-padded manifest IDs {ci3[0]}")
rd = pd.read_csv(R + "/edaic_rerun/part16/POD3/out/readout_direction_q3o_by_arm.csv")
log("readout csv refit-within-arm auc_probe:", dict(zip(rd.arm, rd.auc_probe)))

# =============== 6. LoRA
lo = pd.read_csv(R + "/edaic_rerun/part16/POD2/lora_pitt_oof.csv").rename(columns={"arm": "file_arm"})
lo = attach(lo, "name")
log("lora labels match manifest", bool((lo.label.values == lo.mlabel.values).all()),
    "arm col matches manifest", bool((lo.file_arm.values == lo.arm.values).all()))
a, b, _ = both(lo.label, lo.p_yes)
tv, tg = n_ties(lo.p_yes)
rec("Q25O pitt LoRA (row 16)", 0.8250, a, b, f"n={len(lo)} tied values {tv} in {tg} groups")
ab = pd.read_csv(R + "/omni_final/abl2_pitt_both_oof.csv")
ab = attach(ab, "path")
a, b, _ = both(ab.label, ab.p_yes)
tv, tg = n_ties(ab.p_yes)
rec("Q25O pitt LoRA+projector abl2 csv (row 17)", 0.7696, a, b, f"n={len(ab)} tied values {tv} in {tg} groups")
for arm in ["conflict", "agreement"]:
    s = ab[ab.arm == arm]; a2, b2, _ = both(s.label, s.p_yes)
    log(f"   abl2 csv {arm} {a2:.4f} (abl2 json {json.load(open(R+'/omni_final/abl2_pitt_both.json'))['arms'][arm]}, abl json {json.load(open(R+'/omni_final/abl_pitt_both.json'))['arms'][arm]})")

# =============== 7. lookup diff
def read_rows(p):
    with open(p, newline="") as fh:
        return list(csv.reader(fh))
old = read_rows(BK); new = read_rows(LK)
log("rows (excl header) before", len(old) - 1, "after", len(new) - 1)
key = lambda r: (r[0], r[1], r[2])
ok = {key(r): r for r in old[1:]}; nk = {key(r): r for r in new[1:]}
log("duplicate keys before", len(old) - 1 - len(ok), "after", len(new) - 1 - len(nk))
dropped = sorted(set(ok) - set(nk)); added = sorted(set(nk) - set(ok))
changed = sorted(k for k in set(ok) & set(nk) if ok[k] != nk[k])
same = sorted(k for k in set(ok) & set(nk) if ok[k] == nk[k])
log("dropped", dropped); log("added", len(added)); [log("   +", k, nk[k][3], nk[k][4]) for k in added]
log("changed", len(changed))
for k in changed:
    cols = [c for c, (x, y2) in enumerate(zip(ok[k], nk[k])) if x != y2]
    log("   ~", k, "cols changed", [old[0][c] for c in cols], "auc", ok[k][3], "->", nk[k][3])
log("unchanged", len(same))
# order check: are unchanged rows in the same relative order and byte-identical lines?
ol = open(BK).read().splitlines(); nl = open(LK).read().splitlines()
log("header identical", ol[0] == nl[0])
# every source path in every row
def src_path(s):
    return s.split(" [")[0].strip()
missing = []; stale = 0; stale_old = 0
for r in new[1:]:
    p = src_path(r[6])
    if p.startswith("<local data dir>/paper1_local_runs/"): stale += 1
    if not os.path.exists(p): missing.append((key(r), p))
for r in old[1:]:
    if src_path(r[6]).startswith("<local data dir>/paper1_local_runs/"): stale_old += 1
log("rows with stale G-Drive prefix: before", stale_old, "after", stale)
log("rows whose source path does not exist:", len(missing))
touched = set(added) | set(changed)
missing_touched = [m for m in missing if m[0] in touched]
log("   of which in touched rows:", missing_touched)
# stale rows resolvable at the new prefix?
res = 0
for k, p in missing:
    q = p.replace("<local data dir>/paper1_local_runs/", "<local data dir>/paper work/paper1_local_runs/")
    if os.path.exists(q): res += 1
log("   missing paths that exist under 'paper work/':", res)

# =============== 8. SUPERSEDED.csv
sup = pd.read_csv(SUP)
log("SUPERSEDED rows", len(sup))
for _, r in sup.iterrows():
    log("   file exists", os.path.exists(r.file), "| superseded_by exists", os.path.exists(r.superseded_by), "|", os.path.basename(r.file))


# =============== 9. every interval in rows/T4.tsv, own bootstrap
tsv = pd.read_csv(R + "/edaic_rerun/part17/rows/T4.tsv", sep="\t").set_index("id")
MB["padkey"] = MB.grp + MB.spk          # Control015 style (manifest spelling)
def pitt_keys(df, own_col):
    return {"manifest": MB.loc[df.base, "padkey"].values, "own": df[own_col].astype(str).values}
ci_checks = []
def ci_rec(tid, y, spkd, score):
    row = tsv.loc[tid]
    out = {}
    for kname, spk in spkd.items():
        ci, _, nb = boot(y, spk, [score])
        out[kname] = (float(ci[0][0]), float(ci[0][1]), len(np.unique(spk)), nb)
    hit = [k for k, v in out.items() if abs(v[0] - row.lo) < 1e-9 and abs(v[1] - row.hi) < 1e-9]
    ci_checks.append(dict(id=tid, claimed=[row.value, row.lo, row.hi, row.n_spk], computed=out, match_keys=hit))
    log(f"[{'OK ' if hit else 'BAD'}] CI {tid}: claimed [{row.lo}, {row.hi}] n_spk {row.n_spk} | " +
        " | ".join(f"{k}: [{v[0]}, {v[1]}] spk={v[2]} B={v[3]}" for k, v in out.items()))
# part10
nm2 = nm.copy(); nm2["spk"] = d["spk"]
kk = pitt_keys(nm2, "spk")
ci_rec("T4_pitt_o25_enc_mean5", y, kk, M)
ci_rec("T4_pitt_o25_enc_avgprob_NOTUSED", y, kk, M.mean(0))
# E-DAIC single-score intervals
ci_rec("T4_edaic_o25_llm_avgprob", ey, {"pid": ed.pid.values}, ed.oof_llm.values)
ci_rec("T4_edaic_o25_enc_avgprob", ey, {"pid": ed.pid.values}, ed.oof_enc.values)
# Q3O POD3
for ds, f in [("pcgita", "q3o_pcgita_encoder_nested_oof.csv"), ("pitt", "q3o_pitt_encoder_nested_oof.csv")]:
    q = pd.read_csv(R + "/edaic_rerun/part16/POD3/" + f)
    Mq = q[[f"p_probe_rep{i}" for i in range(5)]].values.T
    if ds == "pitt":
        q = attach(q, "clip"); kk = pitt_keys(q, "speaker")
    else:
        kk = {"own": q.speaker.astype(str).values}
        log("pcgita speakers", q.speaker.nunique(), "clips", len(q))
    ci_rec(f"T4_{ds}_q3o_enc_mean5", q.label.values, kk, Mq)
    ci_rec(f"T4_{ds}_q3o_enc_avgprob_NOTUSED", q.label.values, kk, Mq.mean(0))
# zero-shot
def zci(prefix, df, basecol, spkcol):
    df = attach(df, basecol)
    for arm in ["pooled", "conflict", "agreement"]:
        sub = df if arm == "pooled" else df[df.arm == arm]
        ci_rec(f"{prefix}_{arm}", sub.label.values, pitt_keys(sub, spkcol), sub.p_yes.values)
zci("T4_pitt_kimi", kim, "clip", "speaker")
zci("T4_pitt_kimi_SUPERSEDED_overnight", kold, "clip", "speaker")
zci("T4_pitt_af2", af, "clip_path", "speaker_id")
zci("T4_pitt_af2_SUPERSEDED_audio", afo, "path", "speaker")
for arm in ["conflict", "agreement"]:
    sub = pdf[pdf.arm == arm]
    ci_rec(f"T4_pitt_q3o_{arm}_fullfit", sub.label.values, pitt_keys(sub, "spk"), sub.oof.values)
ci_rec("T4_pitt_o25_lora", lo.label.values, pitt_keys(lo, "spk"), lo.p_yes.values)
ci_rec("T4_pitt_o25_lora_plus_proj", ab.label.values, pitt_keys(ab, "speaker"), ab.p_yes.values)
# value column of T4.tsv vs lookup / recomputation already covered; row counts
log("T4.tsv rows_before", tsv.loc["T4_lookup_rows_before", "value"], "rows_after", tsv.loc["T4_lookup_rows_after", "value"],
    "stale", tsv.loc["T4_lookup_stale_prefix_rows", "value"])
nbad_ci = sum(1 for c in ci_checks if not c["match_keys"])
log("interval checks", len(ci_checks), "unmatched", nbad_ci)
checks.append(dict(what="T4.tsv intervals reproduced", claimed=len(ci_checks), mw=len(ci_checks) - nbad_ci,
                   trap=None, mw_trap_agree=None, agrees=(nbad_ci == 0)))

json.dump(dict(checks=checks, ci_checks=ci_checks, log=LOG, missing_paths=[[list(k), p] for k, p in missing]),
          open(os.path.join(OUT, "T4_verify.json"), "w"), indent=1, default=str)
open(os.path.join(OUT, "T4_verify.log"), "w").write("\n".join(LOG) + "\n")
nbad = sum(1 for c in checks if not c["agrees"])
log("checks", len(checks), "bad", nbad)
