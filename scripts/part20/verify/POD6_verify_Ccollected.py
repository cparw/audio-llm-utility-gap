#!/usr/bin/env python3
"""POD6 verifier, part C collected rows: rebuild every M8 B_pitt row and every T6 row
from the raw per-clip score files with own code, and compare with the collected copies.
usage: POD6_verify_Ccollected.py C_LANDED_DIR OUT.json"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from POD6_verify import auc_mw, boot_ci, _groups, NB
from POD6_verify_C import arms, cells

CD, OUT = sys.argv[1:3]
R = "<local data dir>/release/"
G = "<local data dir>/paper work/paper1_local_runs/"
fix = lambda p: p.replace("<local data dir>/paper1_local_runs/", G)
res = []


def rec(id_, what, comp, ver, lo_c, hi_c, n, nspk, file, base=0.5, has_ci=True):
    if has_ci:
        ag = all(round(float(a), 4) == round(float(b), 4) for a, b in zip((comp, lo_c, hi_c), ver))
        vs = ("above" if ver[1] > base else "below" if ver[2] < base else "spans") + ("" if base == 0.5 else " 0")
        two = f"{ver[0]:.2f} [{ver[1]:.2f}, {ver[2]:.2f}]"; lo, hi = f"{ver[1]:.4f}", f"{ver[2]:.4f}"
    else:
        ag = round(float(comp), 4) == round(float(ver[0]), 4); vs = "n/a"; two = f"{ver[0]:.2f}"; lo = hi = ""
    res.append(dict(id=id_, what=what, value_computed=f"{float(comp):.4f}", value_verified=f"{ver[0]:.4f}", lo=lo, hi=hi,
                    n=n, n_spk=nspk, file=file, two_dp=two, vs_half=vs, agree_4dp="yes" if ag else "NO",
                    _comp_ci=f"[{float(lo_c):.4f}, {float(hi_c):.4f}]" if has_ci else ""))
    print(f"{'OK ' if ag else 'DISAGREE'} {id_}: comp {float(comp):.4f} {res[-1]['_comp_ci']} ver {ver[0]:.4f} "
          f"{'[%.4f, %.4f]' % (ver[1], ver[2]) if has_ci else ''}", flush=True)


# ---------------- M8 B_pitt rows
m8 = pd.read_csv(os.path.join(CD, "C_M8_B_pitt_rows.csv"))
cache = {}
for r in m8.itertuples():
    src = fix(r.source_file)
    if src not in cache:
        cache[src] = cells(arms(src, "p_yes"))
    v = cache[src][r.arm]
    assert v["n"] == r.n and v["n_spk"] == r.n_spk and v["n_pos"] == r.n_pos, (r, v)
    rec(f"C_M8_{r.model.replace(' ', '')}_{r.source_tag}_{r.arm}", f"Pitt {r.model} ({r.source_tag}) zero-shot answer AUC, {r.arm.replace('_', ' ')}",
        r.auc, (v["auc"], v["lo"], v["hi"]), r.lo, r.hi, r.n, r.n_spk, src, base=0.0 if r.arm.startswith("conflict_minus") else 0.5)

# ---------------- T6 rows
s20 = arms(R + "overnight2/text_new/o25_pitt_text.csv", "p_yes")
s15 = arms(G + "omni_pitt_text.csv", "p_yes")
au = arms(R + "omni_final/omni_pitt_zeroshot_scores.csv", "p_yes")
k = s20[["b", "spk", "y", "arm"]].rename(columns={"spk": "spk20", "y": "y20"})
D = k.merge(s20[["b", "s"]].rename(columns={"s": "t20"}), on="b").merge(s15[["b", "s", "y", "spk"]].rename(columns={"s": "t15", "y": "y15", "spk": "spk15"}), on="b") \
     .merge(au[["b", "s", "y", "spk"]].rename(columns={"s": "aud", "y": "yau", "spk": "spkau"}), on="b")
assert len(D) == 468 and (D.y20 == D.y15).all() and (D.y20 == D.yau).all()
print("speaker ids s20==s15:", (D.spk20 == D.spk15).all(), " s20==audio:", (D.spk20 == D.spkau).all())
spk = D.spk20.to_numpy(); y = D.y20.to_numpy(); con = (D.arm == "conflict").to_numpy()
S = {"s20": D.t20.to_numpy(), "s15": D.t15.to_numpy(), "audio": D.aud.to_numpy()}


def boot(fn, mask=None):
    ii0 = np.arange(len(y)) if mask is None else np.where(mask)[0]
    rng = np.random.default_rng(0); u, idx = _groups(spk[ii0]); v = []
    idx = {p: ii0[q] for p, q in idx.items()}
    for _ in range(NB):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        val = fn(ii)
        if val is None: continue
        v.append(val)
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def A(s, ii):
    return auc_mw(y[ii], s[ii]) if len(np.unique(y[ii])) == 2 else None


def gap(s, ii):
    c = ii[con[ii]]; a = ii[~con[ii]]
    if len(np.unique(y[c])) < 2 or len(np.unique(y[a])) < 2: return None
    return auc_mw(y[c], s[c]) - auc_mw(y[a], s[a])


def sub(f, g):
    def h(ii):
        a, b = f(ii), g(ii)
        return None if a is None or b is None else a - b
    return h


t6 = pd.read_csv(os.path.join(CD, "C_T6.tsv"), sep="\t")
ALL = np.ones(len(y), bool); M = {"all468": ALL, "conflict": con, "agreement": ~con}
for r in t6.itertuples():
    i = r.id; parts = i.split("_")
    has = not (isinstance(r.lo, float) and np.isnan(r.lo))
    if i.startswith("T6_perclip_"):
        a, b = S["s20"], S["s15"]
        if "pearson" in i: v = float(np.corrcoef(a, b)[0, 1])
        elif "spearman" in i:
            from scipy.stats import spearmanr; v = float(spearmanr(a, b)[0])
        elif "maxabs" in i: v = float(np.abs(a - b).max())
        else: v = float(np.abs(a - b).mean())
        rec("C_" + i, r.what, r.value, (v,), None, None, r.n, r.n_spk, r.file, has_ci=False); continue
    if i.startswith("T6_armgap_ratio"):
        run = parts[-1]; ii = np.arange(len(y))
        v = gap(S[run], ii) / gap(S["audio"], ii)
        rec("C_" + i, r.what, r.value, (v,), None, None, r.n, r.n_spk, r.file, has_ci=False); continue
    if i.endswith("_conflict_minus_agreement") and (i.startswith("T6_s20_text") or i.startswith("T6_s15_text") or i.startswith("T6_audio")):
        run = "audio" if i.startswith("T6_audio") else parts[1]; s = S[run]
        v = gap(s, np.arange(len(y))); lo, hi = boot(lambda ii: gap(s, ii))
        rec("C_" + i, r.what, r.value, (v, lo, hi), r.lo, r.hi, r.n, r.n_spk, r.file, base=0.0); continue
    if "_text_minus_audio_" in i:
        run = parts[1]; arm = parts[-1]; m = M[arm]; s = S[run]; a_ = S["audio"]
        ii = np.where(m)[0]; v = A(s, ii) - A(a_, ii)
        lo, hi = boot(sub(lambda jj: A(s, jj), lambda jj: A(a_, jj)), None if arm == "all468" else m)
        rec("C_" + i, r.what, r.value, (v, lo, hi), r.lo, r.hi, r.n, r.n_spk, r.file, base=0.0); continue
    if "_gap_text_minus_gap_audio" in i:
        s = S[parts[1]]; a_ = S["audio"]; ii = np.arange(len(y))
        v = gap(s, ii) - gap(a_, ii); lo, hi = boot(sub(lambda jj: gap(s, jj), lambda jj: gap(a_, jj)))
        rec("C_" + i, r.what, r.value, (v, lo, hi), r.lo, r.hi, r.n, r.n_spk, r.file, base=0.0); continue
    if i.startswith("T6_s20_minus_s15_"):
        arm = parts[-1]; a, b = S["s20"], S["s15"]
        if arm == "armgap":
            ii = np.arange(len(y)); v = gap(a, ii) - gap(b, ii); lo, hi = boot(sub(lambda jj: gap(a, jj), lambda jj: gap(b, jj)))
        else:
            m = M[arm]; ii = np.where(m)[0]; v = A(a, ii) - A(b, ii)
            lo, hi = boot(sub(lambda jj: A(a, jj), lambda jj: A(b, jj)), None if arm == "all468" else m)
        rec("C_" + i, r.what, r.value, (v, lo, hi), r.lo, r.hi, r.n, r.n_spk, r.file, base=0.0); continue
    # plain cells: T6_{s20_text|s15_text|audio}_{all468|conflict|agreement}
    run = "audio" if i.startswith("T6_audio") else parts[1]; arm = parts[-1]; m = M[arm]; s = S[run]
    ii = np.where(m)[0]; v = A(s, ii); lo, hi = boot(lambda jj: A(s, jj), None if arm == "all468" else m)
    rec("C_" + i, r.what, r.value, (v, lo, hi), r.lo, r.hi, r.n, r.n_spk, r.file)
json.dump(res, open(OUT, "w"), indent=1)
print("rows", len(res), "agree", sum(r["agree_4dp"] == "yes" for r in res))
