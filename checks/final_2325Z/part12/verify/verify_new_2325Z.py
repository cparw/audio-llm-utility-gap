#!/usr/local/bin/python3
"""Recompute the final-tex numbers that the 152-row verifier does not cover (E-DAIC whole interview FT, window
answers, whole-window probes, projector trained on the E-DAIC pairs). Independent code; same interval rule:
2000 draws, fresh numpy default_rng(0) per cell, rng.choice(N, N, replace=True) over sorted unique speaker ids,
2.5/97.5 percentiles, paired = same idx. AUC = sklearn roc_auc_score. Reads only; writes OUT csv.
usage: verify_new_2325Z.py OUT.csv"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "<local data dir>/release"; G = "<local data dir>"
P23 = G + "/release_from_mac/scores/part23"; P24 = G + "/release_from_mac/scores/part24"
F = dict(
 A2=P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv",
 A3=P23 + "/pull/p23-a-probes/p23a/out/A3_edaic_whole_meanof5_perclip.csv",
 FT0=P24 + "/FT/FT24_whole_seed0_perclip.csv",
 FT1=P24 + "/FT/FT24_whole_seed1all_perclip.csv",
 ZS30=R + "/omni_final/omni_edaic_zeroshot_scores.csv",
 ZSMID30=R + "/edaic_rerun/variants/o25_mid30_zeroshot_scores.csv",
 ZSMID300=R + "/edaic_rerun/variants/o25_mid300_zeroshot_scores.csv",
 ZS300=R + "/edaic_rerun/variants/o25_full_zeroshot_scores.csv",
 NR30_MAC=R + "/omni_final/omni_edaic_nested_repeats.json",
 NR30_GD=G + "/paper1_local_runs/omni_final/omni_edaic_nested_repeats.json",
 NRMID30=R + "/edaic_rerun/variants/o25_mid30_nested_repeats.json",
 NRMID300=R + "/edaic_rerun/variants/o25_mid300_nested_repeats.json",
 T7B=R + "/edaic_rerun/part17/T7b/T7b_edaic300_meanof5_perclip.csv",
 FT966=R + "/scores/part20/POD3/p14_ft966_oof.csv",
 P14O25=R + "/edaic_rerun/part14/p14_o25_zeroshot_scores.csv",
 T1=R + "/edaic_rerun/part17/T1_perclip_distinct_agreement.csv",
)
def rd(path):
    """read csv; some pod files store floats as the text np.float64(x)"""
    d = pd.read_csv(path)
    for c in d.columns:
        if d[c].dtype == object and d[c].astype(str).str.startswith("np.float64(").any():
            d[c] = d[c].astype(str).str.replace(r"np\.float64\((.*)\)", r"\1", regex=True).astype(float)
    return d
def auc(y, s): return float(roc_auc_score(np.asarray(y), np.asarray(s, float)))
def boot(df, spk, fn, nb=2000):
    df = df.reset_index(drop=True); key = df[spk].astype(str).values; ids = np.array(sorted(set(key))); N = len(ids)
    pos = {s: np.flatnonzero(key == s) for s in ids}; rng = np.random.default_rng(0); v = []
    for _ in range(nb):
        idx = rng.choice(N, size=N, replace=True); sub = df.iloc[np.concatenate([pos[ids[i]] for i in idx])]
        try: v.append(fn(sub))
        except ValueError: pass
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)
rows = []
def put(i, v, file, how, lo=None, hi=None, n=None, spk=None):
    rows.append(dict(id=i, value=round(v, 4) if isinstance(v, float) else v, lo=None if lo is None else round(lo, 4),
                     hi=None if hi is None else round(hi, 4), n=n, n_spk=spk, file=file, how=how))
    print(f"{i:32s} {v if not isinstance(v, float) else f'{v:.4f}'}" + (f" [{lo:.4f}, {hi:.4f}]" if lo is not None else "") + f"  n={n} spk={spk}  {file}")
m5 = lambda d, st: float(np.mean([auc(d.label, d[f"{st}_r{r}"]) for r in range(5)]))
# whole interview zero shot and probes
a2 = rd(F["A2"]); lo, hi, _ = boot(a2, "speaker", lambda d: auc(d.label, d.p_yes_whole))
put("edaic_whole_zs", auc(a2.label, a2.p_yes_whole), F["A2"], "AUC p_yes_whole", lo, hi, len(a2), a2.speaker.nunique())
a3 = rd(F["A3"])
for st, nm in [("oof_llm", "edaic_whole_llm_meanof5"), ("oof_enc", "edaic_whole_enc_meanof5"), ("oof_ans", "edaic_whole_ans_meanof5")]:
    put(nm, m5(a3, st), F["A3"], f"mean of five per-repeat AUCs of {st}_r0..r4", n=len(a3), spk=a3.speaker.nunique())
f = lambda d: m5(d, "oof_ans") - auc(d.label, d.p_yes_whole); lo, hi, _ = boot(a3, "speaker", f)
put("edaic_whole_ans_minus_zs", f(a3), F["A3"], "answer-state mean of five minus p_yes_whole, paired", lo, hi, len(a3), a3.speaker.nunique())
# PART 24 fine tune
f0 = rd(F["FT0"]); f1 = rd(F["FT1"])
lo, hi, _ = boot(f0, "pid", lambda d: auc(d.label, d.p_yes))
put("edaic_ft_seed0", auc(f0.label, f0.p_yes), F["FT0"], "AUC p_yes pooled OOF", lo, hi, len(f0), f0.pid.nunique())
j = f0.merge(a2[["pid", "p_yes_whole"]], on="pid"); f = lambda d: auc(d.label, d.p_yes) - auc(d.label, d.p_yes_whole); lo, hi, _ = boot(j, "pid", f)
put("edaic_ft_seed0_minus_zs", f(j), F["FT0"] + " + A2", "paired, join on pid", lo, hi, len(j), j.pid.nunique())
lo, hi, _ = boot(f1, "pid", lambda d: auc(d.label, d.p_yes))
put("edaic_ft_seed1", auc(f1.label, f1.p_yes), F["FT1"], "AUC p_yes pooled OOF", lo, hi, len(f1), f1.pid.nunique())
# window answers
for k, nm in [("ZS30", "edaic_zs_first30"), ("ZSMID30", "edaic_zs_mid30"), ("ZSMID300", "edaic_zs_mid300"), ("ZS300", "edaic_zs_first300")]:
    d = rd(F[k]); lo, hi, _ = boot(d, "speaker", lambda x: auc(x.label, x.p_yes))
    put(nm, auc(d.label, d.p_yes), F[k], "AUC p_yes", lo, hi, len(d), d.speaker.nunique())
# encoder probe per window
for k in ["NR30_MAC", "NR30_GD", "NRMID30", "NRMID300"]:
    js = json.load(open(F[k])); e = js["enc"]
    put(f"edaic_enc_{k}", float(np.mean(e["per_repeat"])), F[k], f"mean of enc.per_repeat {e['per_repeat']}", n=e.get("n"))
t7 = rd(F["T7B"]); put("edaic_enc_first300", m5(t7, "oof_enc"), F["T7B"], "mean of five oof_enc", n=len(t7))
# projector trained on the E-DAIC pairs
ft = rd(F["FT966"]); c = ft[ft.arm == "conflict"]; ag = ft[ft.arm == "agreement"].drop_duplicates("seg_uid")
lo, hi, _ = boot(c, "speaker", lambda d: auc(d.label, d.p_yes))
put("pairs_ft_conflict", auc(c.label, c.p_yes), F["FT966"], "AUC p_yes, 483 conflict rows", lo, hi, len(c), c.speaker.nunique())
lo, hi, _ = boot(ag, "speaker", lambda d: auc(d.label, d.p_yes))
put("pairs_ft_agree_distinct", auc(ag.label, ag.p_yes), F["FT966"], "AUC p_yes, distinct agreement seg_uid", lo, hi, len(ag), ag.speaker.nunique())
t1 = rd(F["T1"]); ta = t1[t1.set == "agreement"]
lo, hi, _ = boot(ta, "speaker_id", lambda d: auc(d.label, d.p_o25))
put("pairs_zs_agree_distinct", auc(ta.label, ta.p_o25), F["T1"], "AUC p_o25, 311 distinct agreement", lo, hi, len(ta), ta.speaker_id.nunique())
pz = rd(F["P14O25"]); pz["arm"] = pz["clip"].astype(str).str.split("_").str[0]; pc = pz[pz.arm == "conflict"]
lo, hi, _ = boot(pc, "speaker", lambda d: auc(d.label, d.p_yes))
put("pairs_zs_conflict", auc(pc.label, pc.p_yes), F["P14O25"], "AUC p_yes, conflict rows", lo, hi, len(pc), pc.speaker.nunique())
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); print("wrote", sys.argv[1])
