"""accel_24sep othermodels: pick one probe per cell from om_cells.tsv, compare with the TSV, count."""
import json, os
import numpy as np, pandas as pd

OUT = "checks/accel_24sep/othermodels"
TSV = "scores/part25/C/C_other_models.tsv"
d = pd.read_csv(f"{OUT}/om_cells.tsv", sep="\t")
T = pd.read_csv(TSV, sep="\t")
lines = []
P = lambda s: lines.append(s)

# 1) zero-shot
P("ZERO-SHOT")
zs = d.drop_duplicates(["model", "dataset"])
for _, r in zs.iterrows():
    P(f"  {r['model']:<19} {r['dataset']:<12} tsv {r['tsv_zs_master']:.4f}  recomputed {r['zs_recomputed']:.4f}  n={r['zs_n']}  match={r['zs_matches']}")
P(f"  zero-shot matches: {int((zs['zs_matches']=='yes').sum())} of {len(zs)}")

# 2) TSV interval reproduced from the TSV's own probe source
P("TSV INTERVAL REPRODUCED FROM ITS OWN PROBE SOURCE")
for _, t in T.iterrows():
    if not isinstance(t["probe_perclip_source"], str):
        P(f"  {t['model']:<19} {t['dataset']:<12} TSV has no probe per-clip source and no interval"); continue
    m = d[(d.model == t.model) & (d.dataset == t.dataset) & (d.probe_file == t.probe_perclip_source)]
    if m.empty:
        P(f"  {t['model']:<19} {t['dataset']:<12} probe source not among the candidates: {t.probe_perclip_source}"); continue
    r = m.iloc[0]
    ok = (round(r.point_diff, 4) == round(t.diff_point_from_perclip, 4)) and (r.diff_lo == round(t.diff_lo, 4)) and (r.diff_hi == round(t.diff_hi, 4))
    if os.path.isfile(t.draws_file):
        cd = np.load(t.draws_file); md = np.load(f"{OUT}/draws/{r.key}.npz")
        a, b = cd["diff"], md["diff"]; both = ~np.isnan(a) & ~np.isnan(b)
        mx = float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("nan")
    else:
        mx = float("nan")  # track C removed this draws file after the first run (00:27 UTC run gave 0.0)
    P(f"  {t['model']:<19} {t['dataset']:<12} TSV {t.diff_point_from_perclip:+.4f} [{t.diff_lo:+.4f}, {t.diff_hi:+.4f}]  indep {r.point_diff:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]  agree={'yes' if ok else 'NO'}  draws max abs diff {mx:.1e}")


# 3) primary probe per cell
def rank(r):
    f = r.probe_file
    if r.kind == "rep5" and r.per_repeat_matches_json == "yes" and "superseded" not in f: return 0
    if r.kind == "rep5" and r.probe_mean_matches_master == "yes" and "superseded" not in f: return 1
    if r.kind == "rep5": return 2
    return 3


d["rank"] = d.apply(rank, axis=1)
d["is_out_dup"] = d.probe_file.str.contains("/POD3/out/") | d.probe_file.str.contains("podD2_final")
prim = d.sort_values(["rank", "is_out_dup"]).drop_duplicates(["model", "dataset"])
order = {k: i for i, k in enumerate(T.model.unique())}; dorder = {k: i for i, k in enumerate(T.dataset.unique())}
prim = prim.assign(o1=prim.model.map(order), o2=prim.dataset.map(dorder)).sort_values(["o1", "o2"])
lab = {0: "exact five-repeat OOF, per-repeat AUCs reproduce master", 1: "five-repeat OOF, mean reproduces master, per-repeat do not match the json",
       2: "five-repeat OOF from a superseded run, does NOT reproduce master", 3: "SINGLE-SPLIT OOF only"}
P("PRIMARY PROBE PER CELL (best available per-clip file)")
for _, r in prim.iterrows():
    P(f"  {r['model']:<19} {r['dataset']:<12} probe {r.probe_recomputed:.4f} (master {r.tsv_probe_master:.4f})  answer {r.zs_recomputed:.4f}  diff {r.point_diff:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]  excl0={r.excludes_zero}  | {lab[r['rank']]} | {r.probe_file}")
prim.drop(columns=["o1", "o2"]).to_csv(f"{OUT}/om_primary.tsv", sep="\t", index=False)

pdad = prim[prim.condition.isin(["PD", "AD"])]
P("COUNTS, 18 PD/AD cells, primary probe")
P(f"  probe above answer: {int((pdad.point_diff > 0).sum())} of {len(pdad)}")
P(f"  interval excludes 0 (and probe above): {int(((pdad.diff_lo > 0)).sum())} of {len(pdad)}")
P(f"  interval excludes 0 on the answer side: {int(((pdad.diff_hi < 0)).sum())}")
for mdl in T.model.unique():
    s = pdad[pdad.model == mdl]
    P(f"    {mdl:<19} above {int((s.point_diff > 0).sum())}/6  excl0 {int((s.diff_lo > 0).sum())}/6  ranks {sorted(s['rank'].tolist())}")
ex = pdad[pdad["rank"] == 0]
P(f"  restricted to cells whose five-repeat OOF reproduces master per-repeat exactly ({len(ex)} cells): above {int((ex.point_diff > 0).sum())}, excl0 {int((ex.diff_lo > 0).sum())}")
P("  TSV's own claim: above 17 of 18, interval excluding 0 in 10")
P("E-DAIC per model (primary probe)")
for _, r in prim[prim.dataset == "E-DAIC"].iterrows():
    P(f"  {r['model']:<19} probe {r.probe_recomputed:.4f}  answer {r.zs_recomputed:.4f}  diff {r.point_diff:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]  excl0={r.excludes_zero}  | {lab[r['rank']]}")
txt = "\n".join(lines)
open(f"{OUT}/om_summary.txt", "w").write(txt + "\n")
print(txt)
