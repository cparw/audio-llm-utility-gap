"""Print paste lines and append provisional rows from a stats_p20 sidecar.
usage: report_p20.py SIDECAR ID_PREFIX RUN_LABEL REF_LABEL"""
import sys, json, os
SC, IDP, RUN, REF = sys.argv[1:5]
ROWS = os.environ.get("ROWS_TSV") or "reports/part20/rows/POD1_provisional.tsv"
d = json.load(open(SC)); res = d["results"]; f = d["per_clip_csv"]
def two(v, lo, hi): return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
def vs_half(lo, hi, v=None):
    if lo > 0.5: return "above 0.5"
    if hi < 0.5: return "below 0.5"
    return "interval includes 0.5" + ("" if v is None else f" (point {'above' if v > 0.5 else 'below'} 0.5)")
def vs_zero(lo, hi):
    return "interval excludes zero (above 0)" if lo > 0 else ("interval excludes zero (below 0)" if hi < 0 else "interval includes zero")
new = not os.path.exists(ROWS)
out = open(ROWS, "a")
if new: out.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\ttwo_dp\tvs_half\n")
for cell in ("all", "conflict", "agreement"):
    r = res[cell]; v = r["balanced"]; lo, hi = r["balanced_ci"]
    print(f"{RUN} AUC {cell}: {v:.4f} [{lo:.4f}, {hi:.4f}] n={r['n']} n_spk={r['n_speakers']} file={f} | "
          f"{two(v, lo, hi)} {vs_half(lo, hi, v)} | beside {REF} {r['standard']:.4f} [{r['standard_ci'][0]:.4f}, {r['standard_ci'][1]:.4f}]")
    if not os.environ.get("SKIP_CELLS"): out.write(f"{IDP}.{cell}\t{RUN} AUC {cell}\t{v:.4f}\t{lo:.4f}\t{hi:.4f}\t{r['n']}\t{r['n_speakers']}\t{f}\t{two(v, lo, hi)}\t{vs_half(lo, hi, v)}\n")
for cell in ("conflict", "all", "agreement"):
    r = res[cell]; v = r["diff_bal_minus_std"]; lo, hi = r["diff_ci"]
    print(f"PAIRED {RUN} minus {REF} {cell}: {v:+.4f} [{lo:+.4f}, {hi:+.4f}] n={r['n']} n_spk={r['n_speakers']} file={f} | "
          f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}] {vs_zero(lo, hi)}")
    out.write(f"{IDP}.diff_vs_{REF.replace(' ', '_')}.{cell}\tPAIRED {RUN} minus {REF} AUC {cell}\t{v:.4f}\t{lo:.4f}\t{hi:.4f}\t{r['n']}\t"
              f"{r['n_speakers']}\t{f}\t{v:+.2f} [{lo:+.2f}, {hi:+.2f}]\t{vs_zero(lo, hi)}\n")
for cell in ("all", "conflict", "agreement"):
    r = res[cell]; v = r["balanced_multivariant"]; lo, hi = r["balanced_multivariant_ci"]
    print(f"{RUN} AUC {cell} (rule-3 multivariant p_yes): {v:.4f} [{lo:.4f}, {hi:.4f}] n={r['n']} n_spk={r['n_speakers']} | {two(v, lo, hi)} {vs_half(lo, hi, v)}")
g = d["extra_conflict_minus_agreement_gap"]
print(f"GAP conflict minus agreement: {RUN} {g['balanced_gap']:+.4f} [{g['balanced_gap_ci'][0]:+.4f}, {g['balanced_gap_ci'][1]:+.4f}]; "
      f"{REF} {g['standard_gap']:+.4f} [{g['standard_gap_ci'][0]:+.4f}, {g['standard_gap_ci'][1]:+.4f}]; change {g['gap_change']:+.4f} "
      f"[{g['gap_change_ci'][0]:+.4f}, {g['gap_change_ci'][1]:+.4f}] {vs_zero(*g['gap_change_ci'])}")
print(f"answer_mass median {d['answer_mass_median']:.4f} min {d['answer_mass_min']:.4f}")
out.close()
