"""PART17 T7b: E-DAIC 300 s window probe, mean of five per-repeat AUCs, with a speaker bootstrap
and the PAIRED difference against the Qwen2.5-Omni zero-shot answer.

Input vectors: edaic_full_oof_repeats.npz written by the UNMODIFIED part16/EDAICFULL/probe_boot.py
(sha256 4ecca1f8...) rerun on a Linux x86_64 pod with scikit-learn 1.9.1 / numpy 2.1.2 / scipy 1.18.1.

Step 1, reproduction gate (no interval is written unless it passes):
  per-repeat AUCs recomputed from the npz, rounded to 4 dp, must equal edaic_full_nested_repeats.json;
  layers_picked of the rerun json must equal the published json; clip order, labels and p_yes must
  equal the published per-clip csv; the mean over the five OOF vectors must equal the published
  oof_<stream> columns.
Step 2, bootstrap: 2000 draws, a fresh numpy.random.default_rng(0) per cell, speakers resampled with
replacement using the same draw code as probe_boot.py boot() (rng.choice over sorted unique speakers),
percentiles 2.5 / 97.5. Mean-of-five cell: inside every draw all five per-repeat AUCs are recomputed
on the drawn speakers and averaged. Paired cell: the same draw also recomputes the answer AUC and
the statistic is (mean of five) minus (answer). The not-used convention (AUC of the averaged OOF) is
recomputed the same way for the side-by-side only.
"""
import argparse, csv, hashlib, json, os, platform, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--rerun_json", required=True, help="edaic_full_nested_repeats.json written by the rerun")
ap.add_argument("--ref_json", required=True, help="published edaic_full_nested_repeats.json")
ap.add_argument("--ref_perclip", required=True, help="published edaic_full_perclip.csv (p_yes_answer)")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--file_label", required=True, help="path written in the file column (Mac location of the per-clip csv)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

NB = 2000
STREAMS = [("llm", "LM probe"), ("enc", "encoder probe"), ("ans", "answer-state probe"), ("proj", "projector probe")]
WIN = "E-DAIC 300 s window"


def auc_rank(y, s):  # identical to probe_boot.py
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); sv = s[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j+1] == sv[i]: j += 1
        r[o[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[y == 1].sum() - n1*(n1+1)/2.0) / (n1*n0))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# ---------------- load ----------------
z = np.load(args.npz)
V = {k: np.asarray(z[f"{k}_oof_repeats"], float) for k, _ in STREAMS}
y = np.asarray(z["label"]).astype(int)
spk = np.asarray(z["speaker"])
clip = [str(c) for c in z["clip"]]
pz_npz = np.asarray(z["p_yes"], float)
pub = json.load(open(args.ref_json))
rer = json.load(open(args.rerun_json))
perclip = list(csv.DictReader(open(args.ref_perclip)))
pid = [r["pid"] for r in perclip]
pz = np.array([float(r["p_yes_answer"]) for r in perclip])
y_pc = np.array([int(r["label"]) for r in perclip])

# ---------------- gate ----------------
gate = dict(checks={})
gate["checks"]["clip_order_equal"] = [c.replace(".wav", "") for c in clip] == pid
gate["checks"]["labels_equal"] = bool(np.array_equal(y, y_pc))
gate["checks"]["p_yes_npz_vs_perclip_maxabs"] = float(np.max(np.abs(pz_npz - pz)))
gate["checks"]["one_clip_per_speaker"] = len(set(spk)) == len(y)
gate["checks"]["n"], gate["checks"]["n_spk"], gate["checks"]["n_pos"] = int(len(y)), int(len(set(spk))), int(y.sum())
ok = all([gate["checks"]["clip_order_equal"], gate["checks"]["labels_equal"],
          gate["checks"]["p_yes_npz_vs_perclip_maxabs"] < 1e-12, gate["checks"]["one_clip_per_speaker"]])
per_full = {}
for k, _ in STREAMS:
    per = [auc_rank(y, v) for v in V[k]]; per_full[k] = per
    mine4 = [round(a, 4) for a in per]
    pubcol = np.array([float(r[f"oof_{k}"]) for r in perclip])
    g = dict(per_repeat_full_precision=per, per_repeat_4dp_rerun=mine4, per_repeat_4dp_published=pub[k]["per_repeat"],
             per_repeat_match_4dp=mine4 == pub[k]["per_repeat"],
             per_repeat_rerun_json=rer[k]["per_repeat"],
             layers_rerun=rer[k]["layers_picked"], layers_published=pub[k]["layers_picked"],
             layers_match=rer[k]["layers_picked"] == pub[k]["layers_picked"],
             mean_of_repeat_aucs_full_precision=float(np.mean(per)),
             mean_of_4dp_repeat_aucs_rounded=round(float(np.mean(mine4)), 4),
             mean_of_repeat_aucs_published=pub[k]["mean_of_repeat_aucs"],
             mean_match_4dp=round(float(np.mean(mine4)), 4) == pub[k]["mean_of_repeat_aucs"],
             auc_of_mean_oof_rerun=auc_rank(y, V[k].mean(axis=0)), auc_of_mean_oof_published=pub[k]["auc_of_mean_oof"],
             max_abs_mean_oof_vs_published_perclip=float(np.max(np.abs(V[k].mean(axis=0) - pubcol))))
    g["reproduced"] = bool(g["per_repeat_match_4dp"] and g["layers_match"] and g["mean_match_4dp"]
                           and g["per_repeat_rerun_json"] == pub[k]["per_repeat"]
                           and g["max_abs_mean_oof_vs_published_perclip"] < 1e-9)
    gate[k] = g
    ok = ok and g["reproduced"]
    print(f"GATE {k:4s} rerun {mine4} pub {pub[k]['per_repeat']}  layers match {g['layers_match']}  "
          f"mean {g['mean_of_4dp_repeat_aucs_rounded']:.4f} vs {pub[k]['mean_of_repeat_aucs']:.4f}  "
          f"max|meanOOF-pub| {g['max_abs_mean_oof_vs_published_perclip']:.1e}  -> {'PASS' if g['reproduced'] else 'FAIL'}", flush=True)
gate["passed"] = bool(ok)
json.dump(gate, open(f"{args.out_dir}/T7b_gate.json", "w"), indent=1)
print("GATE", "PASSED" if ok else "FAILED", flush=True)
if not ok:
    print("gate failed: no intervals written", flush=True); sys.exit(2)

# ---------------- bootstrap ----------------
uspk = np.array(sorted(set(spk)))
idx_of = {s: i for i, s in enumerate(spk)}


def boot(stat):
    """stat(ii, yy) -> float. Fresh default_rng(0) per cell, one speaker draw per replicate."""
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(NB):
        pick = rng.choice(len(uspk), size=len(uspk), replace=True)
        ii = np.array([idx_of[uspk[p]] for p in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        vals.append(stat(ii, yy))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def mean5(k):
    return lambda ii, yy: float(np.mean([auc_rank(yy, v[ii]) for v in V[k]]))


def mean5_minus_ans(k):
    return lambda ii, yy: float(np.mean([auc_rank(yy, v[ii]) for v in V[k]])) - auc_rank(yy, pz[ii])


rows = []
def add(rid, what, value, lo, hi, usable, ez=""):
    r = dict(id=rid, what=what, value=round(value, 4), lo=round(lo, 4), hi=round(hi, 4), n=int(len(y)),
             n_spk=int(len(uspk)), excludes_zero=ez, usable_draws=f"{usable}/{NB}", file=args.file_label,
             value_full=value, lo_full=lo, hi_full=hi)
    rows.append(r)
    print(f"{rid:44s} {value:.4f} [{lo:.4f}, {hi:.4f}] {('excl0 ' + ez) if ez else ''} usable {usable}/{NB}", flush=True)
    return r


a_zs = auc_rank(y, pz)
lo, hi, u = boot(lambda ii, yy: auc_rank(yy, pz[ii]))
add("T7b_zeroshot_answer", f"Qwen2.5-Omni zero-shot answer AUC, {WIN}", a_zs, lo, hi, u)

res = {}
for k, lab in STREAMS:
    m = float(np.mean(per_full[k]))
    lo, hi, u = boot(mean5(k))
    add(f"T7b_{k}_meanof5", f"{lab}, mean of five per-repeat AUCs {[round(a, 4) for a in per_full[k]]} "
        f"(nested GroupKFold(5) x 5 repeats, pod rerun reproduces published), {WIN}", m, lo, hi, u)
    dlo, dhi, du = boot(mean5_minus_ans(k))
    ez = "yes" if (dlo > 0 or dhi < 0) else "no"
    add(f"T7b_{k}_meanof5_minus_answer", f"{lab} mean of five per-repeat AUCs minus zero-shot answer AUC, paired "
        f"(one speaker draw per replicate, all five AUCs recomputed), {WIN}", m - a_zs, dlo, dhi, du, ez)
    res[k] = dict(meanof5=m, lo=lo, hi=hi, d=m - a_zs, dlo=dlo, dhi=dhi, ez=ez)

notused = {}
for k, lab in STREAMS:
    vm = V[k].mean(axis=0); a = auc_rank(y, vm)
    lo, hi, u = boot(lambda ii, yy, vm=vm: auc_rank(yy, vm[ii]))
    add(f"T7b_notused_{k}_aucofmeanoof", f"NOT USED: {lab} AUC of the OOF probabilities averaged over five repeats, {WIN}", a, lo, hi, u)
    dlo, dhi, du = boot(lambda ii, yy, vm=vm: auc_rank(yy, vm[ii]) - auc_rank(yy, pz[ii]))
    ez = "yes" if (dlo > 0 or dhi < 0) else "no"
    add(f"T7b_notused_{k}_aucofmeanoof_minus_answer", f"NOT USED: {lab} (AUC of averaged OOF) minus zero-shot answer, paired, {WIN}",
        a - a_zs, dlo, dhi, du, ez)
    notused[k] = dict(a=a, lo=lo, hi=hi, d=a - a_zs, dlo=dlo, dhi=dhi, ez=ez)

# ---------------- outputs ----------------
cols = ["id", "what", "value", "lo", "hi", "n", "n_spk", "excludes_zero", "usable_draws", "file", "value_full", "lo_full", "hi_full"]
with open(f"{args.out_dir}/T7b_edaic300_meanof5.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
with open(f"{args.out_dir}/T7b.tsv", "w", newline="") as fh:
    fh.write("id\twhat\tvalue\tlo\thi\tn\tn_spk\tfile\n")
    for r in rows:
        fh.write("\t".join(str(r[c]) for c in ["id", "what", "value", "lo", "hi", "n", "n_spk", "file"]) + "\n")
with open(f"{args.out_dir}/T7b_edaic300_meanof5_perclip.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    oc = [f"oof_{k}_r{r}" for k in ("enc", "llm", "ans", "proj") for r in range(5)]
    w.writerow(["pid", "label", "p_yes_answer"] + oc)
    for i in range(len(y)):
        w.writerow([pid[i], int(y[i]), repr(float(pz[i]))] + [repr(float(V[k][r][i])) for k in ("enc", "llm", "ans", "proj") for r in range(5)])

LAB = {"llm": "LM", "enc": "encoder", "ans": "answer-state", "proj": "projector"}
print("\n=== PASTE LINES ===")
paste = []
for k, _ in STREAMS:
    r = res[k]
    s = (f"T7b {LAB[k]} mean-of-5 {r['meanof5']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] | paired minus answer "
         f"{r['d']:.4f} [{r['dlo']:.4f}, {r['dhi']:.4f}] excludes zero {r['ez']} | n={len(y)} n_spk={len(uspk)} | {args.file_label}")
    paste.append(s); print(s)
print(f"T7b answer {a_zs:.4f} [{rows[0]['lo']:.4f}, {rows[0]['hi']:.4f}] | n={len(y)} n_spk={len(uspk)}")
print("\n=== SIDE BY SIDE: USED (mean of five per-repeat AUCs) vs NOT USED (AUC of averaged OOF) ===")
side = []
for k, _ in STREAMS:
    r, q = res[k], notused[k]
    s = (f"{LAB[k]:12s} USED {r['meanof5']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] paired {r['d']:.4f} [{r['dlo']:.4f}, {r['dhi']:.4f}] excl0 {r['ez']}"
         f"   | NOT USED {q['a']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}] paired {q['d']:.4f} [{q['dlo']:.4f}, {q['dhi']:.4f}] excl0 {q['ez']}")
    side.append(s); print(s)

import sklearn, scipy, joblib, threadpoolctl
env = dict(python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
           sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__, joblib=joblib.__version__,
           threadpoolctl=threadpoolctl.__version__, threadpool_info=threadpoolctl.threadpool_info(), cpu_count=os.cpu_count())
run = dict(env_of_this_bootstrap=env, inputs={p: sha256(p) for p in [args.npz, args.rerun_json, args.ref_json, args.ref_perclip]},
           script_sha256=sha256(os.path.abspath(__file__)), argv=sys.argv, paste_lines=paste, side_by_side=side,
           results={k: res[k] for k in res}, notused={k: notused[k] for k in notused}, answer_auc=a_zs,
           answer_ci=[rows[0]["lo_full"], rows[0]["hi_full"]])
json.dump(run, open(f"{args.out_dir}/T7b_boot_run.json", "w"), indent=1)
print("wrote", args.out_dir)
