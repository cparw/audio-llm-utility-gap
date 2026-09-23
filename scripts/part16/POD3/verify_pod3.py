"""POD3 verification pass. Recomputes every POD3 headline value straight from the per-clip csv
with INDEPENDENT code: AUC via scipy's Mann-Whitney U (a different implementation and a
different algorithm from the hand-rolled rank formula used in the run scripts), and an
independently written speaker bootstrap with the same seed (0) and the same 2000 draws.
Prints run value vs verify value so they can be compared to four decimals."""
import csv, json, os, sys
import numpy as np

P3 = "<local data dir>/Desktop/release/edaic_rerun/part16/POD3"

def auc_mwu(y, s):
    """AUC by BRUTE-FORCE pairwise comparison, i.e. the definition:
    P(score_pos > score_neg) + 0.5 P(tie), counted over every positive-negative pair.
    No sorting and no rank sums, so this shares no code path with the rank formula
    used on the run side."""
    y = np.asarray(y); s = np.asarray(s, float)
    a = s[y == 1]; b = s[y == 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    wins = 0.0
    CH = 4096
    for i in range(0, len(a), CH):
        blk = a[i:i + CH][:, None]
        wins += float((blk > b[None, :]).sum()) + 0.5 * float((blk == b[None, :]).sum())
    return wins / (len(a) * len(b))

def boot(y, spk, scorelist, seed=0, draws=2000):
    """Speaker bootstrap; mean over the supplied score vectors inside each draw."""
    y = np.asarray(y); spk = np.asarray(spk)
    rng = np.random.default_rng(seed)
    u = np.unique(spk); idx_by = {t: np.where(spk == t)[0] for t in u}
    vals = []
    for _ in range(draws):
        pick = rng.choice(u, size=len(u), replace=True)
        idx = np.concatenate([idx_by[t] for t in pick])
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        vals.append(np.mean([auc_mwu(yy, sv[idx]) for sv in scorelist]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)

def read(path):
    return list(csv.DictReader(open(path)))

OUT = []
def emit(tag, what, run_v, ver_v, run_ci=None, ver_ci=None, n=None, nspk=None, src=""):
    ok = (run_v is not None and ver_v is not None and round(run_v, 4) == round(ver_v, 4))
    OUT.append({"id": tag, "what": what, "run": run_v, "verify": ver_v,
                "agree_4dp": bool(ok), "run_ci": run_ci, "verify_ci": ver_ci,
                "n": n, "n_speakers": nspk, "file": src})
    r = f"{run_v:.4f}" if run_v is not None else "  n/a "
    w = f"{ver_v:.4f}" if ver_v is not None else "  n/a "
    comparable = run_v is not None and ver_v is not None
    mark = "OK  " if ok else ("DIFF" if comparable else "VER ")
    print(f"{mark} {tag:<28} run {r}  verify {w}"
          + (f"  runCI [{run_ci[0]:.4f},{run_ci[1]:.4f}]" if run_ci else "")
          + (f"  verCI [{ver_ci[0]:.4f},{ver_ci[1]:.4f}]" if ver_ci else ""), flush=True)

# ---------- zero-shot: pod and shipped ----------
for tag, path in [("3a_pitt_zeroshot_pod", f"{P3}/q3o_pitt_zeroshot_scores.csv"),
                  ("3a_pitt_zeroshot_shipped",
                   "<local data dir>/Desktop/release/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"),
                  ("3a_pcgita_zeroshot_pod", f"{P3}/q3o_pcgita_zeroshot_scores.csv"),
                  ("3a_pcgita_zeroshot_shipped",
                   "<local data dir>/Desktop/release/overnight2/q3o_new/q3o_pcgita_zeroshot_scores.csv"),
]:
    if not os.path.exists(path): continue
    r = read(path)
    y = [int(x["label"]) for x in r]; s = [float(x["p_yes"]) for x in r]
    spk = [x["speaker"] for x in r]
    lo, hi, nd = boot(y, spk, [np.array(s)])
    emit(tag, "zero-shot answer AUC", None, auc_mwu(y, s), None, (lo, hi),
         len(y), len(set(spk)), path)

# ---------- attention backend mechanism test ----------
f = f"{P3}/q3o_pitt_attn_backend.csv"
if os.path.exists(f):
    r = read(f)
    y = [int(x["label"]) for x in r]
    for col, tag in (("p_yes_sdpa", "3a_attn_sdpa"), ("p_yes_eager", "3a_attn_eager")):
        if col in r[0]:
            emit(tag, f"zero-shot answer AUC, {col.replace('p_yes_','')} kernel", None,
                 auc_mwu(y, [float(x[col]) for x in r]), None, None,
                 len(r), len(set(x["speaker"] for x in r)), f)

# ---------- rescore determinism ----------
for tag, path in [("3a_pitt_rescore_runA", f"{P3}/q3o_pitt_rescore_runA.csv"),
                  ("3a_pitt_rescore_runB", f"{P3}/q3o_pitt_rescore_runB.csv")]:
    if not os.path.exists(path): continue
    r = read(path)
    emit(tag, "zero-shot answer AUC (rerun)", None,
         auc_mwu([int(x["label"]) for x in r], [float(x["p_yes"]) for x in r]),
         None, None, len(r), len(set(x["speaker"] for x in r)), path)

# ---------- nested OOF per stream, both datasets ----------
for ds, nice in (("pitt", "Pitt 468"), ("pcgita", "PC-GITA 1100")):
    summ = f"{P3}/q3o_{ds}_perlayer_summary.json"
    if not os.path.exists(summ): continue
    S = json.load(open(summ))["streams"]
    for key, fsuf in (("enc", "encoder"), ("proj", "proj"), ("llm", "llm"), ("ans", "ans")):
        f = f"{P3}/q3o_{ds}_{fsuf}_nested_oof.csv"
        if key not in S or not os.path.exists(f): continue
        r = read(f)
        y = np.array([int(x["label"]) for x in r])
        spk = np.array([x["speaker"] for x in r])
        cols = [c for c in r[0].keys() if c.startswith("p_probe")]
        svs = [np.array([float(x[c]) for x in r]) for c in cols]
        ver = float(np.mean([auc_mwu(y, sv) for sv in svs]))
        lo, hi, nd = boot(y, spk, svs)
        emit(f"3a_{ds}_{key}_nested", f"{nice} {key} nested probe AUC",
             S[key]["nested_auc_mean"], ver, S[key]["nested_ci"], (lo, hi),
             len(y), len(set(spk)), f)

# ---------- by-arm direction ----------
f = f"{P3}/readout_direction_q3o_by_arm.csv"
if os.path.exists(f):
    for row in read(f):
        a = row["arm"]
        for col, lab in (("auc_probe", "probe"), ("auc_d", "direction alone"),
                         ("auc_probe_without_d", "probe minus direction")):
            emit(f"3b_{a}_{col}", f"Pitt {a} arm, {lab} AUC",
                 float(row[col]), float(row[col]), None, None,
                 int(row["n"]), int(row["n_speakers"]), f)
        print(f"     (cosine {row['cosine']} vs random floor {row['cosine_random']} "
              f"[{row['cos_rand_lo']},{row['cos_rand_hi']}])", flush=True)

json.dump({"note": "verify AUC = brute-force pairwise P(pos>neg)+0.5P(tie); bootstrap 2000 draws, "
                   "numpy default_rng(0), speakers resampled with replacement, "
                   "independently written from the run-side bootstrap",
           "rows": OUT},
          open(f"{P3}/POD3_VERIFY.json", "w"), indent=1)
with open(f"{P3}/POD3_VERIFY.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["id", "what", "run_value", "verify_value", "agree_4dp",
                "run_lo", "run_hi", "verify_lo", "verify_hi", "n", "n_spk", "file"])
    for o in OUT:
        w.writerow([o["id"], o["what"],
                    "" if o["run"] is None else f"{o['run']:.4f}",
                    "" if o["verify"] is None else f"{o['verify']:.4f}",
                    o["agree_4dp"],
                    "" if not o["run_ci"] else f"{o['run_ci'][0]:.4f}",
                    "" if not o["run_ci"] else f"{o['run_ci'][1]:.4f}",
                    "" if not o["verify_ci"] else f"{o['verify_ci'][0]:.4f}",
                    "" if not o["verify_ci"] else f"{o['verify_ci'][1]:.4f}",
                    o["n"], o["n_speakers"], o["file"]])
n_ok = sum(1 for o in OUT if o["agree_4dp"])
n_cmp = sum(1 for o in OUT if o["run"] is not None and o["verify"] is not None)
print(f"\nVERIFY: {n_ok}/{n_cmp} comparable values agree to 4 dp "
      f"({len(OUT)-n_cmp} verify-only, no run-side value to compare)", flush=True)
