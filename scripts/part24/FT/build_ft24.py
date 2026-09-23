"""PART24 item 1, collect step. Builds the per-clip csvs, sidecars and FT24_values.tsv from the pulled pod outputs.

Reads only part24/pull/<pod>/out/ (pulled by watchdog24.sh) plus the whole-interview zero-shot per-clip file from PART23.
Writes only under part24/FT/.
  FT24_whole_seed0_perclip.csv      seed 0, all five folds (p24-ft0..4)
  FT24_whole_seedrule_perclip.csv   seed 0 with each fold whose seed 0 median answer_mass < 0.5 replaced by its seed 1 run
                                    (only written when at least one fold triggered the rule)
  FT24_whole_seed1all_perclip.csv   EXTRA, seed 1 on every fold from the pre-emptive p24s1 pods (only when all 5 exist);
                                    not the reported number
  FT24_values.tsv                   id value lo hi n file
AUC: sklearn roc_auc_score. Bootstrap: 2000 draws, a fresh numpy default_rng(0) per cell,
idx = rng.choice(N, size=N, replace=True) over speakers in sorted order (E-DAIC: one clip = one speaker),
2.5 and 97.5 percentiles, draws with one class only are skipped (count recorded). Paired: both terms use the same idx.
usage: /usr/local/bin/python3 build_ft24.py
"""
import os, sys, csv, json, re, hashlib, datetime, platform
import numpy as np, sklearn
from sklearn.metrics import roc_auc_score

R = os.environ.get("P24ROOT", "<local data dir>/part24")
PULL, OUT = f"{R}/pull", f"{R}/FT"
ZS = "<local data dir>/part23/pull/p23-a-probes/p23a/out/A2_edaic_whole_zeroshot_perclip.csv"
FOLDF = f"folds/edaic_groupkfold5_pod.csv"
MANF = f"manifests/part24/mf_whole.csv"
NB = 2000
COLS = ["pid", "label", "fold", "seed", "p_yes", "answer_mass", "audio_tok"]
BOOT_RULE = ("2000 draws; each cell a fresh numpy default_rng(0); idx = rng.choice(N, size=N, replace=True) over speakers "
             "(speakers in sorted string order; E-DAIC: one clip = one speaker); 2.5 and 97.5 percentiles (numpy default linear); "
             "draws with a single class are skipped and counted; paired = both terms use the same idx")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def rd(p):
    with open(p, newline="") as fh: return list(csv.DictReader(fh))


def unwrap(s):
    return float(re.sub(r"^np\.float64\((.*)\)$", r"\1", str(s).strip()))


pods = {}
for line in open(f"{R}/pods24.txt"):
    f = line.split()
    if len(f) >= 2: pods[f[0]] = f[1]

fold_true = {r["clip_id"][:-4]: int(r["fold"]) for r in rd(FOLDF)}
lab_true = {r["pid"]: int(r["label"]) for r in rd(MANF)}
assert len(fold_true) == 275 and len(lab_true) == 275


def load_fold(pod, k, s):
    base = f"{PULL}/{pod}/out/FT_whole_s{s}_fold{k}_oof"
    if not (os.path.exists(base + ".csv") and os.path.exists(base + ".sidecar.json")): return None
    rows = rd(base + ".csv"); side = json.load(open(base + ".sidecar.json"))
    assert len(rows) == 55, (pod, k, s, len(rows))
    for r in rows:
        assert int(r["fold"]) == k and int(r["seed"]) == s, (pod, r["pid"])
        assert fold_true[r["pid"]] == k, ("fold mismatch vs fold file", pod, r["pid"])
        assert int(r["label"]) == lab_true[r["pid"]], ("label mismatch vs manifest", pod, r["pid"])
        assert r["speaker"] == r["pid"]
    return {"pod": pod, "pod_id": pods.get(pod), "fold": k, "seed": s, "rows": rows, "side": side,
            "csv": base + ".csv", "csv_sha256": sha(base + ".csv"),
            "sidecar": base + ".sidecar.json", "sidecar_sha256": sha(base + ".sidecar.json"),
            "median_mass": float(np.median([float(r["answer_mass"]) for r in rows]))}


s0 = {k: load_fold(f"p24-ft{k}", k, 0) for k in range(5)}
missing = [k for k in range(5) if s0[k] is None]
if missing: sys.exit(f"seed 0 missing for folds {missing}; nothing written")
s1_pod = {k: load_fold(f"p24-ft{k}", k, 1) for k in range(5)}      # on-pod mass-rule rerun
s1_pre = {k: load_fold(f"p24s1-ft{k}", k, 1) for k in range(5)}    # pre-emptive seed 1 fleet

# mass rule, recomputed here and cross-checked against the pod's own MASS_CHECK file
rerun = []
mass_check = {}
for k in range(5):
    mc = f"{PULL}/p24-ft{k}/out/MASS_CHECK_fold{k}.txt"
    txt = open(mc).read() if os.path.exists(mc) else ""
    mass_check[k] = txt.strip()
    trig = s0[k]["median_mass"] < 0.5
    if txt:
        assert ("decision rerun_seed1" in txt) == trig, ("mass rule disagrees with pod", k, txt, s0[k]["median_mass"])
    if trig: rerun.append(k)

seedrule_src = {}
for k in rerun:
    if s1_pod[k] is not None: seedrule_src[k] = s1_pod[k]
    elif s1_pre[k] is not None: seedrule_src[k] = s1_pre[k]
    else: sys.exit(f"fold {k} triggered the mass rule but no seed 1 output was pulled")


def perclip(parts):
    out = []
    for p in parts:
        for r in p["rows"]:
            out.append({c: r[c] for c in COLS})
    out.sort(key=lambda r: r["pid"])
    assert len(out) == 275 and len({r["pid"] for r in out}) == 275
    return out


def gpu_of(side): return side.get("versions", {}).get("gpu")


def write_set(name, parts, what, extra):
    rows = perclip(parts)
    path = f"{OUT}/{name}_perclip.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    sides = [p["side"] for p in parts]
    snaps = sorted({s.get("hf_snapshot") for s in sides}); mids = sorted({s.get("model_id") for s in sides})
    meta = {"what": what, "result_file": os.path.basename(path), "n": len(rows), "n_speakers": len({r['pid'] for r in rows}),
            "n_positive": int(sum(int(r["label"]) for r in rows)),
            "columns": {"p_yes": "two-logit softmax over ' Yes'(7414) and ' No'(2308) at the answer position, element 0 (recipe)",
                        "answer_mass": "P(Yes set)+P(No set), full-vocabulary softmax, Yes {7414,9454,9693,9834,14004} No {902,2152,2308,2753,8996}",
                        "audio_tok": "count of audio placeholder tokens in the test input"},
            "model_id": mids, "hf_snapshot": snaps,
            "seeds_by_fold": {str(p["fold"]): p["seed"] for p in parts},
            "folds": [{"fold": p["fold"], "seed": p["seed"], "pod": p["pod"], "pod_id": p["pod_id"],
                       "command": p["side"].get("command"), "gpu": gpu_of(p["side"]), "versions": p["side"].get("versions"),
                       "env": p["side"].get("env"), "hf_snapshot": p["side"].get("hf_snapshot"),
                       "median_answer_mass": p["median_mass"], "pod_fold_auc_p_yes": p["side"].get("fold_auc_p_yes_two_logit"),
                       "losses_per_epoch": p["side"].get("losses_per_epoch"), "train_minutes": p["side"].get("train_minutes"),
                       "restarts_from_checkpoint": p["side"].get("restarts_from_checkpoint"),
                       "order_sha256_per_epoch": p["side"].get("order_sha256_per_epoch"),
                       "script_sha256": (p["side"].get("sources", {}).get("script") or [None, None])[1],
                       "cut_windows_sha256_of_list": p["side"].get("sources", {}).get("cut_windows_sha256_of_list"),
                       "source_csv": p["csv"], "source_csv_sha256": p["csv_sha256"],
                       "source_sidecar": p["sidecar"], "source_sidecar_sha256": p["sidecar_sha256"]} for p in parts],
            "auc_rule": "sklearn.metrics.roc_auc_score on p_yes", "bootstrap_rule": BOOT_RULE,
            "fold_file": [FOLDF, sha(FOLDF)], "manifest": [MANF, sha(MANF)],
            "built_by": [os.path.abspath(__file__), sha(os.path.abspath(__file__))],
            "build_command": "/usr/local/bin/python3 " + os.path.abspath(__file__),
            "build_versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                               "platform": platform.platform()},
            "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    meta.update(extra)
    json.dump(meta, open(path[:-4] + ".sidecar.json", "w"), indent=1)
    return path, rows


def spk_order(pids):
    order = sorted(pids)                    # string sort; one clip per speaker
    pos = {p: i for i, p in enumerate(pids)}
    return np.array([pos[p] for p in order])


def boot(y, a, b=None):
    """value, lo, hi, usable. b given -> paired AUC(a) - AUC(b)."""
    y = np.asarray(y, int); a = np.asarray(a, float); b = None if b is None else np.asarray(b, float)
    val = roc_auc_score(y, a) - (0.0 if b is None else roc_auc_score(y, b))
    rng = np.random.default_rng(0); N = len(y); out = []
    for _ in range(NB):
        idx = rng.choice(N, size=N, replace=True)
        yy = y[idx]
        if yy.min() == yy.max(): continue
        v = roc_auc_score(yy, a[idx])
        if b is not None: v -= roc_auc_score(yy, b[idx])
        out.append(v)
    return float(val), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


zs = {r["pid"]: r for r in rd(ZS)}
assert len(zs) == 275
vals = []


def fmt(x): return "" if x is None else repr(float(x))


def values_for(tag, path, rows):
    fn = os.path.basename(path)
    pids = [r["pid"] for r in rows]
    oi = spk_order(pids)                       # rows re-ordered so that row i is the i-th speaker in sorted order
    y = np.array([int(rows[i]["label"]) for i in oi]); p = np.array([float(rows[i]["p_yes"]) for i in oi])
    for i in oi: assert int(zs[rows[i]["pid"]]["label"]) == int(rows[i]["label"]), "label mismatch vs zero-shot file"
    z = np.array([unwrap(zs[rows[i]["pid"]]["p_yes_whole"]) for i in oi])
    v, lo, hi, u = boot(y, p); vals.append((f"{tag}_pooled_oof_auc", v, lo, hi, len(y), fn, u))
    v, lo, hi, u = boot(y, p, z); vals.append((f"{tag}_ft_minus_zs_whole_paired", v, lo, hi, len(y), fn + " ; " + os.path.basename(ZS), u))
    for k in range(5):
        sel = [i for i in oi if int(rows[i]["fold"]) == k]
        yk = np.array([int(rows[i]["label"]) for i in sel]); pk = np.array([float(rows[i]["p_yes"]) for i in sel])
        mk = np.array([float(rows[i]["answer_mass"]) for i in sel])
        v, lo, hi, u = boot(yk, pk); vals.append((f"{tag}_fold{k}_auc", v, lo, hi, len(yk), fn, u))
        vals.append((f"{tag}_fold{k}_median_answer_mass", float(np.median(mk)), None, None, len(yk), fn, None))
    return y, z


os.makedirs(OUT, exist_ok=True)
p0, r0 = write_set("FT24_whole_seed0", [s0[k] for k in range(5)],
                   "Qwen2.5-Omni projector fine-tune, E-DAIC whole interview (275 windows, processor cap lifted), seed 0 on all 5 outer folds, out of fold",
                   {"seed": 0, "seed_rule_folds_triggered": rerun,
                    "mass_rule": "a fold is rerun with seed 1 when its seed 0 median answer_mass over its 55 test clips is < 0.5",
                    "mass_check_files": mass_check})
y0, z0 = values_for("seed0", p0, r0)
zv, zlo, zhi, zu = boot(y0, z0)
vals.append(("ref_zs_whole_auc", zv, zlo, zhi, len(y0), os.path.basename(ZS), zu))

if rerun:
    parts = [seedrule_src[k] if k in seedrule_src else s0[k] for k in range(5)]
    pr, rr = write_set("FT24_whole_seedrule", parts,
                       "Qwen2.5-Omni projector fine-tune, E-DAIC whole interview, seed 0 with each fold whose seed 0 median answer_mass < 0.5 replaced by its seed 1 run",
                       {"seed": "0, and 1 on folds " + ",".join(map(str, rerun)), "folds_replaced_by_seed1": rerun,
                        "seed1_source_by_fold": {str(k): seedrule_src[k]["pod"] for k in rerun},
                        "seed1_source_rule": "the on-pod mass-rule rerun on p24-ft{k} when present, else the pre-emptive p24s1-ft{k} run",
                        "seed0_median_answer_mass_by_fold": {str(k): s0[k]["median_mass"] for k in range(5)},
                        "mass_rule": "a fold is rerun with seed 1 when its seed 0 median answer_mass over its 55 test clips is < 0.5"})
    values_for("seedrule", pr, rr)

if all(s1_pre[k] is not None for k in range(5)):
    pe, re_ = write_set("FT24_whole_seed1all", [s1_pre[k] for k in range(5)],
                        "EXTRA, not the reported number: seed 1 on all 5 folds from the pre-emptive p24s1 pods (see part24/SEED1_PREEMPTIVE.txt)",
                        {"seed": 1, "note": "pre-emptive seed 1 fleet; only folds that triggered the mass rule enter the seed-rule set"})
    values_for("extra_seed1all", pe, re_)

# seed 1 reproducibility where both an on-pod rerun and a pre-emptive run exist
repro = {}
for k in range(5):
    if s1_pod[k] is not None and s1_pre[k] is not None:
        a = {r["pid"]: float(r["p_yes"]) for r in s1_pod[k]["rows"]}; b = {r["pid"]: float(r["p_yes"]) for r in s1_pre[k]["rows"]}
        repro[k] = max(abs(a[p] - b[p]) for p in a)

with open(f"{OUT}/FT24_values.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["id", "value", "lo", "hi", "n", "file"])
    for (i, v, lo, hi, n, f, u) in vals: w.writerow([i, fmt(v), fmt(lo), fmt(hi), n, f])
json.dump({"what": "sidecar for FT24_values.tsv", "bootstrap_rule": BOOT_RULE, "auc_rule": "sklearn.metrics.roc_auc_score",
           "usable_draws": {i: u for (i, v, lo, hi, n, f, u) in vals if u is not None},
           "zero_shot_file": [ZS, sha(ZS)], "zero_shot_column": "p_yes_whole (np.float64(...) wrappers stripped)",
           "seed_rule_folds_triggered": rerun, "seed1_repro_max_abs_p_yes_diff_onpod_vs_preemptive": repro,
           "built_by": [os.path.abspath(__file__), sha(os.path.abspath(__file__))],
           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
          open(f"{OUT}/FT24_values.sidecar.json", "w"), indent=1)
for (i, v, lo, hi, n, f, u) in vals:
    print(f"{i}\t{v:.4f}\t" + ("" if lo is None else f"[{lo:.4f}, {hi:.4f}]") + f"\tn={n}")
print("rerun folds:", rerun, "repro:", repro)
