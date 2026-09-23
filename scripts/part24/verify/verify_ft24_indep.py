#!/usr/local/bin/python3
# PART24 item 1, independent verifier. Written from the rules only; the collect step's code was not opened.
# Reads the per-clip csvs, the pod oof csvs, the fold file, the manifest, the PART23 files and the zero-shot csv.
# Writes part24/verify/FT24_VERIFIED.tsv (id, claimed, recomputed, verified) and a sidecar json.
import csv, glob, hashlib, json, os, platform, re, sys, datetime
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = "<local data dir>/release_from_mac/scores"
P24 = ROOT + "/part24"
P23 = ROOT + "/part23"
OUTDIR = P24 + "/verify"
VALUES = P24 + "/FT/FT24_values.tsv"
PERCLIP = {"seed0": P24 + "/FT/FT24_whole_seed0_perclip.csv", "seed1all": P24 + "/FT/FT24_whole_seed1all_perclip.csv"}
POD = {"seed0": (P24 + "/pull/p24-ft{k}/out/FT_whole_s0_fold{k}_oof.csv", 0),
       "seed1all": (P24 + "/pull/p24s1-ft{k}/out/FT_whole_s1_fold{k}_oof.csv", 1)}
FOLD_FILE = P24 + "/launch_record/scripts/stage/p23/edaic_groupkfold5_pod.csv"
MANIFEST = P24 + "/launch_record/scripts/stage/p23/mf_whole.csv"
ZS = P23 + "/A/A2_edaic_whole_zeroshot_perclip.csv"
ZS_SIDECAR = P23 + "/A/A2_edaic_whole_zeroshot_perclip.sidecar.json"
TOL = 1e-9
NDRAW = 2000

# The 4 dp numbers printed in the collect summary (copied from the task text).
SUMMARY = {
    "seed0_pooled_oof_auc": ("0.8589", "0.8050", "0.9029"),
    "seed0_ft_minus_zs_whole_paired": ("0.0223", "-0.0064", "0.0547"),
    "seed0_fold0_auc": ("0.9167", "0.8267", "0.9783"), "seed0_fold0_median_answer_mass": ("0.9939",),
    "seed0_fold1_auc": ("0.8625", "0.7334", "0.9613"), "seed0_fold1_median_answer_mass": ("0.9934",),
    "seed0_fold2_auc": ("0.8417", "0.6955", "0.9556"), "seed0_fold2_median_answer_mass": ("0.9972",),
    "seed0_fold3_auc": ("0.8372", "0.6955", "0.9464"), "seed0_fold3_median_answer_mass": ("0.9979",),
    "seed0_fold4_auc": ("0.8585", "0.7112", "0.9671"), "seed0_fold4_median_answer_mass": ("0.9970",),
    "ref_zs_whole_auc": ("0.8366", "0.7786", "0.8866"),
    "extra_seed1all_pooled_oof_auc": ("0.8184", "0.7496", "0.8758"),
    "extra_seed1all_ft_minus_zs_whole_paired": ("-0.0182", "-0.0519", "0.0139"),
    "extra_seed1all_fold0_auc": ("0.9225",), "extra_seed1all_fold1_auc": ("0.8117",),
    "extra_seed1all_fold2_auc": ("0.8400",), "extra_seed1all_fold3_auc": ("0.8421",),
    "extra_seed1all_fold4_auc": ("0.8353",),
    "extra_seed1all_fold0_median_answer_mass": ("0.9946",), "extra_seed1all_fold1_median_answer_mass": ("0.9947",),
    "extra_seed1all_fold2_median_answer_mass": ("0.9963",), "extra_seed1all_fold3_median_answer_mass": ("0.9962",),
    "extra_seed1all_fold4_median_answer_mass": ("0.9960",),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def num(s):
    m = re.fullmatch(r"\s*np\.float64\((.*)\)\s*", s)
    return float(m.group(1) if m else s)


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


out = []   # (id, claimed, recomputed, verified)
log = []


def add(i, claimed, recomputed, ok):
    out.append((i, str(claimed), str(recomputed), "yes" if ok else "no"))


def f4(x):
    return f"{x:.4f}"


# ---------- reference files ----------
fold_rows = read(FOLD_FILE)
fold = {}
bad_clip = 0
for r in fold_rows:
    s = int(r["speaker_id"])
    if r["clip_id"] != f"{s}.wav":
        bad_clip += 1
    assert s not in fold
    fold[s] = int(r["fold"])
mf = {int(r["pid"]): int(r["label"]) for r in read(MANIFEST)}
mf_spk_ok = all(r["pid"] == r["speaker"] for r in read(MANIFEST))
zs_rows = read(ZS)
zs = {int(r["pid"]): (int(r["label"]), num(r["p_yes_whole"])) for r in zs_rows}
assert len(zs) == len(zs_rows) == 275

# ---------- load per-clip csvs and pod csvs ----------
sets = {}
for key, path in PERCLIP.items():
    rows = read(path)
    d = {}
    dup = 0
    for r in rows:
        p = int(r["pid"])
        if p in d:
            dup += 1
        d[p] = dict(label=int(r["label"]), fold=int(r["fold"]), seed=int(r["seed"]), p_yes=float(r["p_yes"]),
                    mass=float(r["answer_mass"]), audio_tok=int(r["audio_tok"]), raw=r)
    sets[key] = dict(rows=rows, d=d, dup=dup)

pods = {}
for key, (tmpl, seed) in POD.items():
    allrows = {}
    per_fold = {}
    problems = []
    for k in range(5):
        p = tmpl.format(k=k)
        rows = read(p)
        pids = [int(r["pid"]) for r in rows]
        per_fold[k] = pids
        for r in rows:
            if r["pid"] != r["speaker"]:
                problems.append(f"pid!=speaker {r['pid']}")
            if int(r["fold"]) != k:
                problems.append(f"fold col {r['fold']} in fold {k} file")
            if int(r["seed"]) != seed:
                problems.append(f"seed col {r['seed']} in seed {seed} file")
            pp = int(r["pid"])
            if pp in allrows:
                problems.append(f"pid {pp} in two pod files")
            allrows[pp] = r
    pods[key] = dict(rows=allrows, per_fold=per_fold, problems=problems, seed=seed)

ALL = sorted(fold)
assert [str(x) for x in ALL] == sorted(str(x) for x in ALL)   # string order == numeric order for these pids

# ---------- check: fold file is PART23's ----------
p23_fold_sha = set()
for k in range(5):
    j = json.load(open(f"{P23}/FT/FT_whole_fold{k}_oof.json"))
    p23_fold_sha.add(j["sources"]["fold_file"][1])
p24_copies = [FOLD_FILE] + sorted(glob.glob(P24 + "/pull/*/p23/edaic_groupkfold5_pod.csv"))
p24_fold_sha = {sha(p) for p in p24_copies}
pod_sidecar_fold_sha = set()
for key, (tmpl, seed) in POD.items():
    for k in range(5):
        sc = json.load(open(tmpl.format(k=k).replace("_oof.csv", "_oof.sidecar.json")))
        pod_sidecar_fold_sha.add(sc["sources"]["fold_file"][1])
ok = len(p23_fold_sha) == 1 and p24_fold_sha == p23_fold_sha and pod_sidecar_fold_sha == p23_fold_sha and bad_clip == 0
add("check_fold_file_is_part23", "PART23 FT sidecars fold_file sha256 " + ",".join(sorted(p23_fold_sha))[:16],
    f"PART24 staged + {len(p24_copies)-1} pod copies sha256 {','.join(sorted(p24_fold_sha))[:16]}; all 10 pod sidecars read {','.join(sorted(pod_sidecar_fold_sha))[:16]}", ok)

# PART23 per-fold test sets from the PART23 FT whole oof csvs
p23_sets = {k: {int(r["pid"]) for r in read(f"{P23}/FT/FT_whole_fold{k}_oof.csv")} for k in range(5)}
ff_sets = {k: {s for s, f in fold.items() if f == k} for k in range(5)}
ok23 = all(p23_sets[k] == ff_sets[k] for k in range(5))

for key in ("seed0", "seed1all"):
    pod = pods[key]
    d = sets[key]["d"]
    pod_ok = all(set(pod["per_fold"][k]) == ff_sets[k] for k in range(5))
    col_ok = all(d[p]["fold"] == fold[p] for p in d)
    add(f"check_fold_assignment_{key}", "fold k test set = PART23 fold k (fold file and PART23 FT_whole_fold{k}_oof.csv)",
        f"pod files match fold file: {pod_ok}; PART23 oof sets match fold file: {ok23}; per-clip fold column matches: {col_ok}; sizes {[len(pod['per_fold'][k]) for k in range(5)]}",
        pod_ok and ok23 and col_ok)

# ---------- check: every test clip exactly once per seed set ----------
for key in ("seed0", "seed1all"):
    s = sets[key]
    pod = pods[key]
    pids = [int(r["pid"]) for r in s["rows"]]
    counts = {}
    for p in pids:
        counts[p] = counts.get(p, 0) + 1
    podcount = sum(len(v) for v in pod["per_fold"].values())
    ok = (len(pids) == 275 and s["dup"] == 0 and set(pids) == set(ALL) == set(mf) == set(zs)
          and podcount == 275 and set(pod["rows"]) == set(ALL) and not pod["problems"]
          and all(len(set(v)) == len(v) == 55 for v in pod["per_fold"].values()))
    add(f"check_each_clip_once_{key}", "275 clips, each once",
        f"per-clip csv rows {len(pids)} unique {len(set(pids))} max count {max(counts.values())}; pod rows {podcount} unique {len(pod['rows'])}; "
        f"set == fold file == manifest == zero-shot: {set(pids) == set(ALL) == set(mf) == set(zs)}; pod problems {len(pod['problems'])}", ok)

# ---------- check: collected per-clip csv equals pod oof csvs ----------
for key in ("seed0", "seed1all"):
    d = sets[key]["d"]
    pod = pods[key]
    nbad = 0
    for p, v in d.items():
        r = pod["rows"][p]
        if not (float(r["p_yes"]) == v["p_yes"] and float(r["answer_mass"]) == v["mass"] and int(r["label"]) == v["label"]
                and int(r["audio_tok"]) == v["audio_tok"] and int(r["fold"]) == v["fold"] and int(r["seed"]) == v["seed"] == pod["seed"]):
            nbad += 1
    add(f"check_perclip_equals_pod_{key}", "per-clip csv copied from the 5 pod oof csvs",
        f"{275 - nbad}/275 rows identical in p_yes, answer_mass, label, audio_tok, fold, seed", nbad == 0)

# labels
lab_ok = all(sets[k]["d"][p]["label"] == mf[p] == zs[p][0] for k in sets for p in ALL)
npos = sum(mf[p] for p in ALL)
add("check_labels", "labels agree across manifest, zero-shot csv, both per-clip csvs; n_positive 66",
    f"agree: {lab_ok}; n_positive {npos}; manifest pid==speaker: {mf_spk_ok}", lab_ok and npos == 66 and mf_spk_ok)


# ---------- metrics ----------
def arrays(key, subset):
    d = sets[key]["d"]
    y = np.array([d[p]["label"] for p in subset])
    s = np.array([d[p]["p_yes"] for p in subset])
    return y, s


def boot(y, a, b=None):
    rng = np.random.default_rng(0)
    N = len(y)
    vals = []
    skipped = 0
    for _ in range(NDRAW):
        idx = rng.choice(N, size=N, replace=True)
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        v = roc_auc_score(yy, a[idx])
        if b is not None:
            v -= roc_auc_score(yy, b[idx])
        vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), NDRAW - skipped


rec = {}   # id -> (value, lo, hi, n, usable)
yz = np.array([zs[p][0] for p in ALL])
sz = np.array([zs[p][1] for p in ALL])
v = roc_auc_score(yz, sz)
lo, hi, u = boot(yz, sz)
rec["ref_zs_whole_auc"] = (v, lo, hi, 275, u)
for key, pre in (("seed0", "seed0"), ("seed1all", "extra_seed1all")):
    y, s = arrays(key, ALL)
    assert (y == yz).all()
    v = roc_auc_score(y, s)
    lo, hi, u = boot(y, s)
    rec[f"{pre}_pooled_oof_auc"] = (v, lo, hi, 275, u)
    v = roc_auc_score(y, s) - roc_auc_score(y, sz)
    lo, hi, u = boot(y, s, sz)
    rec[f"{pre}_ft_minus_zs_whole_paired"] = (v, lo, hi, 275, u)
    for k in range(5):
        sub = sorted(ff_sets[k])
        yk, sk = arrays(key, sub)
        v = roc_auc_score(yk, sk)
        lo, hi, u = boot(yk, sk)
        rec[f"{pre}_fold{k}_auc"] = (v, lo, hi, len(sub), u)
        m = float(np.median([sets[key]["d"][p]["mass"] for p in sub]))
        rec[f"{pre}_fold{k}_median_answer_mass"] = (m, None, None, len(sub), None)

# ---------- compare with FT24_values.tsv ----------
with open(VALUES, newline="") as f:
    claimed_rows = list(csv.DictReader(f, delimiter="\t"))
seen = set()
maxdiff = 0.0
for r in claimed_rows:
    i = r["id"]
    seen.add(i)
    if i not in rec:
        add(i, r["value"], "no recomputation for this id", False)
        continue
    val, lo, hi, n, u = rec[i]
    parts_ok = []
    cv = float(r["value"])
    parts_ok.append(abs(cv - val) <= TOL)
    maxdiff = max(maxdiff, abs(cv - val))
    if lo is not None:
        clo, chi = float(r["lo"]), float(r["hi"])
        parts_ok += [abs(clo - lo) <= TOL, abs(chi - hi) <= TOL]
        maxdiff = max(maxdiff, abs(clo - lo), abs(chi - hi))
        parts_ok.append(u == NDRAW)
    else:
        parts_ok.append(r["lo"].strip() == "" and r["hi"].strip() == "")
    parts_ok.append(int(r["n"]) == n)
    disp = SUMMARY.get(i)
    mine4 = (f4(val),) if lo is None else (f4(val), f4(lo), f4(hi))
    if disp is not None:
        parts_ok.append(tuple(disp) == mine4[:len(disp)])
    claimed = f"{r['value']}" + (f" [{r['lo']}, {r['hi']}]" if lo is not None else "") + f" n={r['n']}" + (f" (summary {' '.join(disp)})" if disp else "")
    recomputed = f"{val!r}" + (f" [{lo!r}, {hi!r}]" if lo is not None else "") + f" n={n}" + (f" usable_draws={u}" if u is not None else "")
    add(i, claimed, recomputed, all(parts_ok))
for i in rec:
    if i not in seen:
        add(i, "missing from FT24_values.tsv", repr(rec[i][0]), False)

# ---------- check: median masses and the seed 1 trigger ----------
med = {k: rec[f"seed0_fold{k}_median_answer_mass"][0] for k in range(5)}
trig = [k for k in range(5) if med[k] < 0.5]
mc_ok = True
mc_txt = []
for k in range(5):
    t = open(f"{P24}/pull/p24-ft{k}/out/MASS_CHECK_fold{k}.txt").read().split()
    # "fold K seed 0 median_answer_mass M n 55 decision X"
    m_claim = float(t[t.index("median_answer_mass") + 1])
    dec = t[t.index("decision") + 1]
    exp = "rerun_seed1" if k in trig else "no_rerun"
    mc_ok &= (m_claim == med[k] and dec == exp and t[t.index("n") + 1] == "55")
    mc_txt.append(f"{k}:{dec}")
s1_on_seed0_pods = sorted(glob.glob(P24 + "/pull/p24-ft*/out/*_s1_*"))
seedrule_file = os.path.exists(P24 + "/FT/FT24_whole_seedrule_perclip.csv")
vs = json.load(open(P24 + "/FT/FT24_values.sidecar.json"))
sc0 = json.load(open(PERCLIP["seed0"].replace(".csv", ".sidecar.json")))
ok = (trig == [] and vs["seed_rule_folds_triggered"] == [] and sc0["seed_rule_folds_triggered"] == [] and mc_ok
      and s1_on_seed0_pods == [] and not seedrule_file and all(sc0["seeds_by_fold"][str(k)] == 0 for k in range(5))
      and min(med.values()) > 0.99)
add("check_seed_rule", "no fold triggered (all seed 0 medians > 0.99, rule < 0.5); seed-rule set = seed 0 set; no seedrule csv built",
    f"recomputed medians {[round(med[k], 6) for k in range(5)]} min {min(med.values()):.6f}; triggered {trig}; MASS_CHECK files match exactly and say {' '.join(mc_txt)}: {mc_ok}; "
    f"seed 1 files on p24-ft pods {len(s1_on_seed0_pods)}; seedrule csv exists {seedrule_file}; sidecar seeds_by_fold all 0", ok)

# seed 1 medians against pod sidecars
s1_ok = True
for k in range(5):
    sc = json.load(open(POD["seed1all"][0].format(k=k).replace("_oof.csv", "_oof.sidecar.json")))
    s1_ok &= sc["median_answer_mass"] == rec[f"extra_seed1all_fold{k}_median_answer_mass"][0] and sc["seed"] == 1
add("check_seed1all_pod_sidecars", "seed 1 pod sidecars: seed 1, median answer mass as recomputed", f"match: {s1_ok}", s1_ok)

# pod RESULT fold AUCs
res_ok = True
for key, pre in (("seed0", "seed0"), ("seed1all", "extra_seed1all")):
    for k in range(5):
        tmpl, seed = POD[key]
        t = open(tmpl.format(k=k).replace("_oof.csv", ".RESULT")).read().split()
        res_ok &= abs(float(t[t.index("auc_p_yes") + 1]) - rec[f"{pre}_fold{k}_auc"][0]) <= TOL
add("check_pod_RESULT_fold_auc", "pod .RESULT auc_p_yes for all 10 runs", f"all equal recomputed fold AUC: {res_ok}", res_ok)

# ---------- check: zero-shot vector reproduces 0.8366 ----------
zsc = json.load(open(ZS_SIDECAR))["results"]
v, lo, hi = rec["ref_zs_whole_auc"][:3]
ok = (f4(v) == "0.8366" and abs(v - zsc["auc_whole"]) <= TOL and abs(lo - zsc["ci_whole"][0]) <= TOL
      and abs(hi - zsc["ci_whole"][1]) <= TOL and sha(ZS) == vs["zero_shot_file"][1])
add("check_zero_shot_reproduces_0.8366", f"0.8366 (PART23 sidecar {zsc['auc_whole']!r} [{zsc['ci_whole'][0]!r}, {zsc['ci_whole'][1]!r}])",
    f"{v!r} [{lo!r}, {hi!r}] from p_yes_whole of {os.path.basename(ZS)} sha256 {sha(ZS)[:16]}", ok)

# ---------- extra checks on the collect summary text ----------
at = [sets["seed0"]["d"][p]["audio_tok"] for p in ALL]
add("check_audio_tok_max", "largest audio_tok 22500, 89 clips reach it", f"max {max(at)}, count {sum(a == max(at) for a in at)}",
    max(at) == 22500 and sum(a == 22500 for a in at) == 89)
cut = {sha(p) for p in glob.glob(P24 + "/pull/*/edaicfull/cut_sha256.txt")} | {sha(p) for p in glob.glob(P23 + "/FT/support/pod*/cut_sha256.txt")}
ncut = len(glob.glob(P24 + "/pull/*/edaicfull/cut_sha256.txt"))
add("check_window_cut_hashes_equal_part23", "cut hashes identical to PART23",
    f"{ncut} PART24 pod cut lists + 5 PART23 pod cut lists, distinct sha256 {len(cut)}", len(cut) == 1 and ncut == 10)
m23 = float(np.median([float(r["answer_mass"]) for r in read(f"{P23}/FT/FT_whole_fold3_oof.csv")]))
add("check_part23_fold3_median_mass", "PART23 fold 3 median mass 0 (summary; 0.0000 at 4 dp)", f"{m23!r} = {f4(m23)} at 4 dp; not exactly 0", f4(m23) == "0.0000")
d0 = rec["seed0_ft_minus_zs_whole_paired"]
add("check_paired_interval_includes_0", "interval includes 0", f"[{d0[1]:.4f}, {d0[2]:.4f}]", d0[1] <= 0 <= d0[2])

# ---------- write ----------
os.makedirs(OUTDIR, exist_ok=True)
tsv = OUTDIR + "/FT24_VERIFIED.tsv"
with open(tsv, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "claimed", "recomputed", "verified"])
    w.writerows(out)
side = {
    "what": "independent verification of PART24 item 1 (FT24_values.tsv) plus fold, coverage, seed rule and zero-shot checks",
    "result_file": "FT24_VERIFIED.tsv",
    "script": [os.path.abspath(__file__), sha(os.path.abspath(__file__))],
    "command": "/usr/local/bin/python3 " + os.path.abspath(__file__),
    "inputs": {p: sha(p) for p in [VALUES, PERCLIP["seed0"], PERCLIP["seed1all"], FOLD_FILE, MANIFEST, ZS]
               + [POD[k][0].format(k=i) for k in POD for i in range(5)]},
    "auc_rule": "sklearn.metrics.roc_auc_score on p_yes (FT) and p_yes_whole (zero shot, np.float64 wrapper stripped)",
    "bootstrap_rule": "2000 draws; fresh numpy default_rng(0) per cell; idx = rng.choice(N, size=N, replace=True) over speakers in ascending pid order (same as string order here); one clip = one speaker; draws with one class skipped; np.percentile 2.5 and 97.5; paired = FT and zero shot on the same idx",
    "tolerance_abs": TOL,
    "max_abs_diff_claimed_vs_recomputed": maxdiff,
    "n": 275, "n_speakers": 275, "n_positive": npos,
    "rows": len(out), "rows_verified_yes": sum(r[3] == "yes" for r in out),
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                 "platform": platform.platform()},
    "collect_code_opened": False,
    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
json.dump(side, open(OUTDIR + "/FT24_VERIFIED.sidecar.json", "w"), indent=1)
for r in out:
    print("\t".join(r))
print(f"ROWS {len(out)} YES {sum(r[3] == 'yes' for r in out)} NO {sum(r[3] == 'no' for r in out)} MAXDIFF {maxdiff:.3e}")
