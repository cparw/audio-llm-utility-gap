#!/usr/local/bin/python3
"""PART 25 track T5, second run (r2, written fresh on 24 Sep).

Question: on the ten-pod T5 refit, how many of the 32 Qwen2.5-Omni encoder layers sit
above the zero-shot answer AUC, and above the answer's upper 97.5 bound?

Per dataset row:
  answer AUC      sklearn roc_auc_score(label, p_yes) on the zero-shot per-clip file
  answer interval speaker bootstrap, 2000 draws, fresh np.random.default_rng(0) per cell,
                  speakers = np.unique(speaker as str), idx = rng.choice(n_spk, size=n_spk, replace=True),
                  all clips of every drawn speaker; draws where one class is missing are skipped and counted;
                  2.5 and 97.5 percentiles
  refit curve     the unchanged curves_from_states.py output (<job>_encoder_perlayer.csv) from the pod
                  named in PRIMARY, column auc_oof (the column Figure 1 plots); auc_mean counts also given
  counts          layers with refit auc_oof > answer AUC, and > answer upper bound (strict)
  range           the same counts over every other refit file of that job on every pod
                  (podfile, vrefit, BLAS core type, float64, 8 thread variants, other pods)
  paired delta    per layer, AUC(refit out-of-fold probe score) minus AUC(answer) on the same draw;
                  the out-of-fold scores come from <job>__podfile_oof.npz on the PRIMARY pod after checking
                  that their pooled AUC equals the refit csv
  curve match     max |refit - saved| per curve for auc_oof, auc_mean, auc_std, and layers equal at 4 dp

Reads omni_final only. Writes only under part25/T5.
"""
import csv, glob, hashlib, json, os, platform, re, sys
from datetime import datetime, timezone
import numpy as np
import sklearn, scipy
from sklearn.metrics import roc_auc_score

OUT = "<local data dir>/release_from_mac/scores/part25/T5"
T5 = "<local data dir>/release_from_mac/scores/leftovers_23sep/t5"
PULL = f"{T5}/pull"
OF = "<local data dir>/release/omni_final"
VAR = "<local data dir>/release/edaic_rerun/variants"
P16 = "<local data dir>/release/edaic_rerun/part16/EDAICFULL/full_zeroshot_scores.csv"
P22 = "<local data dir>/release/scores/part22/edaic_lifted_perclip.csv"
P23 = "scores/part23/A/A2_edaic_whole_zeroshot_perclip.csv"
P23_CURVE = "<local data dir>/release_from_mac/scores/part23/pull/p23-a-probes/p23a/out/o25_whole_encoder_perlayer.csv"
NB = 2000
os.makedirs(f"{OUT}/perclip_r2", exist_ok=True)
os.makedirs(f"{OUT}/boot_r2", exist_ok=True)

INPUTS = {}
def sha(p):
    if p not in INPUTS:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        INPUTS[p] = h.hexdigest()
    return INPUTS[p]

NPF = re.compile(r"^np\.float64\((.*)\)$")
def num(s):
    s = s.strip(); m = NPF.match(s)
    return float(m.group(1) if m else s)

def read_zs(path, pcol, spkcol, clipcol):
    """Zero-shot per-clip file -> list of (clip_key, speaker, label, p_yes)."""
    sha(path)
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            clip = r[clipcol].strip()
            key = clip if clip.endswith(".wav") else f"{clip}.wav"
            rows.append((key, r[spkcol].strip(), int(num(r["label"])), num(r[pcol])))
    return rows

def read_curve(path):
    sha(path)
    with open(path, newline="") as fh:
        rd = csv.reader(fh); hdr = next(rd); rows = [list(map(float, r)) for r in rd]
    a = np.array(rows)
    idx = a[:, 0].astype(int)
    assert (idx == np.arange(len(idx))).all(), path
    return {h: a[:, i] for i, h in enumerate(hdr)}

def spk_draws(spk):
    """Fresh rng(0) per cell; return list of clip-index arrays, one per draw."""
    uniq = np.unique(np.asarray(spk).astype(str))
    members = [np.where(np.asarray(spk).astype(str) == s)[0] for s in uniq]
    rng = np.random.default_rng(0)
    draws, spk_idx = [], np.empty((NB, len(uniq)), dtype=np.int32)
    for b in range(NB):
        idx = rng.choice(len(uniq), size=len(uniq), replace=True)
        spk_idx[b] = idx
        draws.append(np.concatenate([members[i] for i in idx]))
    return uniq, draws, spk_idx

def boot_auc(y, s, draws):
    out = np.full(len(draws), np.nan)
    for b, ix in enumerate(draws):
        yb = y[ix]
        if yb.min() != yb.max():
            out[b] = roc_auc_score(yb, s[ix])
    return out

def pod_cpu(pod):
    d = json.load(open(f"{PULL}/{pod}/out/env_{pod}.json"))
    model = [l for l in d["lscpu"] if l.startswith("Model name")][0].split(":", 1)[1].strip()
    arch = sorted({t.get("architecture") for t in d["threadpool_info"] if t.get("internal_api") == "openblas"})
    return model, "/".join(a for a in arch if a), d["numpy"], d["scipy"], d["sklearn"]

# ---------------------------------------------------------------- datasets
ENC = "encoder_perlayer.csv"
DATASETS = [
    dict(ds="neurovoz", display="NeuroVoz", cond="PD", scope="requested",
         zs=(f"{OF}/omni_neurovoz_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_neurovoz_{ENC}", job="neurovoz_encproj", primary="lo-t5-8",
         states="other extraction of the same clips (T5 manifest: p_yes max diff 0.047 vs the saved zero-shot run)"),
    dict(ds="kcl", display="MDVR-KCL", cond="PD", scope="requested",
         zs=(f"{OF}/omni_kcl_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_kcl_{ENC}", job=None, primary=None, states=""),
    dict(ds="adresso", display="ADReSSo", cond="AD", scope="requested",
         zs=(f"{OF}/omni_adresso_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_adresso_{ENC}", job=None, primary=None, states=""),
    dict(ds="adress2020", display="ADReSS-2020", cond="AD", scope="requested",
         zs=(f"{OF}/omni_adress2020_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_adress2020_{ENC}", job="adress2020_encproj", primary="lo-t5-7",
         states="other extraction of the same clips (T5 manifest: p_yes max diff 0.038 vs the saved zero-shot run)"),
    dict(ds="edaic_whole_p23", display="E-DAIC whole interview, lifted limit, answer 0.84 of Table 1 (PART 23 A2 p_yes_whole)",
         cond="MDD", scope="requested",
         zs=(P23, "p_yes_whole", "speaker", "pid"),
         saved=f"{VAR}/o25_full_{ENC}", job="edaicfull_encproj", primary="lo-t5-9",
         states="default input limit whole interview states (part17 T7 shards); the refit is NOT the lifted-limit PART 23 run behind the Figure 1 E-DAIC curve"),
    dict(ds="edaic_lifted_p22", display="E-DAIC whole interview, lifted limit (PART 22 p_yes_lifted)",
         cond="MDD", scope="requested",
         zs=(P22, "p_yes_lifted", "pid", "pid"),
         saved=f"{VAR}/o25_full_{ENC}", job="edaicfull_encproj", primary="lo-t5-9",
         states="default input limit whole interview states (part17 T7 shards)"),
    dict(ds="edaic_full_o25", display="E-DAIC whole interview, default limit (variants o25_full)",
         cond="MDD", scope="requested",
         zs=(f"{VAR}/o25_full_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{VAR}/o25_full_{ENC}", job="edaicfull_encproj", primary="lo-t5-9",
         states="other extraction of the same clips (T5 manifest: p_yes max diff 0.030 vs o25_full)"),
    dict(ds="edaic_full_statesrun", display="E-DAIC whole interview, default limit, answer from the states run (part16 EDAICFULL)",
         cond="MDD", scope="requested",
         zs=(P16, "p_yes", "speaker", "clip"),
         saved=f"{VAR}/o25_full_{ENC}", job="edaicfull_encproj", primary="lo-t5-9",
         states="same run as this answer file (order file of the T5 edaicfull states)"),
    dict(ds="edaic_first30", display="E-DAIC first 30 s (omni_final)", cond="MDD", scope="requested",
         zs=(f"{OF}/omni_edaic_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_edaic_{ENC}", job="edaic30_encproj", primary="lo-t5-10",
         states="same extraction run as the saved zero-shot file"),
    dict(ds="pitt", display="Pitt (extra, not asked)", cond="AD", scope="extra",
         zs=(f"{OF}/omni_pitt_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_pitt_{ENC}", job="pitt_encproj", primary="lo-t5-10",
         states="same extraction run as the saved zero-shot file"),
    dict(ds="pcgita", display="PC-GITA (extra, not asked)", cond="PD", scope="extra",
         zs=(f"{OF}/omni_pcgita_zeroshot_scores.csv", "p_yes", "speaker", "clip"),
         saved=f"{OF}/omni_pcgita_{ENC}", job=None, primary=None, states=""),
]
NO_REFIT = ("no T5 refit: T5 ran only pitt, edaic30, adress2020, neurovoz and edaicfull; the T5 verifier "
            "(leftovers_23sep/verify t5v, v5_notpossible.json) scanned 1461 npz files and found no hidden states "
            "from the run behind the saved curve; saved curve counts are given instead")

def refit_files(job):
    """Every per-layer encoder csv of this job on every pod: (pod, tag, path)."""
    out = []
    for pod in sorted(os.listdir(PULL)):
        d = f"{PULL}/{pod}/out"
        if not os.path.isdir(d):
            continue
        p = f"{d}/{job}_{ENC}"
        if os.path.exists(p):
            out.append((pod, "exact_script", p))
        for p in sorted(glob.glob(f"{d}/{job}__*_enc.csv")):
            tag = os.path.basename(p)[len(job) + 2:-len("_enc.csv")]
            out.append((pod, tag, p))
    return out

def layers_str(mask):
    return ";".join(str(i) for i in np.where(mask)[0])

rows, match_rows, side_ds = [], [], {}
for D in DATASETS:
    ds = D["ds"]
    zpath, pcol, scol, ccol = D["zs"]
    zs = read_zs(zpath, pcol, scol, ccol)
    clips = [r[0] for r in zs]; assert len(set(clips)) == len(clips), ds
    spk = np.array([r[1] for r in zs]); y = np.array([r[2] for r in zs]); p = np.array([r[3] for r in zs])
    ans = float(roc_auc_score(y, p))
    uniq, draws, spk_idx = spk_draws(spk)
    ab = boot_auc(y, p, draws)
    ok = ~np.isnan(ab)
    lo, hi = (float(v) for v in np.percentile(ab[ok], [2.5, 97.5]))
    saved = read_curve(D["saved"])
    s_oof = saved["auc_oof"]; assert len(s_oof) == 32, ds
    R = dict(dataset=ds, display=D["display"], condition=D["cond"], scope=D["scope"],
             n_clips=len(y), n_speakers=len(uniq), n_pos=int(y.sum()),
             answer_auc=ans, answer_lo=lo, answer_hi=hi, boot_usable=int(ok.sum()),
             saved_above_answer=int((s_oof > ans).sum()), saved_above_hi=int((s_oof > hi).sum()),
             saved_min=float(s_oof.min()), saved_min_layer=int(s_oof.argmin()),
             saved_max=float(s_oof.max()), saved_max_layer=int(s_oof.argmax()),
             saved_not_above_answer=layers_str(~(s_oof > ans)), saved_not_above_hi=layers_str(~(s_oof > hi)),
             answer_file=zpath, answer_column=pcol, saved_curve=D["saved"])
    boot = dict(answer_draws=ab, speakers=uniq, speaker_idx=spk_idx, answer_auc=ans)
    pc_cols = ["clip", "speaker", "label", "p_yes_answer"]
    pc_data = [clips, spk.tolist(), y.tolist(), p.tolist()]
    if D["job"] is None:
        R.update(refit_status="NOT LANDED (no refit)", refit_pod="", refit_file="", refit_states="",
                 every_layer_on_refit="no refit", note=NO_REFIT)
    else:
        job, pod = D["job"], D["primary"]
        rf = f"{PULL}/{pod}/out/{job}_{ENC}"
        cur = read_curve(rf); r_oof, r_mean = cur["auc_oof"], cur["auc_mean"]
        assert len(r_oof) == 32
        cpu, arch, npv, spv, skv = pod_cpu(pod)
        above = r_oof > ans; above_hi = r_oof > hi
        # out-of-fold per clip scores (podfile variant on the same pod, same folds)
        oz = np.load(f"{PULL}/{pod}/out/{job}__podfile_oof.npz", allow_pickle=True); sha(f"{PULL}/{pod}/out/{job}__podfile_oof.npz")
        onames = [str(n) for n in oz["name"]]; pos = {n: i for i, n in enumerate(onames)}
        assert sorted(onames) == sorted(clips), f"{ds}: refit clips differ from answer clips"
        order = np.array([pos[c] for c in clips])
        oof = np.asarray(oz["enc"])[order]; ylab = np.asarray(oz["label"])[order]
        assert (ylab == y).all(), f"{ds}: labels differ"
        oof_auc = np.array([roc_auc_score(y, oof[:, l]) for l in range(32)])
        oof_maxabs = float(np.max(np.abs(oof_auc - r_oof)))
        # paired bootstrap per layer, same draws as the answer interval
        pr = np.full((NB, 32), np.nan)
        for b, ix in enumerate(draws):
            yb = y[ix]
            if yb.min() != yb.max():
                for l in range(32):
                    pr[b, l] = roc_auc_score(yb, oof[ix, l])
        dl = pr - ab[:, None]
        dlo = np.nanpercentile(dl, 2.5, axis=0); dhi = np.nanpercentile(dl, 97.5, axis=0)
        boot.update(probe_draws=pr, delta_draws=dl, delta_lo=dlo, delta_hi=dhi,
                    refit_oof_auc=oof_auc, refit_csv_auc_oof=r_oof)
        for l in range(32):
            pc_cols.append(f"refit_oof_enc_L{l}"); pc_data.append(oof[:, l].tolist())
        # every refit file of the job: count range
        allf = refit_files(job)
        cnt_same, cnt_mac = [], []
        for (pd_, tag, path) in allf:
            c = read_curve(path)["auc_oof"]
            t = (pd_, tag, int((c > ans).sum()), int((c > hi).sum()))
            (cnt_mac if tag == "macfile" else cnt_same).append(t)
        def rng_s(lst, i):
            v = [t[i] for t in lst]
            return f"{min(v)} to {max(v)} over {len(v)} files" if v else ""
        d_oof = np.abs(r_oof - s_oof); d_mean = np.abs(r_mean - saved["auc_mean"])
        R.update(refit_status="refit", refit_pod=pod, refit_cpu=cpu, refit_openblas=arch,
                 refit_env=f"numpy {npv} scipy {spv} sklearn {skv}", refit_file=rf, refit_states=D["states"],
                 refit_above_answer=int(above.sum()), refit_above_hi=int(above_hi.sum()),
                 refit_not_above_answer=layers_str(~above), refit_not_above_hi=layers_str(~above_hi),
                 refit_min=float(r_oof.min()), refit_min_layer=int(r_oof.argmin()),
                 refit_max=float(r_oof.max()), refit_max_layer=int(r_oof.argmax()),
                 refit_min_minus_answer=float(r_oof.min() - ans), refit_min_minus_hi=float(r_oof.min() - hi),
                 refit_mean_above_answer=int((r_mean > ans).sum()), refit_mean_above_hi=int((r_mean > hi).sum()),
                 range_above_answer_same_folds=rng_s(cnt_same, 2), range_above_hi_same_folds=rng_s(cnt_same, 3),
                 range_above_answer_mac_folds=rng_s(cnt_mac, 2), range_above_hi_mac_folds=rng_s(cnt_mac, 3),
                 paired_delta_lo_gt0=int((dlo > 0).sum()), paired_delta_hi_lt0=int((dhi < 0).sum()),
                 paired_delta_min_lo=float(dlo.min()),
                 oof_recompute_maxabs=oof_maxabs,
                 refit_vs_saved_maxabs_oof=float(d_oof.max()), refit_vs_saved_match4dp_oof=int((np.round(r_oof, 4) == np.round(s_oof, 4)).sum()),
                 refit_vs_saved_maxabs_mean=float(d_mean.max()),
                 every_layer_on_refit=("holds" if above.all() else f"fails ({int(above.sum())}/32)"),
                 note="")
        R["all_refit_file_counts"] = [dict(pod=a, tag=b, above_answer=c, above_hi=d) for (a, b, c, d) in cnt_same + cnt_mac]
    # per clip csv and sidecar
    pcp = f"{OUT}/perclip_r2/T5_perclip_{ds}.csv"
    with open(pcp, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(pc_cols)
        for i in range(len(clips)):
            w.writerow([col[i] if isinstance(col[i], str) else (repr(col[i]) if isinstance(col[i], float) else col[i]) for col in pc_data])
    bp = f"{OUT}/boot_r2/T5_boot_{ds}.npz"
    np.savez_compressed(bp, **boot)
    R["perclip_csv"] = pcp; R["boot_npz"] = bp
    side = {k: v for k, v in R.items()}
    side.update(task="PART 25 T5 r2 per dataset result", date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                perclip_sha256=hashlib.sha256(open(pcp, "rb").read()).hexdigest(),
                boot_sha256=hashlib.sha256(open(bp, "rb").read()).hexdigest(),
                boot_contents="answer_draws (2000), speaker_idx (2000 x n_spk, rng(0)), speakers; with a refit also probe_draws, delta_draws (2000 x 32, probe minus answer on the same draw), delta_lo, delta_hi")
    json.dump(side, open(f"{OUT}/perclip_r2/T5_perclip_{ds}.sidecar.json", "w"), indent=1, default=str)
    side_ds[ds] = side
    rows.append(R)
    print(f"{ds:22s} ans {ans:.4f} [{lo:.4f}, {hi:.4f}] n_ok {int(ok.sum())} | saved {R['saved_above_answer']}/32 {R['saved_above_hi']}/32"
          + (f" | refit {R['refit_above_answer']}/32 {R['refit_above_hi']}/32 min {R['refit_min']:.4f}@{R['refit_min_layer']} "
             f"range {R['range_above_answer_same_folds']} / {R['range_above_hi_same_folds']} | vs saved {R['refit_vs_saved_maxabs_oof']:.2e} "
             f"({R['refit_vs_saved_match4dp_oof']}/32 at 4dp) | oofchk {R['oof_recompute_maxabs']:.1e} | paired lo>0 {R['paired_delta_lo_gt0']}/32"
             if R["refit_status"] == "refit" else " | NO REFIT"), flush=True)

# ---------------------------------------------------------------- curve match, every refit curve vs its saved curve
SAVED = {"pitt": (OF, "omni_pitt"), "edaic30": (OF, "omni_edaic"), "adress2020": (OF, "omni_adress2020"),
         "neurovoz": (OF, "omni_neurovoz"), "edaicfull": (VAR, "o25_full")}
STREAM = {"encoder_perlayer.csv": ("enc", "encoder_perlayer.csv"), "proj_perlayer.csv": ("proj", "proj_perlayer.csv"),
          "llm_perlayer.csv": ("llm", "llm_perlayer.csv"), "ans_perlayer.csv": ("ans", "ans_perlayer.csv")}
for pod in sorted(os.listdir(PULL), key=lambda s: (len(s), s)):
    d = f"{PULL}/{pod}/out"
    if not os.path.isdir(d):
        continue
    cpu, arch, *_ = pod_cpu(pod)
    for f in sorted(os.listdir(d)):
        m = re.match(r"^(pitt|edaic30|adress2020|neurovoz|edaicfull)_(encproj|llm|ans)_(encoder|proj|llm|ans)_perlayer\.csv$", f)
        if not m:
            continue
        base, stream = m.group(1), m.group(3)
        sdir, spre = SAVED[base]
        sp = f"{sdir}/{spre}_{'encoder' if stream == 'encoder' else stream}_perlayer.csv"
        a, b = read_curve(f"{d}/{f}"), read_curve(sp)
        assert len(a["auc_oof"]) == len(b["auc_oof"]), (f, sp)
        mr = dict(curve=base, stream={"encoder": "enc"}.get(stream, stream), pod=pod, cpu=cpu, openblas=arch,
                  states=("same extraction run" if base in ("pitt", "edaic30") else "other extraction"),
                  n_layers=len(a["auc_oof"]), refit_file=f"{d}/{f}", saved_file=sp)
        for col in ("auc_oof", "auc_mean", "auc_std"):
            dd = np.abs(a[col] - b[col])
            mr[f"{col}_maxabs"] = float(dd.max())
            mr[f"{col}_match4dp"] = int((np.round(a[col], 4) == np.round(b[col], 4)).sum())
        mr["verdict"] = ("exact (max diff below 1e-12)" if mr["auc_oof_maxabs"] < 1e-12 else
                         "close, not exact" if mr["auc_oof_maxabs"] < 0.005 else "does not match")
        match_rows.append(mr)
        print(f"match {base:10s} {mr['stream']:4s} {pod:9s} {arch:9s} oof {mr['auc_oof_maxabs']:.2e} ({mr['auc_oof_match4dp']}/{mr['n_layers']}) "
              f"mean {mr['auc_mean_maxabs']:.2e} std {mr['auc_std_maxabs']:.2e} -> {mr['verdict']}", flush=True)

# PART 23 lifted curve (Figure 1 E-DAIC), saved only, for context
p23c = read_curve(P23_CURVE)["auc_oof"]
p23_ans = [r for r in rows if r["dataset"] == "edaic_whole_p23"][0]
fig1_note = dict(file=P23_CURVE, sha256=sha(P23_CURVE), max=float(p23c.max()), max_layer=int(p23c.argmax()),
                 above_answer=int((p23c > p23_ans["answer_auc"]).sum()), above_hi=int((p23c > p23_ans["answer_hi"]).sum()),
                 note="saved PART 23 lifted-limit curve behind the Figure 1 E-DAIC panel; not part of the T5 refit")
print("P23 figure-1 E-DAIC curve (saved, not refit):", fig1_note, flush=True)

# ---------------------------------------------------------------- write
COLS = ["dataset", "display", "condition", "scope", "refit_status", "n_clips", "n_speakers", "n_pos",
        "answer_auc", "answer_lo", "answer_hi", "boot_usable",
        "refit_above_answer", "refit_above_hi", "every_layer_on_refit",
        "refit_not_above_answer", "refit_not_above_hi", "refit_min", "refit_min_layer", "refit_max", "refit_max_layer",
        "refit_min_minus_answer", "refit_min_minus_hi", "refit_mean_above_answer", "refit_mean_above_hi",
        "range_above_answer_same_folds", "range_above_hi_same_folds", "range_above_answer_mac_folds", "range_above_hi_mac_folds",
        "paired_delta_lo_gt0", "paired_delta_hi_lt0", "paired_delta_min_lo",
        "refit_vs_saved_maxabs_oof", "refit_vs_saved_match4dp_oof", "refit_vs_saved_maxabs_mean", "oof_recompute_maxabs",
        "saved_above_answer", "saved_above_hi", "saved_min", "saved_min_layer", "saved_max", "saved_not_above_answer", "saved_not_above_hi",
        "refit_pod", "refit_cpu", "refit_openblas", "refit_env", "refit_states", "note",
        "refit_file", "saved_curve", "answer_file", "answer_column", "perclip_csv", "boot_npz"]
def fmt(v):
    if v is None: return ""
    if isinstance(v, float): return repr(v)
    return str(v)
with open(f"{OUT}/T5_counts.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(COLS)
    for R in rows:
        w.writerow([fmt(R.get(c, "")) for c in COLS])
MCOLS = ["curve", "stream", "pod", "cpu", "openblas", "states", "n_layers", "auc_oof_maxabs", "auc_oof_match4dp",
         "auc_mean_maxabs", "auc_mean_match4dp", "auc_std_maxabs", "auc_std_match4dp", "verdict", "refit_file", "saved_file"]
with open(f"{OUT}/T5_curve_match.tsv", "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(MCOLS)
    for R in match_rows:
        w.writerow([fmt(R[c]) for c in MCOLS])
outs = [f"{OUT}/T5_counts.tsv", f"{OUT}/T5_curve_match.tsv"] + [r["perclip_csv"] for r in rows] + [r["boot_npz"] for r in rows] \
       + [f"{OUT}/perclip_r2/T5_perclip_{r['dataset']}.sidecar.json" for r in rows]
side = dict(task="PART 25 track T5 (FINAL item 3), run r2: encoder layers above the zero-shot answer on the ten-pod T5 refit",
            date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            script=os.path.abspath(__file__), script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            env=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, machine=platform.machine()),
            definitions=__doc__, no_fits_on_mac="this script fits nothing; every probe value was fitted on the T5 pods (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1)",
            previous_attempt_backup=f"{OUT}/attempt1_2345Z", inputs=INPUTS, datasets=side_ds, curve_match=match_rows,
            figure1_edaic_curve_saved_only=fig1_note,
            output_sha256={p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in outs})
json.dump(side, open(f"{OUT}/T5_counts_sidecar.json", "w"), indent=1, default=str)
print("WROTE", f"{OUT}/T5_counts.tsv", len(rows), "rows;", len(match_rows), "curve rows", flush=True)
