"""PART20 POD5 reporting: builds per-result per-clip csv + sidecar json in /workspace/scores/part20/POD5, prints paste lines,
writes provisional rows. Stats from stats_pod5.py (rank AUC, speaker bootstrap 2000, fresh default_rng(0) per cell).
usage: report_pod5.py WHICH   (WHICH in zs, nested, direction, sft)"""
import os, sys, json, csv, time, numpy as np, pandas as pd
sys.path.insert(0, "/root/p20/scripts")
from stats_pod5 import boot, sha1mb, paste, rauc
W = "/workspace/p20/work"; OUT = "/workspace/scores/part20/POD5"; REF = "/root/p20/ref"; MAC = "scores/part20/POD5"
MID = "Qwen/Qwen2.5-Omni-7B"; PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
MACSRC = {f"{REF}/omnisft_pitt_oof.csv": "<local data dir>/release/omni_final/omnisft_pitt_oof.csv",
          f"{REF}/omni_pitt_zeroshot_scores.csv": "<local data dir>/release/omni_final/omni_pitt_zeroshot_scores.csv",
          f"{REF}/pitt_enc_nested5_oof.npz": "<local data dir>/release/overnight2/part10/pitt_enc_nested5_oof.npz",
          f"{REF}/readout_direction.csv": "<local data dir>/release/omni/readout_direction.csv",
          f"{REF}/omni_pitt_nested_repeats.json": "<local data dir>/release/omni_final/omni_pitt_nested_repeats.json",
          "/root/p20/o25_pitt_states.npz": "<local data dir>/release/overnight2/part10/o25_pitt_states.npz",
          "/root/p20/scripts/pod_manifest_noinv.csv": "manifests/part20/pitt_noinv_cut/manifest.csv (clips) + pitt_conflict_manifest.csv (set)"}
man = pd.read_csv("/root/p20/scripts/pod_manifest_noinv.csv"); SET = dict(zip(man.orig_clip_id, man["set"])); SPK = dict(zip(man.orig_clip_id, man.speaker))
LINES, ROWS = f"{OUT}/POD5_paste_lines.txt", f"{OUT}/POD5_provisional_rows.tsv"
def emit(id_, what, res, n, nspk, fname, diff=False):
    v, lo, hi, ok = res
    line, row = paste(id_, what, v, lo, hi, n, nspk, f"{MAC}/{fname}", diff)
    if not diff: row = row + (", interval excludes 0.5" if (lo > 0.5 or hi < 0.5) else ", interval includes 0.5")
    print(line, flush=True)
    open(LINES, "a").write(line + "\n"); open(ROWS, "a").write(row + "\n")
    return {"id": id_, "what": what, "value": v, "lo": lo, "hi": hi, "valid_draws": ok, "n": n, "n_speakers": nspk}
def cells(prefix, what, df, cols_new, cols_old=None, old_label="original", fname=None):
    """overall + arms for the new side; the old side and paired old-minus-new when cols_old given."""
    out = []
    y = df.label.values.astype(int); spk = df.speaker.values.astype(str); st = df["set"].values.astype(str)
    for arm in ("overall", "conflict", "agreement"):
        m = np.ones(len(y), bool) if arm == "overall" else (st == arm)
        n, ns = int(m.sum()), int(len(np.unique(spk[m])))
        out.append(emit(f"{prefix}_{arm}", f"{what}, interviewer-free, {arm}", boot(y[m], df.loc[m, cols_new].values, spk[m]), n, ns, fname))
        if cols_old:
            out.append(emit(f"{prefix}_{old_label}_{arm}", f"{what}, {old_label} 468 windows, {arm}", boot(y[m], df.loc[m, cols_old].values, spk[m]), n, ns, fname))
            out.append(emit(f"{prefix}_diff_{arm}", f"{what}, paired {old_label} minus interviewer-free, {arm}",
                            boot(y[m], df.loc[m, cols_old].values, spk[m], S2=df.loc[m, cols_new].values), n, ns, fname, diff=True))
    return out
def write(name, df, side, sources, command, extra):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    js = {"result": name, "per_clip_csv": f"{MAC}/{name}.csv", "pod_per_clip_csv": f"{OUT}/{name}.csv", "model": MID, "prompt_verbatim": PROMPT,
          "n": int(len(df)), "n_speakers": int(df.speaker.nunique()), "bootstrap": {"draws": 2000, "seed": 0, "rng": "numpy.random.default_rng(0), fresh per cell",
          "unit": "speaker, resampled with replacement", "interval": "percentile 2.5/97.5", "paired": "one speaker draw per replicate for both sides"},
          "auc": "rank formula, scipy.stats.rankdata, ties averaged", "cells": side, "command": command,
          "sources": [{"path_pod": p, "path_mac": MACSRC.get(p, MAC + "/work/" + os.path.basename(p) if p.startswith(W) else ""), "sha256_first_1MB": sha1mb(p)} for p in sources],
          "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    json.dump(js, open(f"{OUT}/{name}.sidecar.json", "w"), indent=1)
    open(f"{OUT}/{name}.RESULT", "w").write("done\n")
which = sys.argv[1] if len(sys.argv) > 1 else ""
if which == "zs":
    z = pd.read_csv(f"{W}/o25noinv_pitt_zeroshot_scores.csv"); p = pd.read_csv(f"{REF}/omni_pitt_zeroshot_scores.csv").set_index("clip").loc[z.orig_clip_id]
    assert (p.label.values == z.label.values).all() and (p.speaker.values == z.speaker.values).all() and (p.prompt.values == z.prompt.values).all()
    df = z.rename(columns={"p_yes": "p_yes_noinv"}); df["p_yes_original_paper"] = p.p_yes.values; df["mass_original_paper"] = p.mass.values
    side = cells("POD5_zs_o25", "Qwen2.5-Omni zero-shot p_yes (by-product of extraction; POD4 reports it officially)", df, ["p_yes_noinv"], ["p_yes_original_paper"], "paper", "POD5_zs_o25_noinv.csv")
    write("POD5_zs_o25_noinv", df, side, [f"{W}/o25noinv_pitt_zeroshot_scores.csv", f"{REF}/omni_pitt_zeroshot_scores.csv", "/root/p20/scripts/pod_manifest_noinv.csv"],
          "python3 extract_p20.py pod_manifest_noinv.csv o25noinv_pitt ad", {"median_answer_mass_noinv": float(z.answer_mass.median())})
if which == "nested":
    specs = [("enc_mac_seed", "encoder, 5 renumbered Mac-tie splits pitt_groupkfold5_mac_seed0..4 (paper 0.7706)", "paper_npz"),
             ("llm_pod_seed", "LM stages at audio positions, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.8032)", "rerun"),
             ("ans_pod_seed", "answer state, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper 0.7636)", "rerun"),
             ("enc_pod_seed", "encoder SUPPLEMENT, splits pitt_groupkfold5_pod_seed0..4_UNVERIFIED (paper pod json 0.7761)", "rerun")]
    if os.path.exists(f"{W}/noinvP_proj_pod_seed0_oof.csv"):
        specs.append(("proj_pod_seed0", "projector SUPPLEMENT (1 point, seed-0 split as nested_repeats_all.py; paper json 0.7507)", "rerunP"))
    for tag, what, old in specs:
        nw = pd.read_csv(f"{W}/{'noinvP' if old == 'rerunP' else 'noinv'}_{tag}_oof.csv"); pc = [c for c in nw.columns if c.startswith("p_")]
        df = nw.rename(columns={c: c + "__noinv" for c in pc}); new = [c + "__noinv" for c in pc]
        if old == "paper_npz":
            q = np.load(f"{REF}/pitt_enc_nested5_oof.npz", allow_pickle=True); ix = {n: i for i, n in enumerate(q["name"].astype(str))}
            j = [ix[c] for c in df.clip_id]; assert (q["label"][j] == df.label.values).all()
            olds = []
            for r in range(5): df[f"p_rep{r}__paper_mac"] = q["oof"][r][j]; olds.append(f"p_rep{r}__paper_mac")
            srcs = [f"{W}/noinv_{tag}_oof.csv", f"{REF}/pitt_enc_nested5_oof.npz"]; lab = "paper"
        else:
            of = f"{W}/{'origP' if old == 'rerunP' else 'orig'}_{tag}_oof.csv"
            o = pd.read_csv(of).set_index("clip_id").loc[df.clip_id]; assert (o.label.values == df.label.values).all()
            olds = []
            for c in pc: df[c + "__orig_rerun"] = o[c].values; olds.append(c + "__orig_rerun")
            srcs = [f"{W}/{'noinvP' if old == 'rerunP' else 'noinv'}_{tag}_oof.csv", of, "/root/p20/o25_pitt_states.npz"]; lab = "original_rerun"
        df["set"] = [SET[c] for c in df.clip_id]
        name = f"POD5_nested_{tag}"
        side = cells(f"POD5_{tag}", f"Qwen2.5-Omni nested probe, {what}, mean of per-repeat AUCs", df, new, olds, lab, f"{name}.csv")
        summ_new = json.load(open(f"{W}/{'noinvP' if old == 'rerunP' else 'noinv'}_nested_summary.json"))["results"]
        write(name, df, side, srcs + [f"{W}/o25noinv_pitt_states.npz", "/root/p20/scripts/pitt_folds_part20.json", f"{REF}/omni_pitt_nested_repeats.json"],
              f"python3 nested_p20.py o25noinv_pitt_states.npz pitt_folds_part20.json noinv {tag.split('_')[0]}:{tag.split('_',1)[1]}",
              {"noinv_summary": summ_new, "paper_json": json.load(open(f"{REF}/omni_pitt_nested_repeats.json")),
               "original_side": "paper per-clip npz (Mac run)" if old == "paper_npz" else "pod rerun of the same code on overnight2/part10/o25_pitt_states.npz; reproduces the paper per-repeat AUCs",
               "orig_rerun_summary": (json.load(open(f"{W}/{'origP' if old == 'rerunP' else 'orig'}_nested_summary.json"))["results"] if old != "paper_npz" else None)})
if which == "direction":
    d = json.load(open(f"{W}/dir_noinv.json")); pcl = pd.read_csv(f"{W}/dir_noinv_perclip.csv")
    pap = pd.read_csv(f"{REF}/readout_direction.csv").set_index("dataset").loc["pitt"]
    n, ns = d["n"], d["n_speakers"]; fn = "POD5_direction_noinv.csv"; side = []
    for key, id_, what, diff in (("cosine", "POD5_dir_cos", "cos(probe direction, mass-weighted Yes-minus-No unembedding), answer state, interviewer-free (paper -0.0109)", True),
                                 ("probe_auc", "POD5_dir_probe", "answer-state final-stage probe AUC, Mac split, interviewer-free (paper 0.7709)", False),
                                 ("after_removal_auc", "POD5_dir_after_removal", "probe AUC after removing the readout direction, held-out-fold standardised, interviewer-free (paper 0.7662)", False),
                                 ("probe_minus_after_removal", "POD5_dir_probe_minus_removed", "probe AUC minus after-removal AUC, paired, interviewer-free", True),
                                 ("kept_control_auc", "POD5_dir_kept_control", "matched control: same pooling with d kept, interviewer-free", False)):
        side.append(emit(id_, what, tuple(d[key]), n, ns, fn, diff))
    fl = d["random_floor_mean_abs_cos"]; flo, fhi = d["random_floor_2.5_97.5"]
    line = f"POD5_dir_floor | random floor mean |cos| over 2000 Gaussian directions (default_rng(0)), interviewer-free (paper 0.0135) | {fl:.4f} [{flo:.4f}, {fhi:.4f}] (2.5/97.5 of the 2000 |cos|) | n={n} n_spk={ns} | {MAC}/{fn} | {fl:.2f} [{flo:.2f}, {fhi:.2f}]"
    print(line); open(LINES, "a").write(line + "\n")
    open(ROWS, "a").write("\t".join(map(str, ["POD5_dir_floor", "random floor mean |cos|, 2000 Gaussian directions, default_rng(0) (interval = 2.5/97.5 of the 2000 values, not a bootstrap)", f"{fl:.4f}", f"{flo:.4f}", f"{fhi:.4f}", n, ns, f"{MAC}/{fn}", f"{fl:.2f} [{flo:.2f}, {fhi:.2f}]", "n/a"])) + "\n")
    for arm in ("conflict", "agreement"):
        m = pcl["set"].values == arm
        for key, id_ in (("probe", "POD5_dir_probe"), ("after_removal", "POD5_dir_after_removal")):
            side.append(emit(f"{id_}_{arm}", f"{key} AUC, answer state, interviewer-free, {arm} arm (full fit restricted to the arm)", tuple(d["arms"][arm][key]), int(m.sum()), int(pcl.speaker[m].nunique()), fn))
    dor = json.load(open(f"{W}/dir_orig.json")) if os.path.exists(f"{W}/dir_orig.json") else None
    srcs = [f"{W}/dir_noinv.json", f"{W}/dir_noinv_perclip.csv", f"{W}/o25noinv_pitt_states.npz", f"{W}/o25noinv_pitt_readout.pt", f"{REF}/readout_direction.csv", f"{W}/pitt_groupkfold5_mac.csv"]
    if dor: srcs += [f"{W}/dir_orig.json"]
    write("POD5_direction_noinv", pcl, side, srcs, "python3 direction_pod5.py o25noinv_pitt_states.npz o25noinv_pitt_readout.pt o25noinv_pitt_zeroshot_scores.csv pitt_groupkfold5_mac.csv dir_noinv noinv",
          {"paper_row": pap.to_dict(), "noinv_full": d, "original_states_rerun_on_pod": dor})
if which == "sft":
    s = pd.read_csv(f"{W}/omnisft_noinv_pitt_oof.csv"); p = pd.read_csv(f"{REF}/omnisft_pitt_oof.csv"); p["orig_clip_id"] = p.path.map(os.path.basename)
    p = p.set_index("orig_clip_id").loc[s.orig_clip_id]; assert (p.label.values == s.label.values).all() and (p["set"].values == s["set"].values).all() and (p.speaker.values == s.speaker.values).all()
    df = s.rename(columns={"p_yes": "p_yes_noinv"}); df["p_yes_original_paper"] = p.p_yes.values
    side = cells("POD5_sft", "Qwen2.5-Omni projector fine-tune (sft_projector.py recipe), out of fold, pitt_groupkfold5_pod.csv", df, ["p_yes_noinv"], ["p_yes_original_paper"], "paper", "POD5_sft_noinv.csv")
    y = df.label.values; side.append(emit("POD5_sft_rulepyes_overall", "projector fine-tune, interviewer-free, overall, part20-rule p_yes (12 single-token variants, full softmax) instead of the recipe's 2-logit p_yes", boot(y, df[["p_yes_rule"]].values, df.speaker.values), len(y), df.speaker.nunique(), "POD5_sft_noinv.csv"))
    write("POD5_sft_noinv", df, side, [f"{W}/omnisft_noinv_pitt_oof.csv", f"{W}/omnisft_noinv_pitt_sft.json", f"{REF}/omnisft_pitt_oof.csv", f"{W}/pitt_groupkfold5_pod.csv", "/root/p20/scripts/pod_manifest_noinv.csv"],
          "OMP_NUM_THREADS=4 python3 sft_p20.py pod_manifest_noinv.csv omnisft_noinv_pitt pitt_groupkfold5_pod.csv", {"sft_json": json.load(open(f"{W}/omnisft_noinv_pitt_sft.json")), "paper_sft_json_auc": {"overall": 0.7673, "conflict": 0.5189, "agreement": 0.8593}})
print("REPORT", which, "DONE", flush=True)
