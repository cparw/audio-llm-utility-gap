"""PART20 POD5 verifier: direction RESULT cells. AUC cells recomputed from the per-clip csv with own code.
cos / floor compared with the verifier's own independent recomputation from the saved states on the pod
(POD5_dir_verify.py -> dir_mac.json, dir_mac_perclip.npz), which reads lm_head rows straight from the checkpoint."""
import sys; sys.path.insert(0, "<local data dir>/release/scores/part20/verify")
from POD5_verify import *
W = "reports/part20/verify/POD5_work"
REL = "<local data dir>/release"
which = sys.argv[1]   # noinv | orig
DUMP = []; allag = True
def emit(d, ag, det):
    global allag
    allag = allag and ag
    print(("AGREE " if ag else "DISAGREE ") + det, flush=True)
    if ag and d is not None:
        DUMP.append(d); print("PASTE", paste(d), flush=True)

if which == "noinv":
    res_id = "POD5_direction_noinv"
    pc = f"{W}/{res_id}.csv"; sc = json.load(open(f"{W}/{res_id}.sidecar.json"))
    rows = read_csv(pc)
    clips = [q["clip_id"] for q in rows]; spk = np.array([q["speaker"] for q in rows])
    y = np.array([int(q["label"]) for q in rows]); arm = np.array([q["set"] for q in rows])
    bc = basic_checks(clips, spk, y, arm); full = sc["noinv_full"]
    # fold file named = pitt_groupkfold5_mac.csv; check the csv fold column against the release file
    mis, miss = check_folds(clips, spk, [q["fold"] for q in rows], f"{REL}/folds/pitt_groupkfold5_mac.csv")
    # independent per-clip recomputation on the pod
    me = np.load(f"{W}/dir_mac_perclip.npz", allow_pickle=True); mj = json.load(open(f"{W}/dir_mac.json"))
    assert list(me["names"]) == clips
    dp = float(np.max(np.abs(me["oof"] - np.array([float(q["p_probe_oof"]) for q in rows]))))
    dz = float(np.max(np.abs(me["perp_z_rule"] - np.array([float(q["perp_z"]) for q in rows]))))
    dk = float(np.max(np.abs(me["keep_z_rule"] - np.array([float(q["keep_z"]) for q in rows]))))
    dd = float(np.max(np.abs(me["proj_rule"] - np.array([float(q["proj_d"]) for q in rows]))))
    zs = {q["orig_clip_id"]: float(q["p_yes"]) for q in read_csv(f"{W}/o25noinv_pitt_zeroshot_scores.csv")}
    dzs = max(abs(float(q["p_yes_zeroshot"]) - zs[q["clip_id"]]) for q in rows)
    ids_ok = (mj["ids"]["rule"] == [full["yes_ids"], full["no_ids"]])
    print(f"basic {bc} | fold mismatches vs release mac file {mis} missing {miss} | fold_file named {full['fold_file']}")
    print(f"independent recompute vs csv: max|oof| {dp:.2e} max|perp_z| {dz:.2e} max|keep_z| {dk:.2e} max|proj_d| {dd:.2e} "
          f"| zero-shot col vs extraction {dzs:.2e} | yes/no ids match {ids_ok} | prompt {sc['prompt_verbatim'] == PROMPT}")
    base_ok = bc["ok"] and mis == 0 and miss == 0 and max(dp, dz, dk, dd, dzs) < 1e-9 and ids_ok and sc["prompt_verbatim"] == PROMPT \
        and os.path.basename(full["fold_file"]) == "pitt_groupkfold5_mac.csv"
    cells = {c["id"]: c for c in sc["cells"]}
    col = lambda c: np.array([float(q[c]) for q in rows])
    spec = {"POD5_dir_probe": ("single", "p_probe_oof", None, None), "POD5_dir_after_removal": ("single", "perp_z", None, None),
            "POD5_dir_kept_control": ("single", "keep_z", None, None), "POD5_dir_probe_minus_removed": ("paired", "p_probe_oof", "perp_z", None),
            "POD5_dir_probe_conflict": ("single", "p_probe_oof", None, "conflict"), "POD5_dir_after_removal_conflict": ("single", "perp_z", None, "conflict"),
            "POD5_dir_probe_agreement": ("single", "p_probe_oof", None, "agreement"), "POD5_dir_after_removal_agreement": ("single", "perp_z", None, "agreement")}
    for cid, (kind, a, b, ar) in spec.items():
        m = None if ar is None else arm == ar
        if kind == "single":
            r = point_and_ci(spk, y, [col(a)], mask=m); tr_ok = abs(r["value"] - r["value_trap"]) < 1e-12
        else:
            r = paired_ci(spk, y, [col(a)], [col(b)], mask=m); tr_ok = True
        c = cells[cid]
        d, ag = row(cid, c["what"], c["value"], r["value"], c["lo"], c["hi"], r["lo"], r["hi"], r["n"], r["n_spk"], sc["per_clip_csv"])
        ag = ag and tr_ok and base_ok and c["n"] == r["n"] and c["n_speakers"] == r["n_spk"]; d["agree_4dp"] = "yes" if ag else "NO"
        emit(d, ag, f"{cid}: computed {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] | verified {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']}/{r['n_spk']}")
    # cosine (independent refit per draw on the pod) and floor
    c = cells["POD5_dir_cos"]; v = mj["rule"]
    d, ag = row("POD5_dir_cos", c["what"], c["value"], v["cos_wraw_d"], c["lo"], c["hi"], v["cos_lo"], v["cos_hi"], mj["n"], mj["n_spk"], sc["per_clip_csv"])
    ag = ag and base_ok and v["cos_valid"] == c["valid_draws"]; d["agree_4dp"] = "yes" if ag else "NO"
    emit(d, ag, f"POD5_dir_cos: computed {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] | verified {v['cos_wraw_d']:.4f} [{v['cos_lo']:.4f}, {v['cos_hi']:.4f}] (legacy ids {mj['legacy']['cos_wraw_d']:.4f})")
    fl = full["random_floor_2.5_97.5"]
    d, ag = row("POD5_dir_floor", "random floor, mean |cos| of 2000 Gaussian directions (fresh default_rng(0)) with the mass-weighted readout direction, interviewer-free (paper 0.0135)",
                full["random_floor_mean_abs_cos"], v["floor_mean"], fl[0], fl[1], v["floor_lo"], v["floor_hi"], mj["n"], mj["n_spk"], sc["per_clip_csv"])
    ag = ag and base_ok; d["agree_4dp"] = "yes" if ag else "NO"
    emit(d, ag, f"POD5_dir_floor: computed {full['random_floor_mean_abs_cos']:.4f} [{fl[0]:.4f}, {fl[1]:.4f}] | verified {v['floor_mean']:.4f} [{v['floor_lo']:.4f}, {v['floor_hi']:.4f}]")
    # auc_d (projection) and gate
    ad = full["auc_d"]; r = point_and_ci(spk, y, [col("proj_d")])
    d, ag = row("POD5_dir_auc_d", "AUC of the projection on the readout direction (gate: vs zero-shot 0.7231), interviewer-free",
                ad[0], r["value"], ad[1], ad[2], r["lo"], r["hi"], r["n"], r["n_spk"], sc["per_clip_csv"])
    ag = ag and base_ok; d["agree_4dp"] = "yes" if ag else "NO"
    emit(d, ag, f"POD5_dir_auc_d: computed {ad[0]:.4f} [{ad[1]:.4f}, {ad[2]:.4f}] | verified {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] gate delta {r['value'] - auc_mw(y, col('p_yes_zeroshot')):+.4f}")
else:
    res_id = "POD5_direction_origrerun_check"
    pc = f"{W}/{res_id}.csv"; sc = json.load(open(f"{W}/{res_id}.sidecar.json"))
    rows = read_csv(pc)
    clips = [q["clip_id"] for q in rows]; spk = np.array([q["speaker"] for q in rows])
    y = np.array([int(q["label"]) for q in rows]); arm = np.array([q["set"] for q in rows])
    bc = basic_checks(clips, spk, y, arm)
    mis, miss = check_folds(clips, spk, [q["fold"] for q in rows], f"{REL}/folds/pitt_groupkfold5_mac.csv")
    # noinv side identical to the noinv RESULT csv; orig side zero-shot identical to the paper file
    nv = {q["clip_id"]: q for q in read_csv(f"{W}/POD5_direction_noinv.csv")}
    dn = max(abs(float(q[f"{c}__noinv"]) - float(nv[q["clip_id"]][c])) for q in rows for c in ("p_probe_oof", "perp_z", "keep_z", "proj_d"))
    pap = {q["clip"]: float(q["p_yes"]) for q in read_csv(f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv")}
    dpz = max(abs(float(q["p_yes_zeroshot__orig_rerun"]) - pap[q["clip_id"]]) for q in rows)
    print(f"basic {bc} | fold mismatches {mis}/{miss} | noinv side vs noinv RESULT {dn:.2e} | orig zero-shot col vs omni_final {dpz:.2e}")
    base_ok = bc["ok"] and mis == 0 and miss == 0 and dn < 1e-12 and dpz < 1e-12 and sc["prompt_verbatim"] == PROMPT
    cells = {c["id"]: c for c in sc["cells"]}
    col = lambda c: np.array([float(q[c]) for q in rows])
    spec = {"POD5_dir_kept_minus_removed": ("paired", "keep_z__noinv", "perp_z__noinv"),
            "POD5_dir_orig_probe": ("single", "p_probe_oof__orig_rerun", None), "POD5_dir_orig_after_removal": ("single", "perp_z__orig_rerun", None),
            "POD5_dir_orig_kept_minus_removed": ("paired", "keep_z__orig_rerun", "perp_z__orig_rerun"),
            "POD5_dir_probe_diff": ("paired", "p_probe_oof__orig_rerun", "p_probe_oof__noinv"),
            "POD5_dir_after_removal_diff": ("paired", "perp_z__orig_rerun", "perp_z__noinv")}
    for cid, (kind, a, b) in spec.items():
        if kind == "single":
            r = point_and_ci(spk, y, [col(a)]); tr_ok = abs(r["value"] - r["value_trap"]) < 1e-12
        else:
            r = paired_ci(spk, y, [col(a)], [col(b)]); tr_ok = True
        c = cells[cid]
        d, ag = row(cid, c["what"], c["value"], r["value"], c["lo"], c["hi"], r["lo"], r["hi"], r["n"], r["n_spk"], sc["per_clip_csv"])
        ag = ag and tr_ok and base_ok and c["n"] == r["n"] and c["n_speakers"] == r["n_spk"]; d["agree_4dp"] = "yes" if ag else "NO"
        emit(d, ag, f"{cid}: computed {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] | verified {r['value']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] n={r['n']}/{r['n_spk']}")
    # orig cos: needs an independent refit on the original states (POD5_dir_verify.py on the pod, dir_orig.json)
    p = f"{W}/dir_orig_verify.json"
    c = cells["POD5_dir_orig_cos"]
    if os.path.exists(p):
        mj = json.load(open(p)); v = mj["legacy"]
        d, ag = row("POD5_dir_orig_cos", c["what"], c["value"], v["cos_wraw_d"], c["lo"], c["hi"], v["cos_lo"], v["cos_hi"], mj["n"], mj["n_spk"], sc["per_clip_csv"])
        ag = ag and base_ok; d["agree_4dp"] = "yes" if ag else "NO"
        emit(d, ag, f"POD5_dir_orig_cos: computed {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] | verified {v['cos_wraw_d']:.4f} [{v['cos_lo']:.4f}, {v['cos_hi']:.4f}]")
    else:
        emit(None, False, "POD5_dir_orig_cos: PENDING independent refit on the original states")
json.dump({"rows": DUMP, "all_agree": allag}, open(f"POD5_check_dir_{which}.rows.json", "w"), indent=1)
