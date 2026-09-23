#!/usr/bin/env python3
"""POD4c independent verifier (part 20).

Recomputes every value from the per-clip csv files. Nothing is imported from
the compute side.

AUC      explicit Mann-Whitney: every positive vs every negative, ties 0.5.
         Cross-checked by a trapezoid ROC built by hand (thresholds at every
         distinct score, ties handled by moving both rates at once).
CI       2000 draws, fresh numpy default_rng(0) per cell, SPEAKERS resampled
         with replacement (rng.choice over np.unique(speaker)), a draw is used
         only if both classes are present, percentile 2.5 / 97.5.
PAIRED   one speaker draw per replicate, both AUCs recomputed inside that draw,
         difference = original minus interviewer-free.
Arm      conflict / agreement read from the clip file name prefix and checked
         against the part16 set column.
"""
import csv, os, sys, json
import numpy as np

NB = 2000


def auc_mw(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    tot = 0.0
    for i in range(0, len(pos), 1024):
        b = pos[i:i + 1024][:, None]
        tot += (b > neg[None, :]).sum() + 0.5 * (b == neg[None, :]).sum()
    return float(tot / (len(pos) * len(neg)))


def auc_trap(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    P = (y == 1).sum(); N = (y == 0).sum()
    thr = np.unique(s)[::-1]
    tpr = [0.0]; fpr = [0.0]
    for t in thr:
        tpr.append(((s >= t) & (y == 1)).sum() / P)
        fpr.append(((s >= t) & (y == 0)).sum() / N)
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def _groups(spk):
    u = np.unique(spk)
    return u, {p: np.where(spk == p)[0] for p in u}


def boot_ci(spk, y, s, nb=NB, seed=0):
    rng = np.random.default_rng(seed)
    u, idx = _groups(spk)
    v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        v.append(auc_mw(y[ii], s[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def boot_paired(spk, y, s_orig, s_cut, nb=NB, seed=0):
    """original minus cut, one speaker draw per replicate"""
    rng = np.random.default_rng(seed)
    u, idx = _groups(spk)
    v = []
    for _ in range(nb):
        pick = rng.choice(u, size=len(u), replace=True)
        ii = np.concatenate([idx[p] for p in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        v.append(auc_mw(y[ii], s_orig[ii]) - auc_mw(y[ii], s_cut[ii]))
    v = np.array(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def arm_of(clip):
    b = os.path.basename(clip)
    if b.startswith("conflict_"):
        return "conflict"
    if b.startswith("agreement_"):
        return "agreement"
    return None


def load(path, score_col, spk_col=None, label_col="label", clip_col=None):
    rows = list(csv.DictReader(open(path)))
    h = rows[0].keys()
    spk_col = spk_col or ("speaker_id" if "speaker_id" in h else "speaker")
    clip_col = clip_col or next(c for c in ("clip_path", "clip", "path", "wav") if c in h)
    d = {}
    for r in rows:
        k = os.path.basename(r[clip_col])
        assert k not in d, ("duplicate clip", k)
        d[k] = (r[spk_col], int(float(r[label_col])), float(r[score_col]))
    return d, rows


def arrays(d, keys):
    return (np.array([d[k][0] for k in keys]),
            np.array([d[k][1] for k in keys]),
            np.array([d[k][2] for k in keys]))


def cell(d, arm=None):
    keys = sorted(k for k in d if arm is None or arm_of(k) == arm)
    spk, y, s = arrays(d, keys)
    a = auc_mw(y, s); t = auc_trap(y, s)
    assert abs(a - t) < 1e-9, ("MW vs trapezoid disagree", a, t)
    lo, hi, nu = boot_ci(spk, y, s)
    return dict(value=a, trap=t, lo=lo, hi=hi, n=len(keys),
                n_spk=int(len(np.unique(spk))), draws=nu,
                n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()))


def paired(d_orig, d_cut, arm=None):
    keys = sorted(k for k in set(d_orig) & set(d_cut) if arm is None or arm_of(k) == arm)
    spk, y, so = arrays(d_orig, keys)
    spk2, y2, sc = arrays(d_cut, keys)
    assert (spk == spk2).all() and (y == y2).all(), "speaker/label mismatch between orig and cut"
    a_o = auc_mw(y, so); a_c = auc_mw(y, sc)
    lo, hi, nu = boot_paired(spk, y, so, sc)
    return dict(value=a_o - a_c, a_orig=a_o, a_cut=a_c, lo=lo, hi=hi, n=len(keys),
                n_spk=int(len(np.unique(spk))), draws=nu)


if __name__ == "__main__":
    # usage: POD4c_verify.py cell <csv> <score_col>
    #        POD4c_verify.py paired <orig_csv> <cut_csv> <score_col>
    mode = sys.argv[1]
    if mode == "cell":
        d, _ = load(sys.argv[2], sys.argv[3])
        for arm in (None, "conflict", "agreement"):
            r = cell(d, arm)
            print(json.dumps(dict(arm=arm or "overall", **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})))
    elif mode == "paired":
        do, _ = load(sys.argv[2], sys.argv[4]); dc, _ = load(sys.argv[3], sys.argv[4])
        for arm in (None, "conflict", "agreement"):
            r = paired(do, dc, arm)
            print(json.dumps(dict(arm=arm or "overall", **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})))


# ---------------------------------------------------------------------------
# POD4c driver: recompute everything from the raw scorer csv + the Mac paper
# files, then compare with every <id>.json sidecar that has a <id>.RESULT.
# usage: POD4c_verify.py driver <synced POD4c dir> [--append]
# ---------------------------------------------------------------------------
import hashlib, glob

PROMPT = "Based only on this recording, does this speaker show signs of dementia? Answer with one word, Yes or No."
MAC_REF = {
    "af3_part3": "<local data dir>/release/overnight2/part3/af3_pitt.csv",
    "af3_part10": "<local data dir>/release/overnight2/part10/af3_pitt.csv",
    "af2_af2pitt468": "<local data dir>/paper work/paper1_local_runs/af2_results/af2_pitt468.csv",
}
CONFLICT_MAN = "<local data dir>/DementiaBank/pitt_conflict_manifest.csv"
P16_SETS = "<local data dir>/release/edaic_rerun/part16/POD4/pitt_no_interviewer.csv"
CUT_MAN = "manifests/part20/pitt_noinv_cut/manifest.csv"
ROWS = "reports/part20/rows/POD4c.tsv"


def sha1mb(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read(1 << 20)).hexdigest()


def arms_independent():
    a = {os.path.basename(r["segment_path"]): r["set"] for r in csv.DictReader(open(CONFLICT_MAN))}
    b = {r["clip"]: r["set"] for r in csv.DictReader(open(P16_SETS))}
    assert set(a) == set(b) and all(a[k] == b[k] == k.split("_")[0] for k in a), "arm sources disagree"
    return a


def load_raw(path, col="p_yes"):
    rows = list(csv.DictReader(open(path)))
    d = {}
    for r in rows:
        k = r["orig_clip_id"]
        assert k not in d
        d[k] = (r["speaker_id"], int(r["label"]), float(r[col]))
    return d, rows


def load_ref(path):
    d = {}
    for r in csv.DictReader(open(path)):
        k = os.path.basename(r["clip_path"])
        assert k not in d
        d[k] = (r["speaker_id"], int(r["label"]), float(r["p_yes"]))
    return d


def vs_half(v, lo, hi, diff):
    if diff:
        return "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
    return "CI above 0.5" if lo > 0.5 else ("CI below 0.5" if hi < 0.5 else "CI spans 0.5")


def compute_all(model, raw_csv):
    ARM = arms_independent()
    new, rows = load_raw(raw_csv)
    assert len(new) == 468, len(new)
    cut = {r["orig_clip_id"]: r for r in csv.DictReader(open(CUT_MAN))}
    checks = {}
    checks["n_raw"] = len(new)
    checks["prompt_all_verbatim"] = all(r["prompt"] == PROMPT for r in rows)
    checks["arm_col_matches_independent"] = all(r["arm"] == ARM[r["orig_clip_id"]] for r in rows)
    checks["cut_manifest_match"] = all(cut[k]["spk"] == new[k][0] and int(cut[k]["label"]) == new[k][1]
                                       and r_clip == cut[k]["clip"] for k, r_clip in
                                       ((r["orig_clip_id"], r["clip_id"]) for r in rows))
    refs = {t: load_ref(p) for t, p in MAC_REF.items() if t.startswith(model)}
    for t, d in refs.items():
        assert set(d) == set(new), t
        assert all(d[k][0] == new[k][0] and d[k][1] == new[k][1] for k in d), ("spk/label", t)
    out = {}
    for arm in ("all", "conflict", "agreement"):
        a = None if arm == "all" else arm
        dn = {k: v for k, v in new.items() if a is None or ARM[k] == a}
        keys = sorted(dn)
        spk, y, s = arrays(dn, keys)
        av = auc_mw(y, s); at = auc_trap(y, s); assert abs(av - at) < 1e-9
        lo, hi, nu = boot_ci(spk, y, s)
        out[f"{model}_noinv_{arm}"] = dict(value=av, lo=lo, hi=hi, n=len(keys), n_spk=len(np.unique(spk)), draws=nu,
                                           diff=False, file=f"{model}_noinv_{arm}.csv")
        if model == "af2":
            d3, _ = load_raw(raw_csv, "p_yes_rule3")
            _, y3, s3 = arrays(d3, keys)
            out[f"{model}_noinv_{arm}"]["auc_rule3"] = auc_mw(y3, s3)
        for t, d in refs.items():
            tag = t.split("_", 1)[1]
            _, yo, so = arrays(d, keys)
            assert (yo == y).all()
            ao = auc_mw(y, so); assert abs(ao - auc_trap(y, so)) < 1e-9
            lo2, hi2, nu2 = boot_ci(spk, y, so)
            out[f"{model}_orig_{tag}_{arm}"] = dict(value=ao, lo=lo2, hi=hi2, n=len(keys), n_spk=len(np.unique(spk)),
                                                    draws=nu2, diff=False, file=f"{model}_orig_{tag}_{arm}.csv")
            lo3, hi3, nu3 = boot_paired(spk, y, so, s)
            out[f"{model}_diff_{tag}_minus_noinv_{arm}"] = dict(value=ao - av, lo=lo3, hi=hi3, n=len(keys),
                                                                n_spk=len(np.unique(spk)), draws=nu3, diff=True,
                                                                a_orig=ao, a_cut=av,
                                                                file=f"{model}_diff_{tag}_minus_noinv_{arm}.csv")
    return out, checks, new, refs, ARM


def check_result_csv(path, rid, new, refs, ARM):
    """the per-result csv written by the compute side must equal the raw scores"""
    rows = list(csv.DictReader(open(path)))
    bad = 0
    for r in rows:
        k = r["orig_clip_id"]
        if ARM[k] != r.get("arm", ARM[k]): bad += 1
        if "p_yes_noinv" in r:
            if float(r["p_yes_noinv"]) != new[k][2]: bad += 1
            tag = rid.split("_diff_")[1].split("_minus")[0]
            if float(r["p_yes_original"]) != refs[f"{rid[:3]}_{tag}"][k][2]: bad += 1
        elif "_orig_" in rid:
            tag = rid.split("_orig_")[1].rsplit("_", 1)[0]
            if float(r["p_yes"]) != refs[f"{rid[:3]}_{tag}"][k][2]: bad += 1
        else:
            if float(r["p_yes"]) != new[k][2]: bad += 1
    return len(rows), bad


# ---- control: AF2 on the ORIGINAL windows cut to their first 30 s --------
ORIG_WAV_DIR = "<local data dir>/Desktop/clips_for_af3/pitt468"


def compute_ctrl(raw_noinv, raw_o30):
    import wave
    ARM = arms_independent()
    new, _ = load_raw(raw_noinv)
    o30, rows = load_raw(raw_o30)
    full = load_ref(MAC_REF["af2_af2pitt468"])
    assert set(o30) == set(new) == set(full) and len(o30) == 468
    checks = {"o30_n": len(o30),
              "o30_prompt_verbatim": all(r["prompt"] == PROMPT for r in rows),
              "o30_arm_col": all(r["arm"] == ARM[r["orig_clip_id"]] for r in rows),
              "o30_spk_label": all(o30[k][:2] == full[k][:2] == new[k][:2] for k in o30),
              "o30_scored_le_30": all(float(r["scored_s"]) <= 30.0 + 1e-9 for r in rows)}
    # source duration straight from the Mac originals
    dd = []
    for r in rows:
        w = wave.open(os.path.join(ORIG_WAV_DIR, r["orig_clip_id"]))
        dd.append(abs(w.getnframes() / w.getframerate() - float(r["dur_s"])))
        w.close()
    checks["o30_dur_matches_mac_original_max_abs"] = max(dd) < 1e-3
    # clips no longer than 30 s get no cut, so they must reproduce af2_pitt468.csv
    short = [r["orig_clip_id"] for r in rows if float(r["dur_s"]) <= 30.0]
    checks["o30_short_clips_reproduce_full"] = all(abs(o30[k][2] - full[k][2]) < 1e-3 for k in short)
    checks["o30_n_short_clips"] = len(short)
    out = {}
    for arm in ("all", "conflict", "agreement"):
        keys = sorted(k for k in o30 if arm == "all" or ARM[k] == arm)
        spk, y, s30 = arrays(o30, keys)
        _, _, sn = arrays(new, keys)
        _, _, sf_ = arrays(full, keys)
        nsp = len(np.unique(spk))
        a30 = auc_mw(y, s30); assert abs(a30 - auc_trap(y, s30)) < 1e-9
        an = auc_mw(y, sn); af = auc_mw(y, sf_)
        lo, hi, nu = boot_ci(spk, y, s30)
        d3, _ = load_raw(raw_o30, "p_yes_rule3"); _, _, s3 = arrays(d3, keys)
        out[f"af2_orig30_{arm}"] = dict(value=a30, lo=lo, hi=hi, n=len(keys), n_spk=nsp, draws=nu, diff=False,
                                        auc_rule3=auc_mw(y, s3), kind="o30")
        lo, hi, nu = boot_paired(spk, y, s30, sn)
        out[f"af2_diff_orig30_minus_noinv_{arm}"] = dict(value=a30 - an, lo=lo, hi=hi, n=len(keys), n_spk=nsp, draws=nu,
                                                         diff=True, a_orig=a30, a_cut=an, kind="o30_minus_noinv")
        lo, hi, nu = boot_paired(spk, y, sf_, s30)
        out[f"af2_diff_origfull_minus_orig30_{arm}"] = dict(value=af - a30, lo=lo, hi=hi, n=len(keys), n_spk=nsp, draws=nu,
                                                            diff=True, a_orig=af, a_cut=a30, kind="full_minus_o30")
    return out, checks, o30, full


def check_ctrl_csv(path, m, new, o30, full, ARM):
    rows = list(csv.DictReader(open(path)))
    bad = 0
    for r in rows:
        k = r["orig_clip_id"]
        if r["arm"] != ARM[k]: bad += 1
        if m["kind"] == "o30":
            bad += float(r["p_yes"]) != o30[k][2]
        elif m["kind"] == "o30_minus_noinv":
            bad += (float(r["p_yes_orig30"]) != o30[k][2]) + (float(r["p_yes_noinv"]) != new[k][2])
        else:
            bad += (float(r["p_yes_full"]) != full[k][2]) + (float(r["p_yes_orig30"]) != o30[k][2])
    return len(rows), bad


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "driver":
    D = sys.argv[2]; append = "--append" in sys.argv
    done_ids = set()
    if os.path.exists(ROWS):
        done_ids = {l.split("\t")[0] for l in open(ROWS).read().splitlines()[1:]}
    for model in ("af3", "af2"):
        raw = os.path.join(D, "work", f"{model}_noinv468.csv")
        results = sorted(glob.glob(os.path.join(D, f"{model}_*.RESULT")))
        if not results:
            print(f"[{model}] no RESULT yet"); continue
        mine, checks, new, refs, ARM = compute_all(model, raw)
        o30 = full = None
        if model == "af2" and any("orig30" in r for r in results):
            raw30 = next(p for p in (os.path.join(D, "work", "af2_orig30_468.csv"),
                                     os.path.join(D, "..", "raw_from_workspace_work", "af2_orig30_468.csv")) if os.path.exists(p))
            mc, cchecks, o30, full = compute_ctrl(raw, raw30)
            mine.update(mc); checks.update(cchecks)
        print(f"[{model}] checks {checks}")
        for rp in results:
            rid = os.path.basename(rp)[:-7]
            sj = json.load(open(os.path.join(D, rid + ".json")))
            m = mine.get(rid)
            if m is None:
                print(f"UNEXPECTED RESULT {rid} (no independent counterpart)"); continue
            if "kind" in m:
                nrow, bad = check_ctrl_csv(os.path.join(D, rid + ".csv"), m, new, o30, full, ARM)
            else:
                nrow, bad = check_result_csv(os.path.join(D, rid + ".csv"), rid, new, refs, ARM)
            cv, (clo, chi) = sj["value"], sj["ci95"]
            ok = (round(m["value"], 4) == cv and round(m["lo"], 4) == clo and round(m["hi"], 4) == chi
                  and sj["n"] == m["n"] and sj["n_speakers"] == m["n_spk"] and sj["prompt_verbatim"] == PROMPT
                  and sj["bootstrap"]["usable_draws"] == m["draws"] and nrow == m["n"] and bad == 0
                  and all(v for v in checks.values() if isinstance(v, bool)))
            exp_n = {"all": 468, "conflict": 146, "agreement": 322}[rid.rsplit("_", 1)[1]]
            ok = ok and m["n"] == exp_n
            # every named source that has a Mac original must be byte-identical (first 1 MB sha256)
            src_bad = []
            for s_ in sj.get("sources", []):
                mp = s_.get("mac_path", "")
                if os.path.exists(mp) and sha1mb(mp) != s_["sha256_first_1MB"]:
                    src_bad.append(mp)
                if s_["pod_path"].endswith("af2_orig30_468.csv") and sha1mb(raw30) != s_["sha256_first_1MB"]:
                    src_bad.append("raw orig30 csv differs from the synced copy")
                if s_["pod_path"].endswith(f"{model}_noinv468.csv") and sha1mb(raw) != s_["sha256_first_1MB"]:
                    src_bad.append("raw noinv csv differs from the synced copy")
            ok = ok and not src_bad
            if src_bad:
                print(f"SOURCE MISMATCH {rid}: {src_bad}")
            tag = "AGREE" if ok else "DISAGREE"
            line = (f"{tag} {rid}: computed {cv:.4f} [{clo:.4f}, {chi:.4f}] | verified {m['value']:.4f} "
                    f"[{m['lo']:.4f}, {m['hi']:.4f}] n={m['n']} n_spk={m['n_spk']} draws={m['draws']} "
                    f"csv_rows={nrow} csv_mismatch={bad}")
            if "auc_rule3" in m:
                line += f" rule3_verified={m['auc_rule3']:.4f} rule3_computed={sj.get('auc_rule3_ids')}"
            if m["diff"]:
                line += f" (orig {m['a_orig']:.4f} - noinv {m['a_cut']:.4f})"
            print(line)
            if ok and append and rid not in done_ids:
                macfile = f"scores/part20/POD4c{rid}.csv"
                with open(ROWS, "a") as f:
                    f.write("\t".join([rid, sj["what"], f"{cv:.4f}", f"{m['value']:.4f}", f"{m['lo']:.4f}",
                                       f"{m['hi']:.4f}", str(m["n"]), str(m["n_spk"]), macfile,
                                       f"{m['value']:.2f}", vs_half(m["value"], m["lo"], m["hi"], m["diff"]),
                                       "yes"]) + "\n")
                done_ids.add(rid)
                print(f"PASTE {rid}: {m['value']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}] n={m['n']} n_spk={m['n_spk']} "
                      f"| {m['value']:.2f} {vs_half(m['value'], m['lo'], m['hi'], m['diff'])} | FINAL (agrees to 4 dp)")
        missing = sorted(set(mine) - {os.path.basename(r)[:-7] for r in results})
        if missing:
            print(f"[{model}] not yet delivered: {missing}")
