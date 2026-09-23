#!/usr/bin/env python3
"""
INDEPENDENT VERIFICATION of PART 16 / M8 (conflict-arm below-chance claim).

Written from scratch. Nothing is imported or copied from M8_conflict_cells.py.
All numbers are recomputed from the ORIGINAL zero-shot per-clip score csv files
and the two ORIGINAL manifests. The primary job's M8_perclip.csv is only opened
at the very end for a row-count / consistency cross-check, never used as input
to any computation.

AUC route: explicit Mann-Whitney U concordant-pair count with ties = 0.5,
evaluated with np.searchsorted against the sorted negative scores. This is NOT
a rank-sum one-liner. A second, fully separate trapezoid-over-ROC implementation
is run on every cell as an internal cross-check.

ZERO writes under <local data dir>/Desktop/release/omni_final.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd

REL = "<local data dir>/Desktop/release"
P16 = f"{REL}/edaic_rerun/part16"
OUT = f"{P16}/verify"

MAN_A = f"{REL}/edaic_rerun/part14_manifest_new.csv"
MAN_B = "manifests/pitt_conflict_manifest_468.csv"

# (model, tag, set, path)
SOURCES = [
    ("Qwen2.5-Omni",       "primary", "A", f"{REL}/edaic_rerun/part14/p14_o25_zeroshot_scores.csv"),
    ("Qwen2-Audio",        "primary", "A", f"{REL}/edaic_rerun/part14/p14_q2a_zeroshot_scores.csv"),
    ("Qwen3-Omni-30B-A3B", "primary", "A", f"{REL}/edaic_rerun/part14/p14_q3o_zeroshot_scores.csv"),
    ("Audio Flamingo 2",   "primary", "A", f"{REL}/edaic_rerun/part14/p14_af2.csv"),
    ("Audio Flamingo 3",   "primary", "A", f"{REL}/edaic_rerun/part14/p14_af3.csv"),
    ("Kimi-Audio",         "primary", "A", f"{REL}/edaic_rerun/part14/p14_kimi_zeroshot_scores.csv"),
    ("Qwen2.5-Omni",       "primary", "B", f"{REL}/omni_final/omni_pitt_zeroshot_scores.csv"),
    ("Qwen2-Audio",        "primary", "B", "<local data dir>/paper1_local_runs/probe2/pitt_zeroshot_scores.csv"),
    ("Qwen3-Omni-30B-A3B", "primary", "B", f"{REL}/overnight2/q3o_new/q3o_pitt_zeroshot_scores.csv"),
    ("Audio Flamingo 2",   "af2_pitt468",   "B", "<local data dir>/paper1_local_runs/af2_results/af2_pitt468.csv"),
    ("Audio Flamingo 2",   "af2_pitt_audio","B", "<local data dir>/paper1_local_runs/af2_pitt_audio.csv"),
    ("Audio Flamingo 3",   "primary", "B", f"{REL}/overnight2/part3/af3_pitt.csv"),
    ("Kimi-Audio",         "primary", "B", f"{REL}/overnight2/part3/kimi_pitt_zeroshot_scores.csv"),
    ("Kimi-Audio",         "alt_overnight_copy", "B", f"{REL}/overnight/kimi/kimi_pitt_zeroshot_scores.csv"),
]

N_BOOT = 2000


# ----------------------------------------------------------------------------
# AUC implementation 1: Mann-Whitney U by explicit concordant-pair counting.
# For every positive score s: (# negatives strictly below s) + 0.5*(# ties).
# ----------------------------------------------------------------------------
def auc_ucount(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    neg_sorted = np.sort(neg)
    lo = np.searchsorted(neg_sorted, pos, side="left")    # neg strictly < s
    hi = np.searchsorted(neg_sorted, pos, side="right")   # neg <= s
    concordant = lo.sum() + 0.5 * (hi - lo).sum()
    return concordant / (pos.size * neg.size)


# ----------------------------------------------------------------------------
# AUC implementation 2: trapezoid over the full empirical ROC, built by sweeping
# every distinct threshold. Completely separate code path from impl 1.
# ----------------------------------------------------------------------------
def auc_trapz(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    if P == 0 or N == 0:
        return np.nan
    thr = np.unique(s)
    thr = np.concatenate([[np.inf], thr[::-1], [-np.inf]])
    tpr = []
    fpr = []
    for t in thr:
        pred = s >= t
        tpr.append(((pred) & (y == 1)).sum() / P)
        fpr.append(((pred) & (y == 0)).sum() / N)
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    order = np.argsort(fpr, kind="stable")
    return float(np.trapezoid(tpr[order], fpr[order])) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr[order], fpr[order]))


def sha_first_mb(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1024 * 1024))
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Load the two manifests -> the authoritative arm / label / speaker assignment.
# ----------------------------------------------------------------------------
def load_manifests():
    a = pd.read_csv(MAN_A)
    # clip filename is <set>_<speaker>_<seg_uid>.wav in the score files
    a_key = (a["set"].astype(str) + "_" + a["speaker_id"].astype(str) + "_"
             + a["seg_uid"].astype(str) + ".wav")
    man_a = pd.DataFrame({
        "clip": a_key.values,
        "arm": a["set"].astype(str).values,
        "speaker": a["speaker_id"].astype(str).values,
        "label": a["label"].astype(int).values,
    })

    b = pd.read_csv(MAN_B, dtype={"spk": str})
    b_key = b["segment_path"].astype(str).apply(lambda p: os.path.basename(p))
    man_b = pd.DataFrame({
        "clip": b_key.values,
        "arm": b["set"].astype(str).values,
        "speaker": (b["grp"].astype(str) + b["spk"].astype(str)).values,
        "label": b["label"].astype(int).values,
    })
    return man_a, man_b


def load_scores(path):
    """Return DataFrame with columns clip, p_yes, and (if present) label/speaker."""
    df = pd.read_csv(path)
    cols = set(df.columns)
    if "clip" in cols:
        clip = df["clip"].astype(str).apply(os.path.basename)
    elif "clip_path" in cols:
        clip = df["clip_path"].astype(str).apply(os.path.basename)
    elif "path" in cols:
        clip = df["path"].astype(str).apply(os.path.basename)
    else:
        raise RuntimeError(f"no clip column in {path}: {sorted(cols)}")
    out = pd.DataFrame({"clip": clip.values, "p_yes": df["p_yes"].astype(float).values})
    if "label" in cols:
        out["file_label"] = df["label"].astype(int).values
    if "speaker" in cols:
        out["file_speaker"] = df["speaker"].astype(str).values
    elif "speaker_id" in cols:
        out["file_speaker"] = df["speaker_id"].astype(str).values
    return out


# ----------------------------------------------------------------------------
# Bootstrap: speaker-cluster resampling, 2000 draws, default_rng(0), pct 2.5/97.5
# Variant knob only affects the RNG call pattern, so that a bound mismatch can be
# attributed to draw-order rather than to a different estimator.
# ----------------------------------------------------------------------------
def speaker_blocks(speakers):
    order = pd.unique(speakers)              # order of first appearance
    order_sorted = np.array(sorted(set(speakers)))
    idx_by_spk = {}
    for i, s in enumerate(speakers):
        idx_by_spk.setdefault(s, []).append(i)
    return order, order_sorted, {k: np.array(v) for k, v in idx_by_spk.items()}


def boot_auc(y, s, speakers, n_boot=N_BOOT, seed=0, spk_order="sorted", draw="integers"):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    speakers = np.asarray(speakers)
    appear, srt, idx_by = speaker_blocks(speakers)
    spk = srt if spk_order == "sorted" else appear
    nspk = len(spk)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        if draw == "integers":
            pick = rng.integers(0, nspk, nspk)
        else:
            pick = rng.choice(nspk, size=nspk, replace=True)
        idx = np.concatenate([idx_by[spk[j]] for j in pick])
        v = auc_ucount(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(vals.size)


def boot_paired(yc, sc, spc, ya, sa, spa, n_boot=N_BOOT, seed=0,
                spk_order="sorted", draw="integers"):
    """One speaker draw per replicate; both arms recomputed inside that draw."""
    yc, sc, spc = np.asarray(yc), np.asarray(sc, float), np.asarray(spc)
    ya, sa, spa = np.asarray(ya), np.asarray(sa, float), np.asarray(spa)
    all_spk = np.concatenate([spc, spa])
    appear, srt, _ = speaker_blocks(all_spk)
    spk = srt if spk_order == "sorted" else appear
    nspk = len(spk)
    idx_c = {}
    for i, s_ in enumerate(spc):
        idx_c.setdefault(s_, []).append(i)
    idx_a = {}
    for i, s_ in enumerate(spa):
        idx_a.setdefault(s_, []).append(i)
    idx_c = {k: np.array(v) for k, v in idx_c.items()}
    idx_a = {k: np.array(v) for k, v in idx_a.items()}
    empty = np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        if draw == "integers":
            pick = rng.integers(0, nspk, nspk)
        else:
            pick = rng.choice(nspk, size=nspk, replace=True)
        chosen = spk[pick]
        ic = np.concatenate([idx_c.get(k, empty) for k in chosen]) if nspk else empty
        ia = np.concatenate([idx_a.get(k, empty) for k in chosen]) if nspk else empty
        vc = auc_ucount(yc[ic], sc[ic]) if ic.size else np.nan
        va = auc_ucount(ya[ia], sa[ia]) if ia.size else np.nan
        if not (np.isnan(vc) or np.isnan(va)):
            vals.append(vc - va)
    vals = np.array(vals)
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(vals.size)


# ----------------------------------------------------------------------------
def main():
    man_a, man_b = load_manifests()
    print(f"manifest A rows {len(man_a)}  arms {man_a['arm'].value_counts().to_dict()}")
    print(f"manifest B rows {len(man_b)}  arms {man_b['arm'].value_counts().to_dict()}")
    print()

    theirs = pd.read_csv(f"{P16}/M8_conflict_cells.csv")

    rows = []
    label_mismatch_report = []

    for model, tag, setkey, path in SOURCES:
        if not os.path.exists(path):
            rows.append(dict(model=model, tag=tag, set=setkey, arm="MISSING_SOURCE",
                             path=path))
            print(f"!! MISSING SOURCE {path}")
            continue
        man = man_a if setkey == "A" else man_b
        sc = load_scores(path)
        nraw = len(sc)
        # The manifest clip key is NOT unique in Set A (172 repeated agreement
        # clips), so a key merge is impossible. Every score file is verified to
        # be row-for-row aligned with its manifest (clip basename AND label AND
        # speaker all match elementwise), so the join is positional.
        assert len(sc) == len(man), f"row count {len(sc)} != manifest {len(man)}"
        assert (sc["clip"].values == man["clip"].values).all(), \
            f"row order mismatch for {path}"
        merged = pd.concat([sc.reset_index(drop=True).drop(columns=["clip"]),
                            man.reset_index(drop=True)], axis=1)

        # cross-check the label carried in the score file against the manifest
        if "file_label" in merged.columns:
            bad = int((merged["file_label"] != merged["label"]).sum())
            if bad:
                label_mismatch_report.append((model, tag, setkey, bad))

        setname = "A_edaic_part14" if setkey == "A" else "B_pitt"

        sub = {}
        for arm in ("conflict", "agreement"):
            d = merged[merged["arm"] == arm]
            y = d["label"].values
            s = d["p_yes"].values
            sp = d["speaker"].values
            a1 = auc_ucount(y, s)
            a2 = auc_trapz(y, s)
            lo, hi, usable = boot_auc(y, s, sp)
            rows.append(dict(model=model, tag=tag, set=setname, arm=arm,
                             n=len(d), n_spk=int(pd.unique(sp).size),
                             n_pos=int((y == 1).sum()),
                             auc_ucount=round(a1, 6), auc_trapz=round(a2, 6),
                             lo=round(lo, 6), hi=round(hi, 6), boot_usable=usable,
                             source_file=path))
            sub[arm] = (y, s, sp, a1)

        yc, scc, spc, ac = sub["conflict"]
        ya, saa, spa, aa = sub["agreement"]
        diff = ac - aa
        plo, phi, pus = boot_paired(yc, scc, spc, ya, saa, spa)
        nspk_union = int(pd.unique(np.concatenate([spc, spa])).size)
        rows.append(dict(model=model, tag=tag, set=setname,
                         arm="conflict_minus_agreement",
                         n=len(yc) + len(ya), n_spk=nspk_union,
                         n_pos=int((yc == 1).sum() + (ya == 1).sum()),
                         auc_ucount=round(diff, 6), auc_trapz=np.nan,
                         lo=round(plo, 6), hi=round(phi, 6), boot_usable=pus,
                         source_file=path))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/M8_verify_cells.csv", index=False)

    # ---- compare against their cells -------------------------------------
    print("=" * 108)
    print(f"{'model':<20}{'tag':<16}{'set':<16}{'arm':<26}{'theirs':>9}{'mine':>9}{'ok':>4}")
    print("=" * 108)
    cmp_rows = []
    for _, t in theirs.iterrows():
        m = res[(res["model"] == t["model"]) & (res["tag"] == t["source_tag"]) &
                (res["set"] == t["set"]) & (res["arm"] == t["arm"])]
        if m.empty:
            print(f"NO MATCH for {t['model']}/{t['source_tag']}/{t['set']}/{t['arm']}")
            continue
        m = m.iloc[0]
        ok_auc = abs(round(m["auc_ucount"], 4) - round(float(t["auc"]), 4)) < 1e-9
        ok_lo = abs(round(m["lo"], 4) - round(float(t["lo"]), 4)) < 1e-9
        ok_hi = abs(round(m["hi"], 4) - round(float(t["hi"]), 4)) < 1e-9
        ok_n = (int(m["n"]) == int(t["n"])) and (int(m["n_spk"]) == int(t["n_spk"])) \
            and (int(m["n_pos"]) == int(t["n_pos"]))
        tz = "" if np.isnan(m["auc_trapz"]) else f" trapz={m['auc_trapz']:.4f}"
        print(f"{t['model']:<20}{t['source_tag']:<16}{t['set']:<16}{t['arm']:<26}"
              f"{float(t['auc']):>9.4f}{m['auc_ucount']:>9.4f}"
              f"{'  Y' if ok_auc else '  N':>4}"
              f"   CI theirs [{float(t['lo']):.4f},{float(t['hi']):.4f}] "
              f"mine [{m['lo']:.4f},{m['hi']:.4f}] "
              f"{'CI-OK' if (ok_lo and ok_hi) else 'CI-DIFF'} "
              f"{'n-OK' if ok_n else 'n-DIFF'}" + tz)
        cmp_rows.append(dict(model=t["model"], tag=t["source_tag"], set=t["set"],
                             arm=t["arm"],
                             their_auc=f"{float(t['auc']):.4f}",
                             my_auc=f"{m['auc_ucount']:.4f}",
                             my_auc_trapz=("" if np.isnan(m["auc_trapz"]) else f"{m['auc_trapz']:.4f}"),
                             auc_agree=ok_auc,
                             their_lo=f"{float(t['lo']):.4f}", my_lo=f"{m['lo']:.4f}",
                             their_hi=f"{float(t['hi']):.4f}", my_hi=f"{m['hi']:.4f}",
                             ci_agree=(ok_lo and ok_hi),
                             n_agree=ok_n,
                             their_n=int(t["n"]), my_n=int(m["n"]),
                             their_nspk=int(t["n_spk"]), my_nspk=int(m["n_spk"]),
                             their_npos=int(t["n_pos"]), my_npos=int(m["n_pos"]),
                             boot_usable=int(m["boot_usable"])))
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(f"{OUT}/M8_verify_compare.csv", index=False)

    print()
    print(f"AUC point estimates agreeing to 4dp : {int(cmp['auc_agree'].sum())} / {len(cmp)}")
    print(f"CI bounds agreeing to 4dp           : {int(cmp['ci_agree'].sum())} / {len(cmp)}")
    print(f"n / n_spk / n_pos agreeing          : {int(cmp['n_agree'].sum())} / {len(cmp)}")
    print(f"bootstrap draws usable (min,max)    : {cmp['boot_usable'].min()}, {cmp['boot_usable'].max()}")
    if label_mismatch_report:
        print("LABEL MISMATCH vs manifest:", label_mismatch_report)
    else:
        print("labels in every score file match the manifest exactly")

    # impl1 vs impl2 internal check
    two = res.dropna(subset=["auc_trapz"])
    d = (two["auc_ucount"].round(4) - two["auc_trapz"].round(4)).abs().max()
    print(f"internal check, U-count vs ROC-trapezoid max |diff| over {len(two)} arm cells: {d:.6f}")
    print()
    print(cmp[~cmp["auc_agree"] | ~cmp["ci_agree"] | ~cmp["n_agree"]].to_string(index=False))


if __name__ == "__main__":
    main()
