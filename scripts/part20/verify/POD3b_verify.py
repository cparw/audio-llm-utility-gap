#!/usr/bin/env python3
"""PART 20 POD3b independent verifier.
Recomputes every POD3b value from the per-clip csv alone, with its own code.
AUC: explicit Mann-Whitney over all pos x neg pairs (ties 0.5), cross-checked by a trapezoid ROC.
Intervals: 2000 draws, fresh numpy default_rng(0) per cell, speakers resampled with replacement
(rng.integers(0,K,K) over np.unique speaker ids), degenerate draws skipped and counted,
percentile 2.5 / 97.5. Paired delta: one speaker draw per replicate, conflict minus agreement.
The bootstrap AUC per draw is the same explicit pairwise Mann-Whitney, weighted by how many
times each row's speaker was drawn (identical to recomputing on the concatenated resample).
Imports nothing from the compute job.
"""
import sys, json, argparse
import numpy as np, pandas as pd

PROMPT = ("Based only on how this person's voice sounds, does this speaker show signs of "
          "depression? Answer with one word, Yes or No.")
DRAWS = 2000


def mw_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum(); eq = (p[:, None] == n[None, :]).sum()
    return (gt + 0.5 * eq) / (len(p) * len(n))


def trap_auc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    p = np.sort(s[y == 1]); n = np.sort(s[y == 0])
    th = np.unique(s)[::-1]
    tpr = [0.0] + [1.0 - np.searchsorted(p, t, side="left") / len(p) for t in th]
    fpr = [0.0] + [1.0 - np.searchsorted(n, t, side="left") / len(n) for t in th]
    tpr = np.array(tpr); fpr = np.array(fpr)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


class Arm:
    """Pairwise comparison matrix for one set of rows, for weighted bootstrap MW."""
    def __init__(self, y, s, spk_idx):
        y = np.asarray(y).astype(int); s = np.asarray(s, float)
        self.pi = spk_idx[y == 1]; self.ni = spk_idx[y == 0]
        p = s[y == 1]; n = s[y == 0]
        self.M = (p[:, None] > n[None, :]).astype(float) + 0.5 * (p[:, None] == n[None, :])
        self.point = self.M.mean() if self.M.size else float("nan")

    def draw(self, cnt):
        wp = cnt[self.pi].astype(float); wn = cnt[self.ni].astype(float)
        sp, sn = wp.sum(), wn.sum()
        if sp == 0 or sn == 0:
            return float("nan")
        return float(wp @ self.M @ wn) / (sp * sn)


def spk_index(spk):
    spk = np.asarray(spk).astype(str)
    uniq, inv = np.unique(spk, return_inverse=True)
    return uniq, inv


def boot_single(y, s, spk, draws=DRAWS, seed=0):
    uniq, inv = spk_index(spk); K = len(uniq)
    arm = Arm(y, s, inv); rng = np.random.default_rng(seed)
    out = []; bad = 0
    for _ in range(draws):
        d = rng.integers(0, K, K); cnt = np.bincount(d, minlength=K)
        a = arm.draw(cnt)
        if np.isnan(a): bad += 1
        else: out.append(a)
    out = np.array(out)
    return dict(auc=mw_auc(y, s), trap=trap_auc(y, s), lo=float(np.percentile(out, 2.5)),
                hi=float(np.percentile(out, 97.5)), n=int(len(y)), n_spk=int(K), bad=bad,
                n_pos=int((np.asarray(y) == 1).sum()), n_neg=int((np.asarray(y) == 0).sum()))


def boot_paired(yc, sc, spkc, ya, sa, spka, draws=DRAWS, seed=0):
    allspk = np.concatenate([np.asarray(spkc).astype(str), np.asarray(spka).astype(str)])
    uniq, inv = np.unique(allspk, return_inverse=True); K = len(uniq)
    ic = inv[:len(yc)]; ia = inv[len(yc):]
    A = Arm(yc, sc, ic); B = Arm(ya, sa, ia); rng = np.random.default_rng(seed)
    dc, da, dd = [], [], []; bad = 0
    for _ in range(draws):
        d = rng.integers(0, K, K); cnt = np.bincount(d, minlength=K)
        c = A.draw(cnt); g = B.draw(cnt)
        if np.isnan(c) or np.isnan(g): bad += 1; continue
        dc.append(c); da.append(g); dd.append(c - g)
    dc, da, dd = map(np.array, (dc, da, dd))
    pc, pg = mw_auc(yc, sc), mw_auc(ya, sa)
    return dict(conflict=pc, c_lo=np.percentile(dc, 2.5), c_hi=np.percentile(dc, 97.5),
                agreement=pg, a_lo=np.percentile(da, 2.5), a_hi=np.percentile(da, 97.5),
                delta=pc - pg, d_lo=float(np.percentile(dd, 2.5)), d_hi=float(np.percentile(dd, 97.5)),
                n_spk=int(K), bad=bad)


def f4(x):
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def load_join(perclip, manifest):
    m = pd.read_csv(manifest)
    m["clip"] = m["set"] + "_" + m["speaker_id"].astype(str) + "_" + m["seg_uid"] + ".wav"
    c = pd.read_csv(perclip)
    return m, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perclip", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--score_col", default="p_yes")
    ap.add_argument("--key", default="clip", help="manifest column to join on: clip or seg_uid")
    ap.add_argument("--csv_key", default=None, help="per-clip csv column holding that key (default: same name)")
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    m["clip"] = m["set"] + "_" + m["speaker_id"].astype(str) + "_" + m["seg_uid"] + ".wav"
    c = pd.read_csv(a.perclip)
    rep = {}
    rep["manifest_rows"] = len(m); rep["manifest_pairs"] = int(m.pair_id.nunique())
    rep["manifest_speakers"] = int(m.speaker_id.nunique())
    rep["perclip_rows"] = len(c)
    key = a.key
    if a.csv_key and a.csv_key != key:
        c[key] = c[a.csv_key]
    if key == "clip":
        c["clip"] = c["clip"].astype(str).str.split("/").str[-1]
    rep["perclip_unique_keys"] = int(c[key].nunique())
    dup = c[c.duplicated(key, keep=False)]
    if len(dup):
        spread = dup.groupby(key)[a.score_col].agg(lambda v: float(v.max() - v.min())).max()
        rep["perclip_dup_key_max_score_spread"] = spread
    cu = c.drop_duplicates(key).set_index(key)
    missing = sorted(set(m[key]) - set(cu.index))
    extra = sorted(set(cu.index) - set(m[key]))
    rep["manifest_keys_missing_in_perclip"] = len(missing); rep["perclip_keys_not_in_manifest"] = len(extra)
    rep["missing_examples"] = missing[:5]
    j = m.copy(); j["s"] = j[key].map(cu[a.score_col]).astype(float)
    rep["nan_scores_after_join"] = int(j["s"].isna().sum())
    # label / speaker cross-checks where the per-clip csv carries them
    for col, mcol in (("label", "label"), ("speaker", "speaker_id"), ("speaker_id", "speaker_id"), ("set", "set"), ("arm", "set")):
        if col in cu.columns:
            v = j[key].map(cu[col])
            rep[f"mismatch_{col}"] = int((v.astype(str) != j[mcol].astype(str)).sum())
    mcol_ = "mass" if "mass" in cu.columns else ("answer_mass" if "answer_mass" in cu.columns else None)
    if mcol_:
        mass = j[key].map(cu[mcol_]).astype(float)
        rep["mass_median"] = float(mass.median()); rep["mass_min"] = float(mass.min())
    s = j["s"].values
    rep["p_yes_range"] = [float(np.nanmin(s)), float(np.nanmax(s))]
    if a.sidecar:
        sc = json.load(open(a.sidecar))
        flat = json.dumps(sc)
        rep["sidecar_prompt_verbatim"] = PROMPT in flat or sc.get("prompt") == PROMPT
        rep["sidecar_prompt"] = sc.get("prompt")
        rep["sidecar_checkpoint"] = sc.get("checkpoint") or sc.get("model")
        rep["sidecar_window"] = sc.get("window_seconds")
    if "prompt" in cu.columns:
        pr = c["prompt"].astype(str).unique().tolist()
        rep["perclip_prompts"] = pr
        rep["perclip_prompt_verbatim"] = (len(pr) == 1 and pr[0] == PROMPT)

    con = j[j.set == "conflict"]; agr = j[j.set == "agreement"]
    agr_d = agr.drop_duplicates("seg_uid"); con_d = con.drop_duplicates("seg_uid")
    res = {}
    for name, df in (("conflict", con), ("agreement_all", agr), ("agreement_distinct", agr_d),
                     ("conflict_distinct", con_d), ("pooled_all", j),
                     ("pooled_distinct", pd.concat([con_d, agr_d]))):
        r = boot_single(df.label.values, df.s.values, df.speaker_id.values)
        r["n_distinct_seg"] = int(df.seg_uid.nunique())
        res[name] = r
    pr = boot_paired(con.label.values, con.s.values, con.speaker_id.values,
                     agr.label.values, agr.s.values, agr.speaker_id.values)
    prd = boot_paired(con_d.label.values, con_d.s.values, con_d.speaker_id.values,
                      agr_d.label.values, agr_d.s.values, agr_d.speaker_id.values)
    res["paired_all"] = pr; res["paired_distinct"] = prd
    out = dict(report=rep, results=res)
    print(json.dumps(rep, indent=1, default=str))
    print(f"{'cell':22s} {'AUC':>7s} {'trap':>7s} {'lo':>7s} {'hi':>7s} {'n':>5s} {'pos':>4s} {'neg':>4s} {'dseg':>5s} {'spk':>4s} {'bad':>3s}")
    for k, r in res.items():
        if k.startswith("paired"):
            continue
        print(f"{k:22s} {f4(r['auc']):>7s} {f4(r['trap']):>7s} {f4(r['lo']):>7s} {f4(r['hi']):>7s} {r['n']:5d} {r['n_pos']:4d} {r['n_neg']:4d} {r['n_distinct_seg']:5d} {r['n_spk']:4d} {r['bad']:3d}")
    for k in ("paired_all", "paired_distinct"):
        r = res[k]
        print(f"{k:22s} conflict {f4(r['conflict'])} [{f4(r['c_lo'])}, {f4(r['c_hi'])}]  agreement {f4(r['agreement'])} [{f4(r['a_lo'])}, {f4(r['a_hi'])}]  delta {f4(r['delta'])} [{f4(r['d_lo'])}, {f4(r['d_hi'])}]  spk {r['n_spk']} bad {r['bad']}")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
