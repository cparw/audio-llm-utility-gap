"""PART 25 track C, second attempt (r2). Own code, written fresh; it does not import or copy C_build.py or C_verify.py.

For Qwen2-Audio, Qwen3-Omni-30B-A3B and Kimi-Audio on PC-GITA, NeuroVoz, MDVR-KCL, E-DAIC, Pitt, ADReSSo, ADReSS-2020:
  probe  = master_lookup "encoder probe" row (rows 17..178, 1-based data rows). Kimi-Audio has no encoder probe row,
           so its "LM probe" row is used and labelled as a substitute.
  answer = master_lookup "zero shot answer" row.
Each master value is recomputed from its per-clip file:
  answer: the zero-shot scores csv named by the row (a *_zeroshot.json source maps to its *_zeroshot_scores.csv).
  probe : a per-clip file with the five per-repeat out-of-fold probabilities. On disk these are the PART 25 Linux refits
          (pull/p25C-*/out/TAG_DS_STREAM_rep5_oof.csv) and, for Qwen3-Omni PC-GITA and Pitt, the PART 16 POD3 files the
          master rows name, and for Qwen3-Omni E-DAIC the PART 26 re-extraction plus five-repeat refit
          (part26/Qwen3-Omni-30B-A3B_E-DAIC/Q3O_EDAIC_perclip.csv). A refit counts only if its five per-repeat AUCs equal the per-repeat AUCs saved in the
          original nested_repeats json at 4 dp (that json was written by the original run, not by the refit).
AUC = sklearn.metrics.roc_auc_score; every point AUC is also recomputed by an exact pairwise count (P(pos > neg) + 0.5 ties).
Paired speaker bootstrap: 2000 draws, a fresh numpy default_rng(0) per cell, unique speaker ids as strings in np.unique
order (primary: the probe file's speaker column), idx = rng.choice(n_spk, size=n_spk, replace=True), all clips of each
drawn speaker, both terms in the same draw, in each draw the mean of the five per-repeat AUCs minus the answer AUC,
draws with one class only skipped and counted, 2.5 and 97.5 percentiles (numpy default linear).
Where the probe and answer files spell the same speaker grouping with different ids (one to one), the interval is also computed
with the answer file's ids; where the answer file's speaker column is a different grouping (e.g. one id per clip), it is not used.
Writes, per cell: r2/perclip/*.csv, r2/sidecars/*.json, r2/draws/*.npz; plus r2/C_r2_cells.json.
usage: /usr/local/bin/python3 C_r2_recheck.py"""
import os, sys, csv, json, glob, hashlib, datetime, warnings
import numpy as np, pandas as pd
import sklearn, scipy
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

C = "<local data dir>/release_from_mac/scores/part25/C"
R2 = f"{C}/r2"
ML = "<local data dir>/release/master/master_lookup.csv"
POD3 = "<local data dir>/release/edaic_rerun/part16/POD3"
Q3O_NEW = "<local data dir>/release/overnight2/q3o_new"
P26_Q3O_EDAIC = "<local data dir>/release_from_mac/scores/part26/Qwen3-Omni-30B-A3B_E-DAIC/Q3O_EDAIC_perclip.csv"
for d in ("perclip", "sidecars", "draws"): os.makedirs(f"{R2}/{d}", exist_ok=True)
NB = 2000
MODELS = [("Qwen2-Audio", "q2a"), ("Qwen3-Omni-30B-A3B", "q3o"), ("Kimi-Audio", "kimi")]
DSETS = [("pcgita", "PC-GITA", "PD"), ("neurovoz", "NeuroVoz", "PD"), ("kcl", "MDVR-KCL", "PD"), ("edaic", "E-DAIC", "MDD"),
         ("pitt", "Pitt", "AD"), ("adresso", "ADReSSo", "AD"), ("adress2020", "ADReSS-2020", "AD")]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def pair_auc(y, s):
    """exact pairwise AUC: share of (positive, negative) pairs ordered correctly, ties count one half"""
    y = np.asarray(y, int); s = np.asarray(s, float)
    pos = s[y == 1]; neg = s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))

def r4(x): return None if x is None else float(f"{x:.4f}")

# master_lookup: csv module, 1-based data rows
with open(ML, newline="") as f:
    rd = csv.reader(f); header = next(rd); data = list(rd)
H = {h: i for i, h in enumerate(header)}
def master(model, ds, stream):
    hit = [(k, r) for k, r in enumerate(data, start=1) if 17 <= k <= 178 and r[H["model"]] == model and r[H["dataset"]] == ds and r[H["stream"]] == stream]
    if len(hit) > 1: raise SystemExit(f"duplicate master rows {model} {ds} {stream}: {[k for k, _ in hit]}")
    return hit[0] if hit else (None, None)

def answer_csv(src):
    if src.endswith(".csv"): return src
    if src.endswith("_zeroshot.json"): return src.replace("_zeroshot.json", "_zeroshot_scores.csv")
    raise SystemExit("unknown answer source " + src)

def probe_perclip(tag, ds, stream_short, master_src):
    """returns (path, list of five probability columns, origin text) or (None, None, reason)"""
    if master_src.endswith("_nested_oof.csv") and os.path.exists(master_src):
        cols = [f"p_probe_rep{k}" for k in range(5)]
        return master_src, cols, "per-clip file named by the master row (PART 16 POD3, five per-repeat columns)"
    if tag == "q3o" and ds == "edaic" and stream_short == "enc" and os.path.exists(P26_Q3O_EDAIC):
        return P26_Q3O_EDAIC, [f"p_probe_seed{k}" for k in range(5)], ("PART 26 Qwen3-Omni E-DAIC re-extraction (GPU, zero-shot p_yes reproduced with max abs diff 0.0) "
                                                                        "and five-repeat Linux CPU refit (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1)")
    hits = sorted(glob.glob(f"{C}/pull/p25C-*/out/{tag}_{ds}_{stream_short}_rep5_oof.csv"))
    if len(hits) > 1: raise SystemExit(f"two refit files for {tag} {ds}: {hits}")
    if hits: return hits[0], [f"p_seed{k}" for k in range(5)], "PART 25 Linux CPU refit (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), " + hits[0].split("/pull/")[1].split("/")[0]
    return None, None, "no per-clip file with the five per-repeat probabilities on disk"

def json_per_repeat(src, tag, ds, stream_short):
    """per-repeat AUCs saved by the ORIGINAL run for this master row"""
    if src.endswith(".json"):
        j = json.load(open(src)); return j[stream_short]["per_repeat"], src
    if tag == "q3o" and ds == "pcgita":
        p = f"{POD3}/POD3_SIDECAR.json"
        return json.load(open(p))["streams_pcgita1100"]["enc"]["nested_per_repeat"], p + " [streams_pcgita1100.enc.nested_per_repeat]"
    if tag == "q3o" and ds == "pitt":
        p = "<local data dir>/release/edaic_rerun/part16/pod_sync/q3o_pitt_nested_repeats.json"
        return json.load(open(p))["enc"]["per_repeat"], p
    return None, None

def boot(y, S, z, spk):
    """paired speaker bootstrap; S = n x 5 probe probabilities, z = answer probabilities"""
    ids = np.unique(spk.astype(str)); where = {s: np.flatnonzero(spk == s) for s in ids}
    rng = np.random.default_rng(0)
    d = np.full(NB, np.nan); a = np.full(NB, np.nan); b = np.full(NB, np.nan)
    for t in range(NB):
        idx = rng.choice(len(ids), size=len(ids), replace=True)
        ii = np.concatenate([where[ids[i]] for i in idx])
        yy = y[ii]
        if yy.min() == yy.max(): continue
        a[t] = np.mean([roc_auc_score(yy, S[ii, k]) for k in range(S.shape[1])]); b[t] = roc_auc_score(yy, z[ii]); d[t] = a[t] - b[t]
    ok = ~np.isnan(d)
    lo, hi = np.percentile(d[ok], [2.5, 97.5])
    return dict(diff=d, probe=a, answer=b, lo=float(lo), hi=float(hi), usable=int(ok.sum()), n_spk=int(len(ids)),
                probe_lo=float(np.percentile(a[ok], 2.5)), probe_hi=float(np.percentile(a[ok], 97.5)),
                ans_lo=float(np.percentile(b[ok], 2.5)), ans_hi=float(np.percentile(b[ok], 97.5)))

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
env = dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, pandas=pd.__version__)
cells = []
for model, tag in MODELS:
    for ds, dname, cond in DSETS:
        rec = dict(model=model, dataset=dname, ds_key=ds, condition=cond, created_utc=now, script=os.path.abspath(__file__), env=env,
                   master_lookup=ML, master_lookup_sha256=sha256(ML))
        prow_i, prow = master(model, ds, "encoder probe"); probe_stream = "encoder probe"; sshort = "enc"
        if prow is None and tag == "kimi":
            prow_i, prow = master(model, ds, "LM probe"); probe_stream = "LM probe (substitute: Kimi-Audio has no encoder probe row)"; sshort = "llm"
        arow_i, arow = master(model, ds, "zero shot answer")
        rec.update(probe_stream=probe_stream, master_row_probe=prow_i, master_probe=float(prow[H["auc"]]), master_probe_n=int(prow[H["n"]]),
                   master_probe_estimator=prow[H["estimator"]], master_probe_source=prow[H["source"]],
                   master_row_answer=arow_i, master_answer=float(arow[H["auc"]]), master_answer_n=int(arow[H["n"]]),
                   master_answer_source=arow[H["source"]])
        # ---------- answer ----------
        af = answer_csv(arow[H["source"]]); A = pd.read_csv(af, dtype={"clip": str, "speaker": str})
        rec.update(answer_file=af, answer_file_sha256=sha256(af), answer_rows=int(len(A)), answer_dup_clips=int(A["clip"].duplicated().sum()))
        ya = A["label"].astype(int).values; za = A["p_yes"].astype(float).values
        rec["answer_auc_sklearn"] = float(roc_auc_score(ya, za)); rec["answer_auc_pairs"] = pair_auc(ya, za)
        rec["answer_verified"] = bool(r4(rec["answer_auc_sklearn"]) == r4(rec["master_answer"]) and abs(rec["answer_auc_sklearn"] - rec["answer_auc_pairs"]) < 1e-12
                                      and len(A) == rec["master_answer_n"])
        # ---------- probe ----------
        pf, cols, origin = probe_perclip(tag, ds, sshort, prow[H["source"]])
        jr, jsrc = json_per_repeat(prow[H["source"]], tag, ds, sshort)
        rec.update(probe_file=pf, probe_origin=origin, original_per_repeat=jr, original_per_repeat_source=jsrc)
        if jr is not None: rec["original_per_repeat_mean_4dp"] = r4(float(np.mean(jr)))
        single = False
        if pf is None:
            # the master mean-of-five cannot be recomputed; fall back to the single-split nested OOF file, labelled as a different estimator
            sf = f"{Q3O_NEW}/{tag}_{ds}_{sshort}_nested_oof.csv"; sj = f"{Q3O_NEW}/{tag}_{ds}_nested.json"
            if not os.path.exists(sf): raise SystemExit("no probe per-clip file at all for " + tag + ds)
            pf, cols, single = sf, ["p_probe"], True
            rec.update(probe_file=sf, probe_origin="single-split nested OOF (one repeat), NOT the master mean-of-five estimator",
                       single_split_json=sj, single_split_json_auc=json.load(open(sj))[sshort]["auc_nested_oof"])
        P = pd.read_csv(pf, dtype={"clip": str, "speaker": str})
        rec.update(probe_file_sha256=sha256(pf), probe_rows=int(len(P)), probe_dup_clips=int(P["clip"].duplicated().sum()))
        yp = P["label"].astype(int).values
        per = [float(roc_auc_score(yp, P[c].astype(float).values)) for c in cols]
        per_pairs = [pair_auc(yp, P[c].astype(float).values) for c in cols]
        rec["per_repeat_recomputed"] = per; rec["per_repeat_recomputed_4dp"] = [r4(v) for v in per]
        rec["per_repeat_pairs_maxabs_vs_sklearn"] = float(np.max(np.abs(np.array(per) - np.array(per_pairs))))
        rec["probe_recomputed"] = float(np.mean(per))
        if single:
            rec["per_repeat_equal_original_4dp"] = None
            rec["single_split_reproduced_4dp"] = bool(r4(per[0]) == r4(rec["single_split_json_auc"]))
            rec["probe_verified"] = False
            rec["probe_not_verified_reason"] = ("the master value is the mean of five per-repeat AUCs saved only as numbers in the nested_repeats json; "
                                                "the five per-repeat per-clip probabilities were never pulled and the 275-clip encoder states "
                                                "(q3o_edaic_states.npz) are not on the Mac or the G-Drive, so no refit is possible")
        else:
            rec["per_repeat_equal_original_4dp"] = bool(jr is not None and [r4(v) for v in per] == [r4(v) for v in jr])
            rec["probe_verified"] = bool(rec["per_repeat_equal_original_4dp"] and r4(rec["probe_recomputed"]) == r4(rec["master_probe"])
                                         and len(P) == rec["master_probe_n"] and rec["per_repeat_pairs_maxabs_vs_sklearn"] < 1e-12)
        # ---------- join on clip ----------
        Am = A.set_index("clip")
        keep = P["clip"].isin(Am.index).values
        J = P.loc[keep].reset_index(drop=True)
        rec.update(n_joined=int(len(J)), probe_only=int((~keep).sum()), answer_only=int((~A["clip"].isin(P["clip"])).sum()))
        la = Am.loc[J["clip"], "label"].astype(int).values; sa = Am.loc[J["clip"], "speaker"].astype(str).values; z = Am.loc[J["clip"], "p_yes"].astype(float).values
        y = J["label"].astype(int).values
        rec["label_disagreements"] = int((la != y).sum())
        sp = J["speaker"].astype(str).values
        pairs = set(zip(sp, sa))
        rec["speaker_ids_identical"] = bool((sp == sa).all())
        rec["speaker_one_to_one"] = bool(len(pairs) == len(set(sp)) == len(set(sa)))
        S = np.stack([J[c].astype(float).values for c in cols], 1)
        rec["probe_joined"] = float(np.mean([roc_auc_score(y, S[:, k]) for k in range(S.shape[1])]))
        rec["answer_joined"] = float(roc_auc_score(y, z))
        rec["diff_point"] = rec["probe_joined"] - rec["answer_joined"]
        rec["diff_master"] = rec["master_probe"] - rec["master_answer"]
        rec["n_pos"] = int(y.sum()); rec["n_neg"] = int(len(y) - y.sum())
        b = boot(y, S, z, sp)
        rec.update(n_spk=b["n_spk"], boot_usable=b["usable"], diff_lo=b["lo"], diff_hi=b["hi"], probe_lo=b["probe_lo"], probe_hi=b["probe_hi"],
                   answer_lo=b["ans_lo"], answer_hi=b["ans_hi"], excludes_zero=bool(b["lo"] > 0 or b["hi"] < 0),
                   share_draws_le0=float(np.mean(b["diff"][~np.isnan(b["diff"])] <= 0)),
                   interval_kind=("paired speaker bootstrap, mean of the five per-repeat AUCs minus the answer AUC in each draw" if not single else
                                  "SINGLE-SPLIT paired interval (one repeat); a different estimator from the master mean of five"),
                   speaker_ids_used="probe file speaker column")
        rec["n_unique_speaker_probe_file"] = int(len(set(sp))); rec["n_unique_speaker_answer_file"] = int(len(set(sa)))
        if not rec["speaker_ids_identical"] and rec["speaker_one_to_one"]:
            b2 = boot(y, S, z, sa)
            rec.update(alt_answer_speaker_ids_lo=b2["lo"], alt_answer_speaker_ids_hi=b2["hi"], alt_excludes_zero=bool(b2["lo"] > 0 or b2["hi"] < 0))
        elif not rec["speaker_one_to_one"]:
            rec["answer_file_speaker_note"] = (f"the answer file speaker column has {len(set(sa))} ids for {len(set(sp))} probe-file speakers; "
                                               "it is not the same grouping, so only the probe-file speaker ids are used")
        stem = f"C_r2_{tag}_{ds}_{sshort}" + ("_singlesplit" if single else "")
        dr = f"{R2}/draws/{stem}_draws.npz"
        np.savez(dr, diff=b["diff"], probe_mean5=b["probe"], answer=b["answer"], point_diff=rec["diff_point"], seed=0, n_boot=NB, n_spk=b["n_spk"],
                 speaker_ids=np.unique(sp))
        rec["draws_file"] = dr
        # compare with the first attempt's saved draws (same rule, other code)
        prev = sorted(glob.glob(f"{C}/draws/C_{tag}_{ds}_{sshort}*_draws.npz"))
        if prev:
            Pd = np.load(prev[0])["diff"]
            rec["first_attempt_draws"] = prev[0]
            rec["first_attempt_draws_maxabs"] = float(np.nanmax(np.abs(Pd - b["diff"]))) if Pd.shape == b["diff"].shape else None
        pc = f"{R2}/perclip/{stem}_perclip.csv"
        out = pd.DataFrame({"clip": J["clip"], "speaker_probe_file": sp, "speaker_answer_file": sa, "label": y})
        for k in range(S.shape[1]): out[f"p_probe_rep{k}" if not single else "p_probe_single_split"] = S[:, k]
        out["p_yes_answer"] = z
        out.to_csv(pc, index=False, float_format="%.10g"); rec["perclip_file"] = pc
        sc = f"{R2}/sidecars/{stem}.sidecar.json"; rec["sidecar_file"] = sc
        rec["method"] = __doc__
        json.dump(rec, open(sc, "w"), indent=1, default=str)
        cells.append(rec)
        print(f"{tag:4s} {ds:10s} {sshort} probe {rec['probe_recomputed']:.4f}/{rec['master_probe']} v={rec['probe_verified']} "
              f"ans {rec['answer_auc_sklearn']:.4f}/{rec['master_answer']} v={rec['answer_verified']} n={rec['n_joined']} spk={rec['n_spk']} "
              f"diff {rec['diff_point']:+.4f} [{rec['diff_lo']:+.4f}, {rec['diff_hi']:+.4f}] usable {rec['boot_usable']} "
              f"prev_draws_maxabs {rec.get('first_attempt_draws_maxabs')} spk_same {rec['speaker_ids_identical']} "
              + (f"alt [{rec['alt_answer_speaker_ids_lo']:+.4f}, {rec['alt_answer_speaker_ids_hi']:+.4f}]" if 'alt_answer_speaker_ids_lo' in rec else ""), flush=True)
json.dump(cells, open(f"{R2}/C_r2_cells.json", "w"), indent=1, default=str)
print("R2 DONE", len(cells))
