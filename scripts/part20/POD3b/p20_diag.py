"""PART20 POD3b diagnostic (extra, not requested): M2 DistilBERT arms split by the RoBERTa rule arm of the same segment,
and M2 restricted to the 138 Part 14 speakers. Same AUC/bootstrap rules as p20_analyze.py."""
import sys, json, hashlib, datetime, os, numpy as np, pandas as pd
from scipy.stats import rankdata
IN, STEM, MACDIR = sys.argv[1:4]
def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return float((r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0))
def boot(df, B=2000):
    spk = df.speaker_id.values; u = np.unique(spk); by = {k: np.where(spk == k)[0] for k in u}
    rng = np.random.default_rng(0); out = []
    for _ in range(B):
        ii = np.concatenate([by[k] for k in rng.choice(u, size=len(u), replace=True)])
        v = auc(df.label.values[ii], df.p_yes.values[ii])
        if np.isfinite(v): out.append(v)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), len(out)
d = pd.read_csv(IN)
cells = [("conflict_robconflict", "M2 conflict rows whose RoBERTa rule arm is also conflict", d[(d.set=="conflict")&(d.rob_rule_arm=="conflict")]),
         ("conflict_robnone", "M2 conflict rows whose RoBERTa rule arm is none", d[(d.set=="conflict")&(d.rob_rule_arm=="none")]),
         ("agreement_robagreement_distinct", "M2 agreement distinct segments whose RoBERTa rule arm is also agreement", d[(d.set=="agreement")&(d.rob_rule_arm=="agreement")].drop_duplicates("seg_uid")),
         ("agreement_robnone_distinct", "M2 agreement distinct segments whose RoBERTa rule arm is none", d[(d.set=="agreement")&(d.rob_rule_arm=="none")].drop_duplicates("seg_uid")),
         ("conflict_p14speakers", "M2 conflict rows, 138 Part 14 speakers only", d[(d.set=="conflict")&d.in_part14_speakers]),
         ("agreement_distinct_p14speakers", "M2 agreement distinct segments, 138 Part 14 speakers only", d[(d.set=="agreement")&d.in_part14_speakers].drop_duplicates("seg_uid"))]
res = {}; rows = []; mac = f"{MACDIR}/{os.path.basename(STEM)}.csv"
for k, what, x in cells:
    a = auc(x.label, x.p_yes); lo, hi, nu = boot(x)
    res[k] = dict(what=what, value=a, lo=lo, hi=hi, n=len(x), n_distinct_segments=int(x.seg_uid.nunique()), n_speakers=int(x.speaker_id.nunique()), n_pos=int(x.label.sum()), usable_draws=nu)
    side = "above 0.5" if a > 0.5 else "below 0.5"; ex = "interval excludes 0.5" if (lo > .5 or hi < .5) else "interval includes 0.5"
    print(f"POD3b_diag_{k}: AUC {a:.4f} [{lo:.4f}, {hi:.4f}] n={len(x)} distinct_segments={x.seg_uid.nunique()} n_speakers={x.speaker_id.nunique()} file={mac} | {a:.2f} [{lo:.2f}, {hi:.2f}] {side} ({ex})")
    rows.append([f"POD3b_diag_{k}", what + " (diagnostic)", f"{a:.4f}", f"{lo:.4f}", f"{hi:.4f}", str(len(x)), str(x.speaker_id.nunique()), mac, f"{a:.2f} [{lo:.2f}, {hi:.2f}]", side])
d.to_csv(f"{STEM}.csv", index=False)
json.dump({"results": res, "input": os.path.abspath(IN), "input_sha256_first_1MB": hashlib.sha256(open(IN,"rb").read(1<<20)).hexdigest(),
  "roberta_rule_arm": "from part14_segments_sentiment.csv clearly_positive/clearly_negative by segment index: conflict = (label 1 and clearly_positive) or (label 0 and clearly_negative); agreement = the reverse; else none",
  "model_id": "Qwen/Qwen2.5-Omni-7B", "prompt_verbatim": d.prompt.iloc[0], "seed": 0, "bootstrap": "2000 draws, fresh default_rng(0) per cell, speakers resampled with replacement, percentile 2.5/97.5",
  "command": " ".join(["python3"] + sys.argv), "date": datetime.datetime.now(datetime.timezone.utc).isoformat()}, open(f"{STEM}.json", "w"), indent=1, default=float)
open(f"{STEM}.rows.tsv", "w").write("\n".join("\t".join(r) for r in rows) + "\n")
