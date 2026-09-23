#!/usr/bin/env python3
"""PART 20 POD3b verifier, diagnostic cells by RoBERTa rule arm and by Part 14 speaker set.
Own code: derives the RoBERTa rule arm from part14_segments_sentiment.csv (row index = seg_uid number),
takes p_yes from the verified sst2_pairs_o25_audio.csv, recomputes AUC (explicit MW + trapezoid) and
speaker bootstrap via POD3b_verify.boot_single (verifier's own module)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "<local data dir>/release/scores/part20/verify")
from POD3b_verify import boot_single
R = "<local data dir>/release/edaic_rerun"
res = pd.read_csv("scores/part20/POD3b/sst2_pairs_o25_audio.csv")
man = pd.read_csv(f"{R}/part16/M2_distilbert_manifest.csv")
seg = pd.read_csv(f"{R}/part14_segments_sentiment.csv")
p14 = pd.read_csv(f"{R}/part14_manifest_new.csv")
j = man.merge(res[["seg_uid", "set", "pair_id", "p_yes"]], on=["seg_uid", "set", "pair_id"], how="left")
assert j.p_yes.notna().all() and len(j) == 2668
idx = j.seg_uid.str[1:].astype(int).values
s = seg.loc[idx].reset_index(drop=True)
assert (s.pid.values == j.speaker_id.values).all() and np.allclose(s.start.values, j.start.values)
print("label y vs manifest label equal:", bool((s.y.values == j.label.values).all()))
print("clearly_positive == p_pos>=0.70:", bool(((s.p_pos >= 0.70).astype(int) == s.clearly_positive).all()),
      " clearly_negative == p_neg>=0.70:", bool(((s.p_neg >= 0.70).astype(int) == s.clearly_negative).all()))
cp = s.clearly_positive.values.astype(bool); cn = s.clearly_negative.values.astype(bool); y = j.label.values
rob = np.where((y == 1) & cp | (y == 0) & cn, "conflict", np.where((y == 1) & cn | (y == 0) & cp, "agreement", "none"))
j["rob"] = rob
j["in14"] = j.speaker_id.isin(set(p14.speaker_id))
con = j[j.set == "conflict"]; agd = j[j.set == "agreement"].drop_duplicates("seg_uid")
print("conflict rows by rob arm", con.rob.value_counts().to_dict(), " agreement distinct by rob arm", agd.rob.value_counts().to_dict())
cells = {
 "conflict_robconflict": con[con.rob == "conflict"],
 "conflict_robnone": con[con.rob == "none"],
 "agreement_robagreement_distinct": agd[agd.rob == "agreement"],
 "agreement_robnone_distinct": agd[agd.rob == "none"],
 "conflict_p14speakers": con[con.in14],
 "agreement_distinct_p14speakers": agd[agd.in14],
}
comp = json.load(open("scores/part20/POD3b/sst2_pairs_o25_audio_diag_by_roberta_arm.json"))["results"]
out = {}
for k, df in cells.items():
    r = boot_single(df.label.values, df.p_yes.values, df.speaker_id.values)
    c = comp[k]
    md = max(abs(c["value"] - r["auc"]), abs(c["lo"] - r["lo"]), abs(c["hi"] - r["hi"]))
    ok = all(f"{a:.4f}" == f"{b:.4f}" for a, b in ((c["value"], r["auc"]), (c["lo"], r["lo"]), (c["hi"], r["hi"]))) and c["n"] == r["n"] and c["n_speakers"] == r["n_spk"]
    print(f"{k:34s} mine {r['auc']:.4f} (trap {r['trap']:.4f}) [{r['lo']:.4f}, {r['hi']:.4f}] n {r['n']} pos {r['n_pos']} spk {r['n_spk']} bad {r['bad']} | comp {c['value']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}] n {c['n']} spk {c['n_speakers']} | maxabs {md:.2e} agree4dp {ok}")
    out[k] = dict(mine=r, comp=c, agree_4dp=ok, maxabs=md)
json.dump(out, open("reports/part20/verify/POD3b_verify_diag_out.json", "w"), indent=1, default=float)
