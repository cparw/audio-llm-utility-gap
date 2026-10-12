"""
SILENCE-ONLY ARTIFACT PROBE  --  stage 3: is the DISCARD itself the artifact?

The silence probe only sees clips that had >= 0.3 s of usable silence. In Italian and
Neurovoz the discard rate is very different for patients and controls, so the surviving
subset is already selected. This script closes that loophole by probing EVERY clip in
the manifest (nothing discarded) using only two numbers that describe the recording and
contain no speech content:
    dyn_range_db  = (95th pct frame level) - (10th pct frame level) of the clip
    was_discarded = 1 if the clip had no usable silence
Same protocol: GroupKFold(5) by speaker, speaker-level label shuffle, speaker bootstrap CI.

If this alone separates the groups, then even "which clips have silence at all" is a
recording artifact and no subset analysis can be trusted.
"""
import os, csv, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import silence_probe as SP

R = SP.R
OUT = SP.OUT
DATASETS = SP.DATASETS


def main():
    rng = np.random.RandomState(11)
    rep, table = {}, []
    for ds in DATASETS:
        fp = f"{OUT}/silence_feats_{ds}.npz"
        if not os.path.exists(fp):
            continue
        d = np.load(fp, allow_pickle=True)
        pn = list(d["pause_feats"])
        dri = pn.index("dyn_range_db")
        P, y, g, task = d["P"], d["y"], d["g"], d["task"]
        X = [[float(v[dri]), 0.0] for v in P]
        Y = list(y.astype(int)); G = list(g); T = list(task)
        for r in csv.DictReader(open(f"{OUT}/silence_discards_{ds}.csv")):
            try:
                dr = float(r["dyn_range_db"])
            except Exception:
                dr = 0.0
            X.append([dr, 1.0]); Y.append(int(r["label"]))
            G.append(r["speaker_id"]); T.append(r["task_type"])
        X = np.nan_to_num(np.array(X, float)); Y = np.array(Y); G = np.array(G); T = np.array(T)
        res = SP.run_block(X, Y, G, f"{ds}|SELECTION", rng)
        # descriptive
        res["mean_dyn_range_patient"] = round(float(X[Y == 1, 0].mean()), 2)
        res["mean_dyn_range_control"] = round(float(X[Y == 0, 0].mean()), 2)
        res["frac_discarded_patient"] = round(float(X[Y == 1, 1].mean()), 3)
        res["frac_discarded_control"] = round(float(X[Y == 0, 1].mean()), 3)
        rep[ds] = res
        print(f"\n### {ds.upper()}  ALL {len(Y)} clips, nothing discarded")
        print(f"    dyn range (dB)  patient {res['mean_dyn_range_patient']:>7}   "
              f"control {res['mean_dyn_range_control']:>7}")
        print(f"    frac no-usable-silence  patient {res['frac_discarded_patient']:>6}   "
              f"control {res['frac_discarded_control']:>6}")
        if "auc_clip" in res:
            print(f"    AUC from (dynamic range, has-silence) ONLY = {res['auc_clip']:.3f} "
                  f"[{res['auc_clip_ci'][0]:.3f},{res['auc_clip_ci'][1]:.3f}]  "
                  f"spkAUC={res['auc_speaker']:.3f}  shuffle={res['shuffle_mean']:.3f} "
                  f"perm_p={res['perm_p']:.3f}")
            table.append(dict(dataset=ds, n_clips=res["n_clips"], n_speakers=res["n_speakers"],
                              selection_only_AUC=res["auc_clip"],
                              ci_lo=res["auc_clip_ci"][0], ci_hi=res["auc_clip_ci"][1],
                              speaker_AUC=res["auc_speaker"],
                              shuffle_mean=res["shuffle_mean"], perm_p=res["perm_p"],
                              dynrange_patient=res["mean_dyn_range_patient"],
                              dynrange_control=res["mean_dyn_range_control"],
                              frac_nosilence_patient=res["frac_discarded_patient"],
                              frac_nosilence_control=res["frac_discarded_control"]))
    json.dump(rep, open(f"{OUT}/silence_selection_check.json", "w"), indent=2)
    with open(f"{OUT}/silence_selection_auc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    print(f"\nwrote {OUT}/silence_selection_check.json and silence_selection_auc.csv")


if __name__ == "__main__":
    main()
