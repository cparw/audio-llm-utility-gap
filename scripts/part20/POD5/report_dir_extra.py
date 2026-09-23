"""PART20 POD5: direction test extras. (a) matched-control paired difference kept_z minus perp_z on the interviewer-free states;
(b) the same test rerun on the ORIGINAL states (overnight2/part10/o25_pitt_states.npz) on this pod as a pipeline check against
the paper row (cos -0.0109, floor 0.0135, probe 0.7709, after removal 0.7662), with the paired original-minus-interviewer-free
differences of probe and after-removal AUCs (same clips, one speaker draw per replicate)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/root/p20/scripts")
from report_pod5 import emit, write, W, REF, OUT
from stats_pod5 import boot
a = pd.read_csv(f"{W}/dir_noinv_perclip.csv"); o = pd.read_csv(f"{W}/dir_orig_perclip.csv").set_index("clip_id").loc[a.clip_id].reset_index()
assert (o.label.values == a.label.values).all() and (o.speaker.values == a.speaker.values).all() and (o.fold.values == a.fold.values).all()
y = a.label.values; spk = a.speaker.values.astype(str); n, ns = len(y), a.speaker.nunique(); fn = "POD5_direction_origrerun_check.csv"
df = a.rename(columns={c: c + "__noinv" for c in ("p_probe_oof", "perp_z", "keep_z", "proj_d", "p_yes_zeroshot")})
for c in ("p_probe_oof", "perp_z", "keep_z", "proj_d", "p_yes_zeroshot"): df[c + "__orig_rerun"] = o[c].values
do = json.load(open(f"{W}/dir_orig.json")); side = []
side.append(emit("POD5_dir_kept_minus_removed", "matched control minus after-removal (same held-out-fold pooling), paired, interviewer-free", boot(y, a[["keep_z"]].values, spk, S2=a[["perp_z"]].values), n, ns, fn, True))
for key, id_, what, diff in (("cosine", "POD5_dir_orig_cos", "cos, ORIGINAL states rerun on this pod (paper -0.0109)", True), ("probe_auc", "POD5_dir_orig_probe", "probe AUC, ORIGINAL states rerun (paper 0.7709)", False),
                             ("after_removal_auc", "POD5_dir_orig_after_removal", "after-removal AUC, ORIGINAL states rerun (paper 0.7662)", False)):
    side.append(emit(id_, what, tuple(do[key]), n, ns, fn, diff))
side.append(emit("POD5_dir_orig_kept_minus_removed", "matched control minus after-removal, paired, ORIGINAL states rerun", boot(y, o[["keep_z"]].values, spk, S2=o[["perp_z"]].values), n, ns, fn, True))
side.append(emit("POD5_dir_probe_diff", "probe AUC, paired original(rerun) minus interviewer-free", boot(y, o[["p_probe_oof"]].values, spk, S2=a[["p_probe_oof"]].values), n, ns, fn, True))
side.append(emit("POD5_dir_after_removal_diff", "after-removal AUC, paired original(rerun) minus interviewer-free", boot(y, o[["perp_z"]].values, spk, S2=a[["perp_z"]].values), n, ns, fn, True))
write("POD5_direction_origrerun_check", df, side, [f"{W}/dir_noinv_perclip.csv", f"{W}/dir_orig_perclip.csv", f"{W}/dir_orig.json", "/root/p20/o25_pitt_states.npz", "/root/p20/omni25_readout.pt", f"{REF}/readout_direction.csv"],
      "python3 direction_pod5.py /root/p20/o25_pitt_states.npz /root/p20/omni25_readout.pt /root/p20/o25_pitt_zeroshot_scores.csv pitt_groupkfold5_mac.csv dir_orig orig ; python3 report_dir_extra.py",
      {"orig_rerun_full": do, "random_floor_orig": do["random_floor_mean_abs_cos"]})
print("DIR EXTRA DONE")
