"""Rebuild the E-DAIC depression conflict/agreement set.
Mirrors RECOVERED_part14_select.py exactly except for the two knobs this part varies:
  MODE=phq10  -> label := (sev>=10), no separate severity gate (1d, the dataset's own cutoff)
  MODE=paper  -> label := y, conflict gates sev>=15 / sev<=4  (the Table 2 rule, used for 1e)
  THR         -> the sentiment threshold replacing 0.70
usage: rebuild.py MODE THR OUTPREFIX"""
import sys, json, numpy as np, pandas as pd
MODE, THR, OUTP = sys.argv[1], float(sys.argv[2]), sys.argv[3]
R = "<local data dir>/Desktop/release/edaic_rerun"
S = pd.read_csv(f"{R}/part14_segments_sentiment.csv")
S["span"] = S.end - S.start
# px_build eligibility gate, unchanged
E = S[(S.speech_s >= 5) & (S.nw >= 25) & (S.span >= 8)].copy()
# sentiment flags re-derived at THR (verified clearly_positive == p_pos>=0.70 exactly)
POS = E.p_pos >= THR
NEG = E.p_neg >= THR
if MODE == "phq10":
    E["ylab"] = (E.sev >= 10).astype(int)
    CONF = ((E.ylab == 1) & POS) | ((E.ylab == 0) & NEG)
    AGR  = ((E.ylab == 1) & NEG) | ((E.ylab == 0) & POS)
else:
    E["ylab"] = E.y
    CONF = ((E.y == 1) & (E.sev >= 15) & POS) | ((E.y == 0) & (E.sev <= 4) & NEG)
    AGR  = ((E.y == 1) & NEG) | ((E.y == 0) & POS)
print(f"MODE={MODE} THR={THR}  eligible {len(E)} segs / {E.pid.nunique()} speakers")
conf = E[CONF].copy(); pool = E[AGR]
print(f"  conflict candidates {len(conf)} from {conf.pid.nunique()} speakers")
pairs = []
for i, r in conf.iterrows():
    cand = pool[(pool.pid == r.pid) & (~pool.index.isin([i]))]
    if len(cand):
        pairs.append((i, (cand.span - r.span).abs().idxmin()))
    else:
        pairs.append((i, None))
Pp = pd.DataFrame(pairs, columns=["conf_idx", "ctrl_idx"])
unmatched = int(Pp.ctrl_idx.isna().sum())
Pp = Pp[Pp.ctrl_idx.notna()].copy(); Pp["ctrl_idx"] = Pp.ctrl_idx.astype(int)
cm = E.loc[Pp.conf_idx].copy(); am = E.loc[Pp.ctrl_idx].copy()
keep = pd.concat([
    cm.assign(**{"set": "conflict"}, pair_id=range(len(cm)), seg_uid=["c%06d" % k for k in cm.index]),
    am.assign(**{"set": "agreement"}, pair_id=range(len(am)), seg_uid=["a%06d" % k for k in am.index])])
keep["speaker_id"] = keep.pid
keep["label"] = keep.ylab
print(f"  matched pairs {len(Pp)}, unmatched {unmatched}")
print(f"  rows {len(keep)}  speakers {keep.speaker_id.nunique()}  "
      f"arms {keep['set'].value_counts().to_dict()}  pos {int(keep.label.sum())}")
cols = ["seg_uid","set","pair_id","speaker_id","label","y","sev","start","end","span","speech_s","nw",
        "n_turns","p_pos","p_neg","p_neu","val","npos","nneg","oof","text"]
keep[[c for c in cols if c in keep.columns]].to_csv(f"{OUTP}_manifest.csv", index=False)
json.dump({"mode":MODE,"sentiment_threshold":THR,
  "source":f"{R}/part14_segments_sentiment.csv",
  "builder_mirrored":f"{R}/part16/scripts/RECOVERED_part14_select.py",
  "eligibility":"speech_s>=5, nw>=25, span>=8",
  "label_rule":"sev>=10" if MODE=="phq10" else "released y, conflict gated sev>=15 / sev<=4",
  "conflict_rule":("(label==1 & p_pos>=THR) or (label==0 & p_neg>=THR)" if MODE=="phq10"
                   else "(y==1 & sev>=15 & p_pos>=THR) or (y==0 & sev<=4 & p_neg>=THR)"),
  "agreement_rule":"same speaker's closest-span segment whose sentiment matches the label",
  "n_pairs":len(Pp),"unmatched":unmatched,"n_rows":len(keep),
  "n_speakers":int(keep.speaker_id.nunique()),"n_positive_rows":int(keep.label.sum())},
  open(f"{OUTP}_build.json","w"), indent=1)
print(f"  wrote {OUTP}_manifest.csv")
