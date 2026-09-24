"""
Build the DAIC lexical-conflict ("paradox") segment set plus a matched-agreement
control set, cut the audio, and write everything to $R/committed/paradox/.

Rebuilt end to end from the original E-DAIC tarballs. Nothing is taken on faith
from earlier scratch files except the lexicon module dq_lex.py.
"""
import os, io, csv, sys, glob, json, tarfile, wave, warnings, re
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/scratch1/parwatka/pd_probing/code_clean")
import dq_lex as LX
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

R    = "/project2/msoleyma_946/speech_health/results_chaitanya"
BASE = "/project2/msoleyma_946/speech_health/DCAPS_challenge/data"
LAB  = "/project2/msoleyma_946/speech_health/DCAPS_challenge/labels/detailed_lables.csv"
OUT  = f"{R}/committed/paradox"
CLIP = f"{OUT}/clips"
os.makedirs(CLIP, exist_ok=True)
SR = 16000

# ---------------------------------------------------------------- 1. turns
def read_turns(tgz):
    pid = int(os.path.basename(tgz).split("_")[0])
    rows, adur = [], float("nan")
    try:
        with tarfile.open(tgz, "r:gz") as t:
            got_tx = got_au = False
            for m in t:
                nm = m.name.lower()
                if nm.endswith("transcript.csv") and not got_tx:
                    d = t.extractfile(m).read().decode("utf8", "ignore")
                    for i, r in enumerate(csv.DictReader(io.StringIO(d))):
                        try:
                            a, b = float(r["Start_Time"]), float(r["End_Time"])
                        except Exception:
                            continue
                        rows.append((pid, i, a, b, (r.get("Text") or "").strip()))
                    got_tx = True
                elif nm.endswith("audio.wav") and not got_au:
                    w = wave.open(io.BytesIO(t.extractfile(m).read()))
                    adur = w.getnframes() / w.getframerate()
                    got_au = True
                if got_tx and got_au:
                    break
    except Exception as e:
        print("ERR", pid, repr(e)[:100], flush=True)
    return rows, (pid, adur)

files = sorted(glob.glob(f"{BASE}/*_P.tar.gz"))
print("tarballs:", len(files), flush=True)
with Pool(8) as p:
    res = p.map(read_turns, files, chunksize=1)
T = pd.DataFrame([r for sub, _ in res for r in sub],
                 columns=["pid", "turn_idx", "start", "end", "text"])
AUD = dict(d for _, d in res)
print("raw transcript rows: %d from %d speakers" % (len(T), T.pid.nunique()), flush=True)

lab = pd.read_csv(LAB)
lab["pid"] = lab.Participant.astype(int)
T = T.merge(lab[["pid", "Depression_label", "Depression_severity", "gender", "age", "split"]],
            on="pid", how="left")
T["dur"] = T.end - T.start
T["nw"]  = T.text.fillna("").str.split().str.len()

# ---- audio/timestamp alignment audit -------------------------------------
mx = T.groupby("pid").end.max()
align = pd.DataFrame({"tx_max_end": mx, "tar_audio_s": pd.Series(AUD)})
align["over"] = align.tx_max_end - align.tar_audio_s
proc = {}
for pid in align.index:
    p = f"/scratch1/parwatka/dcaps_proc/{pid}.wav"
    if os.path.exists(p):
        w = wave.open(p); proc[pid] = w.getnframes() / w.getframerate()
align["dcaps_proc_s"] = pd.Series(proc)
align.to_csv(f"{OUT}/audio_alignment_audit.csv")
print("\n--- AUDIO SOURCE AUDIT ---")
print("tarball *_AUDIO.wav vs transcript max end: median overshoot %.2fs, "
      "n with transcript past end of audio: %d/%d"
      % (align.over.median(), (align.over > 0.5).sum(), len(align)))
print("dcaps_proc/<id>.wav vs tarball audio, duration ratio: "
      "median %.3f  min %.3f  max %.3f  (n within 1%%: %d/%d)"
      % ((align.dcaps_proc_s / align.tar_audio_s).median(),
         (align.dcaps_proc_s / align.tar_audio_s).min(),
         (align.dcaps_proc_s / align.tar_audio_s).max(),
         ((align.dcaps_proc_s / align.tar_audio_s - 1).abs() < .01).sum(), len(align)))
print("-> transcript timestamps index the TARBALL audio, not dcaps_proc. "
      "Clips are cut from the tarball *_AUDIO.wav.")

# ---------------------------------------------------------------- 2. interviewer filter
# E-DAIC *_Transcript.csv is raw ASR of the room with no speaker column, so the
# virtual interviewer's questions are interleaved with the participant's answers.
# Rule: drop any turn whose text STARTS with a question stem or an interviewer
# discourse marker. Also drop turns longer than 60s (malformed spanning rows).
Q = re.compile(r"^\s*(what|how|why|where|when|who|which|do you|are you|did you|have you|"
               r"can you|could you|would you|will you|tell me|has (there|that)|is (there|it)|"
               r"does |i'd love to hear|that sounds|okay so|so how|and how|and what)\b", re.I)
T["is_q"] = T.text.fillna("").map(lambda s: bool(Q.match(s)))
n_q   = int(T.is_q.sum())
n_bad = int(((~T.is_q) & (T.dur > 60)).sum())
C = T[(~T.is_q) & (T.dur <= 60) & (T.nw >= 1)].copy()
print("\n--- INTERVIEWER FILTER ---")
print("question-stem rule removed %d of %d turns (%.1f%%)" % (n_q, len(T), 100 * n_q / len(T)))
print("plus %d malformed turns with duration >60s" % n_bad)
print("kept %d participant-attributed turns from %d speakers" % (len(C), C.pid.nunique()))

# ---------------------------------------------------------------- 3. segments
def segments(g, target_words=40, max_gap=10.0):
    segs, buf = [], []
    for r in g.itertuples():
        if buf and (r.start - buf[-1].end) > max_gap and sum(x.nw for x in buf) >= 15:
            segs.append(buf); buf = []
        buf.append(r)
        if sum(x.nw for x in buf) >= target_words:
            segs.append(buf); buf = []
    if buf and sum(x.nw for x in buf) >= 15:
        segs.append(buf)
    return [dict(pid=s[0].pid, start=s[0].start, end=s[-1].end, n_turns=len(s),
                 nw=sum(x.nw for x in s), speech_s=sum(x.dur for x in s),
                 text=" ".join(x.text for x in s),
                 y=s[0].Depression_label, sev=s[0].Depression_severity)
            for s in segs]

S = pd.DataFrame([d for _, g in C.sort_values(["pid", "start"]).groupby("pid")
                  for d in segments(g)])
S["span"] = S.end - S.start
print("\nsegments: %d from %d speakers (median %d words, span %.1fs, speech %.1fs)"
      % (len(S), S.pid.nunique(), S.nw.median(), S.span.median(), S.speech_s.median()))

# ---------------------------------------------------------------- 4. two lexical scorers
sc = S.text.fillna("").map(LX.score)
S["val"]  = [x[0] for x in sc]; S["nsent"] = [x[1] for x in sc]
S["npos"] = [x[2] for x in sc]; S["nneg"]  = [x[3] for x in sc]

X, y, g = S.text.fillna("").tolist(), S.y.values, S.pid.values
oof = np.zeros(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups=g):
    assert not (set(g[tr]) & set(g[te])), "speaker leak"
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit([X[i] for i in tr], y[tr]); oof[te] = m.predict_proba([X[i] for i in te])[:, 1]
S["oof"] = oof
print("TF-IDF segment-level OOF AUC (speaker-disjoint GroupKFold): %.3f" % roc_auc_score(y, oof))
print("lexicon valence vs TF-IDF oof correlation: %.3f" % np.corrcoef(S.val, S.oof)[0, 1])
S.to_csv(f"{OUT}/daic_segments_scored.csv", index=False)

# ---------------------------------------------------------------- 5. conflict + agreement
E = S[(S.speech_s >= 5) & (S.nw >= 25) & (S.span >= 8)].copy()   # enough audio to score
print("\neligible segments (>=5s speech, >=25 words, >=8s span): %d from %d speakers"
      % (len(E), E.pid.nunique()))

# strong lexical signal, and the speaker is unambiguous on PHQ-8
POSW = (E.val >=  5) & (E.npos >= 3) & (E.nneg == 0)
NEGW = (E.val <= -4) & (E.nneg >= 2) & (E.npos == 0)
CONF = ((E.y == 1) & (E.sev >= 15) & POSW) | ((E.y == 0) & (E.sev <= 4) & NEGW)
AGR  = ((E.y == 1) & (E.sev >= 15) & NEGW) | ((E.y == 0) & (E.sev <= 4) & POSW)
# looser mirror, used only as fallback when a speaker has no strict agreement segment
POSW2 = (E.val >=  3) & (E.npos >= 2) & (E.npos > E.nneg)
NEGW2 = (E.val <= -3) & (E.nneg >= 2) & (E.nneg > E.npos)
AGR2  = ((E.y == 1) & NEGW2) | ((E.y == 0) & POSW2)

conf = E[CONF].copy()
conf["direction"] = np.where(conf.y == 1, "depressed_positive_words", "control_negative_words")
print("\n=== CONFLICT SET ===")
print("segments %d ; speakers %d" % (len(conf), conf.pid.nunique()))
print(conf.groupby("direction").agg(segments=("pid", "size"), speakers=("pid", "nunique"),
                                    median_span_s=("span", "median"),
                                    median_speech_s=("speech_s", "median")).round(2))
print("median clip duration overall: %.1fs" % conf.span.median())
print("TF-IDF agrees with the lexical conflict direction: dep-positive %d/%d, control-negative %d/%d"
      % (((conf.y == 1) & (conf.oof < .5)).sum(), (conf.y == 1).sum(),
         ((conf.y == 0) & (conf.oof > .5)).sum(), (conf.y == 0).sum()))

# ---- duration- and speaker-matched agreement partner for every conflict segment
pool_strict = E[AGR]; pool_loose = E[AGR2]
pairs = []
for i, r in conf.iterrows():
    for tier, pool in (("strict_same_speaker", pool_strict), ("loose_same_speaker", pool_loose)):
        cand = pool[(pool.pid == r.pid) & (~pool.index.isin([i]))]
        if len(cand):
            j = (cand.span - r.span).abs().idxmin()
            pairs.append((i, j, tier)); break
    else:
        pairs.append((i, None, "unmatched"))
P = pd.DataFrame(pairs, columns=["conf_idx", "ctrl_idx", "tier"])
print("\n=== MATCHED-AGREEMENT CONTROL ===")
print(P.tier.value_counts().to_dict())
P = P[P.ctrl_idx.notna()].copy(); P["ctrl_idx"] = P.ctrl_idx.astype(int)
conf_m = E.loc[P.conf_idx].copy(); ctrl_m = E.loc[P.ctrl_idx].copy()
print("matched pairs: %d over %d speakers (%d unique control segments)"
      % (len(P), conf_m.pid.nunique(), ctrl_m.index.nunique()))
print("duration: conflict median %.1fs vs matched-agreement median %.1fs ; "
      "median |delta| %.1fs" % (conf_m.span.median(), ctrl_m.span.median(),
                                np.median(np.abs(conf_m.span.values - ctrl_m.span.values))))
print("label mix identical by construction: conflict dep %d / ctl %d ; matched dep %d / ctl %d"
      % ((conf_m.y == 1).sum(), (conf_m.y == 0).sum(),
         (ctrl_m.y == 1).sum(), (ctrl_m.y == 0).sum()))

# ---------------------------------------------------------------- 6. cut audio
keep = pd.concat([
    conf_m.assign(set="conflict", pair_id=range(len(conf_m)),
                  seg_uid=["c%06d" % k for k in conf_m.index]),
    ctrl_m.assign(set="agreement", pair_id=range(len(ctrl_m)),
                  seg_uid=["a%06d" % k for k in ctrl_m.index]),
], ignore_index=False)
keep["lex_direction"] = np.where(keep.val > 0, "positive", "negative")
keep["words_say_depressed"] = (keep.val < 0).astype(int)

need = {}
for _, r in keep.iterrows():
    need.setdefault(int(r.pid), []).append(r)

def cut(pid):
    rs = need[pid]
    try:
        with tarfile.open(f"{BASE}/{pid}_P.tar.gz", "r:gz") as t:
            for m in t:
                if m.name.lower().endswith("audio.wav"):
                    raw = t.extractfile(m).read(); break
            else:
                return []
        w = wave.open(io.BytesIO(raw))
        sr, sw, nch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if nch > 1: data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
        out = []
        for r in rs:
            a, b = int(r.start * sr), int(min(r.end, len(data) / sr) * sr)
            seg = data[a:b]
            if len(seg) < sr:  # under 1s of audio actually present
                print("SHORT", pid, r.seg_uid, len(seg) / sr, flush=True); continue
            fn = f"{CLIP}/{r['set']}_{pid}_{r.seg_uid}.wav"
            ww = wave.open(fn, "wb"); ww.setnchannels(1); ww.setsampwidth(sw)
            ww.setframerate(sr); ww.writeframes(seg.tobytes()); ww.close()
            out.append((r.seg_uid, r["set"], fn, len(seg) / sr))
        return out
    except Exception as e:
        print("CUTERR", pid, repr(e)[:120], flush=True); return []

with Pool(8) as p:
    cuts = p.map(cut, sorted(need), chunksize=1)
CUT = {(u, s): (f, d) for sub in cuts for u, s, f, d in sub}
print("\nwrote %d wav clips to %s" % (len(CUT), CLIP))

keep["filepath"]   = [CUT.get((r.seg_uid, r["set"]), (None, None))[0] for _, r in keep.iterrows()]
keep["clip_dur_s"] = [CUT.get((r.seg_uid, r["set"]), (None, np.nan))[1] for _, r in keep.iterrows()]
keep = keep[keep.filepath.notna()].copy()
# keep only pairs where both halves survived
ok = keep.groupby(["pair_id"]).size()
keep = keep[keep.pair_id.isin(ok[ok == 2].index)]
keep["speaker_id"] = keep.pid
keep["label"] = keep.y
keep["task_type"] = "paradox_segment"
cols = ["seg_uid", "set", "pair_id", "speaker_id", "label", "sev", "start", "end", "span",
        "clip_dur_s", "speech_s", "nw", "n_turns", "val", "npos", "nneg", "oof",
        "lex_direction", "words_say_depressed", "task_type", "filepath", "text"]
keep[cols].to_csv(f"{OUT}/manifest.csv", index=False)

print("\n=== FINAL MANIFEST ===")
print(keep.groupby("set").agg(clips=("seg_uid", "size"), speakers=("speaker_id", "nunique"),
                              median_dur_s=("clip_dur_s", "median")).round(2))
kc = keep[keep.set == "conflict"]
print("conflict by direction:")
print(kc.groupby("lex_direction").agg(clips=("seg_uid", "size"),
                                      speakers=("speaker_id", "nunique"),
                                      median_dur_s=("clip_dur_s", "median")).round(2))
print("complete matched pairs: %d" % (len(keep) // 2))

json.dump(dict(
    n_conflict=int((keep.set == "conflict").sum()),
    n_agreement=int((keep.set == "agreement").sum()),
    n_pairs=int(len(keep) // 2),
    speakers=int(keep.speaker_id.nunique()),
    conflict_depressed_positive_words=int(((keep.set == "conflict") & (keep.label == 1)).sum()),
    conflict_control_negative_words=int(((keep.set == "conflict") & (keep.label == 0)).sum()),
    conflict_speakers_dep=int(kc[kc.label == 1].speaker_id.nunique()),
    conflict_speakers_ctl=int(kc[kc.label == 0].speaker_id.nunique()),
    median_conflict_dur_s=float(kc.clip_dur_s.median()),
    median_agreement_dur_s=float(keep[keep.set == "agreement"].clip_dur_s.median()),
    interviewer_turns_removed=n_q, raw_turns=int(len(T)),
    kept_participant_turns=int(len(C)), segments_total=int(len(S)),
    tfidf_segment_auc=round(float(roc_auc_score(y, oof)), 3),
    match_tiers=P.tier.value_counts().to_dict(),
), open(f"{OUT}/build_summary.json", "w"), indent=2)
print("\nJOB_DONE", flush=True)
