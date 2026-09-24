"""Build the unified scoring manifest for the multi-model paradox generalisation.

Rows = 297 unique paradox segment clips + 275 whole dcaps interviews.
Every model scores exactly this file, at exactly the same window, with both prompts.
"""
import pandas as pd, os

R = "/project2/msoleyma_946/speech_health/results_chaitanya"
P = f"{R}/committed/paradox"
os.makedirs(f"{R}/final", exist_ok=True)

M = pd.read_csv(f"{P}/manifest.csv")
seg = M.drop_duplicates("filepath")[
    ["seg_uid", "filepath", "speaker_id", "label", "clip_dur_s"]].copy()
seg["task_type"] = "paradox_segment"
seg = seg.rename(columns={"seg_uid": "uid"})
assert seg.filepath.is_unique and len(seg) == 297, (len(seg), seg.filepath.is_unique)

D = pd.read_csv(f"{R}/manifests/dcaps.csv")
D = D[D.task_type == "interview"].copy()
iv = D[["filepath", "speaker_id", "label"]].copy()
iv["uid"] = "iv" + iv.speaker_id.astype(str)
iv["task_type"] = "interview"
iv["clip_dur_s"] = float("nan")
assert iv.filepath.is_unique and len(iv) == 275, (len(iv), iv.filepath.is_unique)

A = pd.concat([seg[["uid", "filepath", "speaker_id", "label", "task_type", "clip_dur_s"]],
               iv[["uid", "filepath", "speaker_id", "label", "task_type", "clip_dur_s"]]],
              ignore_index=True)
assert A.uid.is_unique and A.filepath.is_unique
missing = [f for f in A.filepath if not os.path.exists(f)]
assert not missing, missing[:5]

out = f"{R}/final/px_gen_manifest.csv"
A.to_csv(out, index=False)
print("wrote", out, A.shape)
print(A.task_type.value_counts().to_dict())
print("segment speakers", seg.speaker_id.nunique(), "interview speakers", iv.speaker_id.nunique())
print("JOB_DONE")
