# One protocol for every number in paper 1

Every clip, every model, every probe runs through these steps. If a result was produced
another way it gets rerun, not merged.

## Audio
- Load with librosa at 16 kHz, mono (`librosa.load(path, sr=16000)`). No loudness or peak
  normalisation, no trimming, no denoising. The model gets the waveform as decoded.
- Window: 30 seconds. Pitt segments are cut at the manifest's start_ms/end_ms
  (`DementiaBank/pitt_conflict_manifest.csv`, `scripts/dementiabank/cut_segments.py`).
  ADReSSo, ADReSS-2020 and KCL: the first 30 s of the recording (`ffmpeg -ac 1 -ar 16000 -t 30`).
  DAIC: participant turns merged to about 30 s (`scripts/prep/dcaps_prep.py`).
- Synthetic voice condition: the same transcript spoken by one fixed Piper voice, then treated
  exactly like a real clip above (16 kHz mono, same window rule).

## The question and the score
- Prompt, verbatim, no transcript in any audio condition:
  "Based only on this recording, does this speaker show signs of X? Answer with one word, Yes or No."
  X = dementia / depression / Parkinson's disease.
- Transcript-only condition: `Here is a transcript of what a person said during a picture
  description task: "<text>" Based only on this transcript, does this speaker show signs of X?
  Answer with one word, Yes or No.` (`scripts/dementiabank/textonly_ad.py`).
- Score: p_yes = P(Yes) / (P(Yes) + P(No)) from the logits at the first answer position, one
  forward pass, no generation. Yes/No token ids: first token of "Yes", " Yes", "yes", " yes",
  "YES" (same for No). Also store the mass P(Yes)+P(No) and report the yes-rate next to every AUC.
- Metric: AUC against the clinical label. Bootstrap 2000 resamples for intervals
  (`results/health/bootstrap_cis.csv` shows the format).

## Probes
- Hidden states mean-pooled over time, one vector per layer (`scripts/adresso/extract_probe_score.py`).
- StandardScaler fitted on the training fold only, LogisticRegression(max_iter>=2000,
  class_weight="balanced", C=1.0).
- Folds: 5, speaker-disjoint (GroupKFold on speaker id; StratifiedKFold only when every speaker
  has exactly one clip).
- Layer choice inside the training folds (nested). Report the nested number; if a peak-of-curve
  number is shown, label it as such.

## Projector retraining (the fix)
- Everything frozen except `multi_modal_projector`; projector held in fp32 inside the fp16 model.
- Encoder layer 24 fed to the projector (chosen from the encoder probe curve on Pitt before the
  projector experiments; not tuned since). Control run: default final layer, no plug.
- AdamW lr 1e-4, 3 epochs, batch 1, cross entropy on the Yes/No logits, GroupKFold 5, seed = fold*10+epoch.
  (`scripts/dementiabank/train_projector.py`, `scripts/kcl/train_projector_kcl.py`)

## Files that must exist for a result to count
- the manifest with clip ids, speaker ids, labels, and window times
- the per-clip score csv (p_yes, mass) or the per-layer probe csv
- the script that produced it, in `scripts/`
