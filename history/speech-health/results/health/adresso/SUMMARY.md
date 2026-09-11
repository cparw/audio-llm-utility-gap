# adresso (237 new speakers, second alzheimers dataset)

- probe on encoder layers: best 0.904 at layer 29 (adresso_encoder_perlayer.csv)
- the model's own answer with audio: 0.636 (adresso_scores_qwen2audio.csv)
- transcript only, no audio, whisper text: 0.617 (adresso_textonly_scores.csv)

Same pattern as pitt: the answer with audio lands where the transcript alone
lands, both far under what the layers hold. The gap and the word reading both
replicate on completely new speakers.

## words check on adresso (adresso_words_check.csv)
Text model alone, out of fold: 0.880. Split by whether the words agree with the label:
139 agreement clips, 19 conflict clips (small, terciles on 237). A probe trained on
agreement reaches 0.99 within agreement and falls to 0.39 to 0.61 on conflict across
layers (best-within layer 21: 0.988 within, 0.389 on conflict). Same picture as pitt:
what the probe learns on this task is largely the words. Conflict n is small, treat as
consistent-with, not as a standalone number.

## patient-onset windows and whisper-large-v3 (11 Sep)
Clips re-cut to start at the patient's first words (interviewer prompt skipped). Probe best layer 0.894,
the model's own answer 0.665 [0.594, 0.734], transcript-only with whisper-large-v3 text 0.682
(small.en text gave 0.617, so the transcript quality mattered). Same gap.
