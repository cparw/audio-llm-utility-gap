# adresso (237 new speakers, second alzheimers dataset)

- probe on encoder layers: best 0.904 at layer 29 (adresso_encoder_perlayer.csv)
- the model's own answer with audio: 0.636 (adresso_scores_qwen2audio.csv)
- transcript only, no audio, whisper text: 0.617 (adresso_textonly_scores.csv)

Same pattern as pitt: the answer with audio lands where the transcript alone
lands, both far under what the layers hold. The gap and the word reading both
replicate on completely new speakers.
