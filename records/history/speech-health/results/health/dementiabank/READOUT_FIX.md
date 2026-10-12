# The fix that works: a small trained readout on the frozen model

Every inference time fix failed (prompts, mixer, text subtraction, steering, direct
layer feed). The one thing that closes the gap is also the simplest: keep the model
frozen, take the hidden states it already computes, and train a small linear readout
on them. That readout is exactly the probe, so its numbers are already measured,
speaker disjoint, on the same clips the model answers.

| task | trained readout (best layer) | the model's own answer | source |
|---|---|---|---|
| kcl read | 0.97 | 0.66 | probing_qwen/kcl_encoder_read.csv, reliance_qwen/kcl_pd_read_clips.csv |
| kcl spontaneous | 0.92 | 0.74 | probing_qwen/kcl_encoder_spont.csv, kcl fresh rescore |
| neurovoz read | 0.95 | 0.55 | probing_qwen/neurovoz_read.csv, reliance_qwen/nv_pd_read_clips.csv |
| pc gita read | 0.93 | 0.54 | probing_qwen/pcgita_read.csv, reliance_qwen/pcg_pd_read_clips.csv |
| daic interview | 0.70 | 0.65 | probing_qwen/dcaps_interview.csv, reliance_qwen/dcaps_dep_clips.csv |
| pitt alzheimers | 0.82 | 0.63 | dementiabank/ad_encoder_perlayer.csv, ad_scores_qwen2audio.csv |

On the pitt LLM layers the readout reaches 0.86, so the gap to the answer grows
with depth. Honest notes that travel with the table: peaks read off the full
curves, nested selection costs about 0.03 on average; the daic readout sees only
30 seconds of a long interview; on alzheimers the readout reads words as well as
voice, which is what the task carries.
