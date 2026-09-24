# Reliance Without Supplying Text

Chaitanya Parwatkar. On 17 July Prof. Soleymani rejected the design my first reliance results were
built on. His words: *"You shouldn't actually give the text at all. You should let the model
essentially read it."* Audio language models transcribe internally, so handing them an accompanying
transcript is redundant and injects a signal that is not in the recording. This documents what was
wrong, what replaced it, and how the numbers moved.

## What the original protocol did

Each clip was scored with a prompt of the form:

```
"Here is what the person said: '<TEXT>'. Based on this recording,
 does this speaker show signs of Parkinson's disease?"
```

where `<TEXT>` was one of three **invented** sentences the speaker never uttered: a neutral one, a
sick-sounding one, and a healthy-sounding one. Word reliance was the gap between the sick and
healthy answers, and "voice-only" AUC was the neutral-text condition.

Two things follow. The neutral condition is not voice-only, because a sentence is still being
supplied. And the sick and healthy conditions measure sensitivity to text that is not in the audio
at all, which is not the same thing as reading the transcript.

## Measured cost of supplying the text

The original runs also contained a genuinely text-free prompt, so the two can be compared directly
on the same clips.

| Corpus | No text supplied | Neutral text supplied |
|---|---|---|
| MDVR-KCL | **0.658** | 0.524 |
| Italian | 0.720 | 0.713 |
| Neurovoz | 0.552 | 0.452 |
| PC-GITA | 0.539 | 0.448 |
| DAIC-WOZ | 0.647 | 0.601 |

Supplying a neutral sentence **degraded** the model's judgment on four of five corpora. The text
was never inert, so any metric built on the neutral condition is measuring interference as well as
acoustics.

## What replaced it

`scripts/audiollm/audio_ablation_bench.py`. One fixed question, no transcript ever supplied:

```
"Based only on this recording, does this speaker show signs of X?
 Answer with one word, Yes or No."
```

Instead of varying the text, the **audio** is ablated:

- **original**
- **low-pass at 400 Hz**, which destroys intelligibility while leaving prosody and voice quality
- **time-reversed**, which destroys word order while preserving the magnitude spectrum exactly
- **heavy additive noise**
- **sustained vowels**, which carry no lexical content by construction

Answer probability is read from the yes-versus-no logits rather than parsed from generated text.
Confidence intervals come from a speaker-clustered bootstrap over 2000 resamples
(`scripts/audiollm/bootstrap_ablation.py`).

## Text-free results

| Corpus | Model | Original AUC | 95% CI |
|---|---|---|---|
| DAIC-WOZ (depression) | Qwen2.5-Omni | **0.835** | [0.779, 0.889] |
| DAIC-WOZ (depression) | Qwen2-Audio | 0.643 | [0.561, 0.726] |
| MDVR-KCL | Qwen2.5-Omni | 0.759 | [0.596, 0.904] |
| MDVR-KCL | Qwen2-Audio | 0.604 | [0.407, 0.793] |
| Neurovoz vowel | Qwen2.5-Omni | 0.604 | [0.550, 0.656] |
| Neurovoz vowel | Qwen2-Audio | 0.573 | [0.516, 0.630] |
| PC-GITA vowel | Qwen2-Audio | 0.553 | [0.509, 0.603] |
| PC-GITA vowel | Qwen2.5-Omni | 0.498 | [0.427, 0.573] |

Two findings.

**Parkinson's is close to chance.** Both models sit between 0.50 and 0.76, and the two
well-powered vowel corpora land at 0.50 to 0.60. MDVR-KCL has only 37 speakers and its interval
spans chance, so it cannot support a claim in either direction.

**Depression is not.** Qwen2.5-Omni reaches 0.835 from the voice with no text at all, and drops to
0.621 when the audio is time-reversed, a significant fall of 0.216. Reversal preserves the
magnitude spectrum exactly, so this is evidence the model is using temporal structure, which is
what the depression literature would predict for slowed speech and flat affect.

Note the asymmetry with the confound audit: the depression corpus has a recording-properties floor
of 0.453, below chance, so 0.835 there is not a recording artifact. The Parkinson's corpora have
floors of 0.68 to 0.995.

## Why the old modulation score has to be withdrawn

The first headline was the drop in word reliance from depression to Parkinson's, reported as +0.00
for Qwen2-Audio and +0.98 for Qwen2.5-Omni. That number does not mean what it appeared to mean.

Mean probability assigned to "yes", across all speakers:

- Qwen2-Audio answers yes for **85 to 99 percent** of everyone
- Qwen2.5-Omni answers no for about **96 percent** of everyone

Qwen2-Audio sits just above the decision threshold, so any nudge from injected text crosses it and
registers as a flip. Omni sits far below it, so nothing crosses. The contrast was measuring
**distance from the decision threshold**, not listening ability. A model that stops flipping
because it is listening and a model that stops flipping because it answers the same way every time
are indistinguishable under that metric.

Voice-only AUC measured text-free separates them; the flip rate does not.

## Offer for the rest of arm 2

The added models were scored with the original supplied-transcript script, which
the result-file field names show (`acoustic_neutral_AUC`, `sick_score_shuffled_audio`,
`lexical_fraction`, `mean_answer_mass`). Given the measured cost of supplying text
above, it is probably worth having text-free numbers next to them before the
headline rests on either set.

`audio_ablation_bench.py` takes a `--model` argument and runs unchanged on any of
them, so I am happy to rerun Voxtral, DiVA and Granite text-free if useful.

## Reproducing

```bash
python scripts/audiollm/audio_ablation_bench.py \
  --manifest $R/manifests/kcl.csv --tasks read --cond parkinsons \
  --model Qwen/Qwen2-Audio-7B-Instruct --tag abl_kcl_read_qwen
python scripts/audiollm/bootstrap_ablation.py --n 2000
```
