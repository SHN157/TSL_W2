# 05 — Results

Main model: **`runs/B_v2`**, Phase B, best at epoch 62 of 80, trained on both batches with
Thai targets. Decoding: beam 5, `no_repeat_ngram_size=3`.

## How these numbers should be read

The SONAR-SLT paper reports **BLEU, BLEURT and ROUGE**; it does not report chrF. Standard
word-BLEU returns ~0 on Thai because Thai has no word spaces, so chrF is used here. Our chrF
therefore has **no counterpart in the paper**, and in particular:

> The paper's BLEU of 22.01 on PHOENIX-2014T and our chrF of 22.1 are different metrics that
> happen to look alike. They must never be compared.

Two valid reference points are used instead: the **achievable ceiling** on this data, and
**trivial baselines**.

## Phase A — can the model find the right region of meaning space?

| | dev ccos | R@1 | R@5 | median rank |
|---|---|---|---|---|
| random | 0.000 | 0.3% | 1.5% | 163 |
| category-average predictor | 0.357 | 4.3% | 16.8% | 22 |
| **A_v1** | **0.373** | 7.3% | ~28% | 13 |

The model beats the category-average predictor on retrieval (R@5 28% vs 17%, median rank 13
vs 22), so it carries genuine sentence-specific signal — but modestly. Plain cosine alone
(0.553) is roughly what guessing the category mean achieves, which is why it is not reported
as a headline.

Epochs 1-2 produce pure "mean direction" output (ccos 0, R@5 1.5%) before anything is learned.

## Phase B — text output

Test set, 347 clips (split v1), the configuration the main results were produced under:

| | chrF | char-BLEU | cosine | ccos | R@1 | R@5 | median rank |
|---|---|---|---|---|---|---|---|
| **all** | **22.1** | 22.9 | 0.535 | 0.362 | 18.4% | 45.0% | 7 |
| batch 1 (cooking, n=103) | 17.3 | 15.3 | 0.428 | | | | |
| batch 2 (medical, n=244) | 26.5 | 29.1 | 0.580 | | | | |

Dev 22.3 vs test 22.1 — no overfitting to the dev set.

**Against the ceiling.** Feeding the decoder the true target vector gives 37.9 (cooking) and
60.1 (medical). The model therefore reaches **46% and 44%** of what is achievable with this
decoder. That ratio, not the raw chrF, is the meaningful number.

Output length is well calibrated: 1.15x the reference for cooking, 1.00x for medical.

## Per-domain fine-tuning: does not help

Starting from `B_v2` and fine-tuning on one batch only (lr 1e-4, 30 epochs):

| | combined model | fine-tuned |
|---|---|---|
| medical dev chrF | 27.1 | 26.6 |
| cooking dev chrF | 17.4 | 17.3 (early-stopped) |

This reproduces the paper's finding that "post-fine-tuning performance remains largely
unchanged". **One combined model is kept.**

Note this answers "does extra specialisation help?", not "does the other batch help at all?".
The latter would require training from scratch on one batch, which has not been run.

## Error analysis

The typical error is **the right sentence shape with one wrong content sign**: nausea
rendered as stomach-ache or dizziness, egg as curry paste, fear as fatigue. Cosine stays at
0.75-0.84 in those cases. Genuinely wrong-sentence cases fall to 0.47-0.59 and are usually
short clips mapped to a different short sentence.

This matches the limitation the paper states: the feature extractor is tuned for global cues
and is less accurate where fine handshape or facial detail is decisive.

A second observation from the paper reproduces here too: fluent output can mask weak
grounding. The decoder is a pretrained language model and will complete a plausible sentence
even when the visual evidence is thin, so surface metrics understate meaning preservation in
some cases and overstate it in others.

## Figures

In `results/figures/`:

| figure | shows |
|---|---|
| fig1 | Phase A dev centred cosine per epoch, against the category-average bar |
| fig2 | Phase A dev R@5, against category-average and random |
| fig3 | Phase B train vs dev cross-entropy — the memorisation gap |
| fig4 | Phase B dev chrF per epoch for both runs, best epoch marked |
| fig5 | test chrF against the achievable ceiling, per domain |
| fig6 | per-sentence chrF distribution, cooking vs medical |

fig3 is worth a look: dev cross-entropy bottoms out around epoch 20-38 and then rises, so
the model is **data-limited, not schedule-limited**. More epochs will not help; more data
would.

## What is still missing

- **BLEURT** — the one metric that would allow direct comparison with the paper (0.545 on
  PHOENIX-2014T). Language-agnostic, works on Thai.
- the 30 / 60 / 120 fps frame-rate ablation
- a spot-check of ~20 clips by a Thai Sign Language reader, the only way to catch a
  single swapped pair
- a batch-only training run, to answer whether the two domains help each other