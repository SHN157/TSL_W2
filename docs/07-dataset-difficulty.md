# 07 — Why the scores are lower than published benchmarks

Notebook: `notebooks/check_phoenixT.ipynb`.
Outputs: `results/analysis/vocab_comparison.csv`, `results/analysis/oov_comparison.csv`.

## The question

Our chrF is ~21 while published results on PHOENIX-2014T look higher. Is the method weak, or
is the task harder?

Scores cannot settle it: the paper reports BLEU / BLEURT / ROUGE, not chrF, and word-BLEU is
unusable on Thai. So the **datasets** were compared instead of the **scores**.

## Method

A model can only learn a sign it has seen several times. For each dataset we counted how
often each word repeats in training, and how many test words never appear in training.

PHOENIX-2014T parsed cleanly: 7,096 / 519 / 642 sentences and a German vocabulary of 2,887,
matching the published figures exactly.

Tokenisation: Thai with `pythainlp` (newmm), German and English by whitespace, punctuation
stripped in all three. **The English-vs-German comparison is the primary evidence**, since
both are whitespace-tokenised European languages; Thai is supporting.

## Results

| | PHOENIX German | TSL English (medical) | TSL Thai (cooking) |
|---|---|---|---|
| sentences | 7,096 | 1,772 | 805 |
| tokens | 99,081 | 14,448 | 15,308 |
| unique words | 2,887 | 2,017 | 2,167 |
| **avg times a word appears** | **34.3x** | **7.2x** | **7.1x** |
| text made of words seen >= 5 times | 97.1% | 81.7% | 82.3% |
| words appearing only once | 37.3% | 45.0% | 47.2% |

| | PHOENIX German | TSL English | TSL Thai (all) | TSL Thai (cooking) |
|---|---|---|---|---|
| test words never seen in training | 0.8% | 5.6% | 5.0% | 8.3% |
| **test sentences containing such a word** | **7.6%** | **35.9%** | **36.2%** | **71.8%** |

**Size control.** Larger corpora repeat more by construction, so PHOENIX was re-measured on a
random 1,772 sentences — the same count as our English. It still repeats each word **14.7x**
against our **7.2x**, with 92.7% coverage against 81.7%. Stable across five random draws
(14.3-14.7).

So roughly **half the gap is corpus size and half is domain breadth**. Both should be claimed,
not just the first.

## Interpretation

PHOENIX-2014T is weather forecasts only. Its frequent words are content words that recur
constantly — *regen* (rain) 838 times, the gloss REGEN 2,149 times, plus region, north, sun,
cloud, degrees, night, south, snow, thunderstorm. Our data covers cooking plus 14 medical
topics, so our frequent words are grammar (you, to, your, a, have) and the content words sit
in the long tail.

The consequence is concrete: **one in three of our test sentences contains a word the model
has never seen**, against one in thirteen for PHOENIX. Those sentences cannot be fully correct
whatever the architecture.

Cooking is the extreme case — 71.8%, with sentences averaging 19 words against medical's 8.
That accounts for cooking's chrF of 15.6 against medical's 25.5 without invoking any
modelling failure.

## Recommendation for future recording

**Write the sentence list so that core vocabulary recurs**, instead of letting every sentence
introduce new words.

> "Take this medicine after meals."
> "Take this medicine with water."
> "Don't stop taking this medicine."

Three different sentences, but the sign for *medicine* is seen three times. PHOENIX gets this
for free: 7,096 different sentences, all about weather. Filming the same sentence with
different signers also helps; one signer repeating an identical sentence helps least.

A ten-minute check before filming: tokenise the planned sentence list and compute the average
repeats per word. Below about 5 means the list is spread too thin.

## Two bugs found and fixed during this analysis

Recorded because both would have produced wrong conclusions:

1. **Punctuation glued to English words** inflated our out-of-vocabulary rate — `cramps.` and
   `cramps` counted as different words. Stripping punctuation moved the figure from 10.6% to
   **5.6%**.
2. **The first cleaning pass destroyed Thai.** The regex `[^\w\s]` deletes Thai tone marks and
   vowels, because Python's `\w` does not match Unicode combining marks. Thai token counts
   jumped 64% before this was caught. The fix keeps the whole Thai block:
   `[^\u0E00-\u0E7F\w\s]`.

All figures above are the corrected values.

## Limitation

This describes task difficulty. It does **not** excuse model performance — the result of
reaching ~46% of the achievable SONAR ceiling stands as a separate finding, measured on our
own data.