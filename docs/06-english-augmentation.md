# 06 — English target augmentation

**Question.** Can batch 2's human English sentences be used as a second training target for
the same clips, and does it help?

Runs: `A_aug`, `B_aug`. Code: the `--en_prob` option in `src/train/sonar_slt.py`.

## The idea

SONAR is language-agnostic: a Thai sentence and its English translation land at nearly the
same point in embedding space. So one clip has two valid targets.

| clip `b2_A_09_0056` | target vector | target text |
|---|---|---|
| `tha_Thai` | SONAR(Thai) | (Thai sentence) |
| `eng_Latn` | SONAR(English) | "Take this medicine after meals." |

At each training visit, a clip that has English uses the English target with probability
`--en_prob`; otherwise Thai. Batch 1 has no English and always uses Thai. **Dev and test
always use Thai**, so the metrics stay comparable with the baseline.

Two reasons to expect a gain:

1. **Regularisation.** With one fixed target the encoder can memorise a single point. With
   two nearby-but-distinct targets, the best solution is the region they share — the meaning.
2. **It repairs a measured defect.** The baseline trained cross-entropy on Thai only, so its
   English output contained Thai fragments, e.g. *"This is a medication for your [Thai word]"*.

## Scope

Deliberately narrower than the paper: **human English only**. No NLLB machine translation,
no French or Spanish, and no video augmentation — so the measured effect is attributable to
one change.

## Measured

`--en_prob 0.5`, otherwise an identical recipe. Both models scored on the same clips with
the same decoding; the baseline was re-evaluated on split v2 for fairness.

**Thai output**

| model | split | chrF | cooking | medical | R@1 | R@5 |
|---|---|---|---|---|---|---|
| B_v2 (Thai only) | dev | **21.94** | 17.39 | 26.64 | **20.5%** | **48.6%** |
| B_aug (+English) | dev | 20.72 | 16.31 | 25.22 | 13.9% | 41.0% |
| B_v2 (Thai only) | test | **21.60** | 17.25 | 25.63 | **17.8%** | **44.2%** |
| B_aug (+English) | test | 20.79 | 15.60 | 25.51 | 12.2% | 40.7% |

**English output** (234 medical test clips with a human English reference)

| model | English chrF | English BLEU | Thai characters in the English output |
|---|---|---|---|
| B_v2 | 22.99 | 4.92 | **9.0%** |
| **B_aug** | **27.99** | **13.20** | **0.0%** |

## Reading the result

**English became usable.** +5.0 chrF, BLEU nearly tripled, Thai leakage eliminated. BLEU
rewards correct word sequences, so tripling it means properly formed English sentences rather
than English-looking fragments.

| reference | B_v2 | B_aug |
|---|---|---|
| This eye drop will help with your eye problem. | "This is a medication for your [Thai]" | "This is for your prescription." |
| Cover your mouth when coughing or sneezing. | "Use handwashing lipstick when washing..." | "Do not wash your hands with soap or water." |

What B_aug still gets wrong is **content, not language**: "Take this medicine with a glass of
steamed bread" is fluent, grammatical medical English and not the reference sentence. That is
the same recognition limit the Thai side has. Augmentation fixed the language problem, not
the recognition problem.

**Thai cost 0.81 chrF on test, and more on retrieval.** The direction is consistent across
dev, test and both domains, so it is not noise. The cause is mechanical: at `en_prob 0.5`
each batch-2 clip trained on its Thai target only half as often, with the same number of
optimisation steps. At 9 hours of data, the regularisation benefit did not pay for that.

Cooking lost the most (17.25 -> 15.60) because batch 1 gained no English target and simply
received less attention. Medical was nearly unchanged (25.63 -> 25.51).

One detail worth noting: B_aug has **lower** Thai cross-entropy (2.886 vs 3.015) and slightly
higher centred cosine, yet lower chrF. Its vectors and token probabilities are not worse —
what degraded is the beam-search output and the ability to pin down one exact sentence.

## Caveats

- B_v2 was trained on the 96 category-8 clips that B_aug no longer sees. That favours B_v2
  and pushes in the same direction as the Thai gap, though it does not explain the English gain.
- Only one value of `en_prob` and one seed were tried.
- No video augmentation, so this is not the paper's full recipe.
- English is evaluated on batch 2 only — batch 1 has no human English reference.

## Verdict and next step

The trade buys a project requirement — working English subtitles — for about 0.8 Thai chrF.

The obvious follow-up is **`--en_prob 0.25`**: keep three-quarters of the Thai supervision
while still teaching English. Nothing new needs building, only the two training commands with
a different number.