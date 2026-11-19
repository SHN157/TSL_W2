# 03 — SONAR targets and data splits

The model is not trained to produce text directly. It is trained to produce the **sentence
embedding** `s` that the frozen SONAR text encoder assigns to the reference sentence. This
page covers how those targets were made and checked, and how the splits were built.

Scripts: `src/targets/`, `src/splits/`.

## Two environments

The official `sonar-space` package pins an older torch than the training environment needs,
so targets are encoded in a separate venv and the result is a `.npy` file both can read.

| env | torch | packages |
|---|---|---|
| TSL (training) | 2.14.0+cu130 | transformers 4.57.6, peft 0.21.0, sacrebleu |
| SONAR (targets) | 2.9.1+cu128 | fairseq2 0.8.1, sonar-space 0.5.0 |

The SONAR env needs `libsndfile`, which could not be installed system-wide without root.
Workaround: symlink the bundled library from the `soundfile` wheel into
`SONAR/lib/sndfile/libsndfile.so.1` and append the `LD_LIBRARY_PATH` export to
`SONAR/bin/activate`. **Consequence: SONAR steps must be run from an activated terminal,
not from Jupyter.**

## Verifying the HF port

Training uses `mtmlt/sonar-nllb-200-1.3B`, the SONAR-SLT authors' HuggingFace port, while
targets come from the official package. These must agree, so it was measured on all 3,373
sentences:

- encoder cosine **>= 0.999993** (MSE 5e-12)
- decoder: 60% byte-identical output, chrF agreement 91.7 (the gap is beam-search detail)

That is well inside the 0.9998 bar used elsewhere in this project, so the port is trusted.

## Encoding

`encode_targets.py` -> `targets/sonar_th.npy`, shape **(3373, 1024)**, covering every clip
including excluded ones, with `targets_index.csv` mapping row to clip id. Vector norm ~0.19.

Two Thai fixes applied before encoding: the sequence U+0E4D U+0E32 normalised to U+0E33, and
missing tone marks in `b2_B_12_0124` restored. The decoder's output is normalised the same
way before scoring.

English (`make_en_targets.py` then `encode_en.py`) -> `targets/sonar_en.npy`, same row order,
zeros where no English exists. 2,320 of 3,373 clips have a human English sentence.

> `.npy` target files are deliberately **not** in this repository: SONAR vectors can be
> decoded back into the sentences, so shipping them is equivalent to shipping the dataset.

## The ceiling — how good could any model be?

Feed the decoder the *true* target vector `s` and see how well it reconstructs the sentence.
Nothing downstream of the visual encoder can beat this.

| | chrF | char-BLEU |
|---|---|---|
| batch 1 (cooking) | **37.9** | 42.7 |
| batch 2 (medical) | **60.1** | 65.0 |

Cooking's ceiling is far lower because its sentences are long — the ceiling falls from 46 to
31 as sentence length grows. Eight outputs came back in Chinese rather than Thai, and four
contained repetitions; these are SONAR decoder artefacts, not model errors.

This ceiling is the reference our results are quoted against, since no cross-dataset
comparison is available (see `docs/05-results.md`).

## Quality check: does each English sentence belong to its own clip?

Thai and English describe the same video, so their vectors should be close. Measured over
2,320 clips:

| | |
|---|---|
| cos(Thai, English) mean / median | 0.764 / 0.787 |
| English vector retrieves its own Thai vector, rank 1 | **89.7%** |
| within top 5 / top 20 | 97.0% / 99.1% |
| flagged (own Thai outside top 20) | 20 clips (0.9%) |

The cosine is below 0.9 because the human English is a *free* translation, not a literal one.
That is not a defect — it is exactly the regularisation signal the augmentation experiment
relies on.

**Reviewing the flags without reading Thai.** Each flagged clip was printed beside `rt_en` —
the English you get by decoding that clip's *Thai* vector — turning a Thai-vs-English
comparison into an English-vs-English one. Most flags turned out to be SONAR decoding
artefacts, not misalignments:

| clip | human English | Thai decoded to English | verdict |
|---|---|---|---|
| `b2_B_02_0195` | numbness | "allergic to **tea**" | Thai *cha* means both -> **correct pair** |
| `b2_A_01_0035` | translator | "some **hunting**" | *laam* vs *laa*, one letter -> **correct pair** |
| `b2_A_02_0061` | weight | "the **rain** drop or rise" | *nam-nak* begins with *nam* (water) -> **correct pair** |
| `b2_A_01_0094` | vaccination records | "Where's the pharmacy?" | **genuinely mismatched** |

The test independently re-found `b2_A_01_0094`, the one clip the supervisor had already
flagged — good evidence that the test works. Five clips remain for a Thai reader to confirm:
`b2_B_02_0167`, `b2_A_05_0051`, `b2_A_06_0072`, `b2_B_12_0164`, `b2_B_02_0200`.

## Splits

`make_split.py` -> `splits/split_v1.csv`. Random, signer-mixed, 80/10/10, stratified by
batch / category / station, with identical sentences grouped so a duplicate never appears on
both sides. Seed 42. Verified: zero leakage, all feature files exist, and the text in the
split matches the text that was encoded.

**v1:** train 2,654 / dev 327 / test 347.

`exclude_cat8B.py` and `fix_clips.py` then produce **`split_v2.csv`** by *removing* the 97
excluded clips from v1 — not by reshuffling. Every remaining clip keeps its original
assignment, so the only difference between v1 and v2 data is those 97 clips.

**v2:** train 2,577 / dev 317 / test 337 = **3,231**.

| | train | dev | test |
|---|---|---|---|
| batch 1 (Thai only) | 805 | 100 | 103 |
| batch 2 (Thai + English) | 1,772 | 217 | 234 |

Each row carries `row` (the index into `sonar_th.npy`), the feature paths, and
`text_th_final`, so the dataset loader needs nothing else.