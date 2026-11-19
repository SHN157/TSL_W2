# TSL_W2 — Thai Sign Language → Text

Gloss-free Thai Sign Language translation, following **SONAR-SLT** (Hamidullah et al.,
WMT 2025) with the **SpaMo** visual front end (Hwang et al., NAACL 2025).

The model maps a sign-language video to a single language-neutral sentence embedding,
which a pretrained multilingual decoder renders as **Thai or English** text.

> **Private repository.** `data/` and `results/runs/*/pred_*.csv` contain the actual Thai
> and English sentences and signer names. See [Data and privacy](#data-and-privacy)
> before sharing anything from here.

---

## Results

Test set, 337 unseen clips, beam 5, `no_repeat_ngram_size=3`, split v2.

| model | chrF | cooking | medical | R@1 | R@5 | median rank |
|---|---|---|---|---|---|---|
| **B_v2** — Thai targets only | **21.6** | 17.3 | 25.6 | 17.8% | 44.2% | 7 |
| B_aug — + English targets | 20.8 | 15.6 | 25.5 | 12.2% | 40.7% | 8 |

English output, 234 medical test clips with a human English reference:

| model | English chrF | English BLEU | Thai characters leaking into English |
|---|---|---|---|
| B_v2 | 23.0 | 4.9 | 9.0% |
| **B_aug** | **28.0** | **13.2** | **0.0%** |

**How to read these numbers.** chrF is used because Thai has no word spaces, so standard
word-BLEU returns ~0 even for a perfect translation. The SONAR-SLT paper reports BLEU and
BLEURT, not chrF, so our scores have no direct counterpart there — do not compare our
chrF with their BLEU.

Instead we measure against an **achievable ceiling**: feeding the decoder the *true*
sentence embedding gives chrF 37.9 (cooking) and 60.1 (medical). The model reaches about
**46% and 44%** of that. Trivial baselines (always predicting the average sentence) score
0.46–0.56 plain cosine, which is why centred cosine and retrieval are reported instead.

---

## Pipeline

```
TSL video clip  ->  resampled to 30 fps, centre-cropped, 448x448 JPEG
        |
        |  offline feature extraction (both frozen)
        +------------------------------+
        v                              v
  CLIP ViT-L/14 with S^2         VideoMAE-L (window 16, overlap 8)
  spatial, (T, 2048)             motion, (N, 1024)
        +------------------------------+
                      v
   SpaMo visual fusion block  (10.1M params, trained)
   2048->768 | 1024->768 | concat in time | TemporalConv K5-P2-K5-P2 | MLP ->1024
                      v  (M, 1024)
          SONAR encoder + LoRA (q_proj, v_proj)
                      v
   Shallow-decoder pooling  (3 copied decoder layers + learnable EOS query
                             + learnable output scale)   -- NOT mean pooling
                      v
                  z  (1024)        language-neutral sentence embedding
                      v
   SONAR text decoder - frozen (Phase A) / LoRA (Phase B)
             + language tag: tha_Thai or eng_Latn
                      v
            Thai or English sentence
```

Base model: `mtmlt/sonar-nllb-200-1.3B` (the SONAR-SLT authors' port). Verified against
the official SONAR encoder: cosine >= 0.999993 on all 3,373 sentences.

**Training supervision.** The reference sentence is encoded by the frozen SONAR text
encoder into a target vector `s`. Loss:

```
L = 7000 * MSE(z, s) + 2.7 * (1 - cos(z, s)) + lambda_ce * CE(text | z)
                                             + lambda_ae * CE(text | s)   [Phase B]
```

| phase | trains | lambda_ce | lambda_ae | model selection |
|---|---|---|---|---|
| A | fusion + encoder LoRA + pooler (decoder frozen) | 0.1 | - | dev centred cosine |
| B | the above + decoder LoRA | 1.0 | 1.0 | dev chrF |

Trainable parameters: 87.2M (Phase A), 90.4M (Phase B). Everything else is frozen.

---

## Dataset

| | batch 1 | batch 2 |
|---|---|---|
| domain | cooking (Thai PBS "Cook Culture") | medical, 14 categories, 2 stations |
| clips used | 1,008 | 2,223 |
| text | Thai only | Thai **and** human English |
| mean sentence | 84 characters | 37 characters |

**3,231 clips, ~9.3 hours of signing, 4 signers** (PHOENIX-2014T is ~11 hours for
comparison). Split 80/10/10, stratified by batch/category/station, identical sentences
kept on one side: **train 2,577 / dev 317 / test 337**.

The clip-to-sentence pairing was *verified*, not assumed — see
[docs/02-clip-sentence-mapping.md](docs/02-clip-sentence-mapping.md). Because the author
does not read Thai Sign Language, pairing was checked by a measurable test: clip duration
correlates with sentence length only under the correct alignment (r = 0.33-0.79 correct
vs ~0.00 shifted). This found several real spreadsheet misalignments.

Video and features are **not** in this repository (41 GB of frames, 7.4 GB of runs).
See [docs/01-data-preparation.md](docs/01-data-preparation.md) to regenerate them.

---

## Repository layout

```
src/data/      frame extraction, fps normalisation, feature extraction
src/targets/   SONAR target encoding (Thai and English) + QC
src/splits/    split construction and the supervisor's clip exclusions
src/train/     sonar_slt.py - the model, both training phases, evaluation
src/eval/      scoring, baselines, run comparison, figures, inference demo
notebooks/     dataset checks and the PHOENIX vocabulary comparison
patches/       historical patches, already applied to sonar_slt.py (kept for provenance)
data/          sentence tables and splits   <- PRIVATE, see below
results/runs/  per-run args, logs, metrics and predictions (no checkpoints)
results/analysis/  dataset statistics, vocabulary and OOV comparisons
results/figures/   training and result figures
docs/          the detailed write-ups
```

---

## Reproducing

Two environments are needed because the official SONAR package pins an older torch.

```bash
python -m venv TSL   && source TSL/bin/activate   && pip install -r requirements-tsl.txt
python -m venv SONAR && source SONAR/bin/activate && pip install -r requirements-sonar.txt
```

| env | torch | key packages |
|---|---|---|
| TSL (training) | 2.14.0+cu130 | transformers 4.57.6, peft 0.21.0, sacrebleu |
| SONAR (targets) | 2.9.1+cu128 | fairseq2 0.8.1, sonar-space 0.5.0 |

Full run, in order:

```bash
# 1. frames and features              (TSL env, GPU; ~24 h total for both batches)
python src/data/extract_frames.py
python src/data/normalize_batch1.py
python src/data/extract_features.py --gpu 0

# 2. SONAR targets                    (SONAR env, activated so LD_LIBRARY_PATH is set)
python src/targets/encode_targets.py          # Thai  -> targets/sonar_th.npy
python src/targets/make_en_targets.py         # English table   (TSL env)
python src/targets/encode_en.py               # English -> targets/sonar_en.npy

# 3. splits and exclusions            (TSL env)
python src/splits/make_split.py
python src/splits/exclude_cat8B.py --apply
python src/splits/fix_clips.py --apply

# 4. training                         (TSL env, ~2.5 h per phase on one A6000)
python src/train/sonar_slt.py --phase A --out runs/A_v1
python src/train/sonar_slt.py --phase B --out runs/B_v2 --init runs/A_v1/best.pt \
       --no_repeat_ngram 3

# 5. evaluation
python src/train/sonar_slt.py --phase B --eval_only --split test \
       --init runs/B_v2/best.pt --out runs/TEST_B_v2 --no_repeat_ngram 3
python src/eval/analyse_test.py
python src/eval/make_plots.py
```

English augmentation adds `--en_prob 0.5` to both training commands;
`--en_prob 0` reproduces the Thai-only baseline exactly.

Hardware used: 2x NVIDIA RTX A6000 48 GB. Phase A ~22 GB, Phase B ~24 GB,
roughly 2 minutes per epoch at 80 epochs.

---

## Runs

| run | what it is |
|---|---|
| `A_v1` | Phase A, Thai targets. Best dev centred cosine 0.373 |
| `B_v1` / `B_v1_nrg3` | Phase B, 40 epochs. Showed `no_repeat_ngram_size=3` is worth +4.8 chrF |
| **`B_v2`** | **the main model** — Phase B, 80 epochs, best at epoch 62 |
| `FT_cook`, `FT_med` | per-domain fine-tuning. **Did not help**, matching the paper |
| `A_aug`, `B_aug` | English target augmentation, `--en_prob 0.5` |
| `TEST_*` | test-set evaluations, including B_v2 re-scored on split v2 for a fair comparison |

Checkpoints are not in the repository (360 MB each, above GitHub's limit). They are on the
server under `~/work/signlanguage/runs/`.

---

## Findings

1. **Per-domain fine-tuning does not help** (medical 27.1 -> 26.6, cooking 17.4 -> 17.3),
   reproducing the paper's result. One combined model is kept.
2. **English augmentation fixes English output but costs a little Thai.** +5.0 English
   chrF, BLEU 4.9 -> 13.2, Thai leakage 9% -> 0%, at the price of 0.8 Thai chrF — each
   clip then saw its Thai target only half as often.
3. **The task is measurably harder than PHOENIX-2014T.** Each word repeats ~7 times in our
   data against 34 in PHOENIX, and **one in three of our test sentences contains a word the
   model never saw** (PHOENIX: one in thirteen). Part of the test set is therefore
   unwinnable regardless of architecture. See
   [docs/07-dataset-difficulty.md](docs/07-dataset-difficulty.md).
4. **Typical error: the right sentence shape with one wrong content sign** — e.g. nausea
   rendered as dizziness. This matches the limitation stated in the paper.

---

## Deviations from the paper

| | paper | here | why |
|---|---|---|---|
| frame cap | 512-frame random crop | none | 139 clips exceed 512 frames, mostly batch 1 |
| video augmentation | frame mask / dropout / noise / shuffle | not used | so the English augmentation effect could be attributed cleanly |
| target augmentation | NLLB en/fr/es machine translation | human English only, batch 2 | real human translations were already available |
| metric | BLEU, BLEURT, ROUGE | chrF, char-BLEU | word-BLEU is unusable on unsegmented Thai |

BLEURT is the next evaluation to add, since it is language-agnostic and reported in the
paper (0.545 on PHOENIX-2014T), making it the one metric that would allow direct comparison.

---

## Data and privacy

These paths contain material that is **not** cleared for public release:

| path | contains |
|---|---|
| `data/master_clips.csv` | every Thai and English sentence, plus signer names |
| `data/splits/*`, `data/targets_index.csv`, `data/en_targets.csv` | the sentences again |
| `data/*_sentences_raw.csv`, `data/text_review.csv` | raw spreadsheet text |
| `results/runs/*/pred_*.csv`, `results/runs/EN_*.csv` | reference sentences and model output |

Git history is permanent. **If a public version of this work is ever needed, create a new
repository containing only `src/`, `docs/` and `results/figures/` — do not make this one
public**, because deleting files in a later commit does not remove them from history.

Not in the repository at all: raw video, extracted frames, CNN features, SONAR target
`.npy` files (they can be decoded back into the sentences), and model checkpoints.

---

## References

- Hamidullah et al., *SONAR-SLT: Sign Language Translation via Language-Agnostic Sentence
  Embeddings*, WMT 2025
- Hwang et al., *SpaMo: Spatial and Motion Representations for Gloss-free Sign Language
  Translation*, NAACL 2025
- Duquenne et al., *SONAR: Sentence-Level Multimodal and Language-Agnostic Representations*
- Camgöz et al., *Neural Sign Language Translation* (PHOENIX-2014T), CVPR 2018