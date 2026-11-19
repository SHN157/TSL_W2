# 04 — Model and training

Everything in this page lives in one file: `src/train/sonar_slt.py` (~420 lines), which
contains the model, both training phases, evaluation and the CLI.

## Architecture

```
spatial (T, 2048)      motion (N, 1024)        [both precomputed, frozen]
      |                      |
   Linear ->768          Linear ->768
      +----- concat in TIME ---+                (M = T + N)
                 |
         TemporalConv  K5 - P2 - K5 - P2        (Conv1d + BatchNorm + ReLU, MaxPool)
                 |                              length ~ (T + N) / 4
            MLP 768 -> 1024 (GELU) -> 1024
                 |  (M', 1024)
       SONAR encoder, LoRA on q_proj / v_proj
                 |
      Shallow-decoder pooling  ->  z (1024)
                 |
       SONAR decoder, LoRA, + language tag  ->  text
```

The fusion block is SpaMo's, reproduced from their released code (`fusion_mode='joint'`,
`inter_hidden=768`, `conv_type=2`). It is **10.1 M parameters**.

### Shallow-decoder pooling, not mean pooling

The paper (section 4.4) pools by feeding an EOS token to a shallow decoder initialised from
the first three SONAR decoder layers, cross-attending to the encoder output; the first output
position is the sentence embedding.

Implementation notes:

- the query is a learnable parameter initialised to the scaled EOS embedding plus position 0
- three deep copies of decoder layers 0-2, plus the final layer norm, all trained
- the layers are copied **before** LoRA is attached, so the pooler is independent of the
  decoder adapters
- a **learnable output scale** (`log_scale`), calibrated once on 10 training batches so that
  mean `|z|` equals mean `|s|`. Without it the MSE term is unusable: LayerNorm output has
  magnitude ~32 while SONAR targets have magnitude ~0.19, and Adam would need thousands of
  steps just to shrink the scale.

Mean pooling was used in an earlier batch-1 codebase and is one of the bugs that codebase
contained. Do not reintroduce it.

### A transformers-version detail

`create_bidirectional_mask` was removed in transformers 4.57, so the cross-attention mask is
built by hand as a 4D additive mask:

```python
cross = (1.0 - enc_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
```

with tuple-return handling for layers. Tested under both transformers 4.57 and 5.x.

## Losses

```
L_sem = 7000 * MSE(z, s) + 2.7 * (1 - cos(z, s))
L     = L_sem + lambda_ce * CE(text | z) + lambda_ae * CE(text | s)
```

`CE(text | s)` is the auto-encoding term: it decodes from the *true* target vector, which
anchors the decoder and, as the paper reports, accelerates convergence. `lambda_nce` is 0.

MSE is the mean over elements (`F.mse_loss`), consistent with the paper's statement that MSE
falls to ~1e-5.

## Two phases

| | Phase A | Phase B |
|---|---|---|
| trains | fusion + encoder LoRA + pooler | the above + decoder LoRA |
| decoder | frozen | LoRA adapters trained |
| `lambda_ce` | 0.1 | 1.0 |
| `lambda_ae` | - | 1.0 |
| batch | 8 x accum 8 = 64 (the paper's effective batch) | 8, no accumulation |
| selection | best dev **centred cosine** | best dev **chrF** |
| trainable | 87.2 M | 90.4 M |

Phase B starts from Phase A's best checkpoint (`--init runs/A_v1/best.pt`).

Shared: LoRA r=16 alpha=32 dropout 0.1 on `q_proj` / `v_proj`; AdamW lr 3e-4, weight decay
0.01, cosine schedule with 5% warm-up, gradient clipping 1.0; fp32 (no TF32); seed 42;
80 epochs with patience; beam 5, `max_new_tokens` 256.

## Decoding

`--no_repeat_ngram 3` is **worth +4.8 chrF** and is used for model selection and every final
evaluation. Without it, batch-1 output rambles — 40% of cooking predictions contained
repetitions and outputs ran 1.77x the reference length. The paper does not specify decoding
settings, so this is reported as our choice.

Decoder output is normalised (U+0E4D U+0E32 -> U+0E33) before scoring.

## Why the metrics are what they are

Plain cosine is **not** evidence of understanding. SONAR sentence vectors share a strong
common direction, so trivial predictors score well:

| predictor | dev cosine |
|---|---|
| always predict the mean of all targets | 0.461 |
| always predict the batch mean | 0.502 |
| always predict the category mean | 0.556 |

So three honest metrics are used instead:

- **ccos** — cosine after subtracting the training-target mean. An "average sentence"
  predictor scores ~0.
- **R@1 / R@5 / median rank** — can the predicted vector retrieve its own sentence among all
  dev targets?
- **chrF and char-BLEU** — `sacrebleu` with `tokenize='char'`, because the default tokeniser
  and `rouge_score` both return **0.00 even for identical Thai sentences** (Thai has no word
  spaces; rouge strips non-ASCII).

Reference points on dev: the category-average predictor gets ccos 0.357, R@1 4.3%, R@5 16.8%,
median rank 22; random gets ccos 0, R@5 1.5%, median rank 163.

## Practical notes

- Phase A uses ~22 GB of GPU memory, Phase B ~24 GB. Both GPUs on the shared server may hold
  another user's job — check `nvidia-smi` and pick a free one.
- About 2 minutes per epoch; 80 epochs is roughly 2.5 hours per phase on an unshared A6000.
- Use `nohup ... &` and `sleep 2 && tail -f logfile` (pasting the two together races — the
  log may not exist yet).
- `best.pt` and `last.pt` are written every epoch, so an interrupted run loses little.
- **One silent segfault was observed** during Phase B at epoch 25 — no traceback, no CUDA
  error — most likely a DataLoader worker reading `.npy` files over the network mount.
  Re-running with `--workers 2` completed without trouble.
- Checkpoints store only what is trained (fusion, pooler, LoRA), not the frozen base model.