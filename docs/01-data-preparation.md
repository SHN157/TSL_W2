# 01 — Data preparation

How raw recordings become the `(T, 2048)` spatial and `(N, 1024)` motion features the
model consumes. Scripts: `src/data/`.

## Sources

| | batch 1 — cooking | batch 2 — medical |
|---|---|---|
| source | Thai PBS "Cook Culture", RealSense `.bag` -> PNG | studio mp4, green screen, 1920x1080 |
| raw | 686,845 PNG frames, 599 GB | 2,320 clips, `-m` = RGB (used), `-r` = RealSense (backup) |
| clips | 1,053 filmed | 2,320 filmed |
| usable | 1,008 | 2,223 |
| script | `normalize_batch1.py` | `extract_frames.py` |

**Incident, resolved.** 1,329 batch-2 RGB files were exact multiples of 1 MB with no
`moov` index — truncated copies. Originals were recovered from Google Drive via rclone.
All 2,320 are now readable.

## Frame rate: everything is resampled to 30 fps

Batch 2 was recorded at 60 fps (967 clips) or 120 fps (1,353), constant per category, so
integer decimation is exact: stride = round(fps/30), i.e. 2 or 4.

Batch 1 is harder. Frame rates measured from the `.bag` timestamps range **12.6 to 71.3
fps**, varying per clip but steady within each clip (no dropped frames — the sessions
simply used different camera settings). Integer stride fails here: 766 clips would land
at 32-40 fps and only 92 near 30. So frames are picked by position:

```python
n_out = floor(n_frames * 30 / fps)
idx   = round(arange(n_out) * fps / 30)
```

This leaves up to half a frame (~13 ms) of jitter — effectively exactly 30 fps.

**40 clips excluded**, all in `6_leaf`, recorded at 12.6-13.3 fps (a second session; the
other 114 leaf clips are 66-70 fps). Threshold `fps < 20`. At 13 fps, VideoMAE's 16-frame
window spans 1.2 s — two or three signs instead of one. Slowest clip kept: 29.0 fps.

### Why 30 fps

1. It is the only rate that uses 100% of batch 2 (60 fps clips cannot become 120).
2. VideoMAE is frozen, and its 16-frame window equals roughly one sign only at 25-30 fps
   (30 fps -> 0.53 s; 120 fps -> 0.13 s, near-identical frames, almost no motion).
3. A larger window at 120 fps is arithmetically the same as every 4th frame at 30 fps,
   at four times the cost.
4. It matches the standard datasets: PHOENIX-2014T 25, CSL-Daily 30, How2Sign 30.

A 30 / 60 / 120 fps ablation on the 1,353 clips recorded at 120 fps is still outstanding.

## Frame extraction

| setting | value |
|---|---|
| crop | centre square, 1080x1080 of 1920x1080 |
| resize | 448x448, bicubic |
| format | JPEG q95 |
| frame cap | none |
| layout | one flat folder per clip |

The crop was validated by green-screen detection: the signer never comes within 0.07 of
the crop edge and is never cut vertically.

*Correction recorded for honesty:* 448 was first justified by claiming S^2 would use that
detail. That was wrong — `AutoImageProcessor` resizes to 224 first and S^2 interpolates
up. 448 is still marginally better than 256, and nothing needed redoing.

Runtime: batch 1 3 h 20 m, batch 2 2 h 36 m. **8 parallel workers is optimal; 16 was
slower** because the job is disk-bound on 872 KB PNGs.

## Feature extraction

`extract_features.py` mirrors SpaMo's released code exactly.

| stream | model | output |
|---|---|---|
| spatial | CLIP ViT-L/14, S^2 scales [1,2], `num_prefix_token=1`, CLS token | `(T, 2048)` |
| motion | VideoMAE-L, window 16, overlap 8, CLS token | `(N, 1024)`, N = (T-16)//8 + 1 |

2048 = 1024 x 2 scales; seeing 1024 means S^2 was not applied. Short clips are padded by
repeating the last frame; an incomplete tail window is dropped. fp32, batch 32. Batch size
does not change speed — the A6000 is compute-bound here.

The "slow image processor" warning is *desirable*: it matches SpaMo's transformers v4
behaviour.

Runtime: batch 1 7 h 49 m, batch 2 12 h 47 m.

| | batch 1 | batch 2 |
|---|---|---|
| frames per clip | min 173, median 355, max 1,036 | min 112, median ~255, max 681 |

### Why batch 1's older features were not reused

A colleague had already extracted features for batch 1. They were rejected after measurement:

1. They used **native fps and every frame**, so motion windows covered 0.23-1.2 s
   depending on the clip, instead of a constant 0.53 s.
2. The pixel path differed: old was 1080 -> 224 in one step, new is 1080 -> 448 -> JPEG
   -> 224. **Measured cosine 0.982-0.987 with 17-20% mean magnitude difference** — far
   outside the 0.9998 agreement bar used elsewhere in this project.
3. Motion features cannot be derived from them at all; the windows do not line up.

## Resulting dataset

| | clips | signing time |
|---|---|---|
| batch 1 | 1,008 | ~3.7 h |
| batch 2 | 2,223 | ~5.7 h |
| **total** | **3,231** | **~9.3 h** |

Four signers appear in both batches (dear, gig, guide, toktak), well balanced at
1.8-2.0 h each. 473 batch-1 clips have no signer recorded in the spreadsheet.