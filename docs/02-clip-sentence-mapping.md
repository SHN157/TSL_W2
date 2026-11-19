# 02 — Clip-to-sentence mapping (verified)

**These rules were measured, not assumed.** Getting them wrong silently pairs videos with
the wrong sentences — no error is raised, the model simply trains on nonsense. That is why
each rule was proven before use.

## The verification test

**Idea.** A longer sentence takes longer to sign, so clip duration should correlate with
sentence character length — **but only under the correct pairing.** A wrong pairing gives
a correlation near zero.

**Method.** For each candidate mapping, pair every clip's duration with its sentence's Thai
character count and compute Pearson r and Spearman rho. Calibrate first on groups already
known to align, comparing the correct mapping against a deliberate one-row shift; only
trust the test if the shift clearly destroys the correlation. Then apply it to the
ambiguous cases.

**Measured.** Batch 2 calibration on 10 exact-match categories: correct r = 0.38-0.70
(mean 0.540), shifted r ~ 0.00. The test is discriminative.

This test needed no sign-language knowledge, which mattered: the author does not read Thai
Sign Language and so could not verify alignment by watching the videos.

**Local check.** The same test in sliding windows of 30 clips (step 10) per
category/station, flagging where a shift beats the current pairing by > 0.30 Spearman:

- control (a group deliberately shifted by one): **211 / 247 windows flagged (85%)** — the
  test can see a local shift
- real data: **0 / 257 windows flagged** — no drift anywhere

*Limitation:* no duration test can detect a single swapped pair or a 1-3 clip error. Only
a Thai Sign Language reader can. A spot-check of ~20 clips is still outstanding.

## Batch 2 — medical

Source: one spreadsheet per category, 14 files. Columns: ID / `text_en` / `text_th` /
`station` / `signer`. 2,439 sentences parsed, 0 empty.

Parsing traps that cost real time:

- **The header row must be detected, not assumed.** It is row 1 in 13 files but **row 0 in
  `2_doctor_patient_interaction`**. Find the row containing `text_en`.
- **Filenames and sheet names lie.** `8_admin_and_billing_193rows.xlsx` has a sheet named
  `200rows` containing 193 rows. Trust only the content.
- The `station` cell reads e.g. `"A\n= 94 rows"` — parse the number out of it.
- `signer` is sparse and must be forward-filled.

### The rule

```python
SHIFT_CATS = {1, 2, 8}

# station A, every category:
excel_row = video_id

# station B:
if cat in SHIFT_CATS:
    b_ids = sorted(station-B video ids in this category)
    excel_row = A_ROWS[cat] + 1 + b_ids.index(video_id)
else:
    excel_row = video_id

A_ROWS = {1:94, 2:100, 3:99, 4:98, 5:100, 6:94, 7:100, 8:97,
          9:99, 10:25, 11:25, 12:101, 13:90, 14:101}
```

Categories 1, 2 and 8 differ because at the station A -> B changeover **one video number
was consumed with no file saved** (probably a discarded test take), so station B's numbers
run one ahead of the spreadsheet rows from that point on.

| cat | A ends | number lost | B starts | B videos | B rows |
|---|---|---|---|---|---|
| 1 | 94 | 95 | 96 | 94 | 94 |
| 2 | 100 | 101 | 102 | 99 | 99 |
| 8 | 97 | 98 | 99 | 96 | 96 |

Counts close exactly in all three. **Category 14 does not shift** despite looking similar.

Two kinds of gap behave differently: *a row with no video* (cat 1 row 79) leaves following
ids aligned, while *a video number with no row* (cat 1 number 95) shifts everything after it.

## Batch 1 — cooking

Source: one spreadsheet, 12 sheets = 10 episode sheets + 2 vocabulary sheets (ignore those).

- **Continuation rows:** some rows have a blank clip number and text continuing the previous
  clip. Rows must be **grouped by clip number and joined**, not read one by one. Sheet 1 has
  59 such rows (312 rows -> 253 clips), sheet 3 has 1.
- The signer column moves between sheets and is absent in sheets 3, 5, 6, 8, 9, 10.
- Episodes 7, 9 and 10 were never filmed.

**Rule: `excel clip_num = video clip id`, in every episode.** No shift anywhere
(r = 0.33-0.79 for id = id, ~0 when shifted).

Batch 1 has **no English** — the spreadsheet is Thai only.

## Supervisor decisions, 2026-10-05

| decision | effect |
|---|---|
| cat 1 lost id 95, cat 2 lost id 101 — confirmed, ignore | no clips removed |
| **cat 8: exclude ids 98-194 (all of station B)**, keep A ids 1-97 | **96 clips removed** |
| **`b2_A_01_0094` excluded** — text and video disagree; clip to be re-shot | 1 clip removed |
| **`b2_A_06_0015`** English replaced with a validated sentence | 1 correction |

Category 8 station B had the weakest alignment evidence in the whole dataset (Spearman
0.293 against 0.536 for its own station A, and 0.35-0.79 everywhere else), so this removes
the pairing we were least sure of.

Implemented by `src/splits/exclude_cat8B.py` and `src/splits/fix_clips.py`. Both mark
exclusions in `master_clips.csv` and remove the clips from the split **without reshuffling**,
so every remaining clip keeps its original train/dev/test assignment. The SONAR target index
is untouched, so no re-encoding was needed.

**Correction recorded.** An earlier note claimed two mismatched English sentences had
already been fixed in `master_clips.csv`. They had not — re-reading the file on 5 October
showed the original wrong text still in place. The lesson: verify a claimed data fix by
re-reading the file, not by trusting the note.

## Text QC

Columns in `master_clips.csv`: `text_th_clean` (NFC, zero-width characters removed,
whitespace collapsed), **`text_th_final`** (the training target), **`text_ok`**.

| finding | decision |
|---|---|
| batch 2 trailing `**` (22 clips) | spreadsheet note mark -> removed |
| batch 1 steak `**name**` (3) | bold formatting -> removed, words kept |
| batch 1 salt `***` (5 clips) | transcriber unsure of the word -> **excluded from training** |
| batch 1 worldcooking brackets (59) | `Central Thai (Northern-dialect verbatim)` -> bracket removed, Central Thai kept |
| batch 2 brackets (25) | clarifications such as `5 ml. (1 teaspoon)` -> kept |
| duplicates (68 clips / 31 sentences) | each duplicate sentence kept on one side of the split |

Two Thai typing errors were found by character-level search rather than by reading: the
sequence U+0E4D U+0E32 where U+0E33 was intended, and missing tone marks in
`b2_B_12_0124` caused by a PDF copy-paste.

**Final: 3,231 training clips.**