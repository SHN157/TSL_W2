#!/usr/bin/env python
"""Step 1 of English augmentation: build targets/en_targets.csv.

One row per clip, keyed by `row` (the same row index as targets/sonar_th.npy).
Batch 2 carries its human English sentence (text_en_clean, falling back to text_en);
batch 1 has none (en_ok = False).  No translation is done anywhere in this script.
"""
import argparse, re, unicodedata
from pathlib import Path

import pandas as pd

W = Path.home() / "work/signlanguage"
CHECK = ["b2_A_06_0015", "b2_A_01_0094"]      # the two English sentences we corrected earlier

ap = argparse.ArgumentParser()
ap.add_argument("--master", default=str(W / "reports/master_clips.csv"))
ap.add_argument("--index", default=str(W / "targets/targets_index.csv"))
ap.add_argument("--split", default=str(W / "splits/split_v2.csv"))
ap.add_argument("--out", default=str(W / "targets/en_targets.csv"))
a = ap.parse_args()

nfc = lambda s: unicodedata.normalize("NFC", str(s)).strip()
THAI = re.compile(r"[\u0E00-\u0E7F]")

idx = pd.read_csv(a.index)[["row", "clip_id", "batch", "text_th_final"]]
mas = pd.read_csv(a.master)[["clip_id", "text_en", "text_en_clean"]]
print("index %d rows | master %d rows" % (len(idx), len(mas)))

df = idx.merge(mas, on="clip_id", how="left", validate="one_to_one")
assert len(df) == len(idx), "merge changed the row count"

raw = df["text_en"].fillna("").map(nfc)
clean = df["text_en_clean"].fillna("").map(nfc)
print("rows where text_en_clean differs from text_en: %d" % int((raw != clean).sum()))

df["text_th"] = df["text_th_final"].fillna("").map(nfc)
df["text_en"] = clean.where(clean.str.len() > 0, raw)        # cleaned, fall back to raw
df["en_ok"] = (df["batch"] == "batch2") & (df["text_en"].str.len() > 0)

# ---------------------------------------------------------------- QC
print("\nEnglish available (all 3,373 clips in the index):")
print(df.groupby("batch")["en_ok"].agg(["size", "sum"]).rename(
    columns={"size": "clips", "sum": "with_english"}).to_string())

miss = df[(df["batch"] == "batch2") & ~df["en_ok"]]
print("\nbatch2 clips with NO English: %d" % len(miss))
if len(miss):
    print(miss[["row", "clip_id"]].head(20).to_string(index=False))

thai_in_en = df[df["en_ok"] & df["text_en"].str.contains(THAI)]
print("English containing Thai characters (column mix-up?): %d" % len(thai_in_en))
if len(thai_in_en):
    print(thai_in_en[["clip_id", "text_en"]].head(10).to_string(index=False))

short = df[df["en_ok"] & (df["text_en"].str.len() < 5)]
print("suspiciously short English (<5 chars): %d" % len(short))
if len(short):
    print(short[["clip_id", "text_th", "text_en"]].to_string(index=False))

e = df[df["en_ok"]]
print("\nEnglish length: mean %.1f  min %d  max %d chars | unique %d of %d"
      % (e["text_en"].str.len().mean(), e["text_en"].str.len().min(),
         e["text_en"].str.len().max(), e["text_en"].nunique(), len(e)))

print("\nthe two clips whose English we corrected earlier:")
print(df[df["clip_id"].isin(CHECK)][["clip_id", "text_th", "text_en"]].to_string(index=False))

# ---------------------------------------------------------------- coverage of the real split
sp = pd.read_csv(a.split)[["clip_id", "split"]]
cov = sp.merge(df[["clip_id", "batch", "en_ok"]], on="clip_id", how="left")
assert cov["en_ok"].notna().all(), "a split clip is missing from the target index"
print("\ncoverage of split_v2 (what training will actually see):")
print(cov.groupby(["split", "batch"])["en_ok"].agg(["size", "sum"]).rename(
    columns={"size": "clips", "sum": "with_english"}).to_string())

cols = ["row", "clip_id", "batch", "text_th", "text_en", "en_ok"]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df[cols].to_csv(a.out, index=False)

print("\nexamples")
for _, r in e.sample(5, random_state=0).iterrows():
    print(" %s\n   th: %s\n   en: %s" % (r["clip_id"], r["text_th"], r["text_en"]))
print("\nsaved %d rows (%d with English) -> %s" % (len(df), int(df["en_ok"].sum()), a.out))