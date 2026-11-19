#!/usr/bin/env python
"""Supervisor updates 2026-10-05:
  b2_A_01_0094 : Thai and English do not match the video -> EXCLUDE (clip will be re-shot)
  b2_A_06_0015 : replace the English with the validated sentence
Updates master_clips.csv and removes the excluded clip from splits/split_v2.csv.
targets/sonar_th.npy and its row index are NOT touched.
"""
import argparse, shutil
from pathlib import Path

import pandas as pd

W = Path.home() / "work/signlanguage"
DROP = "b2_A_01_0094"
DROP_REASON = "text_misaligned_retake_pending"
FIX_ID = "b2_A_06_0015"
FIX_EN = "Are you constantly thirsty even after drinking water?"

ap = argparse.ArgumentParser()
ap.add_argument("--master", default=str(W / "reports/master_clips.csv"))
ap.add_argument("--split", default=str(W / "splits/split_v2.csv"))
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

mas = pd.read_csv(a.master)
d = mas["clip_id"] == DROP
f = mas["clip_id"] == FIX_ID
assert int(d.sum()) == 1, "%s found %d times" % (DROP, int(d.sum()))
assert int(f.sum()) == 1, "%s found %d times" % (FIX_ID, int(f.sum()))

print("EXCLUDE %s" % DROP)
print(mas.loc[d, ["text_th_final", "text_en_clean"]].to_string(index=False))
print("\nFIX %s" % FIX_ID)
print("  thai   : %s" % mas.loc[f, "text_th_final"].iloc[0])
print("  en old : %s" % mas.loc[f, "text_en_clean"].iloc[0])
print("  en new : %s" % FIX_EN)

mas.loc[d, "text_ok"] = False
mas.loc[d, "exclude_reason"] = DROP_REASON
mas.loc[f, ["text_en", "text_en_clean"]] = FIX_EN

sp = pd.read_csv(a.split)
in_split = sp["clip_id"] == DROP
print("\n%s is in split: %s" % (DROP, sp.loc[in_split, "split"].tolist() or "not in split"))
new = sp[~in_split].reset_index(drop=True)
print("split_v2: %d -> %d clips" % (len(sp), len(new)))
print(new.groupby("split").size().to_string())

if not a.apply:
    print("\nDRY RUN - nothing written. Re-run with --apply.")
    raise SystemExit

shutil.copy(a.master, a.master + ".bak_1005")
shutil.copy(a.split, a.split + ".bak_1005")
mas.to_csv(a.master, index=False)
new.to_csv(a.split, index=False)
for s in ["train", "dev", "test"]:
    p = Path(a.split).parent / ("%s_v2.txt" % s)
    p.write_text("\n".join(new.loc[new["split"] == s, "clip_id"].astype(str)) + "\n")
print("\nwrote %s and %s (backups *.bak_1005)" % (a.master, a.split))