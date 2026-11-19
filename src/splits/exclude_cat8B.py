#!/usr/bin/env python
"""Step 0: exclude batch2 category 8 station B (video ids 98-194) and rebuild the split.

- master_clips.csv : text_ok = False, exclude_reason = 'cat8_stationB_excluded'
- splits/split_v2.csv : split_v1.csv with those clips removed (no reshuffle)
Nothing is re-encoded; targets/sonar_th.npy and its row index are untouched.
"""
import argparse, shutil
from pathlib import Path

import pandas as pd

W = Path.home() / "work/signlanguage"
PREFIX = "b2_B_08_"
REASON = "cat8_stationB_excluded"

ap = argparse.ArgumentParser()
ap.add_argument("--master", default=str(W / "reports/master_clips.csv"))
ap.add_argument("--split_in", default=str(W / "splits/split_v1.csv"))
ap.add_argument("--split_out", default=str(W / "splits/split_v2.csv"))
ap.add_argument("--apply", action="store_true", help="write the files (default: dry run)")
a = ap.parse_args()

# ---------------------------------------------------------------- master_clips
mas = pd.read_csv(a.master)
hit = mas["clip_id"].astype(str).str.startswith(PREFIX)
print("master_clips: %d rows | matching %s: %d" % (len(mas), PREFIX, int(hit.sum())))

ids = (mas.loc[hit, "clip_id"].astype(str).str[-4:].astype(int)).sort_values()
print("  video ids: min %d max %d  (expected 99 .. 194)" % (ids.min(), ids.max()))
assert ids.min() >= 99 and ids.max() <= 194, "unexpected id range - stop and check"
assert int(hit.sum()) == 96, "expected 96 clips, found %d - stop and check" % int(hit.sum())

if "exclude_reason" not in mas.columns:
    mas["exclude_reason"] = ""
already_out = int((~mas.loc[hit, "text_ok"].astype(bool)).sum())
print("  of those, already text_ok=False: %d" % already_out)

mas.loc[hit, "text_ok"] = False
mas.loc[hit, "exclude_reason"] = REASON

usable = mas[mas["text_ok"].astype(bool)]
print("\nusable clips after exclusion:")
print(usable.groupby("batch").size().to_string())
print("  total %d" % len(usable))

# ---------------------------------------------------------------- split
sp = pd.read_csv(a.split_in)
drop = sp["clip_id"].astype(str).str.startswith(PREFIX)
print("\nsplit_v1: %d clips | removing %d" % (len(sp), int(drop.sum())))
before = sp.groupby("split").size()
new = sp[~drop].reset_index(drop=True)
after = new.groupby("split").size()
print(pd.DataFrame({"before": before, "removed": before - after, "after": after}).to_string())

assert not new["clip_id"].duplicated().any()
assert new["row"].is_unique, "duplicate row index in the new split"

if not a.apply:
    print("\nDRY RUN - nothing written. Re-run with --apply when the numbers look right.")
    raise SystemExit

shutil.copy(a.master, a.master + ".bak_precat8")
mas.to_csv(a.master, index=False)
Path(a.split_out).parent.mkdir(parents=True, exist_ok=True)
new.to_csv(a.split_out, index=False)
for s in ["train", "dev", "test"]:
    f = Path(a.split_out).parent / ("%s_v2.txt" % s)
    f.write_text("\n".join(new.loc[new["split"] == s, "clip_id"].astype(str)) + "\n")
print("\nwrote %s (backup %s.bak_precat8)" % (a.master, a.master))
print("wrote %s and train/dev/test_v2.txt" % a.split_out)