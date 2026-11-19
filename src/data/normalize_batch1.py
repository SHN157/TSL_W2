"""
normalize_batch1.py — batch 1 PNG frames -> 30 fps, 448x448 JPEG.

Selects frames by position (not by integer stride), because batch 1's rates
(33-71 fps) do not divide evenly into 30. Output matches batch 2 exactly.

Skips clips recorded below 20 fps.
Safe to re-run: finished clips are skipped.

Usage:
    python normalize_batch1.py --dry-run
    python normalize_batch1.py --limit 5
    python normalize_batch1.py
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

HOME     = Path.home()
ARCH     = HOME / "dataset_arch" / "SignLanguage"
REPORTS  = HOME / "work" / "signlanguage" / "reports"

SRC      = ARCH / "frames"                      # nested: <category>/<clip>/*.png
OUT_ROOT = ARCH / "frames_batch1_30fps"         # flat, like batch 2

TARGET_FPS = 30
SIZE       = 448
QUALITY    = 95
MIN_FPS    = 20


def center_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def pick_indices(n_frames, src_fps):
    """Frame indices that land closest to each 1/30 s tick."""
    if src_fps <= TARGET_FPS:
        return np.arange(n_frames)
    n_out = int(np.floor(n_frames * TARGET_FPS / src_fps))
    idx = np.round(np.arange(n_out) * src_fps / TARGET_FPS).astype(int)
    return np.clip(idx, 0, n_frames - 1)


def do_clip(row, out_dir):
    src_dir = SRC / row["cat_name"] / f"{row['cat_name']}_clip_{int(row['clipnum']):03d}"
    files = sorted(src_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(str(src_dir))

    idx = pick_indices(len(files), float(row["fps"]))

    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    for k, i in enumerate(idx):
        img = Image.open(files[i]).convert("RGB")
        center_square(img).resize((SIZE, SIZE), Image.BICUBIC) \
            .save(tmp / ("%06d.jpg" % k), quality=QUALITY)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return len(files), len(idx)

def worker(rec):
    try:
        src_n, out_n = do_clip(rec, OUT_ROOT / rec["clip_id"])
        return {"clip_id": rec["clip_id"], "fps": rec["fps"],
                "src_frames": src_n, "out_frames": out_n,
                "ok": True, "error": ""}
    except Exception as e:
        return {"clip_id": rec["clip_id"], "fps": rec["fps"],
                "src_frames": -1, "out_frames": 0,
                "ok": False, "error": "%s: %s" % (type(e).__name__, e)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    fps = pd.read_csv(REPORTS / "batch1_fps.csv")
    fps = fps.rename(columns={"cat": "cat_name", "clip": "clipnum"})
    fps = fps[fps["fps"].notna()]

    keep = fps[fps["fps"] >= MIN_FPS].copy()
    drop = fps[fps["fps"] <  MIN_FPS]
    keep["clip_id"] = keep.apply(
        lambda r: f"b1_{r['cat_name']}_{int(r['clipnum']):04d}", axis=1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    todo, done = [], 0
    for _, r in keep.iterrows():
        d = OUT_ROOT / r["clip_id"]
        if d.is_dir() and any(d.iterdir()):
            done += 1
        else:
            todo.append(r)
    if args.limit:
        todo = todo[:args.limit]

    print("clips in batch 1    : %d" % len(fps))
    print("excluded (<%d fps)  : %d" % (MIN_FPS, len(drop)))
    print("to process          : %d" % len(keep))
    print("already done        : %d" % done)
    print("remaining           : %d" % len(todo))
    print("output              : %s" % OUT_ROOT)
    print("settings            : %d fps, %dx%d jpeg q%d" % (TARGET_FPS, SIZE, SIZE, QUALITY))

    if args.dry_run:
        print("\n-- first 8 --")
        for r in todo[:8]:
            n_out = len(pick_indices(int(r["bag_frames"]), float(r["fps"])))
            print("  %-34s %6.2f fps  %5d -> %4d frames"
                  % (r["clip_id"], r["fps"], r["bag_frames"], n_out))
        est = sum(len(pick_indices(int(r["bag_frames"]), float(r["fps"]))) for r in todo)
        print("\nestimated frames to write: %s" % format(est, ","))
        print("estimated disk (jpeg q95 @448): %.1f GB" % (est * 55 / 1e6))
        return


    log = []
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    print("using %d parallel workers\n" % n_workers)

    recs = [dict(r) for r in todo]          # plain dicts pickle cleanly
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker, r): r["clip_id"] for r in recs}
        for f in tqdm(as_completed(futs), total=len(futs)):
            res = f.result()
            log.append(res)
            if not res["ok"]:
                print("[FAIL] %s: %s" % (res["clip_id"], res["error"]))

    d = pd.DataFrame(log)
    d.to_csv(REPORTS / "normalize_batch1_log.csv", index=False)
    print("\nOK %d | FAILED %d | frames written %s"
          % (int(d["ok"].sum()), int((~d["ok"]).sum()), format(int(d["out_frames"].sum()), ",")))


if __name__ == "__main__":
    main()