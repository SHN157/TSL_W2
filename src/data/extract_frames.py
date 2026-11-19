"""
extract_frames.py — batch 2 RGB videos -> 30 fps JPEG frames.

Reads each video's own frame rate and keeps every Nth frame.
Safe to re-run: finished clips are skipped.

Usage:
    python extract_frames.py --dry-run
    python extract_frames.py --limit 5
    python extract_frames.py
"""

import argparse
import re
import shutil
from pathlib import Path

import av
import pandas as pd
from PIL import Image
from tqdm import tqdm

HOME     = Path.home()
ARCH     = HOME / "dataset_arch" / "SignLanguage"
REPORTS  = HOME / "work" / "signlanguage" / "reports"

TARGET_FPS = 30
SIZE       = 448
QUALITY    = 95
OUT_ROOT   = ARCH / f"frames_batch2_{TARGET_FPS}fps"

CATNUM = re.compile(r"^(\d+)")


def center_square(img):
    """Crop the middle square out of a wide frame."""
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def clip_id(row):
    station = "A" if "stationA" in row.station else "B"
    catnum = int(CATNUM.match(row.category).group(1))
    return "b2_%s_%02d_%04d" % (station, catnum, int(row.id))


def extract_one(row, out_dir):
    """Decode one video, keep every Nth frame, save as JPEG. Returns frames saved."""
    stride = max(1, round(row.fps / TARGET_FPS))

    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    kept = 0
    with av.open(str(row.path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for i, frame in enumerate(container.decode(video=0)):
            if i % stride:
                continue
            img = center_square(frame.to_image()).resize((SIZE, SIZE), Image.BICUBIC)
            img.save(tmp / ("%06d.jpg" % kept), quality=QUALITY)
            kept += 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)          # rename only when complete
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rgb = pd.read_csv(REPORTS / "batch2_rgb.csv")
    rgb["cid"] = rgb.apply(clip_id, axis=1)
    assert rgb.cid.is_unique, "clip ids are not unique"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    todo, done = [], 0
    for _, r in rgb.iterrows():
        d = OUT_ROOT / r.cid
        if d.is_dir() and any(d.iterdir()):
            done += 1
        else:
            todo.append(r)

    if args.limit:
        todo = todo[:args.limit]

    print("total clips   : %d" % len(rgb))
    print("already done  : %d" % done)
    print("to extract now: %d" % len(todo))
    print("output        : %s" % OUT_ROOT)
    print("settings      : %d fps, %dx%d jpeg q%d" % (TARGET_FPS, SIZE, SIZE, QUALITY))

    if args.dry_run:
        print("\n-- dry run, first 5 --")
        for r in todo[:5]:
            stride = round(r.fps / TARGET_FPS)
            print("  %s  %.0f fps  stride=%d  %d frames -> ~%d"
                  % (r.cid, r.fps, stride, r.nb_frames, round(r.nb_frames / stride)))
        return

    log = []
    for r in tqdm(todo):
        try:
            n = extract_one(r, OUT_ROOT / r.cid)
            log.append({"cid": r.cid, "frames": n, "fps": r.fps, "ok": True, "error": ""})
        except Exception as e:
            log.append({"cid": r.cid, "frames": 0, "fps": r.fps, "ok": False,
                        "error": "%s: %s" % (type(e).__name__, e)})
            print("[FAIL] %s: %s" % (r.cid, e))

    d = pd.DataFrame(log)
    d.to_csv(REPORTS / "extract_frames_log.csv", index=False)
    print("\nOK %d | FAILED %d | frames written %s"
          % (int(d.ok.sum()), int((~d.ok).sum()), format(int(d.frames.sum()), ",")))
    print("log -> %s" % (REPORTS / "extract_frames_log.csv"))


if __name__ == "__main__":
    main()