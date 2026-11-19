"""
extract_features.py — ViT (spatial) + VideoMAE (motion) features.

Mirrors SpaMo's scripts/vit_extract_feature.py and mae_extract_feature.py:
  spatial : CLIP ViT-L/14, S2 scales [1,2], num_prefix_token=1, CLS token -> (T, 2048)
  motion  : VideoMAE-L, window 16 / overlap 8, CLS token          -> (N, 1024)

Safe to re-run: finished clips are skipped.

Usage:
    python extract_features.py --dry-run
    python extract_features.py --limit 5
    python extract_features.py --gpu 1 --precision bf16
    python extract_features.py --gpu 1 --precision bf16 --shard 0 --num-shards 2
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (AutoImageProcessor, CLIPVisionModel,
                          VideoMAEModel, VideoMAEImageProcessor)

HOME      = Path.home()
ARCH      = HOME / "dataset_arch" / "SignLanguage"
REPORTS   = HOME / "work" / "signlanguage" / "reports"
SPAMO_DIR = HOME / "work" / "SpaMo_ref"

FRAMES   = ARCH / "frames_batch2_30fps"
FEAT_DIR = ARCH / "features_b2_30fps"
SPATIAL  = FEAT_DIR / "spatial"
MOTION   = FEAT_DIR / "motion"

VIT_MODEL   = "openai/clip-vit-large-patch14"
MAE_MODEL   = "MCG-NJU/videomae-large"
S2_SCALES   = [1, 2]
MAE_WINDOW  = 16
MAE_OVERLAP = 8
BATCH_SIZE  = 32

np.random.seed(0)
torch.manual_seed(0)


def load_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--frames-dir", type=str, default=None)
    ap.add_argument("--out-dir",    type=str, default=None)
    args = ap.parse_args()
    global FRAMES, SPATIAL, MOTION, FEAT_DIR
    if args.frames_dir:
        FRAMES = Path(args.frames_dir)
    if args.out_dir:
        FEAT_DIR = Path(args.out_dir)
        SPATIAL  = FEAT_DIR / "spatial"
        MOTION   = FEAT_DIR / "motion"
        
    device = torch.device(f"cuda:{args.gpu}")
    use_bf16 = args.precision == "bf16"

    SPATIAL.mkdir(parents=True, exist_ok=True)
    MOTION.mkdir(parents=True, exist_ok=True)

    # ---- clip list, sharded ----
    clips = sorted(p.name for p in FRAMES.iterdir() if p.is_dir())
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]

    todo, done = [], 0
    for c in clips:
        sp = SPATIAL / f"{c}_s2wrapping.npy"
        mo = MOTION / f"{c}_overlap-{MAE_OVERLAP}.npy"
        if sp.exists() and mo.exists():
            done += 1
        else:
            todo.append(c)
    if args.limit:
        todo = todo[:args.limit]

    print(f"shard {args.shard}/{args.num_shards}  gpu {args.gpu}  {args.precision}  "
          f"batch {args.batch_size}")
    print(f"clips in shard : {len(clips)}")
    print(f"already done   : {done}")
    print(f"to extract now : {len(todo)}")
    print(f"output         : {FEAT_DIR}")

    if args.dry_run:
        for c in todo[:5]:
            n = len(list((FRAMES / c).glob('*.jpg')))
            print(f"  {c}  {n} frames")
        return
    if not todo:
        print("nothing to do")
        return

    # ---- SpaMo helpers, unchanged ----
    s2w = load_from_file("s2wrapper", SPAMO_DIR / "utils" / "s2wrapper.py")
    helpers = load_from_file("helpers", SPAMO_DIR / "utils" / "helpers.py")
    multiscale_forward = s2w.forward
    sliding_window_for_list = helpers.sliding_window_for_list

    # ---- models ----
    print("loading models ...")
    vit_model = CLIPVisionModel.from_pretrained(
        VIT_MODEL, output_hidden_states=True).to(device).eval()
    vit_proc = AutoImageProcessor.from_pretrained(VIT_MODEL)
    mae_model = VideoMAEModel.from_pretrained(MAE_MODEL).to(device).eval()
    mae_proc = VideoMAEImageProcessor.from_pretrained(MAE_MODEL)
    print("   ok")

    @torch.no_grad()
    def vit_forward_features(inputs):
        return vit_model(inputs).hidden_states[-1]

    @torch.no_grad()
    def extract_spatial(images):
        out = []
        for j in range(0, len(images), args.batch_size):
            batch = images[j:j + args.batch_size]
            inputs = vit_proc(list(batch), return_tensors="pt").to(device).pixel_values
            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = multiscale_forward(vit_forward_features, inputs,
                                               scales=S2_SCALES, num_prefix_token=1)
            else:
                feats = multiscale_forward(vit_forward_features, inputs,
                                           scales=S2_SCALES, num_prefix_token=1)
            out.append(feats[:, 0].float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def extract_motion(images):
        imgs = list(images)
        if len(imgs) < MAE_WINDOW:
            imgs = imgs + [imgs[-1]] * (MAE_WINDOW - len(imgs))
        windows = sliding_window_for_list(imgs, window_size=MAE_WINDOW,
                                          overlap_size=MAE_OVERLAP)
        out = []
        for j in range(0, len(windows), args.batch_size):
            batch = windows[j:j + args.batch_size]
            inputs = mae_proc(images=batch, return_tensors="pt").to(device)
            if use_bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = mae_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
            else:
                feats = mae_model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    # ---- run ----
    log = []
    t_start = time.time()
    for cid in tqdm(todo):
        try:
            files = sorted((FRAMES / cid).glob("*.jpg"))
            imgs = [Image.open(f).convert("RGB") for f in files]

            sp = extract_spatial(imgs)
            mo = extract_motion(imgs)

            # write to a temp name, then rename — so a killed job leaves no half file
            sp_path = SPATIAL / f"{cid}_s2wrapping.npy"
            mo_path = MOTION / f"{cid}_overlap-{MAE_OVERLAP}.npy"
            np.save(sp_path.with_suffix(".tmp.npy"), sp)
            np.save(mo_path.with_suffix(".tmp.npy"), mo)
            sp_path.with_suffix(".tmp.npy").rename(sp_path)
            mo_path.with_suffix(".tmp.npy").rename(mo_path)

            log.append({"cid": cid, "T": len(imgs),
                        "spatial": str(sp.shape), "motion": str(mo.shape),
                        "ok": True, "error": ""})
        except Exception as e:
            log.append({"cid": cid, "T": -1, "spatial": "", "motion": "",
                        "ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"[FAIL] {cid}: {e}")
            torch.cuda.empty_cache()

    d = pd.DataFrame(log)
    out_log = REPORTS / f"extract_features_shard{args.shard}.csv"
    d.to_csv(out_log, index=False)
    mins = (time.time() - t_start) / 60
    print(f"\nOK {int(d['ok'].sum())} | FAILED {int((~d['ok']).sum())} | {mins:.1f} min")
    print(f"log -> {out_log}")


if __name__ == "__main__":
    main()