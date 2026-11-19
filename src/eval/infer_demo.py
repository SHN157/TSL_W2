#!/usr/bin/env python
"""Inference demo: run the trained model on chosen clips and print Thai + English output.

    python infer_demo.py --ckpt runs/B_v2/best.pt --split test --n 10
    python infer_demo.py --ckpt runs/B_v2/best.pt --clips b2_A_09_0056,b1_1_worldcooking_0016
"""
import argparse, types
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers.modeling_outputs import BaseModelOutput
import sonar_slt as S

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/B_v2/best.pt")
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--clips", default="", help="comma-separated clip ids (overrides --split/--n)")
ap.add_argument("--langs", default="tha_Thai,eng_Latn")
ap.add_argument("--beams", type=int, default=5)
ap.add_argument("--no_repeat_ngram", type=int, default=3)
ap.add_argument("--out", default="runs/inference_demo.csv")
ap.add_argument("--split_csv", default="splits/split_v1.csv")
ap.add_argument("--targets", default="targets/sonar_th.npy")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
margs = types.SimpleNamespace(lang="tha_Thai", lora_r=16, lora_alpha=32,
                              lora_dropout=0.1, max_txt_tokens=256)
model = S.SonarSLT(margs, dev)
model.load_trainable_state(torch.load(a.ckpt, map_location=dev))
model.eval()
print("loaded", a.ckpt)

sp = pd.read_csv(a.split_csv)
rows = sp[sp["clip_id"].isin(a.clips.split(","))] if a.clips else sp[sp["split"] == a.split].head(a.n)
S_tgt = np.load(a.targets)
langs = a.langs.split(",")

def decode(z, lang):
    bos = model.tok.convert_tokens_to_ids(lang)
    with torch.inference_mode():
        g = model.sonar.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=z.unsqueeze(1)),
            attention_mask=torch.ones(z.shape[0], 1, dtype=torch.long, device=z.device),
            decoder_start_token_id=model.eos_id, forced_bos_token_id=bos,
            num_beams=a.beams, max_new_tokens=256, no_repeat_ngram_size=a.no_repeat_ngram)
    return [t.replace("\u0e4d\u0e32", "\u0e33").strip()
            for t in model.tok.batch_decode(g, skip_special_tokens=True)]

out = []
for _, r in rows.iterrows():
    spat = torch.from_numpy(np.load(r["spatial_path"]).astype(np.float32))
    moti = torch.from_numpy(np.load(r["motion_path"]).astype(np.float32))
    with torch.inference_mode():
        z = model.encode_video([spat], [moti])
    s = torch.from_numpy(S_tgt[int(r["row"])].astype(np.float32)).to(dev)[None]
    cos = torch.nn.functional.cosine_similarity(z, s, dim=-1).item()
    rec = {"clip_id": r["clip_id"], "domain": r["batch"], "frames": spat.shape[0],
           "reference_th": r["text_th_final"], "cos": round(cos, 3)}
    for lg in langs:
        rec["out_" + lg] = decode(z, lg)[0]
    out.append(rec)
    print("\n%s  (%s, %d frames, cos %.3f)" % (rec["clip_id"], rec["domain"], rec["frames"], cos))
    print("  reference : %s" % rec["reference_th"])
    for lg in langs:
        print("  %-9s: %s" % (lg, rec["out_" + lg]))

d = pd.DataFrame(out)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
d.to_csv(a.out, index=False)
print("\nsaved %d rows -> %s" % (len(d), a.out))