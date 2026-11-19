#!/usr/bin/env python
"""Step 5b: decode ENGLISH from the video and score it against the human English reference.

Only batch-2 test clips have a human English sentence, so only those are scored.
Also measures 'Thai leakage': how many English outputs still contain Thai characters.

    python eval_en.py --ckpt runs/B_aug/best.pt --tag B_aug
    python eval_en.py --ckpt runs/B_v2/best.pt  --tag B_v2
"""
import argparse, re, types
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch

import sonar_slt as S

W = Path.home() / "work/signlanguage"
THAI = re.compile(r"[\u0E00-\u0E7F]")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--split_csv", default=str(W / "splits/split_v2.csv"))
ap.add_argument("--en_csv", default=str(W / "targets/en_targets.csv"))
ap.add_argument("--beams", type=int, default=5)
ap.add_argument("--no_repeat_ngram", type=int, default=3)
ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
margs = types.SimpleNamespace(lang="tha_Thai", lora_r=16, lora_alpha=32,
                              lora_dropout=0.1, max_txt_tokens=256)
model = S.SonarSLT(margs, dev)
model.load_trainable_state(torch.load(a.ckpt, map_location=dev))
model.eval()
print("loaded", a.ckpt)

sp = pd.read_csv(a.split_csv)
sp = sp[sp["split"] == a.split]
en = pd.read_csv(a.en_csv)
en = en[en["en_ok"].astype(bool)][["clip_id", "text_en"]]
d = sp.merge(en, on="clip_id", how="inner").reset_index(drop=True)
print("%s: %d clips with a human English reference" % (a.split, len(d)))

rows = []
for i in range(0, len(d), a.bs):
    chunk = d.iloc[i:i + a.bs]
    spat = [torch.from_numpy(np.load(p).astype(np.float32)) for p in chunk["spatial_path"]]
    moti = [torch.from_numpy(np.load(p).astype(np.float32)) for p in chunk["motion_path"]]
    with torch.inference_mode():
        z = model.encode_video(spat, moti)
        hyp_en = model.generate(z, beams=a.beams, no_repeat_ngram_size=a.no_repeat_ngram,
                                lang="eng_Latn")
        hyp_th = model.generate(z, beams=a.beams, no_repeat_ngram_size=a.no_repeat_ngram,
                                lang="tha_Thai")
    for (_, r), he, ht in zip(chunk.iterrows(), hyp_en, hyp_th):
        rows.append({"clip_id": r["clip_id"], "ref_en": r["text_en"], "hyp_en": he,
                     "ref_th": r["text_th_final"], "hyp_th": ht,
                     "thai_chars_in_en": bool(THAI.search(he))})
    print("  %d/%d" % (min(i + a.bs, len(d)), len(d)), end="\r", flush=True)
print()

p = pd.DataFrame(rows)
out = W / "runs" / ("EN_%s_%s.csv" % (a.tag, a.split))
p.to_csv(out, index=False)

chrf_en = sacrebleu.corpus_chrf(p["hyp_en"].tolist(), [p["ref_en"].tolist()]).score
bleu_en = sacrebleu.corpus_bleu(p["hyp_en"].tolist(), [p["ref_en"].tolist()]).score
leak = 100 * p["thai_chars_in_en"].mean()
print("\n=== %s | %s | %d batch-2 clips ===" % (a.tag, a.split, len(p)))
print("English chrF      : %.2f" % chrf_en)
print("English BLEU      : %.2f" % bleu_en)
print("Thai leakage      : %.1f%% of English outputs contain Thai characters" % leak)

print("\nexamples")
for _, r in p.sample(min(8, len(p)), random_state=0).iterrows():
    print("\n %s" % r["clip_id"])
    print("   ref  en : %s" % r["ref_en"])
    print("   model en: %s" % r["hyp_en"])
print("\nsaved -> %s" % out)