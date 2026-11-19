#!/usr/bin/env python
"""Step 2 of English augmentation: encode the English sentences with the OFFICIAL SONAR encoder.

Writes targets/sonar_en.npy, aligned row-for-row with targets/sonar_th.npy
(zeros for clips that have no English, i.e. all of batch 1).
QC: cos(th, en) per clip + a retrieval test, flagged rows written to targets/en_qc.csv.

Run from an ACTIVATED SONAR env (it needs the LD_LIBRARY_PATH export in SONAR/bin/activate),
not from Jupyter.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

W = Path.home() / "work/signlanguage"

ap = argparse.ArgumentParser()
ap.add_argument("--en_csv", default=str(W / "targets/en_targets.csv"))
ap.add_argument("--index", default=str(W / "targets/targets_index.csv"))
ap.add_argument("--th_npy", default=str(W / "targets/sonar_th.npy"))
ap.add_argument("--out", default=str(W / "targets/sonar_en.npy"))
ap.add_argument("--qc", default=str(W / "targets/en_qc.csv"))
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--flag_rank", type=int, default=20)
a = ap.parse_args()

df = pd.read_csv(a.en_csv)
S_th = np.load(a.th_npy)
N, D = S_th.shape
assert len(df) == N and df["row"].is_unique, "en_targets.csv does not line up with sonar_th.npy"
assert (df["row"].values == np.arange(N)).all(), "rows are not 0..N-1 in order"

e = df[df["en_ok"].astype(bool)].reset_index(drop=True)
print("encoding %d English sentences (of %d clips)" % (len(e), N))

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                    tokenizer="text_sonar_basic_encoder", device=dev)
emb = pipe.predict(e["text_en"].astype(str).tolist(), source_lang="eng_Latn",
                   batch_size=a.bs).cpu().numpy().astype(np.float32)
assert emb.shape == (len(e), D), "unexpected embedding shape %s" % (emb.shape,)

E = np.zeros((N, D), dtype=np.float32)
E[e["row"].values] = emb
np.save(a.out, E)
print("saved %s  shape %s" % (a.out, E.shape))

# ---------------------------------------------------------------- QC
TH = S_th[e["row"].values]
nrm = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
print("\nmean |v| : english %.4f   thai %.4f   (should both be about 0.19)"
      % (np.linalg.norm(emb, axis=1).mean(), np.linalg.norm(TH, axis=1).mean()))

cos = (nrm(emb) * nrm(TH)).sum(1)
print("cos(th,en): mean %.3f  median %.3f  min %.3f  | <0.70: %d  <0.50: %d"
      % (cos.mean(), np.median(cos), cos.min(), int((cos < 0.70).sum()), int((cos < 0.50).sum())))

sim = nrm(emb) @ nrm(TH).T                      # english -> all thai
rank = (sim > np.diag(sim)[:, None]).sum(1) + 1
print("retrieval (english vector finds its own thai vector, of %d):" % len(e))
print("  rank 1: %.1f%%  | rank<=5: %.1f%%  | rank<=%d: %.1f%%  | median rank %.0f"
      % (100 * (rank == 1).mean(), 100 * (rank <= 5).mean(), a.flag_rank,
         100 * (rank <= a.flag_rank).mean(), np.median(rank)))

idx = pd.read_csv(a.index)[["clip_id", "rt_en"]]
qc = e[["row", "clip_id", "batch", "text_th", "text_en"]].copy()
qc["cos"] = cos.round(3)
qc["rank"] = rank
qc = qc.merge(idx, on="clip_id", how="left").sort_values("rank", ascending=False)
qc.to_csv(a.qc, index=False)

flag = qc[qc["rank"] > a.flag_rank]
print("\nFLAGGED (own thai not in the top %d): %d clips" % (a.flag_rank, len(flag)))
pd.set_option("display.width", 200, "display.max_colwidth", 70)
for _, r in flag.head(30).iterrows():
    print("\n %s  rank %d  cos %.3f" % (r["clip_id"], r["rank"], r["cos"]))
    print("   human en : %s" % r["text_en"])
    print("   thai says: %s" % r["rt_en"])
print("\nfull QC table -> %s" % a.qc)