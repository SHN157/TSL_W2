#!/usr/bin/env python
"""S1: random, signer-mixed train/dev/test split (80/10/10).
Stratified by batch + category + station; identical sentences always stay in the same split."""
from pathlib import Path
import json
import numpy as np, pandas as pd

SEED, P_TRAIN, P_DEV = 42, 0.8, 0.1
W = Path.home() / "work/signlanguage"
D = Path.home() / "dataset_arch/SignLanguage"
OUT = W / "splits"; OUT.mkdir(exist_ok=True)

m = pd.read_csv(W / "reports/master_clips.csv")
t = pd.read_csv(W / "targets/targets_index.csv")[["row", "clip_id", "text_th_final"]]
t = t.rename(columns={"text_th_final": "text_encoded"})
m = m.merge(t, on="clip_id", how="left")
is_true = lambda s: s.astype(str).str.lower().isin(["true", "1"])
m = m[is_true(m["include"]) & is_true(m["text_ok"])].copy()
print("clips used:", len(m))

# 1) target text must be exactly what was encoded into sonar_th.npy
assert m["row"].notna().all(), "some clip is missing from targets_index.csv"
changed = int((m["text_th_final"] != m["text_encoded"]).sum())
print("Thai text changed since encoding:", changed, "(must be 0)")
assert changed == 0
m["row"] = m["row"].astype(int)

# 2) feature files must exist
def feat(batch, cid, kind, suffix):
    b = "b1" if batch == "batch1" else "b2"
    return str(D / ("features_" + b + "_30fps") / kind / (cid + suffix))
m["spatial_path"] = [feat(b, c, "spatial", "_s2wrapping.npy") for b, c in zip(m["batch"], m["clip_id"])]
m["motion_path"] = [feat(b, c, "motion", "_overlap-8.npy") for b, c in zip(m["batch"], m["clip_id"])]
missing = [p for p in list(m["spatial_path"]) + list(m["motion_path"]) if not Path(p).exists()]
print("missing feature files:", len(missing), missing[:5])
assert not missing

# 3) grouped + stratified split (group = identical Thai sentence)
m["stratum"] = m["batch"].astype(str) + "|" + m["cat"].astype(str) + "|" + m["station"].fillna("-").astype(str)
g = (m.groupby("text_th_final", sort=True)
       .agg(n=("clip_id", "size"), stratum=("stratum", "first"))
       .reset_index())
g["split"] = ""
rng = np.random.default_rng(SEED)
for s, d in g.groupby("stratum", sort=True):
    idx = d.index.to_numpy().copy()
    rng.shuffle(idx)
    cum = np.cumsum(g.loc[idx, "n"].to_numpy()) / g.loc[idx, "n"].sum()
    g.loc[idx, "split"] = np.where(cum <= P_TRAIN + 1e-9, "train",
                          np.where(cum <= P_TRAIN + P_DEV + 1e-9, "dev", "test"))
m = m.merge(g[["text_th_final", "split"]], on="text_th_final", how="left")

# 4) checks
leak = m.groupby("text_th_final")["split"].nunique()
print("sentences in more than one split:", int((leak > 1).sum()), "(must be 0)")
assert (leak > 1).sum() == 0

order = ["train", "dev", "test"]
print("\n=== clips / hours per split ===")
tab = m.groupby(["split", "batch"]).agg(clips=("clip_id", "size"), hours=("duration", "sum"))
tab["hours"] = (tab["hours"] / 3600).round(2)
print(tab.unstack("batch").reindex(order))
tot = m.groupby("split").agg(clips=("clip_id", "size"), hours=("duration", "sum")).reindex(order)
tot["hours"] = (tot["hours"] / 3600).round(2)
tot["share_%"] = (100 * tot["clips"] / len(m)).round(1)
print(tot)

print("\n=== clips per category (batch|cat|station) ===")
print(pd.crosstab(m["stratum"], m["split"])[order].to_string())

print("\n=== signers per split ===")
print(pd.crosstab(m["signer"].fillna("unknown"), m["split"])[order])

print("\n=== mean Thai length (chars) ===")
m["n_chars"] = m["text_th_final"].str.len()
print(m.pivot_table(index="batch", columns="split", values="n_chars", aggfunc="mean")[order].round(1))

# 5) save
cols = ["clip_id", "split", "batch", "cat", "station", "signer", "row", "text_th_final",
        "duration", "spatial_path", "motion_path"]
m[cols].sort_values(["split", "clip_id"]).to_csv(OUT / "split_v1.csv", index=False)
for s in order:
    (OUT / (s + "_v1.txt")).write_text("\n".join(sorted(m.loc[m["split"] == s, "clip_id"])) + "\n")
json.dump({"seed": SEED, "ratios": [P_TRAIN, P_DEV, 1 - P_TRAIN - P_DEV],
           "grouping": "identical text_th_final", "strata": "batch|cat|station",
           "filter": "include & text_ok", "targets": "targets/sonar_th.npy (row column)"},
          open(OUT / "split_v1_meta.json", "w"), indent=2)
print("\nsaved:", OUT / "split_v1.csv", "+ train/dev/test_v1.txt + split_v1_meta.json")