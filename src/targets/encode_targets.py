#!/usr/bin/env python
"""T3: SONAR target embeddings (official sonar-space) + round-trip ceiling check."""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, sacrebleu
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="N clips per batch (test run)")
ap.add_argument("--bs", type=int, default=32)
args = ap.parse_args()

R = Path.home() / "work/signlanguage/reports"
OUT = Path.home() / "work/signlanguage/targets" / ("test" if args.limit else "")
OUT.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(R / "master_clips.csv")
m = m[m["text_th_final"].notna()].reset_index(drop=True)
if args.limit:
    m = pd.concat([m[m["batch"] == b].head(args.limit) for b in ("batch1", "batch2")]).reset_index(drop=True)
th = m["text_th_final"].astype(str).tolist()
en_col = m["text_en_clean"] if "text_en_clean" in m else m["text_en"]
has_en = en_col.notna() & (en_col.astype(str).str.strip() != "")
print("sentences:", len(m), "| with English:", int(has_en.sum()), flush=True)

dev = torch.device("cuda")
t2v = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                   tokenizer="text_sonar_basic_encoder", device=dev)
v2t = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder",
                                   tokenizer="text_sonar_basic_encoder", device=dev)

def encode(texts, lang):
    out = []
    for i in range(0, len(texts), args.bs):
        with torch.inference_mode():
            e = t2v.predict(texts[i:i + args.bs], source_lang=lang)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)

def decode(emb, lang):
    out, t0 = [], time.time()
    for i in range(0, len(emb), args.bs):
        with torch.inference_mode():
            out += v2t.predict(torch.from_numpy(emb[i:i + args.bs]).to(dev),
                               target_lang=lang, max_seq_len=256)
        if (i // args.bs) % 10 == 0:
            print("  decode", lang, min(i + args.bs, len(emb)), "/", len(emb),
                  "%.0fs" % (time.time() - t0), flush=True)
    return out

def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

# 1) Thai targets
t0 = time.time()
s_th = encode(th, "tha_Thai")
print("encoded Thai:", s_th.shape, "%.0fs" % (time.time() - t0), flush=True)

# 2) English (batch 2) -> how language-agnostic is SONAR on our pairs
s_en = np.full_like(s_th, np.nan)
idx = np.where(has_en)[0]
if len(idx):
    s_en[idx] = encode(en_col[has_en].astype(str).tolist(), "eng_Latn")
cos_th_en = np.full(len(m), np.nan)
cos_th_en[idx] = (unit(s_th[idx]) * unit(s_en[idx])).sum(1)

# 3) round trip: s -> Thai, s -> English
rt_th = [t.replace("\u0e4d\u0e32", "\u0e33") for t in decode(s_th, "tha_Thai")]   # ํา -> ำ (same vowel, one character)
rt_en = decode(s_th, "eng_Latn")
m["rt_th"], m["rt_en"] = rt_th, rt_en
m["chrf_th"] = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(rt_th, th)]
m["exact_th"] = [h.strip() == r.strip() for h, r in zip(rt_th, th)]
m["cos_th_en"] = cos_th_en
m["norm_s"] = np.linalg.norm(s_th, axis=1)

# 4) save
np.save(OUT / "sonar_th.npy", s_th)
np.save(OUT / "sonar_en.npy", s_en)
m["row"] = np.arange(len(m))
m[["row", "clip_id", "batch", "include", "text_ok", "text_th_final", "rt_th", "rt_en",
   "chrf_th", "exact_th", "cos_th_en", "norm_s"]].to_csv(OUT / "targets_index.csv", index=False)

# 5) summary
print("\n=== SUMMARY ===")
print("NaN in s_th:", int(np.isnan(s_th).sum()), "| norm mean %.3f min %.3f max %.3f"
      % (m["norm_s"].mean(), m["norm_s"].min(), m["norm_s"].max()))
for b, d in m.groupby("batch"):
    refs, hyps = d["text_th_final"].tolist(), d["rt_th"].tolist()
    cb = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score
    cc = sacrebleu.corpus_chrf(hyps, [refs]).score
    print("%s  Thai round trip: chrF %.1f | char-BLEU %.1f | exact %.1f%%"
          % (b, cc, cb, 100 * d["exact_th"].mean()))
e = m[has_en]
if len(e):
    print("batch2 English from Thai vector vs real English: chrF %.1f"
          % sacrebleu.corpus_chrf(e["rt_en"].tolist(), [en_col[has_en].astype(str).tolist()]).score)
    print("cos(Thai vec, English vec): mean %.3f | min %.3f"
          % (e["cos_th_en"].mean(), e["cos_th_en"].min()))
print("\nworst 8 Thai round trips:")
for _, r in m.nsmallest(8, "chrf_th").iterrows():
    print(" %s chrF %.0f\n   ref: %s\n   out: %s" % (r["clip_id"], r["chrf_th"], r["text_th_final"], r["rt_th"]))
print("\nsaved to", OUT)