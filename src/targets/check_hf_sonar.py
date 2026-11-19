#!/usr/bin/env python
"""T5: does the authors' HF port (mtmlt/sonar-nllb-200-1.3B) reproduce official SONAR?"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, sacrebleu
from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ap = argparse.ArgumentParser()
ap.add_argument("--n_decode", type=int, default=200)
ap.add_argument("--bs", type=int, default=32)
args = ap.parse_args()

W = Path.home() / "work/signlanguage"
t = pd.read_csv(W / "targets/targets_index.csv")
S_off = np.load(W / "targets/sonar_th.npy")
texts = t["text_th_final"].astype(str).tolist()
dev = torch.device("cuda")

NAME = "mtmlt/sonar-nllb-200-1.3B"
tok = AutoTokenizer.from_pretrained(NAME)
tok.src_lang = "tha_Thai"
model = M2M100ForConditionalGeneration.from_pretrained(NAME).float().to(dev).eval()
enc_net = model.get_encoder()

ids = tok([texts[0]]).input_ids[0]
print("token format:", tok.convert_ids_to_tokens(ids[:3]), "...", tok.convert_ids_to_tokens(ids[-2:]))
print("decoder_start_token_id:", model.generation_config.decoder_start_token_id,
      "| eos:", tok.eos_token_id, "| tha_Thai id:", tok.convert_tokens_to_ids("tha_Thai"))

# 1) encoder + mean pooling vs official vectors (ALL sentences)
def encode(batch):
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    with torch.inference_mode():
        h = enc_net(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
    mask = enc.attention_mask.unsqueeze(-1).float()
    return ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()

S_hf = np.concatenate([encode(texts[i:i + args.bs]) for i in range(0, len(texts), args.bs)])
cos = (S_hf * S_off).sum(1) / (np.linalg.norm(S_hf, axis=1) * np.linalg.norm(S_off, axis=1))
mse = ((S_hf - S_off) ** 2).mean(1)
print("\n=== ENCODER: HF port vs official ===")
print("cos  mean %.6f  min %.6f" % (cos.mean(), cos.min()))
print("MSE  mean %.2e  max %.2e   (official MSE scale: mean |s|^2/1024 = %.2e)"
      % (mse.mean(), mse.max(), (S_off ** 2).mean()))
print("max abs diff %.2e" % np.abs(S_hf - S_off).max())
worst = np.argsort(cos)[:3]
for i in worst:
    print("  lowest cos %.6f  %s  len %d" % (cos[i], t.loc[i, "clip_id"], len(texts[i])))

# 2) decoder: official s -> text, compare with official round trip
samp = t.groupby("batch", group_keys=False).apply(lambda d: d.sample(args.n_decode // 2, random_state=0))
rows = samp.index.to_numpy()

def decode(emb, lang):
    out, bos = [], tok.convert_tokens_to_ids(lang)
    for i in range(0, len(emb), args.bs):
        e = torch.from_numpy(emb[i:i + args.bs]).to(dev)
        eo = BaseModelOutput(last_hidden_state=e.unsqueeze(1))
        am = torch.ones(e.shape[0], 1, dtype=torch.long, device=dev)
        with torch.inference_mode():
            g = model.generate(encoder_outputs=eo, attention_mask=am, forced_bos_token_id=bos,
                               num_beams=5, max_new_tokens=256)
        if i == 0:
            print("generated prefix:", tok.convert_ids_to_tokens(g[0][:3].tolist()))
        out += tok.batch_decode(g, skip_special_tokens=True)
    return [s.replace("\u0e4d\u0e32", "\u0e33").strip() for s in out]

hf_th = decode(S_off[rows], "tha_Thai")
off_th = samp["rt_th"].astype(str).tolist()
ref = samp["text_th_final"].astype(str).tolist()
print("\n=== DECODER from official s (%d sentences) ===" % len(rows))
print("HF      vs reference: chrF %.1f" % sacrebleu.corpus_chrf(hf_th, [ref]).score)
print("official vs reference: chrF %.1f" % sacrebleu.corpus_chrf(off_th, [ref]).score)
print("HF vs official output: chrF %.1f | identical %.1f%%"
      % (sacrebleu.corpus_chrf(hf_th, [off_th]).score,
         100 * np.mean([a == b for a, b in zip(hf_th, off_th)])))
for k in range(3):
    print(" REF: %s\n  HF: %s\n OFF: %s" % (ref[k], hf_th[k], off_th[k]))