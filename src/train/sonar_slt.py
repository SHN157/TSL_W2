#!/usr/bin/env python
"""SONAR-SLT for Thai Sign Language (Hamidullah et al., WMT 2025) with the SpaMo visual block.

Phase A  (--phase A): train the Visual Block
          fusion (SpaMo) -> SONAR encoder (LoRA q/v) -> shallow-decoder pooling -> z
          loss = 7000*MSE(z,s) + 2.7*(1-cos(z,s)) + 0.1*CE(decode y from z)   [SONAR decoder frozen]
Phase B  (--phase B --init <phase A best.pt>): end-to-end translation
          loss = CE(y|z) + lambda_ae*CE(y|s) + 7000*MSE + 2.7*(1-cos)       [decoder LoRA trained too]
Eval     (--eval_only --init <ckpt> --split test): decode + metrics, writes predictions CSV.
"""
import argparse, copy, csv, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

SONAR_NAME = "mtmlt/sonar-nllb-200-1.3B"
W = Path.home() / "work/signlanguage"


# ----------------------------------------------------------------------------- data
class TSLDataset(Dataset):
    """One clip -> one (target vector, target text) pair.

    Target-language augmentation (paper 4.5), English-only variant: a batch-2 clip has a
    Thai AND an English target and __getitem__ picks one with probability en_prob.  Batch-1
    clips have no English and always use Thai.  en_prob = 0 reproduces the Thai-only run
    exactly, and dev/test always use Thai so the metrics stay comparable.
    """
    def __init__(self, split_csv, split, targets_npy, en_csv=None, en_npy=None, en_prob=0.0):
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.S = np.load(targets_npy)
        self.en_prob = float(en_prob)
        self.S_en, self.en_text = None, {}
        if self.en_prob > 0:
            self.S_en = np.load(en_npy)
            assert self.S_en.shape == self.S.shape, "sonar_en.npy and sonar_th.npy differ in shape"
            e = pd.read_csv(en_csv)
            e = e[e["en_ok"].astype(bool)]
            self.en_text = dict(zip(e["row"].astype(int), e["text_en"].astype(str)))
            have = int(self.df["row"].astype(int).isin(self.en_text).sum())
            print("%-5s: %d/%d clips have English (en_prob %.2f)"
                  % (split, have, len(self.df), self.en_prob))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        row = int(r["row"])
        use_en = (self.en_prob > 0 and row in self.en_text
                  and random.random() < self.en_prob)
        if use_en:
            vec, text, lang = self.S_en[row], self.en_text[row], "eng_Latn"
        else:
            vec, text, lang = self.S[row], str(r["text_th_final"]), "tha_Thai"
        return {
            "clip_id": r["clip_id"],
            "spatial": torch.from_numpy(np.load(r["spatial_path"]).astype(np.float32)),   # (T, 2048)
            "motion": torch.from_numpy(np.load(r["motion_path"]).astype(np.float32)),     # (N, 1024)
            "s": torch.from_numpy(vec.astype(np.float32)),                                # (1024,)
            "text": text,
            "lang": lang,
        }


def collate(batch):
    return {
        "clip_id": [b["clip_id"] for b in batch],
        "spatial": [b["spatial"] for b in batch],
        "motion": [b["motion"] for b in batch],
        "s": torch.stack([b["s"] for b in batch]),
        "text": [b["text"] for b in batch],
        "lang": [b["lang"] for b in batch],
    }


def lengths_to_mask(lengths, device):
    lengths = torch.as_tensor(lengths, device=device)
    return (torch.arange(int(lengths.max()), device=device)[None, :] < lengths[:, None]).long()


# ----------------------------------------------------------------------------- SpaMo visual block (verbatim logic)
class TemporalConv(nn.Module):
    """SpaMo spamo/tconv.py, conv_type=2: K5 P2 K5 P2 (Conv1d+BN+ReLU, MaxPool1d)."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.kernel_size = ["K5", "P2", "K5", "P2"]
        mods = []
        for i, ks in enumerate(self.kernel_size):
            in_sz = input_size if i == 0 else hidden_size
            if ks[0] == "P":
                mods.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            else:
                mods += [nn.Conv1d(in_sz, hidden_size, kernel_size=int(ks[1]), stride=1, padding=0),
                         nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True)]
        self.temporal_conv = nn.Sequential(*mods)

    def update_lgt(self, lgt):
        lgt = lgt.clone()
        for ks in self.kernel_size:
            lgt = torch.div(lgt, 2, rounding_mode="floor") if ks[0] == "P" else lgt - (int(ks[1]) - 1)
        return lgt

    def forward(self, x, lgt):                  # x: (B, C, L)
        return self.temporal_conv(x), self.update_lgt(lgt)


class SpaMoFusion(nn.Module):
    """SpaMo t5_slt.py, fusion_mode='joint', inter_hidden=768, input_size=2048:
    spatial->Linear, motion->Linear, concat in time, TemporalConv, then fusion_proj (mlp2x_gelu)."""
    def __init__(self, input_size=2048, motion_size=1024, inter_hidden=768, out_size=1024):
        super().__init__()
        self.spatio_proj = nn.Linear(input_size, inter_hidden)
        self.spatiotemp_proj = nn.Linear(motion_size, inter_hidden)
        self.temporal_encoder = TemporalConv(inter_hidden, inter_hidden)
        self.fusion_proj = nn.Sequential(nn.Linear(inter_hidden, out_size), nn.GELU(), nn.Linear(out_size, out_size))

    def forward(self, spatial_list, motion_list):
        dev = self.spatio_proj.weight.device
        t_len = torch.tensor([x.shape[0] for x in spatial_list], device=dev)
        n_len = torch.tensor([x.shape[0] for x in motion_list], device=dev)
        sp = self.spatio_proj(pad_sequence([x.to(dev) for x in spatial_list], batch_first=True))
        mo = self.spatiotemp_proj(pad_sequence([x.to(dev) for x in motion_list], batch_first=True))
        joint = pad_sequence([torch.cat((sp[i, :t_len[i]], mo[i, :n_len[i]]), 0) for i in range(len(spatial_list))],
                             batch_first=True)
        out, new_len = self.temporal_encoder(joint.permute(0, 2, 1), t_len + n_len)
        out = self.fusion_proj(out.permute(0, 2, 1))                       # (B, L', 1024)
        mask = lengths_to_mask(new_len, dev)
        return out[:, :mask.shape[1]], mask


# ----------------------------------------------------------------------------- shallow decoder pooling
class ShallowDecoderPooler(nn.Module):
    """Paper 4.4: feed the EOS token to a shallow decoder initialised from the first 3 SONAR decoder
    layers, cross-attending to the encoder outputs; the first output is the sentence embedding."""
    def __init__(self, full_decoder, eos_id, n_layers=3):
        super().__init__()
        self.config = full_decoder.config
        with torch.no_grad():
            ids = torch.tensor([[eos_id]], device=full_decoder.embed_tokens.weight.device)
            x0 = full_decoder.embed_tokens(ids) + full_decoder.embed_positions(ids, None)   # scaled emb + pos 0
        self.query = nn.Parameter(x0.detach().clone().float())                           # (1, 1, D)
        self.layers = nn.ModuleList([copy.deepcopy(full_decoder.layers[i]) for i in range(n_layers)])
        self.layer_norm = copy.deepcopy(full_decoder.layer_norm)
        # SONAR vectors are tiny (|s| ~ 0.19) while LayerNorm outputs have |.| ~ sqrt(1024).
        # A learnable output scale, calibrated once on real data (calibrate_scale), lets MSE work from step 1.
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, enc_h, enc_mask):
        h = self.query.expand(enc_h.shape[0], 1, -1)
        # additive 4D cross-attention mask (B,1,1,L): 0 = attend, very negative = padding.
        # Works with eager and sdpa attention in transformers 4.x and 5.x.
        cross = (1.0 - enc_mask[:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer in self.layers:
            h = layer(hidden_states=h, attention_mask=None, encoder_hidden_states=enc_h,
                      encoder_attention_mask=cross, use_cache=False)
            if isinstance(h, tuple):           # transformers 4.x returns a tuple
                h = h[0]
        return self.layer_norm(h)[:, 0] * self.log_scale.exp()             # (B, D)


# ----------------------------------------------------------------------------- full model
class SonarSLT(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(SONAR_NAME)
        base = M2M100ForConditionalGeneration.from_pretrained(SONAR_NAME).float()
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id
        self.lang_id = self.tok.convert_tokens_to_ids(args.lang)
        self.pooler = ShallowDecoderPooler(base.get_decoder(), self.eos_id, n_layers=3)   # copy BEFORE LoRA
        for p in base.parameters():
            p.requires_grad = False
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=["q_proj", "v_proj"], bias="none")
        self.sonar = get_peft_model(base, lcfg)
        self.fusion = SpaMoFusion(out_size=base.config.d_model)
        self.max_txt_tokens = args.max_txt_tokens
        self.to(device)
        self.device = device

    def set_phase(self, phase):
        """A: decoder LoRA frozen.  B: everything trainable (fusion, encoder LoRA, pooler, decoder LoRA)."""
        for n, p in self.sonar.named_parameters():
            if "lora_" in n:
                p.requires_grad = (phase == "B") or (".decoder." not in n)
            else:
                p.requires_grad = False
        for p in list(self.fusion.parameters()) + list(self.pooler.parameters()):
            p.requires_grad = True

    def encode_video(self, spatial, motion):
        x, mask = self.fusion(spatial, motion)
        h = self.sonar.get_encoder()(inputs_embeds=x, attention_mask=mask).last_hidden_state
        return self.pooler(h, mask)                                         # z: (B, 1024)

    @torch.no_grad()
    def calibrate_scale(self, loader, n_batches=10):
        """Set the pooler output scale so that mean |z| == mean |s| on a few training batches."""
        was = self.training; self.eval()
        zn, sn = [], []
        for i, b in enumerate(loader):
            if i >= n_batches:
                break
            zn.append(self.encode_video(b["spatial"], b["motion"]).norm(dim=-1)); sn.append(b["s"].norm(dim=-1))
        z_mean, s_mean = torch.cat(zn).mean().item(), torch.cat(sn).mean().item()
        self.pooler.log_scale += math.log(s_mean / z_mean)
        self.train(was)
        print("calibrated output scale: mean|z| %.4f -> mean|s| %.4f" % (z_mean, s_mean))

    def labels(self, texts, langs=None):
        """[lang token] ... [eos].  langs=None -> the run's primary language for every row."""
        ids = self.tok(texts, add_special_tokens=False).input_ids
        lg = [self.lang_id] * len(ids) if langs is None else \
             [self.tok.convert_tokens_to_ids(l) for l in langs]
        seqs = [torch.tensor([lg[i]] + x[: self.max_txt_tokens - 2] + [self.eos_id])
                for i, x in enumerate(ids)]
        lab = pad_sequence(seqs, batch_first=True, padding_value=-100).to(self.device)
        return lab

    def ce_from_vector(self, v, lab):
        """Teacher-forced CE of the SONAR decoder conditioned on one vector (as SONAR decodes)."""
        dec_in = shift_tokens_right(lab, self.pad_id, self.eos_id)          # </s> lang tok ... ; -100 -> pad
        out = self.sonar(encoder_outputs=BaseModelOutput(last_hidden_state=v.unsqueeze(1)),
                         attention_mask=torch.ones(v.shape[0], 1, dtype=torch.long, device=v.device),
                         decoder_input_ids=dec_in)
        return F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), lab.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, z, beams=5, max_new_tokens=256, no_repeat_ngram_size=0, lang=None):
        bos = self.lang_id if lang is None else self.tok.convert_tokens_to_ids(lang)
        g = self.sonar.generate(encoder_outputs=BaseModelOutput(last_hidden_state=z.unsqueeze(1)),
                                attention_mask=torch.ones(z.shape[0], 1, dtype=torch.long, device=z.device),
                                decoder_start_token_id=self.eos_id, forced_bos_token_id=bos,
                                num_beams=beams, max_new_tokens=max_new_tokens,
                                no_repeat_ngram_size=no_repeat_ngram_size)
        txt = self.tok.batch_decode(g, skip_special_tokens=True)
        return [t.replace("\u0e4d\u0e32", "\u0e33").strip() for t in txt]

    # ---- checkpoint = only what is trained
    def trainable_state(self):
        return {"fusion": self.fusion.state_dict(), "pooler": self.pooler.state_dict(),
                "lora": get_peft_model_state_dict(self.sonar)}

    def load_trainable_state(self, st):
        self.fusion.load_state_dict(st["fusion"])
        self.pooler.load_state_dict(st["pooler"])
        res = set_peft_model_state_dict(self.sonar, st["lora"])
        unexpected = getattr(res, "unexpected_keys", [])
        assert not unexpected, "unexpected LoRA keys: %s" % unexpected[:5]


def sem_losses(z, s):
    mse = F.mse_loss(z, s)                                   # mean over elements (paper: falls to ~1e-5)
    cos = F.cosine_similarity(z, s, dim=-1)
    return mse, cos


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, args, decode, mu):
    """mu = mean of TRAIN targets. Besides cos, report the sentence-specific signal:
    ccos = cosine after subtracting mu from z and s (an 'average sentence' predictor scores ~0),
    R@1 / R@5 / median rank = retrieve the right sentence among all dev targets from z."""
    import sacrebleu
    model.eval()
    tot = {"mse": 0.0, "ce": 0.0, "n": 0}
    rows, Z, S = [], [], []
    for b in loader:
        z = model.encode_video(b["spatial"], b["motion"])
        s = b["s"].to(model.device)
        mse, cos = sem_losses(z, s)
        ce = model.ce_from_vector(z, model.labels(b["text"], b.get("lang")))
        n = len(b["text"])
        tot["mse"] += mse.item() * n; tot["ce"] += ce.item() * n; tot["n"] += n
        Z.append(z.float().cpu()); S.append(s.float().cpu())
        if decode:
            hyp = model.generate(z, beams=args.beams, no_repeat_ngram_size=args.no_repeat_ngram)
            for cid, ref, h, c in zip(b["clip_id"], b["text"], hyp, cos.tolist()):
                rows.append({"clip_id": cid, "ref": ref, "hyp": h, "cos": c})
    Z, S = torch.cat(Z), torch.cat(S)
    zc, sc = Z - mu, S - mu
    sim = F.normalize(Z, dim=-1) @ F.normalize(S, dim=-1).T
    rank = (sim > sim.diag()[:, None]).sum(1) + 1
    res = {"mse": tot["mse"] / tot["n"], "ce": tot["ce"] / tot["n"],
           "cos": F.cosine_similarity(Z, S, dim=-1).mean().item(),
           "ccos": F.cosine_similarity(zc, sc, dim=-1).mean().item(),
           "r1": (rank <= 1).float().mean().item(), "r5": (rank <= 5).float().mean().item(),
           "medr": rank.float().median().item()}
    if decode:
        refs, hyps = [r["ref"] for r in rows], [r["hyp"] for r in rows]
        res["chrf"] = sacrebleu.corpus_chrf(hyps, [refs]).score
        res["bleu_char"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score
    model.train()
    return res, rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="checkpoint to start from (phase B: phase A best.pt)")
    ap.add_argument("--split_csv", default=str(W / "splits/split_v2.csv"))
    ap.add_argument("--targets", default=str(W / "targets/sonar_th.npy"))
    ap.add_argument("--en_csv", default=str(W / "targets/en_targets.csv"))
    ap.add_argument("--en_npy", default=str(W / "targets/sonar_en.npy"))
    ap.add_argument("--en_prob", type=float, default=0.0,
                    help="training only: chance of using the English target when a clip has one; "
                         "0.0 = Thai-only (the old behaviour), 0.5 = even split")
    ap.add_argument("--lang", default="tha_Thai")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of total steps")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=7000.0)
    ap.add_argument("--lambda_cos", type=float, default=2.7)
    ap.add_argument("--lambda_ce", type=float, default=None, help="default: A 0.1, B 1.0")
    ap.add_argument("--lambda_ae", type=float, default=1.0, help="phase B only")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--max_txt_tokens", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=1, help="epochs between dev evals")
    ap.add_argument("--patience", type=int, default=15, help="dev evals without improvement")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: N clips per split")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--split", default="test", help="split for --eval_only")
    ap.add_argument("--domain", default="", choices=["", "batch1", "batch2"],
                    help="train/evaluate on one domain only (per-domain fine-tuning)")
    args = ap.parse_args()
    if args.lambda_ce is None:
        args.lambda_ce = 0.1 if args.phase == "A" else 1.0

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / ("args_eval.json" if args.eval_only else "args.json"), "w"), indent=2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SonarSLT(args, dev)
    if args.init:
        model.load_trainable_state(torch.load(args.init, map_location=dev))
        print("loaded", args.init)
    model.set_phase(args.phase)

    def loader(split, shuffle):
        ds = TSLDataset(args.split_csv, split, args.targets, en_csv=args.en_csv,
                        en_npy=args.en_npy,
                        en_prob=(args.en_prob if split == "train" else 0.0))
        if args.domain:
            ds.df = ds.df[ds.df["batch"] == args.domain].reset_index(drop=True)
        if args.limit:
            ds.df = ds.df.head(args.limit)
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle, collate_fn=collate,
                          num_workers=args.workers, pin_memory=True)

    _sp = pd.read_csv(args.split_csv)
    mu = torch.from_numpy(np.load(args.targets)[_sp.loc[_sp["split"] == "train", "row"].values].mean(0)).float()

    if args.eval_only:
        res, rows = evaluate(model, loader(args.split, False), args, decode=True, mu=mu)
        pd.DataFrame(rows).to_csv(out / ("pred_%s.csv" % args.split), index=False)
        json.dump(res, open(out / ("metrics_%s.json" % args.split), "w"), indent=2)
        print(args.split, json.dumps(res, indent=1))
        return

    tr, dv = loader("train", True), loader("dev", False)
    if not args.init:
        model.calibrate_scale(tr)
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    print("trainable params: %.2fM | train %d dev %d clips" % (n_tr / 1e6, len(tr.dataset), len(dv.dataset)))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = math.ceil(len(tr) / args.accum) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)

    decode = args.phase == "B"
    best_key = "chrf" if decode else "ccos"      # phase A: sentence-specific (centred) cosine
    best, bad, step = -1e9, 0, 0
    logf = open(out / "log.csv", "a", newline="")
    logw = None
    model.train()
    for ep in range(1, args.epochs + 1):
        t0, agg, nb = time.time(), {}, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(tr):
            z = model.encode_video(b["spatial"], b["motion"])
            s = b["s"].to(dev)
            mse, cos = sem_losses(z, s)
            l_sem = args.lambda_mse * mse + args.lambda_cos * (1 - cos).mean()
            lab = model.labels(b["text"], b["lang"])
            l_ce = model.ce_from_vector(z, lab)
            loss = l_sem + args.lambda_ce * l_ce
            parts = {"loss": loss, "mse": mse, "cos": cos.mean(), "ce": l_ce}
            if args.phase == "B" and args.lambda_ae > 0:
                l_ae = model.ce_from_vector(s, lab)
                loss = loss + args.lambda_ae * l_ae
                parts["loss"], parts["ae"] = loss, l_ae
            parts = {k: v.detach() for k, v in parts.items()}
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(tr):
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            nb += 1
        trm = {("train_" + k): v / nb for k, v in agg.items()}
        row = {"epoch": ep, "step": step, "lr": sch.get_last_lr()[0], "sec": round(time.time() - t0), **trm}
        if ep % args.eval_every == 0 or ep == args.epochs:
            res, rows = evaluate(model, dv, args, decode, mu)
            row.update({("dev_" + k): v for k, v in res.items()})
            if res[best_key] > best:
                best, bad = res[best_key], 0
                torch.save(model.trainable_state(), out / "best.pt")
                if rows:
                    pd.DataFrame(rows).to_csv(out / "pred_dev_best.csv", index=False)
                row["best"] = 1
            else:
                bad += 1
        torch.save(model.trainable_state(), out / "last.pt")
        if logw is None:
            keys = ["epoch", "step", "lr", "sec", "train_loss", "train_mse", "train_cos", "train_ce", "train_ae",
                    "dev_mse", "dev_cos", "dev_ccos", "dev_r1", "dev_r5", "dev_medr", "dev_ce",
                    "dev_chrf", "dev_bleu_char", "best"]
            logw = csv.DictWriter(logf, fieldnames=keys, extrasaction="ignore")
            if logf.tell() == 0:
                logw.writeheader()
        logw.writerow(row); logf.flush()
        print(" | ".join("%s %s" % (k, ("%.4g" % v) if isinstance(v, float) else v) for k, v in row.items()), flush=True)
        if bad >= args.patience:
            print("early stop: no dev %s improvement for %d evals" % (best_key, bad))
            break
    print("best dev %s = %.4f" % (best_key, best))


if __name__ == "__main__":
    main()