#!/usr/bin/env python
"""Patch sonar_slt.py for English target augmentation. Run once: python patch_augment.py"""
from pathlib import Path

P = Path.home() / "work/signlanguage/sonar_slt.py"
src = P.read_text()
if "en_prob" in src:
    raise SystemExit("already patched - nothing to do")

PATCHES = []

# ---------------------------------------------------------------- 1. dataset + collate
PATCHES.append(('''class TSLDataset(Dataset):
    def __init__(self, split_csv, split, targets_npy):
        df = pd.read_csv(split_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.S = np.load(targets_npy)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {
            "clip_id": r["clip_id"],
            "spatial": torch.from_numpy(np.load(r["spatial_path"]).astype(np.float32)),   # (T, 2048)
            "motion": torch.from_numpy(np.load(r["motion_path"]).astype(np.float32)),     # (N, 1024)
            "s": torch.from_numpy(self.S[int(r["row"])].astype(np.float32)),              # (1024,)
            "text": str(r["text_th_final"]),
        }


def collate(batch):
    return {
        "clip_id": [b["clip_id"] for b in batch],
        "spatial": [b["spatial"] for b in batch],
        "motion": [b["motion"] for b in batch],
        "s": torch.stack([b["s"] for b in batch]),
        "text": [b["text"] for b in batch],
    }''',
'''class TSLDataset(Dataset):
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
    }'''))

# ---------------------------------------------------------------- 2. per-example language token
PATCHES.append(('''    def labels(self, texts):
        ids = self.tok(texts, add_special_tokens=False).input_ids
        seqs = [torch.tensor([self.lang_id] + x[: self.max_txt_tokens - 2] + [self.eos_id]) for x in ids]''',
'''    def labels(self, texts, langs=None):
        """[lang token] ... [eos].  langs=None -> the run's primary language for every row."""
        ids = self.tok(texts, add_special_tokens=False).input_ids
        lg = [self.lang_id] * len(ids) if langs is None else \\
             [self.tok.convert_tokens_to_ids(l) for l in langs]
        seqs = [torch.tensor([lg[i]] + x[: self.max_txt_tokens - 2] + [self.eos_id])
                for i, x in enumerate(ids)]'''))

# ---------------------------------------------------------------- 3. decode in any language
PATCHES.append(('''    def generate(self, z, beams=5, max_new_tokens=256, no_repeat_ngram_size=0):
        g = self.sonar.generate(encoder_outputs=BaseModelOutput(last_hidden_state=z.unsqueeze(1)),
                                attention_mask=torch.ones(z.shape[0], 1, dtype=torch.long, device=z.device),
                                decoder_start_token_id=self.eos_id, forced_bos_token_id=self.lang_id,''',
'''    def generate(self, z, beams=5, max_new_tokens=256, no_repeat_ngram_size=0, lang=None):
        bos = self.lang_id if lang is None else self.tok.convert_tokens_to_ids(lang)
        g = self.sonar.generate(encoder_outputs=BaseModelOutput(last_hidden_state=z.unsqueeze(1)),
                                attention_mask=torch.ones(z.shape[0], 1, dtype=torch.long, device=z.device),
                                decoder_start_token_id=self.eos_id, forced_bos_token_id=bos,'''))

PATCHES.append(('        ce = model.ce_from_vector(z, model.labels(b["text"]))',
                '        ce = model.ce_from_vector(z, model.labels(b["text"], b.get("lang")))'))

PATCHES.append(('            lab = model.labels(b["text"])',
                '            lab = model.labels(b["text"], b["lang"])'))

# ---------------------------------------------------------------- 4. wiring
PATCHES.append(('''    def loader(split, shuffle):
        ds = TSLDataset(args.split_csv, split, args.targets)''',
'''    def loader(split, shuffle):
        ds = TSLDataset(args.split_csv, split, args.targets, en_csv=args.en_csv,
                        en_npy=args.en_npy,
                        en_prob=(args.en_prob if split == "train" else 0.0))'''))

PATCHES.append(('''    ap.add_argument("--split_csv", default=str(W / "splits/split_v1.csv"))
    ap.add_argument("--targets", default=str(W / "targets/sonar_th.npy"))''',
'''    ap.add_argument("--split_csv", default=str(W / "splits/split_v2.csv"))
    ap.add_argument("--targets", default=str(W / "targets/sonar_th.npy"))
    ap.add_argument("--en_csv", default=str(W / "targets/en_targets.csv"))
    ap.add_argument("--en_npy", default=str(W / "targets/sonar_en.npy"))
    ap.add_argument("--en_prob", type=float, default=0.0,
                    help="training only: chance of using the English target when a clip has one; "
                         "0.0 = Thai-only (the old behaviour), 0.5 = even split")'''))

for old, new in PATCHES:
    n = src.count(old)
    assert n == 1, "expected 1 occurrence, found %d:\\n%s" % (n, old[:90])
    src = src.replace(old, new)

bak = P.with_suffix(".py.bak_preaug")
if not bak.exists():
    bak.write_text(P.read_text())
P.write_text(src)
print("patched %s (%d edits); backup %s" % (P, len(PATCHES), bak))