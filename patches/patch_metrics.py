p = "sonar_slt.py"; s = open(p).read()
a = s.index("@torch.no_grad()\ndef evaluate(")
b = s.index("# ----------------------------------------------------------------------------- main")
new_eval = '''@torch.no_grad()
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
        ce = model.ce_from_vector(z, model.labels(b["text"]))
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


'''
s = s[:a] + new_eval + s[b:]
reps = [
('''    if args.eval_only:
        res, rows = evaluate(model, loader(args.split, False), args, decode=True)''',
'''    _sp = pd.read_csv(args.split_csv)
    mu = torch.from_numpy(np.load(args.targets)[_sp.loc[_sp["split"] == "train", "row"].values].mean(0)).float()

    if args.eval_only:
        res, rows = evaluate(model, loader(args.split, False), args, decode=True, mu=mu)'''),
('''            res, rows = evaluate(model, dv, args, decode)''',
'''            res, rows = evaluate(model, dv, args, decode, mu)'''),
('''    best_key = "chrf" if decode else "cos"''',
'''    best_key = "chrf" if decode else "ccos"      # phase A: sentence-specific (centred) cosine'''),
('''"dev_mse", "dev_cos", "dev_ce", "dev_chrf", "dev_bleu_char", "best"]''',
'''"dev_mse", "dev_cos", "dev_ccos", "dev_r1", "dev_r5", "dev_medr", "dev_ce",
                    "dev_chrf", "dev_bleu_char", "best"]'''),
]
for old, new in reps:
    assert s.count(old) == 1, "pattern not found:\n" + old
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched: centred cosine + retrieval metrics, phase A selects by dev ccos")