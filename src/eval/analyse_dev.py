import re, pandas as pd, sacrebleu
p = pd.read_csv("runs/B_v1/pred_dev_best.csv")
p["batch"] = p["clip_id"].str[:2].map({"b1": "batch1", "b2": "batch2"})
p["rep"] = p["hyp"].astype(str).str.contains(r"(.{4,15}?)\1{2,}", regex=True)   # a phrase repeated 3+ times
p["len_ratio"] = p["hyp"].str.len() / p["ref"].str.len()
for b, d in p.groupby("batch"):
    print("%s n=%d | chrF %.1f | char-BLEU %.1f | repetition %.0f%% | mean cos %.3f | len ratio %.2f" % (
        b, len(d), sacrebleu.corpus_chrf(d.hyp.tolist(), [d.ref.tolist()]).score,
        sacrebleu.corpus_bleu(d.hyp.tolist(), [d.ref.tolist()], tokenize="char").score,
        100 * d["rep"].mean(), d["cos"].mean(), d["len_ratio"].median()))
ok = p[~p["rep"]]
print("\nwithout the repeating outputs: n=%d | chrF %.1f" % (
    len(ok), sacrebleu.corpus_chrf(ok.hyp.tolist(), [ok.ref.tolist()]).score))
print("\nbest 5 outputs:")
for _, r in p.assign(c=[sacrebleu.sentence_chrf(h, [x]).score for h, x in zip(p.hyp, p.ref)]).nlargest(5, "c").iterrows():
    print(" %s (chrF %.0f)\n   ref: %s\n   hyp: %s" % (r.clip_id, r.c, r.ref, r.hyp))