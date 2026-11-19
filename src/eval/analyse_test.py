import pandas as pd, sacrebleu
p = pd.read_csv("runs/TEST_B_v2/pred_test.csv")
p["batch"] = p["clip_id"].str[:2].map({"b1": "batch1 (cooking)", "b2": "batch2 (medical)"})
for b, d in list(p.groupby("batch")) + [("ALL", p)]:
    print("%-18s n=%3d | chrF %5.1f | char-BLEU %5.1f | mean cos %.3f | len ratio %.2f" % (
        b, len(d), sacrebleu.corpus_chrf(d.hyp.tolist(), [d.ref.tolist()]).score,
        sacrebleu.corpus_bleu(d.hyp.tolist(), [d.ref.tolist()], tokenize="char").score,
        d["cos"].mean(), (d.hyp.str.len() / d.ref.str.len()).median()))
p["c"] = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(p.hyp, p.ref)]
print("\nbest 5:")
for _, r in p.nlargest(5, "c").iterrows():
    print(" %s (chrF %.0f)\n   ref: %s\n   hyp: %s" % (r.clip_id, r.c, r.ref, r.hyp))
print("\nworst 3:")
for _, r in p.nsmallest(3, "c").iterrows():
    print(" %s (chrF %.0f)\n   ref: %s\n   hyp: %s" % (r.clip_id, r.c, r.ref, r.hyp))
p.to_csv("runs/TEST_B_v2/pred_test_scored.csv", index=False)