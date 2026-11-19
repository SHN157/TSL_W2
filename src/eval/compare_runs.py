import pandas as pd, sacrebleu
for run in ["B_v1", "B_v2"]:
    lg = pd.read_csv("runs/%s/log.csv" % run).dropna(subset=["dev_chrf"])
    b = lg.loc[lg["dev_chrf"].idxmax()]
    print("%s: best epoch %d/%d | dev chrF %.1f | R@5 %.3f | medR %.0f | train_ce %.2f dev_ce %.2f"
          % (run, b["epoch"], lg["epoch"].max(), b["dev_chrf"], b["dev_r5"], b["dev_medr"],
             b["train_ce"], b["dev_ce"]))
    p = pd.read_csv("runs/%s/pred_dev_best.csv" % run)
    p["batch"] = p["clip_id"].str[:2]
    for bt, d in p.groupby("batch"):
        print("   %s n=%d chrF %.1f | char-BLEU %.1f | len ratio %.2f" % (
            bt, len(d), sacrebleu.corpus_chrf(d.hyp.tolist(), [d.ref.tolist()]).score,
            sacrebleu.corpus_bleu(d.hyp.tolist(), [d.ref.tolist()], tokenize="char").score,
            (d.hyp.str.len() / d.ref.str.len()).median()))