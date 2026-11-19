p = "sonar_slt.py"; s = open(p).read()
reps = [
('''    ap.add_argument("--split", default="test", help="split for --eval_only")''',
'''    ap.add_argument("--split", default="test", help="split for --eval_only")
    ap.add_argument("--domain", default="", choices=["", "batch1", "batch2"],
                    help="train/evaluate on one domain only (per-domain fine-tuning)")'''),
('''        ds = TSLDataset(args.split_csv, split, args.targets)
        if args.limit:''',
'''        ds = TSLDataset(args.split_csv, split, args.targets)
        if args.domain:
            ds.df = ds.df[ds.df["batch"] == args.domain].reset_index(drop=True)
        if args.limit:'''),
]
for old, new in reps:
    assert s.count(old) == 1, "pattern not found:\n" + old
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched: --domain batch1|batch2")