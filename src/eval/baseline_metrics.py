import numpy as np, pandas as pd
S = np.load("targets/sonar_th.npy"); sp = pd.read_csv("splits/split_v1.csv")
u = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
tr, dv = sp[sp.split == "train"], sp[sp.split == "dev"]
mu = S[tr.row.values].mean(0); s = S[dv.row.values]
m = {k: S[g.row.values].mean(0) for k, g in tr.groupby(["batch", "cat"])}
z = np.stack([m[(r.batch, r.cat)] for r in dv.itertuples()])
ccos = (u(z - mu) * u(s - mu)).sum(1).mean()
sim = u(z) @ u(s).T
rank = (sim > np.diag(sim)[:, None]).sum(1) + 1
print("category-average guess: ccos %.3f | R@1 %.3f | R@5 %.3f | median rank %d"
      % (ccos, (rank <= 1).mean(), (rank <= 5).mean(), np.median(rank)))
print("random guess:           ccos 0.000 | R@1 %.3f | R@5 %.3f | median rank %d"
      % (1 / len(dv), 5 / len(dv), len(dv) // 2))