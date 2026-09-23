import pandas as pd, numpy as np, os, sys
from sklearn.metrics import cohen_kappa_score
sp = sys.argv[1]
s = pd.read_csv(sp, dtype=str, keep_default_na=False)
k = pd.read_csv(os.path.expanduser("~/Desktop/release/overnight/rater_key.csv"), dtype=str, keep_default_na=False)
d = s.merge(k[["id","automatic_decision"]], on="id")
m = {"Yes":"conflict","No":"agreement"}
for ds in ["E-DAIC","Pitt"]:
    g = d[d.dataset==ds]
    a1 = g.rater1.map(m); a2 = g.rater2.map(m); auto = g.automatic_decision
    print(f"{ds}: n={len(g)} r1_hits={(a1==auto).sum()}/{len(g)}={(a1==auto).mean()*100:.1f}%  "
          f"r2_hits={(a2==auto).sum()}/{len(g)}={(a2==auto).mean()*100:.1f}%  "
          f"kappa={cohen_kappa_score(a1,a2):.4f}  raw_ag={(a1==a2).mean()*100:.1f}%")
