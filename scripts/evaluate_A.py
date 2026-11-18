"""Evaluate pipeline output on company A against the golden labels (labels used ONLY here, for scoring)."""
import sys, pandas as pd, numpy as np
out = pd.read_csv(sys.argv[1], dtype={"exclusion_reason": str}).fillna({"exclusion_reason": ""}); lab = pd.read_csv(sys.argv[2])
m = out.merge(lab, on="record_id"); inc = m[m.include == 1]
def pw(pred, gold):
    ct = pd.crosstab(pd.Series(pred).values, pd.Series(gold).values).values.astype(float); c2 = lambda x: (x * (x - 1) / 2).sum()
    tp, pp, gp = c2(ct), c2(ct.sum(1)), c2(ct.sum(0)); p, r = tp / pp, tp / gp; return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3)
big = m.supplier_key_y.value_counts().index[0]
for i in (1, 2, 3):
    ok = inc[f"l{i}"] == inc[f"category_l{i}"]; w = inc.spend.abs()
    print(f"L{i} accuracy {ok.mean():.3f}  spend-weighted {(ok*w).sum()/w.sum():.3f}")
print("supplier pairwise P/R/F1 (all ERP rows):", pw(m.supplier_key_x, m.supplier_key_y))
mm = m[m.supplier_key_y != big]; print(f"supplier pairwise excluding catch-all gold key {big} ({(m.supplier_key_y==big).mean():.0%} of rows):", pw(mm.supplier_key_x, mm.supplier_key_y))
print("item pairwise P/R/F1:", pw(m.item_key_x, m.item_key_y))
print("mean confidence", round(inc.confidence.mean(), 3), "| confidence of correct vs wrong L3:", round(inc[inc.l3 == inc.category_l3].confidence.mean(), 3), round(inc[inc.l3 != inc.category_l3].confidence.mean(), 3))
