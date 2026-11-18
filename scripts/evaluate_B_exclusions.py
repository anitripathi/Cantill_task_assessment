"""Vendor-grouped holdout evaluation of B's NON_PROCUREMENT seed-label model."""
import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spendintel.clean import load_cfg, clean_multi_currency
from spendintel.spend_rules import _feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--config", default="configs/company_B.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    records, _ = clean_multi_currency(cfg, args.data_dir)
    labels = pd.read_csv(os.path.join(args.data_dir, cfg["labels"]["file"]))
    rows = records.merge(labels[["line_id", "include"]].rename(columns={"include": "label_include"}),
                         left_on="record_id", right_on="line_id")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                               .split(rows, groups=rows.supplier_raw.fillna("")))
    cols = cfg.get("exclusion_features", ["gl_account_description", "document_type", "text"])
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(C=10, class_weight="balanced", max_iter=500))
    train, test = rows.iloc[train_idx], rows.iloc[test_idx]
    model.fit(_feat(train, cols), (train.label_include == 0).astype(int))
    pred = model.predict(_feat(test, cols))
    truth = (test.label_include == 0).astype(int).to_numpy()
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(f"vendor-grouped holdout: n={len(test)}, precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")


if __name__ == "__main__":
    main()
