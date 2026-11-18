"""Reproduce A item-grouped supervised and label-hidden zero-shot metrics."""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spendintel.clean import load_cfg, clean_two_system
from spendintel.entities import item_keys
from spendintel.taxonomy import load_taxonomy
from spendintel.classify import label_assisted, zero_shot, Embedder


def report(name, pred, truth, spend):
    print(name)
    for i in range(3):
        correct = np.asarray([p[i] == y[i] for p, y in zip(pred, truth)])
        weighted = (correct * spend).sum() / max(spend.sum(), 1)
        print(f"  L{i + 1}: accuracy={correct.mean():.3f}, spend_weighted={weighted:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--config", default="configs/company_A.yaml")
    ap.add_argument("--embedder", default="tfidf", choices=["auto", "e5", "tfidf"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    records, _ = clean_two_system(cfg, args.data_dir)
    labels = pd.read_csv(os.path.join(args.data_dir, cfg["labels"]["file"]))
    rows = records[records.record_id.str.startswith("ERP:")].merge(labels, on="record_id")
    rows["group_key"] = item_keys(rows.material_number, rows.text)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                               .split(rows, groups=rows.group_key))
    train, test = rows.iloc[train_idx], rows.iloc[test_idx]
    _, leaves = load_taxonomy(os.path.join(args.data_dir, cfg["taxonomy"]["file"]))
    truth = list(zip(test.category_l1, test.category_l2, test.category_l3))
    spend = test.amount_net.fillna(0).abs().to_numpy()

    train_paths = list(zip(train.category_l1, train.category_l2, train.category_l3))
    pred, _ = label_assisted(train.text.to_numpy(), train_paths, test.text.to_numpy(), leaves)
    report("Supervised grouped holdout", pred, truth, spend)

    # No labels are passed to the zero-shot model; labels are used only for scoring afterward.
    pred0, _ = zero_shot(test.text.to_numpy(), leaves, Embedder(args.embedder))
    report(f"Zero-shot {args.embedder} grouped holdout", pred0, truth, spend)


if __name__ == "__main__":
    main()
