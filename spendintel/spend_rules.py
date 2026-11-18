"""NON_PROCUREMENT detection: learned from labels when the company has them, otherwise generic keyword rules from config."""
import re, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def _feat(df, cols): return df[[c for c in cols if c in df.columns]].fillna("").astype(str).agg(" | ".join, axis=1)

def apply_non_procurement(df, cfg, labels=None, train_df=None):
    cols = cfg.get("exclusion_features", ["gl_account_description", "document_type", "text"])
    rules = cfg.get("non_procurement_keywords", [
        r"\bVAT\b", r"\bGST\b", r"sales tax", r"withholding tax", r"royalt(?:y|ies)",
        r"inter[ -]?company", r"intercompany", r"partner billing", r"financing cost",
        r"loan repayment", r"interest expense", r"dividend", r"revenue share", r"bank charge"
    ])
    live = df.include == 1; flag = pd.Series(False, index=df.index); how = "none"
    if rules:
        rx = re.compile("|".join(rules), re.I); flag |= _feat(df, cols).str.contains(rx)
        how = "keywords"
    if labels is not None and "include" in labels.columns and (labels.include == 0).sum() >= 20:
        m = (train_df if train_df is not None else df).merge(labels[["record_id", "include"]].rename(columns={"include": "lab"}), on="record_id", how="left")
        tr = m.dropna(subset=["lab"]); X = _feat(tr, cols)
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(C=10, class_weight="balanced", max_iter=500))
        clf.fit(X, (tr.lab == 0).astype(int))
        pr = clf.predict_proba(_feat(df, cols))[:, 1]; flag |= pd.Series(pr > 0.5, index=df.index); how += "+learned"
    m = flag & live
    df.loc[m, ["include", "exclusion_reason"]] = [0, "NON_PROCUREMENT"]; df.loc[m, "spend"] = 0.0
    return df, dict(method=how, non_procurement=int(m.sum()))
