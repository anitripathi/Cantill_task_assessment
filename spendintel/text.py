"""Steps 6-8: grouped split, feature text building, preprocessing pipeline (fit on train only)."""
import re, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

ABBR = {"vlv": "valve", "elec": "electric", "hyd": "hydraulic", "brg": "bearing", "ancr": "anchor", "rofblt": "roof bolt", "dsl": "diesel"}

def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"p/n:?\s*[\w\-/]+", " ", regex=True).str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    s = s.str.replace(r"\b\d{3,}\b", " ", regex=True)   # long numbers / part codes carry no meaning across clients
    s = s.apply(lambda t: " ".join(ABBR.get(w, w) for w in t.split()))
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

def group_split(df, group_col, seed=42, fr=(0.70, 0.15, 0.15)):
    """Whole groups go to one split only; asserts no group is shared."""
    g = df[group_col].fillna("NA").values
    gss = GroupShuffleSplit(1, test_size=fr[1] + fr[2], random_state=seed)
    tr, rest = next(gss.split(df, groups=g))
    gss2 = GroupShuffleSplit(1, test_size=fr[2] / (fr[1] + fr[2]), random_state=seed)
    va, te = next(gss2.split(df.iloc[rest], groups=g[rest]))
    parts = [df.iloc[tr], df.iloc[rest[va]], df.iloc[rest[te]]]
    sets = [set(p[group_col].fillna("NA")) for p in parts]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "group leakage"
    return parts

def _clean_array(x):
    return clean_text(pd.Series(x)).values

def build_preprocessor():
    """Word + character TF-IDF on cleaned text; character n-grams survive typos, abbreviations and other languages."""
    txt = FunctionTransformer(_clean_array, validate=False)
    return Pipeline([("clean", txt), ("tfidf", FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60000))]))])
