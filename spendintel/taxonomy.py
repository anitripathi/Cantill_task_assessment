"""Dynamic taxonomy parsing: any number of category_l* columns, no names in code."""
import re, pandas as pd

def load_taxonomy(path):
    t = pd.read_csv(path, dtype=str).fillna("")
    levels = sorted([c for c in t.columns if re.fullmatch(r"category_l\d+", c)], key=lambda c: int(c.split("l")[-1]))
    if not levels:   # fallback: any leading columns
        levels = list(t.columns)
    t = t[levels].drop_duplicates().reset_index(drop=True)
    paths = []
    for _, r in t.iterrows():
        p = [v for v in r.values if v != ""]
        paths.append(tuple(p))
    leaves = sorted(set(paths), key=lambda p: (len(p), p))
    return levels, leaves
