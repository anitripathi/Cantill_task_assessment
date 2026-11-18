"""Supplier and item de-duplication (conservative: precision first)."""
import re, pandas as pd
from rapidfuzz import fuzz, process
_SUF = r"\b(PTY|LTD|LIMITED|CC|THE|CO|INC|LLC|GMBH|BV|SA)\b"

def norm_name(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper()); s = re.sub(_SUF, " ", s)
    return re.sub(r"\s+", " ", s).strip()

def supplier_keys(raw: pd.Series, threshold=92):
    """Normalise, then merge names whose token-sorted similarity >= threshold (fixes typos such as 'YARF IELD')."""
    n = raw.fillna("").map(norm_name); names = sorted(set(n)); par = {x: x for x in names}
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for x in names:
        if len(x) < 5: continue
        for o, sc, _ in process.extract(x, names, scorer=fuzz.ratio, score_cutoff=threshold, limit=5):
            if o != x and abs(len(o) - len(x)) <= 3: par[f(o)] = f(x)
    return n.map({x: "SUP:" + f(x) for x in names}).where(n != "", "SUP:UNKNOWN-" + pd.Series(raw.index, index=raw.index).astype(str))

GENERIC = {"PRODUCT", "PARTS", "PART", "MATERIAL", "SERVICE", "SERVICES", "ITEM", "GOODS", ""}
def item_keys(material_number: pd.Series, text: pd.Series):
    """Exact material id when present; else normalised description. Generic words never merge (each stays its own item)."""
    t = text.fillna("").str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    key = "ITM:T:" + t
    key = key.where(~t.isin(GENERIC), "ITM:G:" + pd.Series(text.index, index=text.index).astype(str))
    m = material_number.fillna("").astype(str).str.strip()
    return key.where(m == "", "ITM:M:" + m)
