import pandas as pd
import numpy as np
COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"]
CODES = {"", "CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}

def validate(out, expected_ids, leaves):
    errs = []
    if list(out.columns) != COLS: errs.append("columns differ from template")
    if len(out) != len(expected_ids) or set(out.record_id) != set(expected_ids): errs.append("record ids differ from input")
    if out.record_id.duplicated().any(): errs.append("duplicate record ids")
    if not set(out.exclusion_reason.fillna("")).issubset(CODES): errs.append("invalid exclusion code")
    if ((out.include == 0) & (out.spend != 0)).any(): errs.append("excluded row with spend")
    if ((out.include == 1) & (out.exclusion_reason.fillna("") != "")).any(): errs.append("included row with reason")
    if ((out.include == 0) & (out.exclusion_reason.fillna("") == "")).any(): errs.append("excluded row without reason")
    if not pd.to_numeric(out.confidence, errors="coerce").between(0, 1).all(): errs.append("confidence out of [0,1]")
    spend = pd.to_numeric(out.spend, errors="coerce")
    if spend.isna().any() or not np.isfinite(spend).all(): errs.append("missing or non-finite spend")
    if pd.to_numeric(out.include, errors="coerce").isna().any() or not out.include.isin([0, 1]).all(): errs.append("include must be 0/1")
    for c in ("supplier_key", "item_key"):
        if out[c].isna().any() or out[c].astype(str).str.strip().eq("").any(): errs.append(f"missing {c}")
    ok = set(leaves)
    inc = out[out.include == 1]
    got = list(zip(*[inc[f"l{i}"].fillna("").astype(str) for i in range(1, 5)]))
    bad = 0
    for g in got:
        path = tuple(x for x in g if x)
        padded = path + ("",) * (4 - len(path))
        if tuple(g) != padded or path not in ok:
            bad += 1
    if bad: errs.append(f"{bad} rows with category path not in taxonomy")
    return errs
