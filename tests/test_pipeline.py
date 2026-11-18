import pandas as pd
import numpy as np
from spendintel.clean import parse_amount, parse_date, norm_vat, norm_po
from spendintel.entities import supplier_keys, item_keys
from spendintel.taxonomy import load_taxonomy

def test_amount_formats():
    assert parse_amount(pd.Series(["5,266.63", "1.234,56", "12"])).round(2).tolist() == [5266.63, 1234.56, 12.0]

def test_vat_default_and_formats():
    assert norm_vat(pd.Series(["15", "0.15", "15%", None, "0"]), 0.15).tolist() == [0.15, 0.15, 0.15, 0.15, 0.0]

def test_dates():
    d = parse_date(pd.Series(["2024-01-05", "19/01/2024", "27-Mar-24", None]))
    assert d.notna().tolist() == [True, True, True, False]

def test_po_normalisation():
    assert norm_po(pd.Series(["PO#4510055507", "4510085877/10.0"])).tolist() == ["4510055507", "4510085877"]

def test_supplier_typo_merge_and_no_false_merge():
    k = supplier_keys(pd.Series(["YARFIELD", "YARF IELD", "KELMONT (PTY) LTD", "KELDEN (PTY) LTD"]))
    assert k[0] == k[1] and k[2] != k[3]

def test_generic_items_not_merged():
    k = item_keys(pd.Series(["", ""]), pd.Series(["Product", "Product"]))
    assert k[0] != k[1]

def test_taxonomy_any_depth(tmp_path):
    p = tmp_path / "t.csv"; p.write_text("category_l1,category_l2,category_l3,category_l4\nA,B,C,D\nA,B,E,\n")
    levels, leaves = load_taxonomy(p)
    assert len(levels) == 4 and ("A", "B", "E") in leaves and ("A", "B", "C", "D") in leaves

def test_single_file_status_gross_and_duplicate_rules(tmp_path):
    from spendintel.clean import clean_multi_currency
    p = tmp_path / "lines.csv"
    pd.DataFrame([
        {"id": "C1", "invoice": "I1", "line": "1", "date": "2026-10-01", "amount": "115", "currency": "USD", "rate": "", "supplier": "S1", "description": "steel coil", "vat": "15", "status": "Approved"},
        {"id": "C2", "invoice": "I2", "line": "1", "date": "2026-10-02", "amount": "100", "currency": "USD", "rate": "", "supplier": "S2", "description": "motor", "vat": "0", "status": "Cancelled"},
        {"id": "C3", "invoice": "I3", "line": "1", "date": "2026-10-03", "amount": "100", "currency": "USD", "rate": "", "supplier": "S3", "description": "parts", "vat": "0", "status": "Rejected"},
        {"id": "C4", "invoice": "I1", "line": "1", "date": "2026-10-01", "amount": "115", "currency": "USD", "rate": "", "supplier": "S1", "description": "steel coil", "vat": "15", "status": "Approved"},
    ]).to_csv(p, index=False)
    cfg = {
        "reporting_currency": "USD",
        "primary": {"file": "lines.csv", "id_col": "id", "id_prefix": "", "amount_col": "amount",
                    "amount_is_gross": True, "vat_col": "vat", "currency_col": "currency", "rate_col": "rate",
                    "date_col": "date", "supplier_col": "supplier", "description_cols": ["description"],
                    "status_col": "status", "duplicate_cols": ["invoice", "line"]},
        "status_ok": ["approved", "paid"], "status_cancelled": ["cancelled"]
    }
    out, _ = clean_multi_currency(cfg, str(tmp_path))
    assert out.record_id.tolist() == ["C1", "C2", "C3", "C4"]
    assert np.round(out.spend.to_numpy(), 2).tolist() == [100.0, 0.0, 0.0, 0.0]
    assert out.exclusion_reason.fillna("").tolist() == ["", "CANCELLED", "NOT_APPROVED", "DUPLICATE"]


def test_validate_rejects_missing_spend():
    from spendintel.validate import validate, COLS
    row = {c: "" for c in COLS}
    row.update(record_id="R1", include=1, exclusion_reason="", supplier_key="S1", item_key="I1",
               spend=np.nan, l1="A", l2="B", l3="C", l4="", confidence=0.5)
    errors = validate(pd.DataFrame([row]), ["R1"], [("A", "B", "C")])
    assert "missing or non-finite spend" in errors
