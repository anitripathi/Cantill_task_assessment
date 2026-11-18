"""Steps 4-5: config-driven cleaning + data engineering. No category names, no company-specific code paths."""
import re, numpy as np, pandas as pd, yaml, os

def load_cfg(path): return yaml.safe_load(open(path))

def parse_amount(s):
    s = s.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    # "1.234,56" european vs "1,234.56"
    eu = s.str.contains(r"\d\.\d{3},\d{1,2}$")
    s = s.where(~eu, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")

def parse_date(s):
    s = s.astype(str).str.strip().replace({"nan": None, "": None})
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y"):
        m = out.isna() & s.notna()
        if not m.any(): break
        out[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    return out

def norm_vat(s, default):
    v = pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
    v = v.where(~(v > 1), v / 100)
    return v.fillna(default)

def norm_po(s):
    return s.astype(str).str.extract(r"(\d{8,12})")[0]

def norm_po_item(s):
    x = s.astype(str).str.extract(r"[/\-](\d+)(?:\.0)?$")[0]
    return x

def _read(cfg, d, key): return pd.read_csv(os.path.join(d, cfg[key]["file"]), dtype=str)

def clean_two_system(cfg, d):
    p, t = cfg["primary"], cfg["secondary"]
    P, T = _read(cfg, d, "primary"), _read(cfg, d, "secondary")
    P["record_id"] = p["id_prefix"] + P[p["id_col"]]
    P["amount_net"] = parse_amount(P[p["amount_col"]])
    P["date"] = parse_date(P[p["date_col"]])
    P["supplier_raw"] = P[p["supplier_col"]]
    P["text"] = P[p["description_cols"]].fillna("").agg(" ".join, axis=1).str.strip()
    P["po"] = norm_po(P[p["po_col"]])
    P["include"], P["exclusion_reason"] = 1, ""
    T["record_id"] = t["id_prefix"] + T[t["id_col"]]
    T["gross"] = parse_amount(T[t["amount_col"]])
    T["vat"] = norm_vat(T[t["vat_col"]], t["default_vat"])
    T["amount_net"] = T["gross"] / (1 + T["vat"])
    T["date"] = parse_date(T[t["date_col"]])
    T["supplier_raw"] = T[t["supplier_col"]]
    T["text"] = T[t["description_cols"]].fillna("").agg(" ".join, axis=1).str.strip()
    T["po"] = norm_po(T[t["po_col"]])
    T["include"], T["exclusion_reason"] = 1, ""
    st = T[t["status_col"]].str.lower().str.strip()
    canc = st.isin(cfg["status_cancelled"]); bad = ~st.isin(cfg["status_ok"]) & ~canc
    T.loc[canc, ["include", "exclusion_reason"]] = [0, "CANCELLED"]
    T.loc[bad, ["include", "exclusion_reason"]] = [0, "NOT_APPROVED"]
    # tool exported twice: same invoice + text + amount -> keep first
    dup = T.duplicated([t["invoice_col"], "text", "gross"], keep="first") & (T.include == 1)
    T.loc[dup, ["include", "exclusion_reason"]] = [0, "DUPLICATE"]
    n_export_dup = int(dup.sum())
    # cross-system: same PO and net within tolerance -> already counted in primary
    live = T[(T.include == 1) & T.po.notna()]
    cand = live.reset_index().merge(P[["record_id", "po", "amount_net"]].rename(columns={"record_id": "erp_id", "amount_net": "erp_net"}), on="po")
    tol = t["match_tolerance"]
    cand["rel"] = (cand.amount_net - cand.erp_net).abs() / cand.erp_net.abs().clip(lower=1)
    hit = cand[cand.rel <= tol].sort_values("rel").drop_duplicates("index")
    T.loc[hit["index"].values, ["include", "exclusion_reason"]] = [0, "DUPLICATE"]
    T.loc[hit["index"].values, "matched_erp"] = hit.erp_id.values
    n_cross = len(hit)
    # PO-level fallback: tool lines on the same PO whose sum equals the ERP PO total (split receipts)
    live = T[(T.include == 1) & T.po.notna()]
    g = live.groupby("po").amount_net.sum().rename("t").to_frame().join(P.groupby("po").amount_net.sum().rename("e"), how="inner")
    g = g[((g.t - g.e).abs() / g.e.abs().clip(lower=1)) <= tol].index
    m = T.include.eq(1) & T.po.isin(g)
    T.loc[m, ["include", "exclusion_reason"]] = [0, "DUPLICATE"]; n_po = int(m.sum())
    for D in (P, T): D["spend"] = np.where(D.include == 1, D.amount_net.fillna(0.0), 0.0)
    # Preserve entity identifiers and useful exclusion features for downstream logic.
    for frame, source, prefix in ((P, p, "primary"), (T, t, "secondary")):
        material_col = source.get("material_col")
        frame["material_number"] = frame[material_col].fillna("").astype(str) if material_col and material_col in frame else ""
        for c in ("gl_account", "gl_account_description", "document_type", "spend_type", "material_type"):
            if c not in frame: frame[c] = ""
    keep = ["record_id", "include", "exclusion_reason", "supplier_raw", "text", "po", "date", "amount_net", "spend", "material_number", "gl_account", "gl_account_description", "document_type", "spend_type", "material_type"]
    out = pd.concat([P[keep], T[keep]], ignore_index=True)
    stats = dict(tool_export_duplicates=n_export_dup, cross_system_line_matches=n_cross, po_total_matches=n_po,
                 not_approved=int((T.exclusion_reason == "NOT_APPROVED").sum()), cancelled=int((T.exclusion_reason == "CANCELLED").sum()),
                 tool_dates_unparsed=int(T.date.isna().sum()), tool_vat_defaulted=int(T[t["vat_col"]].isna().sum()))
    return out, stats

def clean_multi_currency(cfg, d):
    p = cfg["primary"]; P = _read(cfg, d, "primary")
    f = cfg.get("fx"); fx = _read(cfg, d, "fx") if f else pd.DataFrame(columns=["c", "m", "r"])
    if f: fx[f["rate_col"]] = pd.to_numeric(fx[f["rate_col"]], errors="coerce")
    else: f = {"currency_col": "c", "month_col": "m", "rate_col": "r"}
    P["record_id"] = p["id_prefix"] + P[p["id_col"]]
    P["amount_local"] = parse_amount(P[p["amount_col"]])
    if p.get("amount_is_gross"):
        vat_col = p.get("vat_col")
        vat = norm_vat(P[vat_col], p.get("default_vat", 0.0)) if vat_col and vat_col in P else pd.Series(float(p.get("default_vat", 0.0)), index=P.index)
        P["amount_local"] = P["amount_local"] / (1 + vat)
    d1 = parse_date(P[p["date_col"]]); d2 = parse_date(P[p["fallback_date_col"]]) if p.get("fallback_date_col") else d1
    bad = d1.isna() | (d1.dt.year < d2.dt.year - 1) | (d1.dt.year > d2.dt.year)   # impossible vs posting date
    P["date"] = d1.where(~bad, d2); P["date_fixed"] = bad
    P["month"] = P["date"].dt.strftime("%Y-%m")
    rate = pd.to_numeric(P[p["rate_col"]], errors="coerce") if p.get("rate_col") else pd.Series(np.nan, index=P.index)
    rep = cfg.get("reporting_currency", "USD")
    cur = P[p["currency_col"]].str.upper().str.strip() if p.get("currency_col") else pd.Series(rep, index=P.index)
    # fallback rate: FX table for the month, else nearest month for that currency
    fxm = fx.set_index([f["currency_col"], f["month_col"]])[f["rate_col"]].to_dict()
    bycur = {c: g.dropna(subset=[f["rate_col"]]).assign(_period=pd.PeriodIndex(g[f["month_col"]], freq="M"))
             for c, g in fx.groupby(f["currency_col"])}
    def lookup_rate(c, m):
        exact = fxm.get((c, m))
        if pd.notna(exact): return exact
        table = bycur.get(c)
        if table is None or not len(table) or pd.isna(m): return np.nan
        try:
            target = pd.Period(m, freq="M")
            distance = (table["_period"] - target).map(lambda x: abs(x.n))
            return table.loc[distance.idxmin(), f["rate_col"]]
        except Exception:
            return float(table[f["rate_col"]].median())
    fb = [lookup_rate(c, m) for c, m in zip(cur, P.month)]
    P["rate_source"] = np.where(cur == rep, "usd", np.where(rate.notna() & (rate > 0), "line", "fx_table"))
    rate = rate.where(rate.notna() & (rate > 0), pd.Series(fb, index=P.index)).where(cur != "USD", 1.0)
    P["amount_net"] = P.amount_local / rate
    P["supplier_raw"] = P[p["supplier_col"]]
    P["text"] = P[p["description_cols"]].fillna("").agg(" ".join, axis=1).str.strip()
    P["include"], P["exclusion_reason"] = 1, ""
    # Optional status rules for a new single-file company (e.g. C).
    status_col = p.get("status_col")
    if status_col and status_col in P:
        status = P[status_col].fillna("").astype(str).str.lower().str.strip()
        ok_status = [str(x).lower() for x in cfg.get("status_ok", ["approved", "paid"])]
        cancelled_status = [str(x).lower() for x in cfg.get("status_cancelled", ["cancelled", "canceled"])]
        cancelled = status.isin(cancelled_status)
        not_ok = ~status.isin(ok_status) & ~cancelled
        P.loc[cancelled, ["include", "exclusion_reason"]] = [0, "CANCELLED"]
        P.loc[not_ok, ["include", "exclusion_reason"]] = [0, "NOT_APPROVED"]
    # Optional exact duplicate key for a single-file client; every source row remains in output.
    dup_cols = [c for c in p.get("duplicate_cols", []) if c in P.columns]
    if dup_cols:
        dup = P.duplicated(dup_cols, keep="first") & P.include.eq(1)
        P.loc[dup, ["include", "exclusion_reason"]] = [0, "DUPLICATE"]
    P["spend"] = np.where(P.include == 1, P.amount_net.fillna(0.0), 0.0)
    stats = dict(rows=len(P), rates_from_line=int((P.rate_source == "line").sum()), rates_from_fx_table=int((P.rate_source == "fx_table").sum()),
                 usd_lines=int((P.rate_source == "usd").sum()), missing_fx_rate=int((rate.isna() & P.amount_local.notna()).sum()), missing_source_amount=int(P.amount_local.isna().sum()),
                 zero_spend_due_to_missing_amount=int((P.amount_local.isna() & P.include.eq(1)).sum()),
                 invoice_dates_replaced_by_posting=int(P.date_fixed.sum()), blank_descriptions=int((P[p["description_cols"][0]].fillna("").str.strip() == "").sum()),
                 cancelled=int((P.exclusion_reason == "CANCELLED").sum()), not_approved=int((P.exclusion_reason == "NOT_APPROVED").sum()),
                 duplicates=int((P.exclusion_reason == "DUPLICATE").sum()))
    keep = ["record_id", "include", "exclusion_reason", "supplier_raw", "text", "date", "amount_net", "spend", "gl_account", "gl_account_description", "document_type", "currency", "source_system"]
    for c in [c for c in keep if c not in P.columns]: P[c] = ""
    extra = [c for c in p.get("extra_cols", []) if c in P.columns]
    return P[keep + extra], stats
