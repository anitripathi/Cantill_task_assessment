"""End-to-end run: config -> clean -> spend rules -> entities -> classify -> validate -> export."""
import os, time, pandas as pd, numpy as np
from .clean import load_cfg, clean_two_system, clean_multi_currency
from .spend_rules import apply_non_procurement
from .entities import supplier_keys, item_keys
from .taxonomy import load_taxonomy
from .classify import Embedder, zero_shot, label_assisted
from .validate import validate, COLS

def _clean(cfg, d): return clean_two_system(cfg, d) if "secondary" in cfg else clean_multi_currency(cfg, d)

def run(config_path, data_dir, output, mode="auto", backend="auto", log=print, train_dir=None):
    t0 = time.time(); cfg = load_cfg(config_path)
    df, st = _clean(cfg, data_dir)
    log(f"cleaned {len(df)} rows: {st}")
    if st.get("missing_fx_rate", 0):
        raise SystemExit(f"VALIDATION FAILED: {st['missing_fx_rate']} rows have amounts but no usable FX rate")
    labels = None; ldir = train_dir or data_dir   # labels + their records come from the original kit when scoring new data
    tdf = _clean(cfg, ldir)[0] if train_dir else df.copy()
    if cfg.get("labels") and mode != "zero_shot":
        lp = os.path.join(ldir, cfg["labels"]["file"])
        if os.path.exists(lp):
            labels = pd.read_csv(lp, dtype=str); labels = labels.rename(columns={cfg["labels"].get("id_col", "record_id"): "record_id"})
            if "include" in labels.columns: labels["include"] = pd.to_numeric(labels["include"], errors="coerce")
            if cfg.get("labels", {}).get("id_prefix"): labels["record_id"] = cfg["labels"]["id_prefix"] + labels["record_id"]
    df, sp = apply_non_procurement(df, cfg, labels if mode != "zero_shot" else None, tdf); log(f"spend rules: {sp}")
    mat = df["material_number"] if "material_number" in df.columns else pd.Series("", index=df.index)
    df["supplier_key"] = supplier_keys(df.supplier_raw); df["item_key"] = item_keys(mat, df.text)
    levels, leaves = load_taxonomy(os.path.join(data_dir, cfg["taxonomy"]["file"]))
    inc = df.include == 1
    cat_ok = labels is not None and any(c in labels.columns for c in ("category_l1",))
    paths = [None] * len(df); conf = np.zeros(len(df))
    if cat_ok:
        lab = labels.dropna(subset=["category_l1"]); lab = lab[lab[levels[0]].notna()] if levels[0] in lab.columns else lab
        m = tdf.merge(lab, on="record_id"); lcols = [c for c in levels if c in m.columns]
        tp = [tuple(v for v in r if isinstance(v, str) and v) for r in m[lcols].values]
        ok = [p in set(leaves) for p in tp]; log(f"label-assisted mode, {sum(ok)} labelled training rows")
        mm = m[ok]; tpp = [p for p, k in zip(tp, ok) if k]
        pp, cc = label_assisted(mm.text.values, tpp, df.loc[inc, "text"].values, leaves)
    else:
        emb = Embedder(backend); log(f"zero-shot mode, embedder={emb.backend}")
        pp, cc = zero_shot(df.loc[inc, "text"].values, leaves, emb)
    idx = np.where(inc.values)[0]
    for i, p, c in zip(idx, pp, cc): paths[i] = p; conf[i] = c
    for i in range(1, 5): df[f"l{i}"] = [p[i - 1] if p is not None and len(p) >= i else "" for p in paths]
    df["confidence"] = np.clip(conf, 0, 1).round(4)
    out = df[COLS].copy(); out["spend"] = out.spend.round(2)
    errs = validate(out, df.record_id, leaves)
    if errs: raise SystemExit("VALIDATION FAILED: " + "; ".join(errs))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True); out.to_csv(output, index=False)
    log(f"wrote {output}: {len(out)} rows, included {int(out.include.sum())}, spend {out.spend.sum():,.0f}, {time.time()-t0:.0f}s")
    return out
