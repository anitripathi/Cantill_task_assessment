# DECISIONS

## Company A

- ERP ZAR values are already net. Tool gross values are converted to net as `gross / (1 + VAT)`; blank VAT uses 15%. Values such as `15`, `0.15`, and `15%` are accepted. The current run defaulted 6,542 missing VAT rates.
- ERP is the count-once anchor. Tool records are excluded as `DUPLICATE` for PO + net amount matches within 1% (16,423), PO-total matches within 1% (14), and repeated invoice + text + gross exports (807). The current A output therefore has 17,244 duplicate tool rows.
- Only Approved/Paid tool invoices count. Cancelled (188) and other statuses (315) are excluded. The tool date is missing/unparseable for 18,660 rows; dates are retained as audit fields and are not used for the PO/amount matching rule.
- The A config explicitly checks `spend_type`, `material_type`, and description text for tax, royalty, intercompany, financing and partner-billing terms. No A rows matched these generic non-procurement rules in this run. ERP records have no status field in the supplied data.
- Supplier keys use normalised supplier names and conservative fuzzy matching. The golden key `SG-f85711` covers approximately 56% of ERP rows and appears to be a catch-all. Pairwise supplier precision/recall/F1 are 1.00/0.10/0.18 overall; excluding that catch-all bucket, they are 1.00/1.00/1.00. This limitation is retained rather than forcing unsafe merges.
- Item keys use material number when available, otherwise normalised description. Generic descriptions remain unique. Pairwise item precision/recall/F1 on ERP rows are 1.00/0.47/0.64.

## Company B

- Reporting currency is USD. USD lines use rate 1; valid line rates take priority; otherwise the nearest available monthly rate for the invoice month is used. The latest run used 19,726 line rates, 3,343 monthly-table fallbacks and 17,959 USD-as-is rows. No FX rate is missing after fallback.
- Three records have missing source amounts, not missing FX rates: `B025101`, `B025867`, and `B026435`. Their currency is RCN and monthly FX is available, but `line_amount` is blank. There is no reliable amount to reconstruct from the supplied fields. They remain included, with spend set to 0 to ensure the output has no missing/non-finite spend; no fabricated estimate is introduced.
- 78 impossible invoice dates were replaced by posting dates. Negative amounts are retained as reversals.
- The 3,000 seed labels include 450 `NON_PROCUREMENT` examples. A logistic-regression text model plus generic keywords predicts non-procurement; current run excluded 4,692 rows. A vendor-grouped 25% holdout gave precision 1.00, recall 0.855 and F1 0.922 (62 excluded examples in the holdout). This learned rule is company-specific and is not used for C without labels.
- Supplier keys use vendor IDs; item keys prefer material IDs.

## General / unseen company

- Exclusion codes are limited to `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, and `DUPLICATE`. Generic keyword rules are a conservative fallback when labels are absent.
- Category paths are loaded from the supplied taxonomy at runtime, with up to four levels. The single-file reader also supports optional status exclusions, gross/VAT conversion and configured duplicate keys. A non-USD amount without a usable line or table FX rate stops the run instead of silently producing a false USD value. A zero-label run does not read or train on the label file.
- E5 is an optional embedding backend. It could not be installed or evaluated in this offline environment; `auto` therefore uses TF-IDF here. Do not describe E5 as tested or depend on it being present.
- Confidence is score-derived and uncalibrated. No API calls or paid services are used.
