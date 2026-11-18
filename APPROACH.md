# APPROACH

## Pipeline

YAML config -> source ingestion -> amount/date/currency normalisation -> count-once matching -> spend exclusions -> supplier/item keys -> taxonomy classification -> strict output validation. The pipeline is deterministic, local, and makes no API calls. A (65,549 combined records) ran in about 7 seconds; B (41,028 records) in about 4 seconds in this environment.

## Matching, spend and entities

A ERP is the primary record. Tool lines are excluded as `DUPLICATE` when an exact normalised PO plus net amount match is within 1%, when remaining tool lines sum to the ERP PO total within 1%, or when an invoice/text/gross-amount export is repeated. A tool statuses outside Approved/Paid are excluded with `CANCELLED` or `NOT_APPROVED`. Tool gross values are divided by (1 + VAT); missing VAT defaults to 15%. B uses line FX when valid, otherwise the nearest available monthly rate for the invoice month; USD is rate 1. Supplier names are normalised and conservatively fuzzy-merged; B vendor IDs are stable supplier keys. Item keys prefer material IDs, then normalised descriptions; generic descriptions are kept distinct.

## Classification

The taxonomy is read at runtime; no category names are hard-coded. With labels, word + character TF-IDF features feed fast hierarchical SGD classifiers: an L1 gate constrains the leaf prediction to a valid branch. With no labels, the tool compares the input text to full taxonomy paths using multilingual E5 when installed/cached, otherwise local character TF-IDF. The fallback is deterministic and requires no external service. Confidence is a score-derived ranking signal, not a calibrated probability.

## A grouped holdout (labels withheld from prediction)

A ERP rows were split 75/25 by item key (seed 42), so an item group does not cross train/test. The supervised model achieved:

| Level | Accuracy | Spend-weighted accuracy |
|---|---:|---:|
| L1 | 0.697 | 0.825 |
| L2 | 0.603 | 0.756 |
| L3 | 0.526 | 0.635 |

B non-procurement classification was also checked on a vendor-grouped 25% holdout: precision 1.000, recall 0.855, F1 0.922 (62 positive examples in the holdout). The zero-label TF-IDF baseline on the same A held-out rows achieved L1/L2/L3 accuracy 0.412/0.202/0.156 and spend-weighted accuracy 0.411/0.312/0.220. These are A-specific measurements, not a claim of C accuracy. An E5 evaluation was attempted but could not be run: `sentence-transformers` was absent and this execution environment had no package-index/network access. E5 remains optional and must not be assumed available during the blind test.

## Limitations

- A has no labelled include/exclude field; generic non-procurement rules were run over spend type, material type and text and flagged zero rows. This is an explicit, auditable rule check, not evidence that all subtle non-procurement cases are solved.
- Three B lines have a missing source `line_amount`. Their FX rates are available, but their monetary amounts cannot be recovered from the supplied fields. They remain included with spend 0 rather than inventing values; IDs are recorded in `DECISIONS.md`.
- A supplier gold key contains a large catch-all group; entity metrics are therefore reported with and without that bucket. Item matching is conservative and does not attempt risky semantic merges.
- A synthetic C-style smoke test with a four-level taxonomy, gross VAT, statuses and duplicate keys passed end-to-end; it validates plumbing only, not blind C classification quality. Missing FX for a non-USD amount is a hard validation failure rather than silently writing a fabricated spend value.
- Zero-shot TF-IDF performance is weak on A. C is unseen and unlabelled, so its category quality cannot be measured before the blind test. If E5 is unavailable, this limitation remains material.
