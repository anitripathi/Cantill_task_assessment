# ML pipeline notes

This folder retains the exploratory figures created during data inspection. The runnable implementation is the config-driven pipeline in `spendintel/`; the current classifier, evaluation numbers and known limitations are documented in `APPROACH.md` and `DECISIONS.md`.

## Current state

| Stage | Implementation |
|---|---|
| Ingestion | YAML-driven A two-source and B multi-currency readers |
| Cleaning | Amount/VAT parsing, date repair, monthly FX fallback |
| Count once | A tool duplicate, PO/net and PO-total matching |
| Exclusions | A status + keyword rules; B seed-label model + keywords |
| Entities | Conservative supplier normalisation and material/description item keys |
| Classification | Hierarchical supervised TF-IDF/SGD; zero-label taxonomy path matcher |
| Validation | IDs, row counts, output schema, numeric spend, entity keys, confidence, exclusion codes and taxonomy paths |

## Evaluation protocol

A supervised evaluation groups rows by item key and holds out 25% of groups (seed 42). Zero-shot evaluation hides labels from the classifier and uses only the category tree. Results and caveats are in `APPROACH.md`. E5 could not be installed/downloaded in the offline environment; the tested no-dependency fallback is TF-IDF.

The blind C dataset is not available before the assessment window, so C accuracy is unknown. No performance claim is made for it.
