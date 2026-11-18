# Cantill.ai spend intelligence pipeline

A config-driven spend pipeline for the Cantill.ai assessment. It ingests one or more source files, normalises spend and dates, applies exclusions, deduplicates suppliers/items, classifies included lines into a run-time taxonomy, validates the result, and writes the required 11-column CSV.

## 1. Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10+ is supported. On macOS/Linux, create and activate the virtual environment with `python -m venv .venv` and `source .venv/bin/activate`.

## 2. Keep assessment data out of Git

Extract the supplied `Cantill_DataML_Task_Kit` folder into the repository as `data/`, so the local layout is:

```text
data/
  company_A/  (A source files, labels_A.csv, taxonomy_A.csv)
  company_B/  (B source files, seed_labels.csv, taxonomy_B.csv, FX rates)
```

The `data/` directory is ignored by `.gitignore`. Do not force-add it or commit any assessment data.

## 3. Run A and B (one command per company)

From the repository root:

```powershell
python main.py --config configs/company_A.yaml --data-dir data --output outputs/anivesh_tripathi_A.csv
python main.py --config configs/company_B.yaml --data-dir data --output outputs/anivesh_tripathi_B.csv
```

The default `--embedder auto` uses multilingual E5 if `sentence-transformers` and its model are already available; otherwise it falls back to the local TF-IDF taxonomy matcher. The supervised A/B path uses the labels configured for the company. To force the local zero-label path:

```powershell
python main.py --config configs/company_A.yaml --data-dir data --output outputs/A_zero_shot.csv --mode zero_shot --embedder tfidf
```

Optional E5 setup (recommended before the blind test if internet is available; downloads and caches the multilingual model):

```powershell
pip install sentence-transformers torch
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
```

Then use `--embedder e5` for the C run to fail clearly if E5 is not available, or `--embedder auto` to keep the TF-IDF fallback.

## 4. Blind-test execution

- **A2/B2:** reuse the A/B configs. Point `--data-dir` at the blind-test folder containing the same `company_A/` or `company_B/` file layout, and use `--train-dir data` to train from the original labelled kit. The taxonomy must be present in the blind-test folder or the corresponding company folder.
- **C:** create `configs/company_C.yaml` describing its single invoice file, currency/date/amount/supplier/description columns, optional FX file, and taxonomy file. Run with `--mode zero_shot`; no label file is required. The taxonomy reader discovers `category_l1` through `category_l4` at runtime.

Example C command:

```powershell
python main.py --config configs/company_C.yaml --data-dir blind_test --output submission/anivesh_tripathi_C.csv --mode zero_shot --embedder auto
```

The code does not contain A/B category names. Company-specific file/column details belong in YAML. Do not modify the frozen code after the assessment build deadline.

## 5. Tests and evaluation

```powershell
pytest -q
python scripts/evaluate_A.py outputs/anivesh_tripathi_A.csv data/company_A/labels_A.csv
python scripts/evaluate_A_holdout.py --data-dir data --embedder tfidf
# after optional E5 installation/model caching:
python scripts/evaluate_A_holdout.py --data-dir data --embedder e5
python scripts/evaluate_B_exclusions.py --data-dir data
```

The pipeline performs validation before writing: exact output columns, one row per input ID, unique IDs, allowed exclusion codes, finite spend, valid confidence, non-empty entity keys, and taxonomy-valid category paths. Excluded rows have zero spend and blank categories.

## Project layout

- `spendintel/clean.py` — config-driven ingestion, VAT, FX, dates, A cross-system matching and duplicate removal
- `spendintel/spend_rules.py` — generic non-procurement keywords and label-assisted exclusion model
- `spendintel/entities.py` — supplier/item keys
- `spendintel/taxonomy.py` — dynamic taxonomy parsing (up to four levels)
- `spendintel/classify.py` — supervised hierarchical text classification and zero-label taxonomy matching
- `spendintel/validate.py` — output contract validation
- `configs/` — company-specific YAML only
- `outputs/` — generated A/B submission CSVs (no source data)
