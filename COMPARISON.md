# Model and pipeline comparison notes

The current production implementation is described in `APPROACH.md`. The supervised classifier is a hierarchical word/character TF-IDF + SGD model selected for the assessment's 60-minute runtime constraint. It achieved 0.697/0.603/0.526 L1/L2/L3 accuracy on an A item-grouped holdout (75/25, seed 42), with spend-weighted scores 0.825/0.756/0.635.

The zero-label TF-IDF path is retained as a portable fallback, not presented as high accuracy: on the same A holdout it achieved 0.412/0.202/0.156 accuracy. E5 code is optional, but the E5 model could not be downloaded or evaluated in the offline build environment. Do not treat earlier notebook/LinearSVC figures as measurements of the current implementation.

A count-once, B FX and exclusions, supplier/item key limitations, and unresolved missing source amounts are documented in `DECISIONS.md`.
