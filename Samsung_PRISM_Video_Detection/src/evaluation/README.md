# `src/evaluation`

**Purpose.** Metrics and evaluation pipelines for both same-dataset and
cross-dataset settings.

Planned responsibilities (not yet implemented):

- Binary classification metrics: accuracy, precision, recall, F1, AUC, FPR.
- Confusion matrix + ROC / PR curve generation.
- Per-manipulation-type breakdowns.
- Cross-dataset generalisation harness (train on A, test on B).
- Report writers that emit JSON / CSV into `outputs/metrics/`.
