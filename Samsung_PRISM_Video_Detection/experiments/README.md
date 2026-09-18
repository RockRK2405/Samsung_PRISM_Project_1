# Experiments

Reproducible experiment records for the video module (Parts 33-34).

Each run gets its own directory:

```
experiments/
    exp_001_<name>/
        config.yaml          # exact config used (copied at run start)
        manifest_hash.txt     # hash of train/val/test manifests
        metrics.json          # accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, FNR
        best_model.pt         # symlink or path to checkpoint
        train_log.txt
        plots/
            confusion_matrix.png
            roc_curve.png
            pr_curve.png
            training_curve.png
            score_distribution.png
```

Rules (Part 35 — do not fake improvement):
- Every model change is compared against the previous best on the SAME
  test manifest. Baseline numbers are saved in `reports/baseline_results.json`.
- If a new model is worse, the record states that plainly and diagnoses why.
- Cross-generator (seen vs unseen) numbers are reported separately, never
  averaged into one headline accuracy.
