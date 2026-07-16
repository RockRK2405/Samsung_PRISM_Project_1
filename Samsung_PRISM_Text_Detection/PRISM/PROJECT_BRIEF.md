# PRISM Worklet Brief

## What the two PPTs say

The project is **26TS08: A cost-aware framework for synthetic data detection in data acquisition pipelines**.

The main goal is to build an automated, explainable system that detects real vs. AI-generated data across four modalities:

- Image
- Audio
- Text
- Video

The target architecture is:

1. Data upload
2. Metadata extraction
3. Modality splitting
4. Parallel modality detectors
5. Per-modality score generation
6. Fusion layer / weighted aggregation
7. Explainable QC decision dashboard

The expected output from each detector is a synthetic probability score, then the fusion engine combines those scores into a final verdict with confidence and explanation.

## Success metrics from the deck

- Accuracy / F1: greater than 92%
- False positive rate: less than 5%
- Latency: less than 5 seconds per sample
- Explainability score: greater than 85%
- Modalities covered: image, audio, text, video
- Stretch outcome: 1 paper or patent

For audit integrity, the false positive rate is especially important: real data wrongly flagged as synthetic is the business-critical failure mode.

## Why HC3 is the right first step

The template deck lists HC3 as the text dataset. Since HC3 is already downloaded and mounted in Drive, the fastest useful workstream is the text detector:

- Load HC3 human and ChatGPT answers.
- Normalize into `text,label` rows.
- Train a baseline detector.
- Report accuracy, F1, false positive rate, and latency.
- Save the trained model and metrics.
- Later expose the model as one per-modality score in the fusion layer.

## Text detector direction

The PPT mentions:

- Perplexity scoring
- Burstiness
- LLM stylometry
- RoBERTa / transformer-based detection
- Hybrid stylometric + semantic approaches

The starter scaffold implements a cheap baseline first:

- Word TF-IDF
- Character n-gram TF-IDF
- Stylometric statistics such as sentence length, lexical diversity, punctuation ratio, and burstiness
- Logistic regression with probability output

This is intentionally cost-aware: it is fast, CPU-friendly, and gives a measurable baseline before moving to heavier transformer models.

## Month 1 practical target

For the current start:

1. Get HC3 loading correctly from Drive.
2. Train the baseline text detector.
3. Record accuracy, F1, false positive rate, ROC-AUC, and latency.
4. Inspect failures where human text is flagged synthetic.
5. Decide whether the next step should be RoBERTa fine-tuning, perplexity scoring, or a two-stage cost-aware router.

## Immediate command path

Install the starter dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the HC3 baseline:

```bash
python -m prism_text_detector.train_baseline --data "/content/drive/MyDrive/path/to/HC3" --out artifacts/hc3_text_baseline --target-fpr 0.05
```

If the first run is too slow, start with a smaller sample:

```bash
python -m prism_text_detector.train_baseline --data "/content/drive/MyDrive/path/to/HC3" --out artifacts/hc3_text_baseline_smoke --sample-per-label 2000 --min-df 1 --target-fpr 0.05
```
