cd "C:\Users\nishanth golakoti\Downloads\audio_authenticity_detector\audio_authenticity_detector"# Audio Authenticity Detector (Real vs AI-Generated Speech)

Part of: **26TS08 — A Cost-Aware Framework for Synthetic Data Detection in Data
Acquisition Pipelines** (Samsung PRISM, Team: Ansh Jerath, Kashish Singh,
Rudra Khale, Golakoti Sri Ram Nishanth)

This module implements **Phase 3, Step 3.3 (Audio Analysis)** from your
architecture: classify a speech clip as REAL (human-recorded) or FAKE
(AI/TTS/voice-cloned), and output an authenticity score + synthetic-speech
probability, in the same style as your Image (EfficientNet-B0) and Text
(Perplexity + RoBERTa) detectors.

Total cost to build and train this: **$0**, using free Colab/Kaggle GPUs and
free public datasets.

---

## 1. Folder structure

```
audio_authenticity_detector/
├── README.md                  <- this file
├── requirements.txt
├── notebooks/
│   └── train_colab.ipynb      <- open this directly in Google Colab (free GPU)
├── src/
│   ├── preprocess.py          <- audio -> mel-spectrogram feature extraction
│   ├── dataset.py             <- PyTorch Dataset / DataLoader
│   ├── model.py                <- lightweight CNN (EfficientNet-B0-style budget)
│   ├── train.py                <- training loop, saves best checkpoint
│   ├── evaluate.py             <- accuracy / F1 / FPR / ROC-AUC, confusion matrix
│   └── infer.py                <- run the trained model on a single .wav file
└── outputs/                   <- checkpoints, logs, plots land here
```

## 2. Get free data (pick ONE to start — FoR is easiest)

| Dataset | Size | Where (free) | Notes |
|---|---|---|---|
| **Fake-or-Real (FoR)** | ~200k clips | Kaggle: search "Fake or Real audio dataset" (Bird et al.) | Easiest: already split real/fake, real/fake folders |
| **ASVspoof 2019 LA** | ~120k clips | https://datashare.ed.ac.uk/handle/10283/3336 (free, official) | The benchmark your own literature review cites |
| **In-The-Wild deepfake audio** | ~38k clips | Kaggle / Zenodo, search "In-the-Wild Audio Deepfake" | More realistic / noisy, good for a second evaluation round |

Expected folder layout after download (the scripts assume this):
```
data/
├── real/   *.wav
└── fake/   *.wav
```
(If FoR/ASVspoof already give you train/dev/eval splits, just point
`--data_dir` at each split folder separately — `preprocess.py` handles both.)

## 3. Where to run it for free

**Option A — Kaggle Notebooks (recommended for this project)**
1. Create a free Kaggle account.
2. Search the dataset above on Kaggle, click "New Notebook" directly from the
   dataset page — the data is then already mounted, **zero download time/cost**.
3. Settings → Accelerator → GPU T4 x2 (free, 30 hrs/week quota).
4. Upload `src/` files or copy-paste their contents into notebook cells, then run.

**Option B — Google Colab (recommended if you already have the dataset as a zip)**
1. Open `notebooks/train_colab.ipynb` in Colab (free T4 GPU, ~12 hr/session).
2. Upload your zipped dataset to Google Drive, mount Drive in the first cell.
3. Run all cells — preprocessing, training, evaluation are all included.

Both are $0. No AWS/GCP/Azure GPU billing needed for a model this size.

## 4. How the model works (Track A: spectrogram CNN — what's included here)

1. **preprocess.py**: loads each `.wav`, resamples to 16kHz, trims/pads to a
   fixed 4-second window, converts to a 128-bin log-Mel spectrogram (via
   `librosa`). This mirrors your slide's "Spectrogram Analysis — Mel
   spectrograms, frequency distributions, vocoder artifacts" step.
2. **model.py**: a small 4-block CNN (~1.2M params — similar compute budget
   to the EfficientNet-B0 you used for the image module), takes the
   spectrogram as a 1-channel image and outputs REAL/FAKE logits.
3. **train.py**: standard PyTorch training loop, weighted loss to handle
   class imbalance, saves the best checkpoint by validation F1.
4. **evaluate.py**: reports Accuracy, F1, Precision, Recall, False Positive
   Rate, ROC-AUC — same metrics your team already reports for Image/Text, so
   results plug straight into your KPI slide.
5. **infer.py**: given one `.wav`, prints `{authenticity_score, label,
   synthetic_speech_probability}` — the exact output schema your
   architecture diagram expects from the Audio Analysis block.

### Track B (optional upgrade, still free): Wav2Vec2 features

For higher accuracy with very little extra compute, swap `model.py`'s
spectrogram CNN for a frozen `facebook/wav2vec2-base` (via HuggingFace
`transformers`, free, ~95MB) as a feature extractor + a 2-layer MLP head you
train. Only the small head is trained, so it's fast even on a free Colab CPU.
This is flagged as a `--backbone wav2vec2` option inside `model.py` /
`train.py` comments — wire it in once Track A is working, if you have time
budget for it.

## 5. Cost comparison (for your project report)

| Item | Paid option | Cost | Free option used here | Cost |
|---|---|---|---|---|
| Compute (training) | AWS p3.2xlarge (V100), ~5 hrs | $15–18 | Kaggle/Colab free T4 GPU | $0 |
| Compute (Wav2Vec2 fine-tune, if attempted) | A100 instance, ~3 hrs | $9–12 | Free Colab/Kaggle GPU (frozen backbone, head-only training) | $0 |
| Dataset | Licensed/purchased speech corpora | $50–500+ | ASVspoof / FoR / In-the-Wild (public, free) | $0 |
| Storage | Cloud bucket | $1–5/mo | Local disk / Kaggle-hosted dataset | $0 |
| **Total for this module** | | **~$70–550** | | **$0** |

## 6. Next steps to match your "Upcoming Month's Plan" slide

- Once this Audio Detector hits your target (Accuracy/F1 ≥92%, FPR <5%,
  matching your project's success metric), wire its output into the
  **Adaptive Detection Controller** (cost-aware scheduler) and the
  **Cross-Modal Consistency Engine** (Audio ↔ Video lip-sync, Audio ↔ Text
  semantic agreement) exactly as in your architecture slides 9–16.
- Keep the FastAPI backend contract identical to your Image/Text modules so
  `infer.py`'s output JSON can be dropped straight into the same REST
  endpoint pattern.

