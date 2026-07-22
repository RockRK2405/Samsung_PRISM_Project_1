# Samsung PRISM Worklet 26TS08 — Project Status & Roadmap

**Worklet:** A Cost-Aware Framework for Synthetic Data Detection in Data
Acquisition Pipelines
**As of:** 2026-07-23
**Branch:** `claude/samsung-prism-video-detection-kctonh`

This document is the single source of truth for where the project stands
and what remains. It is deliberately honest about what is *validated* vs.
what is *built but unproven* — a working demo is not the same as a
success-metric-passing system.

---

## 1. Worklet architecture — component status

```
Data Ingestion → Modality Pipelines → Parallel Detectors → Fusion Engine & QC → Explainable Output
     ✅                 ✅                    ✅ (4/4)            ✅ v1              🟡 partial
```

| Stage | Status | Notes |
|---|---|---|
| Data ingestion | ✅ | Per-modality loaders exist in each module |
| Modality pipelines | ✅ | Preprocessing built for all 4 modalities |
| Parallel detectors | ✅ 4/4 | All four modality models integrated behind adapters |
| Fusion engine | ✅ v1 | Weighted aggregation + confidence-gated cross-modal flags + abstain-on-conflict routing |
| QC dashboard | 🟡 | Engine emits verdict + flags + explanation as structured data; **no UI built yet** |
| Explainable output | 🟡 | Video has GradCAM; fusion emits natural-language explanation; other modalities' explanations not surfaced through fusion |

---

## 2. Per-module status (the four detectors)

| Module | Owner | State | Benchmark accuracy* | Weight | Key caveat |
|---|---|---|---|---|---|
| **Video** | Rudra | ✅ Production | 50% | 0.30 | Baseline (EfficientNet-B0 + mean-pool). OOD on 2022 SOTA deepfakes. |
| **Audio** | Nishanth | ✅ Integrated | 25% | 0.32 | Confidently wrong on voice-cloned video deepfakes (trained on ASVspoof TTS artifacts). Confidence now capped at 0.85. |
| **Image** | Kashish | 🟡 Integrated | 75% (best) | 0.20 | Strongest on the benchmark despite being a 1-epoch CIFAKE proof-of-concept. Retraining script now in repo. |
| **Text** | Ansh | ✅ Trained | 33%** | 0.18 | **Real model trained this session**: 99.26% acc / 0.41% FPR on HC3 held-out. Low benchmark score is because transcripts of scripted deepfake voiceovers genuinely read as borderline. |

\* From the first fusion benchmark (`bench.jsonl`, 4 clips: 1 real + 3
deepfakes). **Tiny sample — directional only, not a validated metric.**
\*\* Text's *own* task performance is 99%+ ; the 33% here is on the
unusual task of judging ASR transcripts of deepfake voiceovers.

---

## 3. What was actually accomplished (this phase)

### Fusion engine (new `Samsung_PRISM_Fusion_Engine/` module)
- **Schema** (`schema.py`): versioned, JSON-serialisable `ModalityResult`
  / `FusionResult` wire format; normalises each module's differing
  class-index conventions into one `prob_synthetic` meaning.
- **Aggregation** (`fusion.py`): confidence-weighted score fusion over
  whatever subset of modalities is available; graceful when any modality
  is missing or fails.
- **Cross-modal consistency (v1)**: confidence-gated disagreement flags +
  **abstain-on-confident-conflict** routing (straddling conflict →
  `uncertain` → human review, instead of silently averaging to a verdict).
- **Adapters**: one per modality, each catching all exceptions internally
  so one bad modality never sinks the engine.
- **CLI + batch benchmark**: `scripts/analyze.py` (single submission),
  `scripts/benchmark.py` (labeled manifest → aggregate metrics with both
  score-based and honest verdict-based / abstention numbers).
- **Tests**: 19 pure-logic tests passing (schema + fusion).
- **Docs**: `INTEGRATION_NOTES.md` (per-module wiring + retraining paths),
  `CROSS_MODAL.md` (v1 scope vs. the worklet's full lip-sync vision).

### Module fixes
- **Text**: trained a real model (HC3, baseline TF-IDF+LogReg,
  99.26% acc / 0.41% FPR); fusion auto-discovers `artifacts/enhanced/model.pkl`.
- **Text/perplexity**: resolved the torch/transformers version conflict
  (cap transformers `<4.45` to keep torch pinned `<2.3` for the video
  module's facenet-pytorch).
- **Image**: added `scripts/train.py` — a real CLI port of the notebook
  training loop, so the 1-epoch checkpoint can be retrained on full CIFAKE.
- **Audio**: capped self-reported confidence at 0.85 (was pathologically
  overconfident at 0.999 while 25% accurate).

### The honest headline finding
On the first real benchmark, the fusion engine's **fused accuracy did not
beat its best single modality** — because its two highest-weighted
detectors (audio, video) are out-of-distribution and confidently wrong on
current SOTA deepfakes. Fusion tuning cannot recover this; it is a
detector-quality bottleneck. The engine now *correctly abstains* (routes
to review) rather than silently passing fakes as real — a safer, honest
failure mode, but it means the system currently provides limited
automation value on this content until the detectors improve.

---

## 4. Success metrics vs. worklet targets

| Metric | Target | Current | Status |
|---|---|---|---|
| F1 / Accuracy | ≥ 92% | Fusion: not met on 4-clip benchmark; Text (own task): 98.9% F1 | 🔴 fusion / ✅ text |
| False Positive Rate | < 5% | Text (own task): 0.41%; fusion: unmeasurable on 1 real clip | 🟡 |
| Explainability | ≥ 85% | Video GradCAM + fusion NL explanation; not yet quantified | 🟡 |
| Latency | < 5s/sample | ~1.3s/clip (4-modality, Whisper excluded) | ✅ |

**Blocker for the top two rows: no adequate validation set.** Everything
downstream (weight tuning, threshold calibration, a real reliability
diagram, a defensible F1/FPR number) is gated on having ≥20 labeled,
same-subject multimodal clips.

---

## 5. Roadmap — milestones to complete the worklet

### Milestone A — Validation data (NEXT, unblocks everything)
- [ ] Acquire ≥20 labeled clips. Options, in order of friction:
  - **DFD dataset** (Kaggle, no gate) — download started, not finished.
  - **FakeAVCeleb** (Google Form, self-service) — best task match (real
    audio+video deepfakes). Form submitted / pending.
  - **FakeSV** (signed agreement + institutional email) — native 4-modality
    but "fake news", not synthetic-media. Slowest.
- [ ] Build a larger `bench.jsonl` and run `scripts/benchmark.py`.
- [ ] Report honest verdict-based accuracy, abstention rate, FPR.

### Milestone B — Detector quality (the real bottleneck)
- [ ] Retrain **audio** on voice-cloned / audio-driven-reenactment deepfakes
      (not just ASVspoof TTS) — it is the worst performer and highest weight.
- [ ] Retrain **image** beyond 1 epoch on full CIFAKE (`scripts/train.py` ready).
- [ ] Evaluate **video** cross-dataset gap; consider fine-tuning on newer
      deepfake generations.
- [ ] Once B done, **re-tune fusion weights on the Milestone-A validation
      set** (currently deliberately un-tuned to avoid overfitting 4 points).

### Milestone C — True cross-modal validation (worklet's stated end-state)
- [ ] Video: expose per-frame timestamps (not just indices).
- [ ] Audio: add voice-activity / phoneme-boundary detection.
- [ ] New `lip_sync.py`: mouth-region motion vs. audio activity alignment.
- [ ] New `transcript_consistency.py`: ASR (whisper-tiny) vs. submitted text.
- These catch deepfakes via cross-modal *inconsistency* rather than any
  single detector's scalar — robust to the OOD problem in Milestone B.

### Milestone D — QC dashboard + explainable output
- [ ] UI over `FusionResult` (verdict, flags, per-modality breakdown,
      explanations) — the worklet's "QC Dashboard".
- [ ] Surface each modality's own explanation through fusion (not just video's).
- [ ] Quantify explainability against the ≥85% target.

### Milestone E — Cost-aware framing (the worklet's headline)
- [ ] Measure per-modality cost (latency + compute) vs. marginal accuracy.
- [ ] Cheap-first cascade: run cheap modalities first, only invoke
      expensive ones (e.g. video GradCAM, ASR) when the cheap ones are
      uncertain or conflict — the literal "cost-aware" contribution.

---

## 6. Known technical debt
- Audio confidence cap (0.85) is a single-point heuristic; replace with a
  proper reliability-diagram calibration once Milestone A data exists.
- Fusion weights are provisional (proportional to each module's *self-
  reported* metrics on *different* datasets) — not jointly validated.
- No `test_audio_adapter.py` / `test_text_adapter.py` yet (need a test
  env with librosa/torch/sklearn).
- QC dashboard is data-only; no UI.
