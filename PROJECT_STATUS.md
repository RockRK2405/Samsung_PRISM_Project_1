# Samsung PRISM 26TS08 — Project Status

> **This file is superseded. Do not read it as current status.**
>
> Its previous contents (dated 2026-07-23) described the short-lived
> four-modality era — audio / image / text module owners and a
> `Samsung_PRISM_Fusion_Engine/` — none of which exist in this repository any
> more. Those modules were removed in commit `a85c52c`
> *"chore: remove non-video modules — keep only Video Detection + Dashboard"*,
> and the project was then redefined so that **general AI-video detection is
> primary and face deepfake is one branch**. Keeping a stale "single source of
> truth" at the repo root actively misled anyone opening the project, so it has
> been replaced by this pointer. The original text remains in git history.

## Current status lives here

| Read this | For |
|---|---|
| **`VIDEO_MODULE_WORK_LOG.md`** (repo root) | Everything built, tested and measured — the portable, self-contained record. Start here. |
| `Samsung_PRISM_Video_Detection/docs/DETECTION_ENABLEMENT_PLAN.md` | Why end-to-end inference was broken, and the phased fix |
| `Samsung_PRISM_Video_Detection/PIPELINE_README.md` | Which pipeline/testing files are current, and the verified Colab environment |
| `Samsung_PRISM_Video_Detection/docs/experiments/` | EXP-001 … EXP-006, each with a metrics JSON in `reports/` |
| `Samsung_PRISM_Video_Detection/checkpoints/MANIFEST.md` | Which checkpoint produced which result (weights are git-ignored) |
| `Samsung_PRISM_Video_Detection/docs/VIDEO_MODULE_MASTERPLAN.md` | The full phase plan (Phases 6-10 still open) |

## Results at a glance

| Experiment | Protocol | ROC-AUC |
|---|---|:-:|
| **EXP-006** | Celeb-DF v2, unseen **identity** | **0.9644** ±0.05 |
| **EXP-004** | GenVidBench, unseen **generator** | **0.9426** |
| EXP-001 | FF++ → Celeb-DF v2, unseen **dataset** | 0.7076 |
| EXP-005 | FF+++Celeb-DF → DFDC, unseen **dataset** | 0.6564 |
| EXP-002 | Dual-stream RGB+FFT | 0.7124 (negative) |
| EXP-003 | Explainability metric | 0.3094 aggregate (metric mismatch) |

These measure **different** generalization axes and are not interchangeable.
Three of six are kept negative results, by design.
