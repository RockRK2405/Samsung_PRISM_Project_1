# Experiment Log — Template

> One file per experiment. Suggested filename:
> `EXP-<NNN>-<short-slug>.md`, e.g. `EXP-001-xception-ff-c23.md`.

## Identifier
- **Experiment ID:** EXP-NNN
- **Date:**
- **Author:**
- **Related milestone:**

## Hypothesis
_What are we testing? What outcome would confirm / disprove it?_

## Setup
- **Dataset(s):**
- **Model / architecture:**
- **Config file(s):** `configs/...`
- **Hardware:**
- **Git commit:**

## Procedure
_Concrete step-by-step description of what was run._

## Results
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Acc    |       |     |      |
| F1     |       |     |      |
| FPR    |       |     |      |

## Observations
_Plots, failure cases, unexpected behaviour._

## Conclusion
_Did the hypothesis hold? What is the next experiment?_

## Artefacts
- Checkpoint: `checkpoints/...`
- Logs: `logs/...`
- Metrics: `outputs/metrics/...`
