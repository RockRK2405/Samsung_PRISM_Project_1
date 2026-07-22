# Cross-Modal Validation — v1 scope vs the worklet's full vision

## What the worklet architecture calls for

Slide 6 of the worklet deck ("Multi Model Fusion Engine") lists, as
step 02, **"Cross-Modal Validation"** with two named examples:

- Audio vs. lip sync
- Text transcript vs. speech

This implies frame-level (or at least segment-level) temporal
alignment between modalities — e.g. does the mouth movement in the
video match the phonemes in the audio track at the same timestamp;
does the text transcript's content match what's actually said in the
audio.

## What v1 of this fusion engine actually does

**A same-submission disagreement heuristic**, implemented in
`fusion.py::_cross_modal_flags`:

> If two available modalities' `prob_synthetic` scores differ by more
> than `cross_modal_disagreement_threshold` (default 0.5), raise a flag
> naming both modalities and the gap.

Example: video module says 95% synthetic, audio module says 5%
synthetic on the same submission → flagged for manual review.

This is a legitimate and useful signal — large disagreement between
independent detectors on the same input is exactly the kind of case a
QC reviewer should look at by hand — but it is **not** the lip-sync or
transcript-alignment check the worklet architecture describes. It
operates on the two modalities' final scalar scores, with no awareness
of *time* within either one.

### Refinements after the first benchmark (bench.jsonl, 4 clips)

The initial version flagged **any** pair differing by ≥0.5, which fired
on 100% of clips — a flag that triggers on everything tells a reviewer
nothing. Two changes made it meaningful:

1. **Confidence gate (`conflict_confidence_gate`, default 0.6)**: both
   modalities in a disagreeing pair must be at least this confident for
   it to count as a conflict. A modality below the gate is treated as
   *abstaining*, not voting — its disagreement with a confident modality
   is not a real conflict, just one detector declining to commit.

2. **Abstain on confident straddling conflict**: when two *confident*
   modalities land on opposite sides of the decision threshold (one says
   synthetic, one says real), the fused verdict is forced to
   `"uncertain"` and routed to human review, rather than emitting a
   low-confidence `"real"`/`"synthetic"` from the averaged scalar. This
   is the cost-aware QC behaviour the worklet's dashboard calls for: the
   engine refuses to silently pass a submission that its own confident
   detectors flatly disagree about. (On the Morgan Freeman deepfake,
   plain averaging returned `"real"` at confidence 0.73 despite the
   image detector confidently flagging it synthetic; with this change it
   returns `"uncertain"` → review.)

Both still operate on scalar scores with no temporal awareness — they
sharpen the *routing*, not the underlying cross-modal modelling. The
lip-sync / transcript-alignment work below remains the real v2 goal.

## Why v1 stops here

Building true lip-sync mismatch detection requires:

1. **Frame-accurate audio/video alignment** — extracting the audio
   track's timestamps to match the video's sampled-frame timestamps
   (the video module currently samples 32 frames uniformly across the
   whole clip's duration, not at fixed real-time intervals — this
   would need to change to support alignment).
2. **A mouth-movement / phoneme correspondence model** — either a
   dedicated lip-sync model (e.g. SyncNet-style architectures) or a
   heuristic comparing audio energy envelopes to detected mouth
   open/close state per frame. Neither exists in any of the four
   modules today.
3. **A transcript-vs-speech comparator** for the text↔audio case —
   would need automatic speech recognition (ASR) on the audio track to
   produce a transcript to compare against the submitted text, or
   forced alignment if a transcript is already given.

All three are non-trivial standalone efforts, each comparable in scope
to one of the four modality detectors already built. Recommended as
**v2 work**, not a v1 corner-case to rush.

## What to build for v2 (recommended plan)

1. **Video module change**: expose per-frame *timestamps* (not just
   frame indices) alongside the existing per-frame scores, so other
   modalities can align to real time.
2. **Audio module change**: add a lightweight voice-activity /
   phoneme-boundary detector (even a simple energy-envelope heuristic
   would beat nothing) that outputs a timestamped activity signal.
3. **New `lip_sync.py`** in this fusion engine: compare the video's
   per-frame mouth-region motion (could reuse the video module's
   existing face-crop pipeline — mouth region is a fixed sub-crop of
   the already-detected face) against the audio's activity signal;
   flag segments where one is active and the other isn't.
4. **New `transcript_consistency.py`**: if a text input is provided
   alongside audio, run a small ASR pass (e.g. `whisper-tiny`, cheap
   enough to fit the worklet's cost-aware framing) on the audio and
   compare word-level overlap/similarity against the submitted text.

Both would slot into `fuse()` the same way `_cross_modal_flags` does
today — as additional entries in `FusionResult.cross_modal_flags` —
so no schema change is needed, only new detection logic.
