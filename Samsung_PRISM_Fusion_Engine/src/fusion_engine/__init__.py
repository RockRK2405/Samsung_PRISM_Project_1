"""PRISM Worklet 26TS08 — Multi-Modal Fusion Engine.

Combines the Image, Audio, Text, and Video synthetic-data detectors
into one verdict, per the worklet's cross-attention/weighted-fusion
architecture (see docs/ARCHITECTURE.md).

Example
-------
>>> from fusion_engine import FusionEngine
>>> engine = FusionEngine()
>>> result = engine.analyze(video_path="clip.mp4", text="the transcript")
>>> result.verdict
'synthetic'
"""

from fusion_engine.engine import FusionEngine
from fusion_engine.schema import FusionResult, ModalityResult, SCHEMA_VERSION

__all__ = ["FusionEngine", "FusionResult", "ModalityResult", "SCHEMA_VERSION"]
