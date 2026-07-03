"""Pytest configuration — makes ``src`` importable without an install step.

Adds the project root to ``sys.path`` so tests can ``from src.<pkg> import ...``
without having to ``pip install -e .`` first. This is intentional — the module
is not (yet) intended to be installed as a pip package (see pyproject.toml).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
