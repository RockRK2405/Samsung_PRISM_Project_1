"""Filesystem path resolution.

Every path used by the pipeline (raw data, processed cache, outputs,
checkpoints, logs) is defined in ``configs/paths.yaml`` relative to
the project root. This module resolves those relative paths against
the project root at runtime so calling code never has to hand-craft
strings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import load_yaml


def project_root() -> Path:
    """Return the absolute path to the project root.

    The root is defined as the directory containing this ``src`` package
    (three levels up from this file: ``.../src/utils/paths.py``).

    Returns:
        Absolute :class:`pathlib.Path` of the project root.
    """
    return Path(__file__).resolve().parents[2]


def load_paths(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/paths.yaml`` and resolve every path against the project root.

    Args:
        config_path: Optional override for the paths YAML location. When
            ``None`` (default) the standard ``configs/paths.yaml`` inside
            the project root is used.

    Returns:
        A dictionary matching the shape of ``paths.yaml`` but with every
        leaf string replaced by an absolute :class:`pathlib.Path`.
    """
    root = project_root()
    cfg_path = Path(config_path) if config_path else root / "configs" / "paths.yaml"
    raw = load_yaml(cfg_path).get("paths", {})
    return _resolve(raw, root)


def _resolve(node: Any, root: Path) -> Any:
    """Recursively convert relative path strings into absolute ``Path`` objects."""
    if isinstance(node, dict):
        return {k: _resolve(v, root) for k, v in node.items()}
    if isinstance(node, str):
        p = Path(node)
        return p if p.is_absolute() else (root / p).resolve()
    return node
