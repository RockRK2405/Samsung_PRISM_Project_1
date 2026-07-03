"""YAML configuration loading helpers.

Kept deliberately small — full dataclass-backed configs live in
:mod:`src.configs` and will be introduced when the training code
lands. For preprocessing we only need to read the YAML files under
``configs/`` as plain dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file. May be a ``str`` or :class:`pathlib.Path`.

    Returns:
        The parsed YAML as a Python ``dict``. Empty or comment-only files
        return an empty dict rather than ``None``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
