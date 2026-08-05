"""Central config loader.

Loads ``config/config.yaml`` once and resolves the handful of paths the
rest of the backend needs. Nothing model-related is hardcoded elsewhere
(requirement 13) — everything flows from here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Dashboard root = parent of the backend/ package.
DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = DASHBOARD_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the parsed config dict (cached)."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_str: str | Path) -> Path:
    """Resolve a config path against the dashboard root (if relative)."""
    p = Path(path_str)
    return p if p.is_absolute() else (DASHBOARD_ROOT / p).resolve()


def video_module_root() -> Path:
    """Absolute path to the sibling Video Detection module."""
    cfg = get_config()
    return resolve(cfg["model"]["video_module_path"])


def checkpoint_path() -> Path:
    """Absolute path to the model checkpoint inside the video module."""
    cfg = get_config()
    return (video_module_root() / cfg["model"]["checkpoint"]).resolve()


def database_path() -> Path:
    cfg = get_config()
    return resolve(cfg["storage"]["database"])


def uploads_dir() -> Path:
    cfg = get_config()
    d = resolve(cfg["storage"]["uploads"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def outputs_dir() -> Path:
    cfg = get_config()
    d = resolve(cfg["storage"]["outputs"])
    d.mkdir(parents=True, exist_ok=True)
    return d
