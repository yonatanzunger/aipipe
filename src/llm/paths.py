"""Filesystem locations for the llm backend's persisted configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """Return the directory for aipipe's persisted data (settings, secrets).

    Honors ``AIPIPE_DATA_DIR`` if set; otherwise uses a platform-appropriate
    per-user location.
    """
    override = os.environ.get("AIPIPE_DATA_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "aipipe"


def env_path() -> Path:
    """Return the path to the optional ``.env`` file in the data directory."""
    return data_dir() / ".env"
