"""Layered YAML configuration loader.

Role in the system: single source of non-secret settings. ``config/base.yaml`` holds the
shared defaults and ``config/<APP_ENV>.yaml`` (default ``dev``) is deep-merged over it,
so an environment file only lists what differs. Secrets stay in ``.env``.
"""

import os
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_ENV = "dev"


def _read_yaml(path: Path) -> dict:
    """Parse a YAML file into a dict.

    Args:
        path: YAML file to read.

    Returns:
        dict: Parsed content (empty dict for an empty file).
    """
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Args:
        base: Default values.
        override: Values that win; nested dicts are merged, anything else is replaced.

    Returns:
        dict: The merged result.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@cache
def load_config() -> dict:
    """Load ``base.yaml`` merged with the file for the active environment.

    The environment comes from ``APP_ENV`` (read from ``.env`` too), default ``dev``.

    Returns:
        dict: The merged, cached configuration.

    Raises:
        FileNotFoundError: If ``config/<APP_ENV>.yaml`` does not exist.
    """
    load_dotenv(REPO_ROOT / ".env")
    env = os.environ.get("APP_ENV", DEFAULT_ENV)
    return _deep_merge(_read_yaml(CONFIG_DIR / "base.yaml"), _read_yaml(CONFIG_DIR / f"{env}.yaml"))
