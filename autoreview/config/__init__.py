from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from autoreview.config.models import DomainConfig

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base. Override values win for non-dict values.
    For dict values, merge recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data else {}


def _apply_env_overrides(config: dict[str, Any], prefix: str = "AUTOREVIEW_") -> dict[str, Any]:
    """Apply environment variable overrides.

    Env vars like AUTOREVIEW_SEARCH__DATE_RANGE override config["search"]["date_range"].
    Double underscore (__) separates nesting levels.
    """
    result = config.copy()
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Try to convert to int/float/bool
        final_key = parts[-1]
        if value.lower() in ("true", "false"):
            current[final_key] = value.lower() == "true"
        else:
            try:
                current[final_key] = int(value)
            except ValueError:
                try:
                    current[final_key] = float(value)
                except ValueError:
                    current[final_key] = value
    return result


def load_config(
    domain: str = "general",
    overrides: dict[str, Any] | None = None,
) -> DomainConfig:
    """Load configuration for a domain.

    Priority: defaults YAML < env vars < explicit overrides
    """
    # Load domain-specific YAML if it exists
    yaml_path = _DEFAULTS_DIR / f"{domain}.yaml"
    base = _load_yaml(yaml_path) if yaml_path.exists() else {"domain": domain}

    # Apply env var overrides
    config = _apply_env_overrides(base)

    # Apply explicit overrides
    if overrides:
        config = deep_merge(config, overrides)

    return DomainConfig.model_validate(config)
