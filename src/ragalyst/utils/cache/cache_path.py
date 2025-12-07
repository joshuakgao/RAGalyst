"""Utility functions for caching based on configuration parameters."""

import hashlib
import json
from pathlib import Path


def make_cache_path(purpose: str, relevant_cfg_dict: dict) -> Path:
    """Create a cache path based on the configuration parameters."""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    key_str = json.dumps(relevant_cfg_dict, sort_keys=True)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()

    dir = cache_dir / purpose / key_hash
    dir.mkdir(parents=True, exist_ok=True)

    (dir / "_config.json").write_text(json.dumps(relevant_cfg_dict, indent=2))

    return dir
