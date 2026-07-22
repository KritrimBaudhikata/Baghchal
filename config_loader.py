"""Load config from config.yaml with optional env overrides."""
import os
from pathlib import Path

_CONFIG = None
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config():
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    try:
        import yaml
        with open(_CONFIG_PATH, "r") as f:
            _CONFIG = yaml.safe_load(f) or {}
    except Exception:
        _CONFIG = {}
    # Env overrides
    if os.environ.get("BAGHCHAL_SAVE_DIR"):
        _CONFIG.setdefault("training", {})["save_dir"] = os.environ["BAGHCHAL_SAVE_DIR"]
    if os.environ.get("BAGHCHAL_MODEL_PATH"):
        _CONFIG.setdefault("paths", {})["default_model"] = os.environ["BAGHCHAL_MODEL_PATH"]
    return _CONFIG


def get(key_path: str, default=None):
    """Get config value by dot path, e.g. 'training.num_iterations'."""
    c = load_config()
    for part in key_path.split("."):
        c = (c or {}).get(part)
        if c is None:
            return default
    return c
