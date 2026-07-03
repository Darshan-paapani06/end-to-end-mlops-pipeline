from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import METADATA_PATH, MODEL_PATH


def save_model(model: Any, model_path: str | Path = MODEL_PATH) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(model_path: str | Path = MODEL_PATH) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m src.train` before starting the API."
        )
    return joblib.load(path)


def save_metadata(metadata: dict[str, Any], metadata_path: str | Path = METADATA_PATH) -> Path:
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_metadata(metadata_path: str | Path = METADATA_PATH) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
