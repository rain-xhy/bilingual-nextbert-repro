"""Lock each auxiliary dataset on first use, reuse for paired evaluations."""
import json
from pathlib import Path
from huggingface_hub import HfApi
from src.common import write_json


def dataset_revision(name):
    path = Path("artifacts/dataset_lock.json")
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if name not in existing:
        existing[name] = HfApi().dataset_info(name).sha
        write_json(path, existing)
    return existing[name]
