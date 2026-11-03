"""Validated prepared data, including first-subword NER labels."""
import json
from pathlib import Path
import torch
from src.common import read_jsonl, sha256


def validated_rows(path, cfg, allow_unverified=False):
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        if not allow_unverified:
            raise ValueError("Prepared data provenance manifest required")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["model"], manifest["revision"]) != (cfg["model"], cfg.get("revision")):
            raise ValueError("Prepared data tokenizer revision mismatch")
        if manifest["files"].get(path.name) != sha256(path):
            raise ValueError("Prepared data hash mismatch")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty data")
    return rows


def token_batch(row, device):
    ids = torch.tensor([row["input_ids"]], device=device, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def balanced_order(rows, rng):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row.get("language", "unknown"), []).append(i)
    for values in groups.values():
        rng.shuffle(values)
    return [values[i] for i in range(max(map(len, groups.values())))
            for _, values in sorted(groups.items()) if i < len(values)]
