"""Validate token layout, manifest integrity and cross-split context separation."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from src.common import read_jsonl, sha256, write_json
from src.interventions import materialize

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data/evidence")
p.add_argument("--output", default="artifacts/evidence_audit.json")
a = p.parse_args()
root = Path(a.data_dir)
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"])
sets, counts = {}, {}
for split in ("train", "dev", "test"):
    path = root / f"{split}.jsonl"
    if sha256(path) != manifest["files"][path.name]:
        raise ValueError("Data hash mismatch")
    rows = read_jsonl(path)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    for row in rows:
        materialize(row, tokenizer, "cpu")
    sets[split] = {g for r in rows for g in [r["group_id"], *r["filler_groups"]]}
    counts[split] = len(rows)
if any(sets[x] & sets[y] for x, y in (("train", "dev"), ("train", "test"), ("dev", "test"))):
    raise ValueError("Context leakage detected")
write_json(a.output, {"status": "passed", "counts": counts, "length": manifest["length"],
                     "checks": ["file hashes", "unique IDs", "token-exact length", "evidence/filler context isolation"],
                     "not_verified": "Semantic irrelevance of generated negative documents requires human assessment."})
