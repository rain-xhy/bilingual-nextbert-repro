"""Paired sensitivity diagnostics; never reported as full-corpus retrieval accuracy."""
import argparse
import json
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from src.common import read_jsonl, write_json, sha256
from src.modeling import Encoder
from src.interventions import materialize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--base-config")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["dev", "test"])
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError("Evaluation output already exists")
    path = Path(a.checkpoint) / "config.json" if a.checkpoint else Path(a.base_config)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(cfg).to(device).eval()
    if a.checkpoint:
        model.load_transfer(a.checkpoint)
    rows = read_jsonl(a.data)
    manifest = Path(a.data).parent / "manifest.json"
    if manifest.exists():
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        if (provenance["model"], provenance["revision"]) != (cfg["model"], cfg.get("revision")):
            raise ValueError("Tokenizer revision mismatch")
        if provenance["files"][Path(a.data).name] != sha256(a.data):
            raise ValueError("Data hash mismatch")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate diagnostic query IDs")
    per_query = {}
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
        for row in rows:
            z = [model.embed(b).float() for b in materialize(row, model.tokenizer, device)]
            scores = [(z[0] * d).sum().item() for d in z[1:]]
            front, middle, back, neg, deleted, sham = scores
            per_query[row["id"]] = {"language": row["language"], "group_id": row["group_id"],
                "position_range": max(scores[:3]) - min(scores[:3]),
                "evidence_margin": min(scores[:3]) - deleted,
                "sham_change": abs(middle - sham),
                "selectivity": middle - deleted - abs(middle - sham),
                "all_positions_beat_negative": float(min(scores[:3]) > neg),
                "all_positions_beat_deleted": float(min(scores[:3]) > deleted),
                "scores": dict(zip(("front", "middle", "back", "negative", "deleted", "sham"), scores))}
    keys = ("position_range", "evidence_margin", "sham_change", "selectivity", "all_positions_beat_negative", "all_positions_beat_deleted")
    groups = {}
    for language in sorted({r["language"] for r in per_query.values()}):
        group_rows = [r for r in per_query.values() if r["language"] == language]
        groups[language] = {k: sum(r[k] for r in group_rows) / len(group_rows) for k in keys}
    if not groups:
        raise ValueError("Empty diagnostic data")
    groups["language_macro"] = {k: sum(g[k] for g in groups.values()) / len(groups) for k in keys}
    write_json(a.output, {"scope": "controlled_evidence_diagnostic", "metrics": groups, "split": a.split,
        "per_query": per_query, "data_sha256": sha256(a.data), "config": cfg, "checkpoint": a.checkpoint,
        "architecture_sha256": sha256(Path(a.checkpoint) / "architecture.pt") if a.checkpoint and (Path(a.checkpoint) / "architecture.pt").exists() else None,
        "adapter_sha256": sha256(Path(a.checkpoint) / "adapter/adapter_model.safetensors") if a.checkpoint else None})
    print(groups)


if __name__ == "__main__":
    main()
