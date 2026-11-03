"""Classification / entity extraction and frozen-backbone retention, from prepared data."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from src.modeling import Encoder
from src.probe import entities
from src.task_data import validated_rows, token_batch
from src.common import write_json, sha256


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--base-config")
    p.add_argument("--task", choices=["classification", "ner"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--probe-train")
    p.add_argument("--probe-steps", type=int, default=100)
    p.add_argument("--probe-seed", type=int, default=42)
    p.add_argument("--probe-size", type=int, default=200, help="Maximum prepared training examples per language")
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError("Evaluation output exists")
    path = Path(a.checkpoint) / "config.json" if a.checkpoint else Path(a.base_config)
    source_cfg = json.loads(path.read_text(encoding="utf-8"))
    manifest = json.loads((Path(a.data).parent / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["labels"][a.task]
    cfg = dict(source_cfg, task=a.task, num_labels=len(names))
    if not a.probe_train and (not a.checkpoint or source_cfg["task"] != a.task):
        raise ValueError("A supervised task checkpoint or a frozen-probe training file is required")
    torch.set_num_threads(4)
    torch.manual_seed(a.probe_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(cfg).to(device).eval()
    if a.checkpoint:
        model.load_transfer(a.checkpoint, load_head=not a.probe_train)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rows = validated_rows(a.data, cfg)
    probe_ids = []
    if a.probe_train:
        model.requires_grad_(False)
        torch.manual_seed(a.probe_seed)
        model.head = torch.nn.Linear(model.backbone.config.hidden_size, len(names)).to(device)
        training = validated_rows(a.probe_train, cfg)
        cache, counts = [], {}
        with torch.no_grad():
            for row in training:
                language = row["language"]
                if counts.get(language, 0) >= a.probe_size:
                    continue
                counts[language] = counts.get(language, 0) + 1
                probe_ids.append(row["id"])
                batch = token_batch(row, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    h = model.hidden(batch)
                if a.task == "classification":
                    x, y = h.mean(1), torch.tensor([row["labels"]])
                else:
                    labels = torch.tensor(row["labels"])
                    x, y = h[0, labels != -100], labels[labels != -100]
                if y.numel():
                    cache.append((x.float().cpu(), y))
        if not cache:
            raise ValueError("No probe training features")
        rng = random.Random(a.probe_seed)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001)
        model.head.train()
        for _ in range(a.probe_steps):
            chosen = rng.choices(cache, k=16)
            x = torch.cat([v[0] for v in chosen]).to(device)
            y = torch.cat([v[1] for v in chosen]).to(device)
            loss = F.cross_entropy(model.head(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        del cache
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    per_query = {}
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        for row in rows:
            batch = token_batch(row, device)
            predictions = model.logits(batch).float().argmax(-1)[0].cpu()
            if a.task == "classification":
                record = {"correct": int(predictions == row["labels"])}
            else:
                labels = torch.tensor(row["labels"])
                selected = labels != -100
                gold = entities([names[i] for i in labels[selected].tolist()])
                guessed = entities([names[i] for i in predictions[selected].tolist()])
                record = {"tp": len(gold & guessed), "fp": len(guessed - gold), "fn": len(gold - guessed)}
            if row["id"] in per_query:
                raise ValueError("Duplicate task example ID")
            per_query[row["id"]] = {"language": row["language"], **record}
    if device.type == "cuda":
        torch.cuda.synchronize()
    if a.task == "classification":
        metrics = {"accuracy": sum(r["correct"] for r in per_query.values()) / len(per_query)}
    else:
        sums = {key: sum(r[key] for r in per_query.values()) for key in ("tp", "fp", "fn")}
        metrics = {"entity_micro_f1": 2 * sums["tp"] / max(1, 2*sums["tp"] + sums["fp"] + sums["fn"])}
    write_json(a.output, {"metrics": metrics, "per_query": per_query, "split": a.split, "config": cfg,
        "checkpoint": a.checkpoint, "data_sha256": sha256(a.data), "seconds": time.monotonic() - start,
        "protocol": "frozen_probe" if a.probe_train else "supervised_lora_task_branch",
        "probe": {"train_sha256": sha256(a.probe_train), "ids": probe_ids, "seed": a.probe_seed,
                  "steps": a.probe_steps} if a.probe_train else None,
        "adapter_sha256": sha256(Path(a.checkpoint) / "adapter/adapter_model.safetensors") if a.checkpoint else None,
        "architecture_sha256": sha256(Path(a.checkpoint) / "architecture.pt") if a.checkpoint and (Path(a.checkpoint) / "architecture.pt").exists() else None,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0})
    print(metrics)


if __name__ == "__main__":
    main()
