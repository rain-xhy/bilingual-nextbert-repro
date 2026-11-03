"""Matched-view adaptation with token-exact evidence interventions and gradient caching."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from src.common import read_jsonl, write_json, sha256
from src.modeling import Encoder
from src.gradient_cache import cached_backward
from src.interventions import materialize
from src.evidence_loss import evidence_objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--steps", type=int)
    p.add_argument("--hours", type=float)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.steps is not None:
        cfg["steps"] = a.steps
    if a.hours is not None:
        cfg["max_hours"] = a.hours
    if cfg["steps"] < 1:
        raise ValueError("steps must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not a.allow_cpu:
        raise ValueError("CUDA required for real training")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory already contains a run")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 42))
    torch.set_num_threads(4)
    rows = read_jsonl(cfg["data"])
    if not rows:
        raise ValueError("Empty training data")
    manifest_path = Path(cfg["data"]).parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model"] != cfg["model"] or manifest["revision"] != cfg.get("revision"):
            raise ValueError("Pretokenized data requires exactly the generating model/tokenizer revision")
        if manifest["files"][Path(cfg["data"]).name] != sha256(cfg["data"]):
            raise ValueError("Prepared data changed after its manifest was saved")
    elif not a.allow_cpu:
        raise ValueError("Real training requires a data provenance manifest")
    model = Encoder(cfg).to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.get("lr", 2e-4), weight_decay=.01)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    def forward(batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            return model.embed(batch)
    order = list(range(len(rows)))
    rng = random.Random(cfg.get("seed", 42))
    cursor = len(order)
    train_tokens = forward_tokens = 0
    actual_lengths = set()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    write_json(out / "input_manifest.json", {"config": cfg, "data_sha256": sha256(cfg["data"]),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "embedding_parameters": model.backbone.get_input_embeddings().weight.numel(),
        "device": str(device), "torch": torch.__version__, "backbone_config": model.backbone.config.to_dict(),
        "precision": "bf16" if amp else "float32", "gradient_cache": "two-pass with RNG replay"})
    start = time.monotonic()
    stop_reason = "steps"
    with (out / "train.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(cfg["steps"]):
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            for _ in range(cfg.get("accumulation", 1)):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                row = rows[order[cursor]]
                cursor += 1
                batches = materialize(row, model.tokenizer, device)
                actual_lengths.add(batches[1]["input_ids"].shape[1])
                teacher = None
                if cfg.get("anchor_weight", 0):
                    model.eval()
                    with torch.no_grad(), model.backbone.disable_adapter():
                        teacher = forward(batches[0]).float()
                    model.train()
                    forward_tokens += batches[0]["input_ids"].numel()
                stats = cached_backward(forward, batches, lambda z: evidence_objective(z, cfg, teacher),
                                        scale=1 / cfg.get("accumulation", 1))
                tokens = sum(b["input_ids"].numel() for b in batches)
                train_tokens += tokens
                forward_tokens += 2 * tokens
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value / cfg.get("accumulation", 1)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite parameter gradients")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            record = {"step": step + 1, **totals, "elapsed_seconds": elapsed,
                "input_tokens": train_tokens, "forward_tokens_including_replay": forward_tokens,
                "actual_lengths": sorted(actual_lengths), "grad_norm": float(norm),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
            if elapsed >= cfg.get("max_hours", 1) * 3600:
                stop_reason = "time"
                break
    model.save(out / "checkpoint")
    write_json(out / "summary.json", {**record, "stop_reason": stop_reason, "ran_actual_8k": 8192 in actual_lengths})


if __name__ == "__main__":
    main()
