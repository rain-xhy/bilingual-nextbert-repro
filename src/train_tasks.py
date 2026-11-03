"""One shared architecture, common MLM adaptation, then independent supervised task branches."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from src.modeling import Encoder, mask_tokens
from src.gradient_cache import cached_backward
from src.task_data import validated_rows, token_batch, balanced_order
from src.common import write_json, sha256
from src.architecture import architecture_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--init-checkpoint")
    p.add_argument("--steps", type=int)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.steps is not None:
        cfg["steps"] = a.steps
    if cfg["steps"] < 1 or cfg.get("accumulation", 1) < 1:
        raise ValueError("Positive steps and accumulation required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not a.allow_cpu:
        raise ValueError("Formal training requires CUDA")
    if cfg.get("attention") == "flash_attention_2" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("This trainer requires BF16-capable CUDA for FA2; use SDPA float32 otherwise")
    output = Path(a.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rows = validated_rows(cfg["data"], cfg, allow_unverified=a.allow_cpu)
    model = Encoder(cfg).to(device)
    if a.init_checkpoint:
        model.load_transfer(a.init_checkpoint)
    model.train()
    # Loading checkpoints must not alter dropout randomness in matched runs.
    torch.manual_seed(seed)
    rng, corruption = random.Random(seed), random.Random(seed + 1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4), weight_decay=.01)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    def embed(batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            return model.embed(batch)
    def score(batch):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            return model.score(batch)
    write_json(output / "input_manifest.json", {"config": cfg, "architecture_id": architecture_id(cfg),
        "data_sha256": sha256(cfg["data"]), "init_checkpoint": a.init_checkpoint,
        "init_adapter_sha256": sha256(Path(a.init_checkpoint) / "adapter/adapter_model.safetensors") if a.init_checkpoint else None,
        "init_architecture_sha256": sha256(Path(a.init_checkpoint) / "architecture.pt") if a.init_checkpoint and (Path(a.init_checkpoint) / "architecture.pt").exists() else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in params),
        "embedding_parameters": model.backbone.get_input_embeddings().weight.numel(),
        "bridge_parameters": sum(p.numel() for p in model.bridge.parameters()) if model.bridge is not None else 0,
        "backbone_config": model.backbone.config.to_dict(), "precision": "bf16" if amp else "float32",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    order, cursor = [], 0
    start = time.monotonic()
    tokens = forward_tokens = max_actual = 0
    lengths, language_samples = {}, {}
    stop_reason = "steps"
    with (output / "train.jsonl").open("w", encoding="utf-8") as log:
        for step in range(cfg["steps"]):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for _ in range(cfg.get("accumulation", 1)):
                if cursor == len(order):
                    order, cursor = balanced_order(rows, rng), 0
                row = rows[order[cursor]]
                cursor += 1
                language = row.get("language", "unknown")
                language_samples[language] = language_samples.get(language, 0) + 1
                scale = 1 / cfg.get("accumulation", 1)
                task = cfg["task"]
                if task in {"mlm", "classification", "ner"}:
                    batch = token_batch(row, device)
                    if batch["input_ids"].shape[1] > cfg.get("max_length", 8192):
                        raise ValueError("Prepared example exceeds declared task length")
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                        if task == "mlm":
                            if batch["input_ids"].shape[1] != cfg.get("max_length", 8192):
                                raise ValueError("Common MLM must use exact-length packed content")
                            masked = mask_tokens(batch, model.tokenizer, cfg.get("masking", "token"), .15, corruption)
                            loss = model.mlm_loss(masked)
                        else:
                            labels = torch.tensor([row["labels"]], device=device)
                            logits = model.logits(batch).float()
                            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite task loss")
                    (loss * scale).backward()
                    loss_value += float(loss.detach()) * scale
                    batches, passes = [batch], 1
                else:
                    positive = row["positive_passages"][0]
                    negative = row["negative_passages"][0]  # fixed negative, independent of model/seed
                    if task == "retrieval":
                        batches = [model.tokenize([row["query"]], cfg.get("query_length", 128))]
                        batches += [model.tokenize([d["text"]], cfg.get("max_length", 8192)) for d in (positive, negative)]
                        def objective(z):
                            pos, neg = (z[0] * z[1]).sum(), (z[0] * z[2]).sum()
                            loss = F.softplus((neg - pos) / cfg.get("temperature", .05))
                            return loss, {}
                        stats = cached_backward(embed, batches, objective, scale=scale)
                    elif task == "rerank":
                        batches = [model.tokenize([row["query"]], cfg.get("max_length", 8192), [d["text"]]) for d in (positive, negative)]
                        stats = cached_backward(score, batches, lambda z: (F.softplus(z[1] - z[0]).mean(), {}), scale=scale)
                    else:
                        raise ValueError(f"Unsupported task {task}")
                    loss_value += stats["loss"] * scale
                    passes = 2
                n = sum(int(b["attention_mask"].sum()) for b in batches)
                tokens += n
                forward_tokens += passes * n
                for batch in batches:
                    length = int(batch["attention_mask"].sum(-1).max())
                    max_actual = max(max_actual, length)
                    lengths[str(length)] = lengths.get(str(length), 0) + 1
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite parameter gradients")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            record = {"step": step + 1, "loss": loss_value, "grad_norm": float(norm), "seconds": elapsed,
                "input_tokens": tokens, "encoder_forward_tokens_excluding_checkpoint_recompute": forward_tokens,
                "max_actual_length": max_actual, "language_samples": language_samples,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if elapsed >= cfg.get("max_hours", .5) * 3600:
                stop_reason = "time"
                break
    model.save(output / "checkpoint")
    write_json(output / "summary.json", {**record, "task": cfg["task"], "architecture_id": architecture_id(cfg),
        "requested_steps": cfg["steps"], "completed_steps": record["step"] == cfg["steps"],
        "stop_reason": stop_reason, "length_histogram": lengths, "ran_actual_8k": max_actual >= 8192})


if __name__ == "__main__":
    main()
