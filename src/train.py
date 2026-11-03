"""Bounded LoRA adaptation. Time caps are checked at optimizer boundaries."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from src.common import read_jsonl, write_json, sha256
from src.modeling import Encoder, mask_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-adapter")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.steps is not None:
        cfg["steps"] = args.steps
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. --allow-cpu is only intended for smoke tests.")
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a fresh output directory; existing runs are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    rng = random.Random(seed)
    order_rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_jsonl(cfg["data"])
    if not rows:
        raise ValueError("Training file is empty")
    if cfg["task"] != "mlm":
        for row in rows:
            if not row.get("positive_passages") or not row.get("negative_passages"):
                raise ValueError("Each training query needs a positive and an explicit negative")
    model = Encoder(cfg).to(device)
    if args.init_adapter:
        model.load_transfer(Path(args.init_adapter).parent)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4), weight_decay=.01)
    use_amp = device == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    start = time.monotonic()
    seen_tokens = 0
    length_histogram = {}
    actual_max_length = 0
    stop_reason = "steps"
    optimizer.zero_grad()
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    write_json(destination / "input_manifest.json", {
        "config": cfg, "data_sha256": sha256(cfg["data"]), "init_adapter": args.init_adapter,
        "init_adapter_sha256": sha256(Path(args.init_adapter) / "adapter_model.safetensors") if args.init_adapter else None,
        "torch": torch.__version__, "device": device,
        "gpu": torch.cuda.get_device_name() if use_amp else None,
        "trainable_parameters": sum(p.numel() for p in params),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "embedding_parameters": model.backbone.get_input_embeddings().weight.numel(),
        "non_embedding_parameters": sum(p.numel() for p in model.parameters()) - model.backbone.get_input_embeddings().weight.numel(),
        "backbone_config": model.backbone.config.to_dict(),
    })
    order = list(range(len(rows)))
    cursor = len(order)
    with (destination / "train.jsonl").open("w", encoding="utf-8") as log:
        for step in range(cfg["steps"]):
            loss_value = 0.0
            schedule = cfg.get("length_schedule", [[1.0, 512]])
            fraction = step / max(1, cfg["steps"])
            length = next(length for until, length in schedule if fraction < until)
            for _ in range(cfg.get("accumulation", 8)):
                if cursor == len(order):
                    order_rng.shuffle(order)
                    cursor = 0
                row = rows[order[cursor]]
                cursor += 1
                length_histogram[str(length)] = length_histogram.get(str(length), 0) + 1
                with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                    if cfg["task"] == "mlm":
                        batch = model.tokenize([row["text"]], length)
                        corrupted = mask_tokens(batch, model.tokenizer, cfg.get("masking", "token"), .15, rng)
                        loss = model.mlm_loss(corrupted)
                    else:
                        positive = rng.choice(row["positive_passages"])
                        negative = rng.choice(row["negative_passages"])
                        documents = [positive["text"], negative["text"]]
                        if cfg["task"] == "retrieval":
                            batch = model.tokenize(documents, length)
                            query = model.tokenize([row["query"]], cfg.get("query_length", 128))
                            q = model.embed(query)
                            d = model.embed(batch)
                            logits = q @ d.T / cfg.get("temperature", .05)
                            loss = F.cross_entropy(logits.float(), torch.zeros(1, device=device, dtype=torch.long))
                            seen_tokens += int(query["attention_mask"].sum())
                            if cfg.get("anchor_weight", 0) > 0:
                                # Disable adapters to use exactly the frozen initialization as teacher.
                                model.backbone.eval()
                                with torch.no_grad(), model.backbone.disable_adapter():
                                    target = model.embed(query)
                                model.backbone.train()
                                loss = loss + cfg["anchor_weight"] * (1 - (q * target).sum(-1)).mean()
                                seen_tokens += int(query["attention_mask"].sum())
                        elif cfg["task"] == "rerank":
                            batch = model.tokenize([row["query"]] * 2, length, documents)
                            logits = model.score(batch).unsqueeze(0)
                            loss = F.cross_entropy(logits.float(), torch.zeros(1, device=device, dtype=torch.long))
                        else:
                            raise ValueError(f"Unknown task {cfg['task']}")
                    seen_tokens += int(batch["attention_mask"].sum())
                    actual_max_length = max(actual_max_length, int(batch["attention_mask"].sum(-1).max()))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                scaler.scale(loss / cfg.get("accumulation", 8)).backward()
                loss_value += loss.item() / cfg.get("accumulation", 8)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            record = {"step": step + 1, "loss": loss_value, "max_length": length,
                      "elapsed_seconds": elapsed, "input_tokens": seen_tokens,
                      "input_tokens_per_second": seen_tokens / elapsed,
                      "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if elapsed >= cfg.get("max_hours", 1) * 3600 or seen_tokens >= cfg.get("max_tokens", 10**12):
                stop_reason = "time" if elapsed >= cfg.get("max_hours", 1) * 3600 else "tokens"
                break
    model.save(destination / "checkpoint")
    write_json(destination / "summary.json", {**record, "stop_reason": stop_reason,
               "length_histogram": length_histogram, "actual_max_length": actual_max_length,
               "completed_schedule": str(max(n for _, n in cfg.get("length_schedule", [[1, 512]]))) in length_histogram,
               "ran_actual_8k": actual_max_length >= 8192})


if __name__ == "__main__":
    main()
