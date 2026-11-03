"""Frozen-backbone linear probes for classification and entity extraction retention."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from src.common import write_json
from src.data_version import dataset_revision


def entities(tags):
    result, active, start = set(), None, 0
    for i, tag in enumerate([*tags, "O"]):
        kind = tag[2:] if tag != "O" else None
        if active is not None and (tag.startswith("B-") or kind != active):
            result.add((active, start, i))
            active = None
        if kind is not None and active is None:
            active, start = kind, i
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["xnli", "ner"], required=True)
    p.add_argument("--language", choices=["en", "zh"], required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--model", default="jhu-clsp/mmBERT-base")
    p.add_argument("--revision")
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError("Use a new probe output path")
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.checkpoint:
        cfg = json.loads((Path(a.checkpoint) / "config.json").read_text(encoding="utf-8"))
        if cfg.get("document_bridge"):
            raise ValueError("Use src.evaluate_tasks --probe-train for structural checkpoints; this legacy probe does not load document bridges")
        a.model, a.revision = cfg["model"], cfg.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    architecture = AutoConfig.from_pretrained(a.model, revision=a.revision)
    if a.checkpoint:
        for key, value in cfg.get("architecture_overrides", {}).items():
            setattr(architecture, key, value)
    encoder = AutoModel.from_pretrained(a.model, revision=a.revision, config=architecture, attn_implementation="sdpa")
    encoder.config.reference_compile = False
    if a.checkpoint:
        encoder = PeftModel.from_pretrained(encoder, str(Path(a.checkpoint) / "adapter"))
    encoder.to(device).eval().requires_grad_(False)
    dataset_name = "facebook/xnli" if a.task == "xnli" else "unimelb-nlp/wikiann"
    data_revision = dataset_revision(dataset_name)
    dataset = load_dataset(dataset_name, a.language, revision=data_revision)
    names = dataset["train"].features["label"].names if a.task == "xnli" else dataset["train"].features["ner_tags"].feature.names
    indices, cache = {}, {}
    for split, count in [("train", a.train_size), (a.eval_split, a.eval_size)]:
        indices[split] = sorted(random.Random(a.seed).sample(range(len(dataset[split])), min(count, len(dataset[split]))))
        items = []
        for index in indices[split]:
            row = dataset[split][index]
            if a.task == "xnli":
                batch = tokenizer(row["premise"], row["hypothesis"], truncation=True, max_length=512, return_tensors="pt")
                labels = torch.tensor([row["label"]])
            else:
                batch = tokenizer(row["tokens"], is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
                positions, word_indices = [], []
                previous = None
                for position, word in enumerate(batch.word_ids()):
                    if word is not None and word != previous:
                        positions.append(position)
                        word_indices.append(word)
                    previous = word
                labels = torch.tensor([row["ner_tags"][i] for i in word_indices])
            batch.pop("token_type_ids", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                hidden = encoder(**batch).last_hidden_state[0]
            features = hidden.mean(0, keepdim=True) if a.task == "xnli" else hidden[positions]
            if len(labels):
                items.append((features.float().cpu(), labels))
        cache[split] = items
    # Loading an adapter consumes RNG; reset so base/adapted probes have identical head initialization.
    torch.manual_seed(a.seed)
    head = torch.nn.Linear(encoder.config.hidden_size, len(names)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    for step in range(a.steps):
        selected = rng.choices(cache["train"], k=16)
        x = torch.cat([pair[0] for pair in selected]).to(device)
        y = torch.cat([pair[1] for pair in selected]).to(device)
        loss = F.cross_entropy(head(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    correct = total = tp = fp = fn = 0
    head.eval()
    with torch.no_grad():
        for x, y in cache[a.eval_split]:
            pred = head(x.to(device)).argmax(-1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
            if a.task == "ner":
                gold, guessed = entities([names[i] for i in y]), entities([names[i] for i in pred])
                tp += len(gold & guessed)
                fp += len(guessed - gold)
                fn += len(gold - guessed)
    score = {"accuracy": correct / total} if a.task == "xnli" else {"entity_micro_f1": 2 * tp / max(1, 2 * tp + fp + fn)}
    write_json(a.output, {"metrics": score, "protocol": "frozen encoder, newly trained linear head; not full-finetuning leaderboard results",
                          "args": vars(a), "dataset": dataset_name, "dataset_revision": data_revision,
                          "sample_indices": indices, "sample_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest()})
    torch.save(head.state_dict(), str(a.output) + ".head.pt")
    print(score)


if __name__ == "__main__":
    main()
