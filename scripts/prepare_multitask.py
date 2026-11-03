"""Build a locked, bilingual four-task dataset and exact-length MLM packs from training text."""
import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from src.common import read_jsonl, write_json, write_jsonl, sha256
from src.data_version import dataset_revision
from src.interventions import special_template


def encode_labeled(row, task, tokenizer, length=512):
    if task == "classification":
        batch = tokenizer(row["premise"], row["hypothesis"], truncation=True, max_length=length)
        return {"input_ids": batch["input_ids"], "labels": row["label"]}
    batch = tokenizer(row["tokens"], is_split_into_words=True, truncation=True, max_length=length)
    labels, previous = [], None
    for word in batch.word_ids():
        labels.append(row["ner_tags"][word] if word is not None and word != previous else -100)
        previous = word
    return {"input_ids": batch["input_ids"], "labels": labels}


def pack_training(texts, tokenizer, length, max_packs):
    prefix, suffix = special_template(tokenizer)
    width = length - len(prefix) - len(suffix)
    if width < 1:
        raise ValueError("Packing budget too small")
    buffer, rows = [], []
    separator = tokenizer.sep_token_id
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        buffer.extend(ids + ([separator] if separator is not None else []))
        while len(buffer) >= width:
            rows.append({"id": f"pack:{len(rows)}", "input_ids": prefix + buffer[:width] + suffix})
            buffer = buffer[width:]
            if len(rows) == max_packs:
                return rows
    return rows  # Never pad a short sample and call it 8K content.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mldr", default="data/mldr")
    p.add_argument("--output", default="data/multitask")
    p.add_argument("--train-per-language", type=int, default=800)
    p.add_argument("--dev-per-language", type=int, default=40)
    p.add_argument("--test-per-language", type=int, default=100)
    p.add_argument("--packs-per-language", type=int, default=64)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root, source = Path(a.output), Path(a.mldr)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory for a new dataset version")
    lock = json.loads(Path("artifacts/model_lock.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(lock["model"], revision=lock["revision"])
    manifest = {"model": lock["model"], "revision": lock["revision"], "args": vars(a),
                "datasets": {}, "labels": {}, "files": {}, "counts": {},
                "mldr_manifest_sha256": sha256(source / "manifest.json")}
    def save(name, rows):
        write_jsonl(root / name, rows)
        manifest["files"][name] = sha256(root / name)
        manifest["counts"][name] = len(rows)
    train = read_jsonl(source / "train.jsonl")
    selected_train, packs = [], []
    for lang in ("en", "zh"):
        rows = [r for r in train if r["language"] == lang]
        random.Random(a.seed).shuffle(rows)
        selected_train.extend(rows[:a.train_per_language])
        unique = list(dict.fromkeys(d["text"] for r in rows[:a.train_per_language] for d in r["positive_passages"]))
        packed = pack_training(unique, tokenizer, a.length, a.packs_per_language)
        if not packed:
            raise ValueError(f"Insufficient {lang} training text for exact-length MLM packs")
        packs.extend({**r, "id": f"{lang}:{r['id']}", "language": lang} for r in packed)
        for split, size in (("dev", a.dev_per_language), ("test", a.test_per_language)):
            rows = read_jsonl(source / f"{lang}_{split}.jsonl")
            random.Random(a.seed).shuffle(rows)
            rows = rows[:size]
            pool = {d["docid"]: d for r in rows for d in r["positive_passages"] + r["negative_passages"]}
            save(f"{lang}_{split}_queries.jsonl", rows)
            save(f"{lang}_{split}_pool.jsonl", list(pool.values()))
    # Alternate languages to keep small-step experiments balanced.
    def interleave(rows):
        by_lang = [[r for r in rows if r["language"] == lang] for lang in ("en", "zh")]
        return [r for i in range(max(map(len, by_lang))) for group in by_lang for r in group[i:i+1]]
    save("retrieval_train.jsonl", interleave(selected_train))
    save("mlm_train.jsonl", interleave(packs))
    for task, name in (("classification", "facebook/xnli"), ("ner", "unimelb-nlp/wikiann")):
        revision = dataset_revision(name)
        manifest["datasets"][name] = revision
        combined_train = []
        for lang in ("en", "zh"):
            dataset = load_dataset(name, lang, revision=revision)
            labels = dataset["train"].features["label"].names if task == "classification" else dataset["train"].features["ner_tags"].feature.names
            if task in manifest["labels"] and manifest["labels"][task] != labels:
                raise ValueError("Label ordering differs across languages")
            manifest["labels"][task] = labels
            for split, original, count in (("train", "train", a.train_per_language),
                    ("dev", "validation", a.dev_per_language), ("test", "test", a.test_per_language)):
                ids = random.Random(a.seed).sample(range(len(dataset[original])), min(count, len(dataset[original])))
                rows = [{"id": f"{task}:{lang}:{original}:{i}", "language": lang,
                         **encode_labeled(dataset[original][i], task, tokenizer)} for i in ids]
                if split == "train":
                    combined_train.extend(rows)
                else:
                    save(f"{task}_{lang}_{split}.jsonl", rows)
            del dataset
        save(f"{task}_train.jsonl", interleave(combined_train))
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
