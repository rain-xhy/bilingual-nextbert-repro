"""Bilingual QA-derived controls, grouped splits, fixed tokenizer and exact token budgets."""
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from src.common import write_json, write_jsonl, sha256
from src.data_version import dataset_revision
from src.interventions import context_id, context_split, normalize, make_views


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="jhu-clsp/mmBERT-base")
    p.add_argument("--revision", default="c5955035435e2bf121cde7f3c8863ef52ff35d82")
    p.add_argument("--output", default="data/evidence")
    p.add_argument("--train-per-language", type=int, default=400)
    p.add_argument("--eval-per-language", type=int, default=100)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--max-evidence", type=int, default=1024)
    a = p.parse_args()
    root = Path(a.output)
    if (root / "manifest.json").exists():
        raise ValueError("Use a new output folder to preserve the existing dataset")
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    content_length = a.length - tokenizer.num_special_tokens_to_add(pair=False)
    outputs = {s: [] for s in ("train", "dev", "test")}
    manifest = {"model": a.model, "revision": a.revision, "length": a.length, "sources": {}, "rejections": {}}
    for language, name in (("en", "rajpurkar/squad"), ("zh", "hfl/cmrc2018")):
        revision = dataset_revision(name)
        dataset = load_dataset(name, revision=revision)
        manifest["sources"][name] = revision
        test_ids = {context_id(row["context"]) for row in dataset["validation"]}
        groups = {s: {} for s in outputs}
        for source_split in ("train", "validation"):
            for row in dataset[source_split]:
                key = context_id(row["context"])
                if source_split == "train" and key in test_ids:
                    continue
                split = "test" if source_split == "validation" else context_split(row["context"])
                # One query per context prevents repeated questions inflating sample size.
                groups[split].setdefault(key, row)
        rng = random.Random(42)
        for split, group in groups.items():
            pool = list(group.items())
            rng.shuffle(pool)
            cap = a.train_per_language if split == "train" else a.eval_per_language
            accepted, rejected = 0, 0
            for key, row in pool:
                if accepted >= cap:
                    break
                answers = row["answers"]["text"]
                evidence = tokenizer.encode(row["context"], add_special_tokens=False)
                if not answers or not evidence or len(evidence) > min(a.max_evidence, content_length // 3):
                    rejected += 1
                    continue
                aliases = [normalize(s) for s in answers if normalize(s)]
                if not aliases or not any(s in normalize(tokenizer.decode(evidence)) for s in aliases):
                    rejected += 1
                    continue
                filler, replacement, filler_sources = [], [], []
                for _ in range(2000):
                    other_key, other = pool[rng.randrange(len(pool))]
                    if other_key == key or any(s in normalize(other["context"]) for s in aliases):
                        continue
                    ids = tokenizer.encode(other["context"], add_special_tokens=False)
                    if not ids:
                        continue
                    filler_sources.append(other_key)
                    if len(filler) < content_length:
                        filler.extend(ids)
                    else:
                        replacement.extend(ids)
                    if len(filler) >= content_length and len(replacement) >= len(evidence):
                        break
                if len(filler) < content_length or len(replacement) < len(evidence):
                    rejected += 1
                    continue
                filler, replacement = filler[:content_length], replacement[:len(evidence)]
                views, spans = make_views(evidence, filler, replacement, content_length)
                if any(alias in normalize(tokenizer.decode(views["deleted"])) or
                       alias in normalize(tokenizer.decode(views["negative"])) for alias in aliases):
                    rejected += 1
                    continue
                outputs[split].append({"id": f"{language}:{key}", "language": language, "group_id": key,
                    "query": row["question"], "query_ids": tokenizer.encode(row["question"], add_special_tokens=False)[:128],
                    "evidence_ids": evidence, "filler_ids": filler, "replacement_ids": replacement,
                    "total_length": a.length, "evidence_ranges_without_specials": spans,
                    "answers": answers, "filler_groups": sorted(set(filler_sources))})
                accepted += 1
            manifest["rejections"][f"{language}_{split}"] = {"accepted": accepted, "rejected": rejected}
            if accepted == 0:
                raise ValueError(f"No usable examples in {language}/{split}")
    for split, rows in outputs.items():
        random.Random(42).shuffle(rows)
        write_jsonl(root / f"{split}.jsonl", rows)
    # Include filler contexts in leakage checks, not just labeled evidence contexts.
    sets = {s: {g for r in rows for g in [r["group_id"], *r["filler_groups"]]} for s, rows in outputs.items()}
    if any(sets[x] & sets[y] for x, y in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Source-context overlap across splits")
    manifest["files"] = {p.name: sha256(p) for p in root.glob("*.jsonl")}
    manifest["limitations"] = "Answer-string absence is a filter, not a proof of semantic irrelevance. Manually audit sampled negatives."
    write_json(root / "manifest.json", manifest)
    print(manifest["rejections"])


if __name__ == "__main__":
    main()
