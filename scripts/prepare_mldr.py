# BERT9_HFMIRROR_PROGRESS_V1
"""Download official MLDR JSONL files without executing dataset loading code."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from scripts.mirror_fetch import fetch_json, download_file
from src.common import write_json, write_jsonl, sha256


def download(url, target):
    download_file(url, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/mldr")
    p.add_argument("--train-per-language", type=int, default=2000)
    p.add_argument("--full-corpus", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--revision", default="d67138e705d963e346253a80e59676ddb418810a")
    a = p.parse_args()
    root = Path(a.output)
    if (root / "manifest.json").exists():
        revision = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["revision"]
    else:
        revision = a.revision
    manifest = {"dataset": "Shitao/MLDR", "revision": revision, "seed": a.seed, "files": []}
    all_train, texts = [], {}
    for language in ("zh", "en"):
        splits = {}
        for split in ("train", "dev", "test"):
            print(f"[MLDR] {language}/{split}: download or verify cache", flush=True)
            name = f"mldr-v1.0-{language}/{split}.jsonl.gz"
            target = root / "raw" / revision / name
            url = f"https://huggingface.co/datasets/Shitao/MLDR/resolve/{revision}/{name}"
            download(url, target)
            manifest["files"].append({"url": url, "sha256": sha256(target)})
            print(f"[MLDR] {language}/{split}: parse {target.name}", flush=True)
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            print(f"[MLDR] {language}/{split}: {len(rows)} records parsed", flush=True)
            for row in rows:
                row["query_id"] = f"{language}:{row['query_id']}"
                row["language"] = language
                for field in ("positive_passages", "negative_passages"):
                    row[field] = row.get(field) or []
                    for document in row[field]:
                        document["docid"] = f"{language}:{document['docid']}"
            splits[split] = rows
        heldout_ids = {d["docid"] for split in ("dev", "test") for row in splits[split]
                       for d in row["positive_passages"] + row["negative_passages"]}
        heldout_hashes = {hashlib.sha256(d["text"].encode()).hexdigest()
                          for split in ("dev", "test") for row in splits[split]
                          for d in row["positive_passages"] + row["negative_passages"]}
        filtered = []
        for row in splits["train"]:
            # Exclude overlap in both positives and negatives from training/DAPT.
            if any(d["docid"] in heldout_ids or hashlib.sha256(d["text"].encode()).hexdigest() in heldout_hashes
                   for d in row["positive_passages"] + row["negative_passages"]):
                continue
            positives = {d["docid"] for d in row["positive_passages"]}
            positive_texts = {d["text"] for d in row["positive_passages"]}
            row["negative_passages"] = [d for d in row["negative_passages"]
                                        if d["docid"] not in positives and d["text"] not in positive_texts]
            if positives and row["negative_passages"]:
                filtered.append(row)
        random.Random(a.seed).shuffle(filtered)
        selected = filtered[:a.train_per_language]
        print(f"[MLDR] {language}: train={len(splits['train'])}, filtered={len(filtered)}, selected={len(selected)}", flush=True)
        manifest[language] = {"original_train": len(splits["train"]), "filtered_train": len(filtered),
                              "selected_train": len(selected)}
        all_train.extend(selected)
        for row in selected:
            for doc in row["positive_passages"]:
                texts[doc["docid"]] = {"id": doc["docid"], "text": doc["text"], "language": language}
        for split in ("dev", "test"):
            write_jsonl(root / f"{language}_{split}.jsonl", splits[split])
            pool = {d["docid"]: d for row in splits[split]
                    for d in row["positive_passages"] + row["negative_passages"]}
            write_jsonl(root / f"{language}_{split}_pool.jsonl", pool.values())
        if a.full_corpus:
            name = f"mldr-v1.0-{language}/corpus.jsonl.gz"
            target = root / "raw" / revision / name
            url = f"https://huggingface.co/datasets/Shitao/MLDR/resolve/{revision}/{name}"
            download(url, target)
            manifest["files"].append({"url": url, "sha256": sha256(target)})
            def corpus_rows():
                with gzip.open(target, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        doc = json.loads(line)
                        doc["docid"] = f"{language}:{doc['docid']}"
                        yield doc
            write_jsonl(root / f"{language}_corpus.jsonl", corpus_rows())
    if not all_train:
        raise ValueError("No usable train examples remain after filtering; inspect dataset overlap")
    random.Random(a.seed).shuffle(all_train)
    write_jsonl(root / "train.jsonl", all_train)
    write_jsonl(root / "mlm_train.jsonl", texts.values())
    manifest["outputs"] = {p.name: sha256(p) for p in root.glob("*.jsonl")}
    manifest["pool_warning"] = "*_pool.jsonl is a judged candidate subset, not the official corpus"
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
