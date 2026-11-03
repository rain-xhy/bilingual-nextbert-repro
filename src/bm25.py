"""Transparent lexical control: English words and Chinese characters, BM25 k1=1.2,b=.75."""
import argparse
import collections
import heapq
import math
import re
import time
from pathlib import Path
from src.common import read_jsonl, write_json, sha256
from src.metrics import evaluate


def tokens(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--scope", choices=["candidate_pool", "full_corpus"], required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.scope == "full_corpus" and "_pool" in Path(a.corpus).stem:
        raise ValueError("A candidate pool cannot be labeled full corpus")
    docs, queries = read_jsonl(a.corpus), read_jsonl(a.queries)
    ids = [d["docid"] for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate corpus IDs")
    postings = collections.defaultdict(list)
    lengths = []
    start = time.monotonic()
    for i, doc in enumerate(docs):
        counts = collections.Counter(tokens(doc["text"]))
        lengths.append(sum(counts.values()))
        for term, count in counts.items():
            postings[term].append((i, count))
    average = sum(lengths) / max(1, len(lengths))
    if not average:
        raise ValueError("Empty searchable corpus")
    runs = {}
    for query in queries:
        scores = collections.defaultdict(float)
        for term in set(tokens(query["query"])):
            posting = postings.get(term, [])
            idf = math.log(1 + (len(docs) - len(posting) + .5) / (len(posting) + .5))
            for i, tf in posting:
                scores[i] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[i] / average))
        ranked = heapq.nsmallest(min(100, len(docs)), scores, key=lambda i: (-scores[i], i))
        selected = set(ranked)
        for i in range(len(docs)):
            if len(ranked) >= min(100, len(docs)):
                break
            if i not in selected:
                ranked.append(i)
        runs[query["query_id"]] = [ids[i] for i in ranked]
    qrels = {q["query_id"]: {d["docid"]: 1 for d in q["positive_passages"]} for q in queries}
    if len(qrels) != len(queries):
        raise ValueError("Duplicate query IDs")
    id_set = set(ids)
    if any(d not in id_set for rel in qrels.values() for d in rel):
        raise ValueError("Judged positive missing from corpus")
    metrics, per_query = evaluate(qrels, runs)
    recall, _ = evaluate(qrels, runs, 100)
    metrics["recall@100"] = recall["recall@100"]
    write_json(a.output, {"metrics": metrics, "per_query": per_query, "runs": runs,
                          "scope": a.scope, "corpus_size": len(docs), "seconds": time.monotonic() - start,
                          "queries_sha256": sha256(a.queries), "corpus_sha256": sha256(a.corpus),
                          "tokenization": "English words and Chinese characters; no language-specific stemming"})
    print(metrics)


if __name__ == "__main__":
    main()
