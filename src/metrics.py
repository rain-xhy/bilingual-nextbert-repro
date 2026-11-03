"""Query-macro retrieval metrics. Missing queries count as zero."""
import math
import random


def evaluate(qrels, runs, k=10):
    if k < 1 or not qrels:
        raise ValueError("k must be positive and qrels must be nonempty")
    per_query = {}
    for qid, relevant in qrels.items():
        relevant = {str(d): float(g) for d, g in relevant.items() if g > 0}
        if not relevant:
            raise ValueError(f"Query {qid} has no positive judgement")
        ranked = list(dict.fromkeys(str(d) for d in runs.get(qid, [])))[:k]
        gains = [relevant.get(d, 0.0) for d in ranked]
        dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        per_query[qid] = {
            f"ndcg@{k}": dcg / idcg,
            f"recall@{k}": sum(g > 0 for g in gains) / len(relevant),
            f"mrr@{k}": next((1 / (i + 1) for i, g in enumerate(gains) if g > 0), 0.0),
        }
    return {key: sum(v[key] for v in per_query.values()) / len(per_query)
            for key in next(iter(per_query.values()))}, per_query


def paired_bootstrap(a, b, key="ndcg@10", samples=2000, seed=42):
    if set(a) != set(b) or not a or samples < 100:
        raise ValueError("Need identical nonempty query sets and >=100 resamples")
    differences = [b[q][key] - a[q][key] for q in sorted(a)]
    rng = random.Random(seed)
    values = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                    for _ in range(samples))
    return {"delta": sum(differences) / len(differences),
            "ci95": [values[int(.025 * samples)], values[int(.975 * samples)]],
            "queries": len(differences), "seed": seed}
