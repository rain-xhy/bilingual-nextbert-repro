"""Confirm a frozen architecture on held-out tests with paired seed/query bootstrap.

NER uses resampled entity counts to recompute micro-F1; it never averages per-example F1.
"""
import argparse
import random
from pathlib import Path
from scripts.run_architecture import read
from scripts.select_architecture import METRICS
from src.common import sha256, write_json


def score(records, task):
    if task == "ner":
        tp, fp, fn = [sum(r[k] for r in records) for k in ("tp", "fp", "fn")]
        return 2*tp / max(1, 2*tp + fp + fn)
    key = "correct" if task == "classification" else "ndcg@10"
    return sum(r[key] for r in records) / len(records)


def paired_task(pairs, task, repeats=2000):
    if repeats < 100 or not pairs:
        raise ValueError("Need run pairs and >=100 resamples")
    query_ids = None
    for pair in pairs:
        ids = {}
        for lang in ("en", "zh"):
            base, candidate = pair[lang]
            if not base or set(base) != set(candidate):
                raise ValueError("Different or empty paired query sets")
            ids[lang] = sorted(base)
        if query_ids is not None and ids != query_ids:
            raise ValueError("Query sets differ across seeds")
        query_ids = ids
    def delta(pair, samples):
        return sum(score([pair[l][1][q] for q in samples[l]], task) -
                   score([pair[l][0][q] for q in samples[l]], task) for l in ("en", "zh"))/2
    observed = [delta(pair, query_ids) for pair in pairs]
    rng = random.Random(42)
    bootstrap = []
    for _ in range(repeats):
        samples = {l: rng.choices(ids, k=len(ids)) for l, ids in query_ids.items()}
        bootstrap.append(sum(delta(pair, samples) for pair in rng.choices(pairs, k=len(pairs)))/len(pairs))
    bootstrap.sort()
    return {"candidate_minus_baseline": sum(observed)/len(observed), "seed_deltas": observed,
            "ci95": [bootstrap[int(.025*repeats)], bootstrap[int(.975*repeats)]],
            "aggregation": "equal en/zh macro with paired seed and query resampling"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/architecture")
    p.add_argument("--selection", default="runs/architecture/selection.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--output", default="runs/architecture/test_confirmation.json")
    a = p.parse_args()
    if Path(a.output).exists() or len(set(a.seeds)) != len(a.seeds) or len(a.seeds) < 3:
        raise ValueError("Use a new output path and at least three distinct training seeds")
    root, selection = Path(a.root), read(a.selection)
    protocol = read(root / "protocol.json")
    digest = sha256(root / "protocol.json")
    if selection["protocol_sha256"] != digest or not selection["selected"]:
        raise ValueError("No valid frozen selection")
    for path, saved_hash in selection["source_reports"].items():
        if sha256(path) != saved_hash:
            raise ValueError("Frozen validation evidence changed")
    names = (protocol["plan"]["baseline"], selection["selected"])
    loaded, signatures = {}, {}
    for name in set(names):
        for seed in a.seeds:
            report = read(root / "evaluations/test" / f"{name}_{seed}/report.json")
            if report["split"] != "test" or report["seed"] != seed or report["name"] != name or report["protocol_sha256"] != digest:
                raise ValueError("Test report provenance mismatch")
            for key, path in report["training"].items():
                if sha256(path) != report["training_sha256"][key] or not read(path)["completed_steps"]:
                    raise ValueError("Incomplete or modified confirmation training")
            loaded[name, seed] = {}
            for key, path in report["files"].items():
                if sha256(path) != report["files_sha256"][key]:
                    raise ValueError("Test evidence changed")
                result = read(path)
                if result["split"] != "test":
                    raise ValueError("Mixed test/validation evidence")
                loaded[name, seed][key] = result
                sig = [result.get(k) for k in ("data_sha256", "queries_sha256", "corpus_sha256", "scope", "max_length", "candidates_sha256", "rerank_k")]
                if key in signatures and signatures[key] != sig:
                    raise ValueError("Confirmation data/scope differ across models or seeds")
                signatures[key] = sig
    metrics = {}
    for task in METRICS:
        pairs = [{lang: [loaded[name, seed][f"{task}_{lang}"]["per_query"] for name in names]
                  for lang in ("en", "zh")} for seed in a.seeds]
        metrics[task] = paired_task(pairs, task)
    # Descriptive retention and long-range results accompany uncertainty in task quality.
    auxiliary = {}
    for name in set(names):
        auxiliary[name] = {str(seed): {k: v["metrics"] for k, v in loaded[name, seed].items()
                                      if k.startswith(("retention_", "reference_")) or k == "diagnostic"} for seed in a.seeds}
    write_json(a.output, {"selected": selection["selected"], "baseline": names[0], "seeds": a.seeds,
        "metrics": metrics, "auxiliary": auxiliary, "selection_sha256": sha256(a.selection),
        "scope": "fixed prepared test candidate pools and sampled classification/NER tests",
        "interpretation": "Four task intervals are reported individually; no automatic all-tasks-better or publication claim"})
    print(metrics)


if __name__ == "__main__":
    main()
