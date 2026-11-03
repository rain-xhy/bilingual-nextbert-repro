"""Fail-closed multi-task architecture selection using validation results only."""
import argparse
import itertools
from pathlib import Path
from scripts.run_architecture import read, STAGES
from src.architecture import architecture_id
from src.common import sha256, write_json

METRICS = {"retrieval": "ndcg@10", "rerank": "ndcg@10", "classification": "accuracy", "ner": "entity_micro_f1"}


def summarize(report, protocol):
    plan = protocol["plan"]
    if report["split"] != "dev":
        raise ValueError("Test results must never drive architecture selection")
    cfg = protocol["configs"][report["name"]]
    files = {}
    for key, path in report["files"].items():
        if sha256(path) != report["files_sha256"][key]:
            raise ValueError("Evaluation result changed after report creation")
        files[key] = read(path)
        if files[key]["split"] != "dev":
            raise ValueError("A test result was placed in a validation report")
        if not key.startswith("reference_"):
            if files[key]["config"].get("seed", 42) != report["seed"] or architecture_id(files[key]["config"]) != architecture_id(cfg):
                raise ValueError("Evaluation checkpoint seed or architecture mismatch")
    train_tokens, steps, peaks = {}, {}, []
    for task in STAGES:
        path = report["training"][task]
        if sha256(path) != report["training_sha256"][task]:
            raise ValueError("Training summary changed")
        result = read(path)
        if not result["completed_steps"] or result["step"] != plan["stages"][task]["steps"]:
            raise ValueError("Incomplete or unequal-step run")
        if result["architecture_id"] != architecture_id(cfg):
            raise ValueError("Training architecture mismatch")
        if task == "mlm" and not result["ran_actual_8k"]:
            raise ValueError("A formal selected model requires actual 8K common adaptation")
        steps[task], train_tokens[task] = result["step"], result["input_tokens"]
        peaks.append(result["peak_allocated_gb"])
    scores, retention, reference, signatures = {}, {}, {}, {}
    eval_seconds = 0.
    for lang in ("en", "zh"):
        for task, metric in METRICS.items():
            key = f"{task}_{lang}"
            item = files[key]
            scores[key] = item["metrics"][metric]
            eval_seconds += item["seconds"]
            peaks.append(item["peak_allocated_gb"])
            if task in {"retrieval", "rerank"}:
                signatures[key] = [item[k] for k in ("queries_sha256", "corpus_sha256", "scope", "max_length")]
                if task == "rerank":
                    signatures[key] += [item["candidates_sha256"], item["rerank_k"]]
            else:
                signatures[key] = [item["data_sha256"], item["protocol"]]
        for task in ("classification", "ner"):
            reference[f"{task}_{lang}"] = files[f"reference_{task}_{lang}"]["metrics"][METRICS[task]]
            for branch in ("mlm", "retrieval"):
                key = f"retention_{branch}_{task}_{lang}"
                item = files[key]
                retention[key] = item["metrics"][METRICS[task]]
                ref = files[f"reference_{task}_{lang}"]
                if item["probe"] != ref["probe"] or item["data_sha256"] != ref["data_sha256"]:
                    raise ValueError("Retention and reference probes used different data or optimization")
                signatures[key] = [item["data_sha256"], item["probe"], ref["metrics"]]
    signatures["diagnostic"] = files["diagnostic"]["data_sha256"]
    diagnostic = {lang: files["diagnostic"]["metrics"][lang] for lang in ("en", "zh")}
    return {"scores": scores, "retention": retention, "reference": reference, "diagnostic": diagnostic,
            "signatures": signatures, "train_tokens": train_tokens, "steps": steps,
            "eval_seconds": eval_seconds, "peak_gb": max(peaks), "seed": report["seed"]}


def assess(candidate, baseline, policy):
    reasons = []
    if candidate["signatures"] != baseline["signatures"] or candidate["steps"] != baseline["steps"] \
            or candidate["train_tokens"] != baseline["train_tokens"]:
        raise ValueError("Comparisons require identical evaluation protocols and training exposure")
    deltas = {k: candidate["scores"][k] - baseline["scores"][k] for k in baseline["scores"]}
    for key, delta in deltas.items():
        if delta < -policy["max_task_drop"]:
            reasons.append(f"task regression: {key}")
    for key, score in candidate["retention"].items():
        reference_key = "_".join(key.split("_")[-2:])
        if score < candidate["reference"][reference_key] - policy["max_retention_drop"]:
            reasons.append(f"retention regression: {key}")
    for lang in ("en", "zh"):
        c, b = candidate["diagnostic"][lang], baseline["diagnostic"][lang]
        if c["position_range"] > b["position_range"] + policy["max_position_range_increase"]:
            reasons.append(f"position regression: {lang}")
        if c["evidence_margin"] < b["evidence_margin"] - policy["max_evidence_margin_drop"]:
            reasons.append(f"evidence regression: {lang}")
    if candidate["eval_seconds"] > baseline["eval_seconds"] * policy["max_eval_time_ratio"]:
        reasons.append("evaluation cost exceeds the predeclared ratio")
    if candidate["peak_gb"] > policy["max_peak_gb"]:
        reasons.append("measured memory exceeds the predeclared ceiling")
    return {"eligible": not reasons, "reasons": reasons, "deltas": deltas,
            "mean_gain": sum(deltas.values()) / len(deltas), "eval_seconds": candidate["eval_seconds"]}


def factorial_effects(configs, means):
    """Balanced 2^3 contrast estimates, descriptive only; no causal generalization or p-values."""
    factors = {}
    for name, cfg in configs.items():
        if name not in means:
            continue
        overrides = cfg["architecture_overrides"]
        bridge = cfg.get("document_bridge", {})
        if bridge and bridge.get("mode") != "global":
            continue
        factors[name] = (1 if overrides["local_attention"] == 256 else -1,
                         1 if overrides["global_attn_every_n_layers"] == 6 else -1,
                         1 if bridge else -1)
    if len(factors) != 8 or len(set(factors.values())) != 8:
        return {"status": "requires_complete_2x2x2_matrix"}
    labels = ("window", "global_interval", "bridge")
    result = {}
    for size in (1, 2, 3):
        for indices in itertools.combinations(range(3), size):
            positive, negative = [], []
            for name, values in factors.items():
                sign = 1
                for index in indices:
                    sign *= values[index]
                (positive if sign == 1 else negative).append(means[name])
            result[" x ".join(labels[i] for i in indices)] = sum(positive)/len(positive) - sum(negative)/len(negative)
    return {"status": "descriptive_factorial_contrasts", "effects_on_equal_task_language_mean": result}


def select(root, seeds):
    root = Path(root)
    protocol = read(root / "protocol.json")
    plan, digest = protocol["plan"], sha256(root / "protocol.json")
    if len(set(seeds)) != len(seeds) or len(seeds) < plan["selection"]["required_seeds"]:
        raise ValueError("Insufficient distinct training seeds")
    runs, sources = {}, {}
    for name in plan["names"]:
        runs[name] = {}
        for seed in seeds:
            path = root / "evaluations/dev" / f"{name}_{seed}" / "report.json"
            report = read(path)
            if report["protocol_sha256"] != digest or report["name"] != name or report["seed"] != seed:
                raise ValueError("Run report provenance mismatch")
            runs[name][seed] = summarize(report, protocol)
            sources[str(path)] = sha256(path)
    baseline = plan["baseline"]
    assessments, means = {}, {}
    for name in plan["names"]:
        entries = [assess(runs[name][s], runs[baseline][s], plan["selection"]) for s in seeds]
        assessments[name] = {"eligible": all(e["eligible"] for e in entries), "by_seed": entries,
            "mean_gain": sum(e["mean_gain"] for e in entries)/len(entries),
            "eval_seconds": sum(e["eval_seconds"] for e in entries)/len(entries)}
        means[name] = sum(sum(runs[name][s]["scores"].values())/8 for s in seeds)/len(seeds)
    candidates = [name for name, a in assessments.items() if a["eligible"] and name != baseline
                  and a["mean_gain"] >= plan["selection"]["minimum_mean_gain"]]
    selected = max(candidates, key=lambda n: (assessments[n]["mean_gain"], -assessments[n]["eval_seconds"])) if candidates else \
        (baseline if assessments[baseline]["eligible"] else None)
    return {"selected": selected, "status": "no_admissible_model" if selected is None else
            ("baseline_retained" if selected == baseline else "validation_candidate_selected"),
            "seeds": seeds, "assessments": assessments, "protocol_sha256": digest,
            "source_reports": sources, "selected_config": protocol["configs"].get(selected),
            "selected_checkpoints": {str(s): {task: str(root / "models" / f"{selected}_{s}" / task / "checkpoint")
                                     for task in STAGES} for s in seeds} if selected else {},
            "factorial": factorial_effects(protocol["configs"], means),
            "claim": "Validation selection among measured architectures; not a test-set or global-best claim"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/architecture")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--output", default="runs/architecture/selection.json")
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError("Selection is immutable; use a new output path")
    result = select(a.root, a.seeds)
    write_json(a.output, result)
    print(result["status"], result["selected"])


if __name__ == "__main__":
    main()
