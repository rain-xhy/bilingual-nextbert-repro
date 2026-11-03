"""Architecture-first orchestration. Every command should be wrapped by scripts.budget_run."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from src.common import write_json, sha256

STAGES = ("mlm", "retrieval", "rerank", "classification", "ner")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def call(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


def freeze_protocol(root, plan_path, plan):
    path = root / "protocol.json"
    data = Path(plan["data_root"])
    diagnostic = Path(plan["diagnostic_root"])
    for folder in (data, diagnostic):
        manifest = read(folder / "manifest.json")
        for name, digest in manifest["files"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Prepared data changed after manifest creation")
    protocol = {"plan": plan, "plan_sha256": sha256(plan_path),
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "diagnostic_manifest_sha256": sha256(diagnostic / "manifest.json"),
        "configs": {n: read(Path(plan["config_root"]) / f"{n}.json") for n in plan["names"]}}
    if path.exists():
        if read(path) != protocol:
            raise ValueError("Protocol changed after experiment initialization; use a new experiment root")
    else:
        write_json(path, protocol)
    return protocol


def train_one(root, plan, cfg, name, seed, allow_cpu=False, capacity=False):
    data = Path(plan["data_root"])
    manifest = read(data / "manifest.json")
    destination = root / ("capacity" if capacity else "models") / f"{name}_{seed}"
    for task in (("mlm",) if capacity else STAGES):
        path = destination / task
        stage = dict(cfg, **plan["stages"][task], task=task, seed=seed)
        stage["data"] = str(data / ("retrieval_train.jsonl" if task == "rerank" else f"{task}_train.jsonl"))
        stage["max_length"] = 512 if task in {"classification", "ner"} else cfg.get("max_length", 8192)
        if task in manifest["labels"]:
            stage["num_labels"] = len(manifest["labels"][task])
        if capacity:
            stage.update(steps=4, max_hours=.25)
        config_path = destination / "configs" / f"{task}.json"
        if (path / "summary.json").exists():
            if read(config_path) != stage or not read(path / "summary.json")["completed_steps"]:
                raise ValueError("Existing stage has a different config or did not complete its step budget")
            continue
        write_json(config_path, stage)
        args = ["--config", config_path, "--output", path]
        if task != "mlm":
            args += ["--init-checkpoint", destination / "mlm/checkpoint"]
        if allow_cpu:
            args += ["--allow-cpu"]
        call("src.train_tasks", *args)
        result = read(path / "summary.json")
        if not result["completed_steps"]:
            raise RuntimeError("Stage hit its time cap. Incomplete runs cannot enter structural selection")
        if task == "mlm" and not allow_cpu and not result["ran_actual_8k"]:
            raise RuntimeError("Common adaptation did not execute 8K content")


def evaluate_one(root, plan, name, seed, split):
    data, diagnostic = Path(plan["data_root"]), Path(plan["diagnostic_root"])
    model_root = root / "models" / f"{name}_{seed}"
    results = root / "evaluations" / split / f"{name}_{seed}"
    baseline_cfg = Path(plan["config_root"]) / f"{plan['baseline']}.json"
    def execute(output, module, *args):
        if not output.exists():
            call(module, *args, "--output", output)
        return str(output)
    files, training = {}, {}
    for task in STAGES:
        training[task] = str(model_root / task / "summary.json")
        if not read(training[task])["completed_steps"]:
            raise ValueError("Cannot evaluate an incomplete task branch for model selection")
    for language in ("en", "zh"):
        queries = data / f"{language}_{split}_queries.jsonl"
        corpus = data / f"{language}_{split}_pool.jsonl"
        fixed = root / "candidates" / f"{language}_{split}.json"
        execute(fixed, "src.bm25", "--queries", queries, "--corpus", corpus, "--scope", "candidate_pool")
        for task in ("retrieval", "rerank"):
            args = ["--checkpoint", model_root / task / "checkpoint", "--queries", queries,
                    "--corpus", corpus, "--scope", "candidate_pool", "--split", split,
                    "--max-length", read(model_root / task / "checkpoint/config.json")["max_length"]]
            if task == "rerank":
                args += ["--candidates", fixed, "--rerank-k", plan["rerank_k"]]
            files[f"{task}_{language}"] = execute(results / f"{task}_{language}.json", "src.evaluate", *args)
        for task in ("classification", "ner"):
            path = data / f"{task}_{language}_{split}.jsonl"
            args = ["--task", task, "--data", path, "--split", split]
            files[f"{task}_{language}"] = execute(results / f"{task}_{language}.json", "src.evaluate_tasks",
                "--checkpoint", model_root / task / "checkpoint", *args)
            probe_args = [*args, "--probe-train", data / f"{task}_train.jsonl", "--probe-steps", plan["probe_steps"],
                          "--probe-size", plan["probe_size_per_language"], "--probe-seed", 42]
            for branch in ("mlm", "retrieval"):
                key = f"retention_{branch}_{task}_{language}"
                files[key] = execute(results / f"{key}.json", "src.evaluate_tasks", "--checkpoint",
                                     model_root / branch / "checkpoint", *probe_args)
            key = f"reference_{task}_{language}"
            files[key] = execute(root / "references" / split / f"{task}_{language}.json", "src.evaluate_tasks",
                                  "--base-config", baseline_cfg, *probe_args)
    files["diagnostic"] = execute(results / "diagnostic.json", "src.evaluate_evidence", "--checkpoint",
                                   model_root / "retrieval/checkpoint", "--data", diagnostic / f"{split}.jsonl", "--split", split)
    report = {"name": name, "seed": seed, "split": split, "files": files, "training": training,
              "protocol_sha256": sha256(root / "protocol.json"),
              "files_sha256": {k: sha256(v) for k, v in files.items()},
              "training_sha256": {k: sha256(v) for k, v in training.items()}}
    write_json(results / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["capacity", "train", "evaluate"], required=True)
    p.add_argument("--plan", default="configs/architecture/plan.json")
    p.add_argument("--root", default="runs/architecture")
    p.add_argument("--names", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--selection", help="Frozen dev selection required before test evaluation")
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()
    plan, root = read(a.plan), Path(a.root)
    protocol = freeze_protocol(root, a.plan, plan)
    names, seeds = a.names or plan["names"], a.seeds or plan["seeds"]
    if len(set(names)) != len(names) or len(set(seeds)) != len(seeds) or not set(names) <= set(plan["names"]):
        raise ValueError("Use unique declared architecture names and seeds")
    if a.phase == "evaluate" and a.split == "test":
        if not a.selection:
            raise ValueError("Freeze validation-based selection before accessing test results")
        selection = read(a.selection)
        if selection["protocol_sha256"] != sha256(root / "protocol.json") or not selection.get("selected"):
            raise ValueError("Selection missing or from a different protocol")
        if not set(names) <= {plan["baseline"], selection["selected"]}:
            raise ValueError("Test only the frozen selected architecture and declared baseline")
        for path, digest in selection["source_reports"].items():
            if sha256(path) != digest:
                raise ValueError("Selection evidence changed after freezing")
    if a.phase == "capacity":
        # The most expensive declared combination by default.
        name = a.names[0] if a.names else "w256_g3_b1"
        train_one(root, plan, protocol["configs"][name], name, seeds[0], a.allow_cpu, capacity=True)
        return
    for name in names:
        for seed in seeds:
            if a.phase == "train":
                train_one(root, plan, protocol["configs"][name], name, seed, a.allow_cpu)
            else:
                evaluate_one(root, plan, name, seed, a.split)


if __name__ == "__main__":
    main()
