"""Offline end-to-end architecture protocol test with tiny random weights and synthetic labels.

Uses the real trainer/evaluators in-process to avoid repeated Python imports on Windows.
Includes an actual 8192-token MLM step for both baseline and bridge variants.
"""
import contextlib
import gc
import importlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast, ModernBertConfig, ModernBertForMaskedLM
from src.common import write_json, write_jsonl, sha256
from scripts.run_architecture import freeze_protocol, train_one, evaluate_one
from scripts.select_architecture import select
from src.modeling import Encoder


def fixture(root):
    vocabulary = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "China", "capital", "Beijing", "France", "Paris", "river", "water", "中国", "首都", "北京"])}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    base = root / "tiny"
    tokenizer.save_pretrained(base)
    config = ModernBertConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8192, local_attention=128,
        global_attn_every_n_layers=3, pad_token_id=0, cls_token_id=2, sep_token_id=3, reference_compile=False)
    ModernBertForMaskedLM(config).save_pretrained(base)
    data = root / "data"
    manifest = {"model": str(base), "revision": None, "files": {},
        "labels": {"classification": ["entailment", "neutral", "contradiction"],
                   "ner": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]}}
    def save(name, rows):
        write_jsonl(data / name, rows)
        manifest["files"][name] = sha256(data / name)
    docs = [{"docid": "a", "text": "China capital Beijing 中国 首都 北京"},
            {"docid": "b", "text": "France capital Paris river water"}]
    retrieval = [{"query_id": f"{l}:q", "language": l, "query": "China capital" if l == "en" else "中国 首都",
                  "positive_passages": [docs[0]], "negative_passages": [docs[1]]} for l in ("en", "zh")]
    save("retrieval_train.jsonl", retrieval)
    save("mlm_train.jsonl", [{"id": l, "language": l, "input_ids": [2] + [5, 6, 7, 10, 11]*1638 + [3]} for l in ("en", "zh")])
    for task in ("classification", "ner"):
        examples = [{"id": f"{task}:{l}", "language": l, "input_ids": [2, 5, 6, 7, 3],
                     "labels": 0 if task == "classification" else [-100, 5, 0, 5, -100]} for l in ("en", "zh")]
        save(f"{task}_train.jsonl", examples)
        for split in ("dev", "test"):
            for row in examples:
                save(f"{task}_{row['language']}_{split}.jsonl", [dict(row, id=row["id"]+split)])
    for split in ("dev", "test"):
        for row in retrieval:
            save(f"{row['language']}_{split}_queries.jsonl", [row])
            save(f"{row['language']}_{split}_pool.jsonl", docs)
    write_json(data / "manifest.json", manifest)
    diagnostic = root / "diagnostic"
    diagnostic_manifest = {"model": str(base), "revision": None, "files": {}}
    for split in ("dev", "test"):
        rows = [{"id": l, "group_id": l, "language": l, "query_ids": [5, 6], "evidence_ids": [5, 6, 7],
                 "filler_ids": [10, 11]*32, "replacement_ids": [8, 9, 10], "total_length": 64} for l in ("en", "zh")]
        write_jsonl(diagnostic / f"{split}.jsonl", rows)
        diagnostic_manifest["files"][f"{split}.jsonl"] = sha256(diagnostic / f"{split}.jsonl")
    write_json(diagnostic / "manifest.json", diagnostic_manifest)
    config_root = root / "configs"
    for name in ("base", "bridge"):
        cfg = {"model": str(base), "revision": None, "task": "mlm", "seed": 42, "lora_rank": 2,
            "architecture_name": name, "architecture_overrides": {"local_attention": 128, "global_attn_every_n_layers": 3},
            "document_bridge": {"chunk_size": 256, "bottleneck": 8, "heads": 2, "mode": "global"} if name == "bridge" else {},
            "max_length": 8192, "mlm_logit_chunk": 256, "gradient_checkpointing": True, "accumulation": 1}
        write_json(config_root / f"{name}.json", cfg)
    plan = {"version": "synthetic-smoke-only", "names": ["base", "bridge"], "baseline": "base", "seeds": [42],
        "config_root": str(config_root), "data_root": str(data), "diagnostic_root": str(diagnostic),
        "stages": {task: {"steps": 2, "max_hours": .1} for task in ("mlm", "retrieval", "rerank", "classification", "ner")},
        "rerank_k": 10, "probe_steps": 2, "probe_size_per_language": 1,
        "selection": {"max_task_drop": 1., "max_retention_drop": 1., "max_position_range_increase": 1.,
            "max_evidence_margin_drop": 1., "max_eval_time_ratio": 100., "max_peak_gb": 30.,
            "minimum_mean_gain": 0., "required_seeds": 1}}
    write_json(config_root / "plan.json", plan)
    return plan, config_root / "plan.json"


def in_process(module, *args):
    with patch.object(sys, "argv", [module, *map(str, args)]):
        importlib.import_module(module).main()
    gc.collect()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    captured = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="architecture-smoke-") as tmp:
        root = Path(tmp)
        plan, plan_path = fixture(root)
        # Inspect actual constructed layers, not just a JSON configuration field.
        architecture_cfg = json.loads((Path(plan["config_root"]) / "base.json").read_text(encoding="utf-8"))
        architecture_cfg.update(task="retrieval", architecture_overrides={"local_attention": 32, "global_attn_every_n_layers": 2})
        constructed = Encoder(architecture_cfg)
        assert constructed.backbone.get_base_model().layers[1].attn.local_attention == (16, 16)
        architecture_cfg["architecture_overrides"]["global_attn_every_n_layers"] = 1
        constructed = Encoder(architecture_cfg)
        assert all(layer.attn.local_attention == (-1, -1) for layer in constructed.backbone.get_base_model().layers)
        del constructed
        experiment = root / "experiment"
        protocol = freeze_protocol(experiment, plan_path, plan)
        with patch("scripts.run_architecture.call", side_effect=in_process), contextlib.redirect_stdout(captured):
            for name in plan["names"]:
                train_one(experiment, plan, protocol["configs"][name], name, 42, allow_cpu=True)
                evaluate_one(experiment, plan, name, 42, "dev")
            chosen = select(experiment, [42])
            assert chosen["selected"] in plan["names"]
            selection_path = experiment / "selection.json"
            write_json(selection_path, chosen)
            # Test is reached only after the validation decision has been written.
            selected_names = list(dict.fromkeys([plan["baseline"], chosen["selected"]]))
            in_process("scripts.run_architecture", "--phase", "evaluate", "--plan", plan_path,
                "--root", experiment, "--split", "test", "--selection", selection_path, "--names", *selected_names)
        summaries = {}
        for name in plan["names"]:
            path = experiment / "models" / f"{name}_42/mlm/summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            assert summary["ran_actual_8k"] and summary["completed_steps"]
            summaries[name] = {k: summary[k] for k in ("step", "max_actual_length", "grad_norm", "seconds")}
        # The structural state must survive reload independently of task-head state.
        assert (experiment / "models/bridge_42/ner/checkpoint/architecture.pt").is_file()
        checkpoint = experiment / "models/bridge_42/ner/checkpoint"
        restored = Encoder(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
        restored.load_transfer(checkpoint, load_head=True)
        assert restored.bridge.gate.detach().abs().sum() > 0
        write_json("artifacts/architecture_smoke.json", {"status": "passed", "purpose": "synthetic tiny random model only",
            "tasks": ["8192_token_mlm", "retrieval", "rerank", "classification", "ner", "frozen_retention", "diagnostic", "selection", "frozen_test_gate", "actual_attention_layout", "bridge_checkpoint_reload"],
            "variants": list(summaries), "actual_mlm_checks": summaries,
            "not_verified": "307M target-GPU capacity, real-task quality or publication novelty"})
    print("Architecture protocol smoke passed, including baseline/bridge 8192-token MLM backward and four tasks")


if __name__ == "__main__":
    main()
