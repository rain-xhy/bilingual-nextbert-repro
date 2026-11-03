import copy
import json
from pathlib import Path
import unittest
import torch
from src.architecture import DocumentBridge, architecture_id
from scripts.select_architecture import assess, factorial_effects, summarize
from scripts.confirm_architecture import score, paired_task


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(10)

    def test_bridge_initially_preserves_checkpoint_function(self):
        module = DocumentBridge(16, chunk_size=4, bottleneck=8, heads=2)
        x = torch.randn(2, 9, 16, requires_grad=True)
        mask = torch.tensor([[1]*9, [1]*5 + [0]*4])
        y = module(x, mask)
        self.assertTrue(torch.equal(x, y))
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(module.gate.grad).all())
        self.assertGreater(module.gate.grad.abs().sum().item(), 0)

    def test_padding_and_empty_rows_do_not_pollute_valid_tokens(self):
        module = DocumentBridge(16, 4, 8, 2)
        module.gate.data.fill_(.3)
        x = torch.randn(2, 9, 16)
        mask = torch.tensor([[1]*5 + [0]*4, [0]*9])
        changed = x.clone()
        changed[~mask.bool()] = 1000 * torch.randn_like(changed[~mask.bool()])
        a, b = module(x, mask), module(changed, mask)
        self.assertTrue(torch.allclose(a[0, :5], b[0, :5], atol=1e-6))
        self.assertTrue(torch.isfinite(a).all())

    def test_global_bridge_communicates_across_chunks_local_control_does_not(self):
        global_block = DocumentBridge(16, 4, 8, 2)
        global_block.gate.data.fill_(.5)
        local = copy.deepcopy(global_block)
        local.mode = "local"
        x = torch.randn(1, 8, 16)
        changed = x.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:])
        mask = torch.ones(1, 8)
        self.assertFalse(torch.allclose(global_block(x, mask)[:, :4], global_block(changed, mask)[:, :4]))
        self.assertTrue(torch.allclose(local(x, mask)[:, :4], local(changed, mask)[:, :4]))
        self.assertEqual(sum(p.numel() for p in local.parameters()), sum(p.numel() for p in global_block.parameters()))

    def test_architecture_identity_tracks_bridge_not_task(self):
        cfg = {"architecture_overrides": {"local_attention": 128}, "task": "mlm"}
        self.assertEqual(architecture_id(cfg), architecture_id(dict(cfg, task="ner")))
        self.assertNotEqual(architecture_id(cfg), architecture_id(dict(cfg, document_bridge={"mode": "global"})))

    def test_factorial_effects_separate_interaction_from_main_effect(self):
        configs, means = {}, {}
        for w in (128, 256):
            for g in (3, 6):
                for b in (0, 1):
                    key = f"{w}_{g}_{b}"
                    configs[key] = {"architecture_overrides": {"local_attention": w, "global_attn_every_n_layers": g},
                                    "document_bridge": {"mode": "global"} if b else {}}
                    means[key] = .5 + .1*(1 if w == 256 else -1)*(1 if b else -1)
        result = factorial_effects(configs, means)["effects_on_equal_task_language_mean"]
        self.assertAlmostEqual(result["window"], 0)
        self.assertAlmostEqual(result["window x bridge"], .2)

    def test_selector_rejects_quality_gain_with_retention_regression(self):
        policy = {"max_task_drop": .01, "max_retention_drop": .01, "max_position_range_increase": .01,
                  "max_evidence_margin_drop": .01, "max_eval_time_ratio": 2, "max_peak_gb": 30}
        baseline = {"signatures": {}, "steps": {}, "train_tokens": {}, "scores": {"retrieval_en": .5},
                    "retention": {"retention_mlm_classification_en": .8}, "reference": {"classification_en": .8},
                    "diagnostic": {l: {"position_range": .1, "evidence_margin": .1} for l in ("en", "zh")},
                    "eval_seconds": 1, "peak_gb": 2}
        candidate = copy.deepcopy(baseline)
        candidate["scores"]["retrieval_en"] = .9
        candidate["retention"]["retention_mlm_classification_en"] = .6
        self.assertFalse(assess(candidate, baseline, policy)["eligible"])
        candidate["train_tokens"] = {"mlm": 1}
        with self.assertRaises(ValueError):
            assess(candidate, baseline, policy)

    def test_declared_primary_plan_is_a_complete_structural_factorial(self):
        plan = json.loads(Path("configs/architecture/plan.json").read_text(encoding="utf-8"))
        self.assertEqual(len(plan["names"]), 8)
        configs = [json.loads(Path(plan["config_root"], f"{name}.json").read_text(encoding="utf-8")) for name in plan["names"]]
        combinations = {(c["architecture_overrides"]["local_attention"],
                         c["architecture_overrides"]["global_attn_every_n_layers"], bool(c["document_bridge"])) for c in configs}
        self.assertEqual(len(combinations), 8)
        self.assertEqual(set(plan["stages"]), {"mlm", "retrieval", "rerank", "classification", "ner"})
        self.assertEqual(len({c["revision"] for c in configs}), 1)

    def test_confirmation_ner_recomputes_micro_counts(self):
        records = [{"tp": 1, "fp": 0, "fn": 0}, {"tp": 0, "fp": 0, "fn": 9}]
        self.assertAlmostEqual(score(records, "ner"), 2/11)
        data = {str(i): r for i, r in enumerate(records)}
        pairs = [{lang: [data, data] for lang in ("en", "zh")} for _ in range(3)]
        self.assertEqual(paired_task(pairs, "ner", repeats=100)["ci95"], [0., 0.])

    def test_confirmation_rejects_nonmatching_query_sets(self):
        pair = {lang: [{"a": {"correct": 1}}, {"b": {"correct": 1}}] for lang in ("en", "zh")}
        with self.assertRaises(ValueError):
            paired_task([pair], "classification", repeats=100)

    def test_test_results_cannot_enter_selection(self):
        with self.assertRaisesRegex(ValueError, "Test results"):
            summarize({"split": "test"}, {"plan": {}})


if __name__ == "__main__":
    unittest.main()
