import unittest
from src.metrics import evaluate, paired_bootstrap


class MetricTests(unittest.TestCase):
    def test_missing_query_is_zero(self):
        mean, _ = evaluate({"a": {"d": 1}, "b": {"e": 1}}, {"a": ["d"]})
        self.assertEqual(mean["ndcg@10"], .5)

    def test_duplicate_document_not_double_counted(self):
        mean, _ = evaluate({"a": {"d": 1, "e": 1}}, {"a": ["d", "d", "e"]}, 2)
        self.assertEqual(mean["recall@2"], 1)
        self.assertEqual(mean["ndcg@2"], 1)

    def test_bootstrap_requires_same_queries(self):
        with self.assertRaises(ValueError):
            paired_bootstrap({"a": {"ndcg@10": 1}}, {"b": {"ndcg@10": 1}})

    def test_identical_runs_have_zero_delta(self):
        scores = {str(i): {"ndcg@10": i / 10} for i in range(10)}
        self.assertEqual(paired_bootstrap(scores, scores)["ci95"], [0, 0])


if __name__ == "__main__":
    unittest.main()
