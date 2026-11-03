import random
import unittest
import torch
from src.modeling import mask_tokens


class FakeTokenizer:
    all_special_ids = [0, 1, 2, 3]
    mask_token_id = 3

    def __len__(self):
        return 40


class MaskingTests(unittest.TestCase):
    def test_same_mask_count_and_protected_tokens(self):
        batch = {"input_ids": torch.tensor([[1, *range(4, 24), 2, 0]]),
                 "attention_mask": torch.tensor([[1] * 22 + [0]])}
        for mode in ("token", "span"):
            result = mask_tokens(batch, FakeTokenizer(), mode, .15, random.Random(42))
            self.assertEqual(int((result["labels"] != -100).sum()), 3)
            self.assertEqual(result["labels"][0, [0, 21, 22]].tolist(), [-100] * 3)
            indices = result["labels"] != -100
            self.assertTrue(torch.equal(result["labels"][indices], batch["input_ids"][indices]))

    def test_no_maskable_text_raises(self):
        with self.assertRaises(ValueError):
            mask_tokens({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)},
                        FakeTokenizer(), "span", .15, random.Random(42))
