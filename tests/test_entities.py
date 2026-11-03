import unittest
from src.probe import entities


class EntityTests(unittest.TestCase):
    def test_types_and_boundaries(self):
        self.assertEqual(entities(["B-PER", "I-PER", "O", "B-ORG", "I-ORG"]),
                         {("PER", 0, 2), ("ORG", 3, 5)})

    def test_adjacent_same_type_entities(self):
        self.assertEqual(entities(["B-PER", "B-PER"]), {("PER", 0, 1), ("PER", 1, 2)})

    def test_illegal_inside_tag_starts_entity(self):
        self.assertEqual(entities(["I-LOC", "O"]), {("LOC", 0, 1)})
