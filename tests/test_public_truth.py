import unittest

import pandas as pd

from tests.public_truth import check_submission


class CheckSubmissionTest(unittest.TestCase):
    """Ревью 26.10, R8: неполный ответ с дублями id не должен получать метрику."""

    truth = pd.DataFrame({"id": [1, 2], "attack_cat": ["A", "B"]})

    def test_duplicate_replaces_missing_id(self):
        pred = pd.DataFrame({"id": [1, 1], "attack_cat": ["A", "A"]})
        problems = check_submission(self.truth, pred, "attack_cat")
        self.assertTrue(any("дубли" in p for p in problems))
        self.assertTrue(any("нет в ответе: 1" in p for p in problems))

    def test_missing_values_and_columns(self):
        self.assertTrue(check_submission(self.truth, pd.DataFrame({"id": [1, 2], "attack_cat": ["A", None]}), "attack_cat"))
        self.assertTrue(check_submission(self.truth, pd.DataFrame({"id": [1, 2]}), "attack_cat"))

    def test_full_answer_in_other_order(self):
        pred = pd.DataFrame({"id": [2, 1], "attack_cat": ["B", "A"]})
        self.assertEqual(check_submission(self.truth, pred, "attack_cat"), [])


if __name__ == "__main__":
    unittest.main()
