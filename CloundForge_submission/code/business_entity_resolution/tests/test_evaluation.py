"""Unit tests for src/evaluation.py (stdlib unittest). Run from code/business_entity_resolution/:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation import evaluate, f_beta, main, parse_id_list, read_id_list_tsv  # noqa: E402


class FBetaTest(unittest.TestCase):
    def test_official_worked_example(self):
        # README: predict {47, 193, 812}, truth {47, 812} -> P=2/3, R=1 -> 0.714
        self.assertAlmostEqual(f_beta(3, 2, 2), 0.7142857, places=6)

    def test_singleton_rules(self):
        self.assertEqual(f_beta(0, 0, 0), 1.0)  # correct "no match"
        self.assertEqual(f_beta(1, 0, 0), 0.0)  # any prediction on a singleton

    def test_zero_cases(self):
        self.assertEqual(f_beta(0, 2, 0), 0.0)  # missed everything
        self.assertEqual(f_beta(2, 2, 0), 0.0)  # all predictions wrong

    def test_precision_weighted_more_than_recall(self):
        half_recall = f_beta(1, 2, 1)   # P=1, R=0.5
        half_precision = f_beta(2, 1, 1)  # P=0.5, R=1
        self.assertAlmostEqual(half_recall, 0.8333333, places=6)
        self.assertAlmostEqual(half_precision, 0.5555556, places=6)
        self.assertGreater(half_recall, half_precision)

    def test_perfect(self):
        self.assertEqual(f_beta(3, 3, 3), 1.0)


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.truth = {
            "S1-1": {"S2-47", "S3-812"},
            "S1-2": {"S3-4"},
            "S1-3": set(),
            "S1-4": set(),
        }

    def test_macro_average_over_all_entities(self):
        pred = {"S1-1": {"S2-47", "S2-193", "S3-812"}, "S1-2": {"S3-4"}, "S1-3": set(), "S1-4": {"S2-9"}}
        r = evaluate(pred, self.truth)
        expected = (0.7142857142857143 + 1.0 + 1.0 + 0.0) / 4
        self.assertAlmostEqual(r["macro_f0.5"], expected)
        self.assertEqual(r["singletons"]["count"], 2)
        self.assertEqual(r["singletons"]["predicted_empty"], 1)
        self.assertEqual(r["exact_set_match_rate"], 0.5)
        self.assertAlmostEqual(r["micro_precision"], 3 / 5)
        self.assertAlmostEqual(r["micro_recall"], 3 / 3)

    def test_missing_prediction_counts_as_empty(self):
        r = evaluate({"S1-1": {"S2-47", "S3-812"}}, self.truth)
        self.assertAlmostEqual(r["macro_f0.5"], (1.0 + 0.0 + 1.0 + 1.0) / 4)
        self.assertTrue(any("no prediction row" in w for w in r["warnings"]))

    def test_only_predicted_restricts_evaluation(self):
        r = evaluate({"S1-2": {"S3-4"}, "S1-3": {"S2-1"}}, self.truth, only_predicted=True)
        self.assertEqual(r["n_entities"], 2)
        self.assertAlmostEqual(r["macro_f0.5"], 0.5)

    def test_all_empty_baseline_equals_singleton_share(self):
        r = evaluate({}, self.truth)
        self.assertAlmostEqual(r["macro_f0.5"], 0.5)

    def test_candidate_diagnostics(self):
        cands = {"S1-1": {"S2-47", "S2-193"}, "S1-2": {"S3-4", "S3-5"}, "S1-3": set(), "S1-4": {"S2-9"}}
        pred = {"S1-1": {"S2-47"}, "S1-2": {"S3-4"}, "S1-3": set(), "S1-4": set()}
        c = evaluate(pred, self.truth, cands)["candidates"]
        self.assertAlmostEqual(c["pair_recall_ceiling"], 2 / 3)
        # Oracle: S1-1 keeps 1 of 2 (P=1, R=0.5 -> 0.8333), the others are perfect.
        self.assertAlmostEqual(c["oracle_macro_f0.5"], (0.8333333333333334 + 3) / 4)
        self.assertEqual(c["total_candidate_pairs"], 5)
        self.assertEqual(c["max_per_s1"], 2)
        self.assertEqual(c["s1_with_no_candidates"], 1)

    def test_prediction_outside_candidates_warns(self):
        r = evaluate({"S1-2": {"S3-99"}}, self.truth, {"S1-2": {"S3-4"}})
        self.assertTrue(any("not among their candidates" in w for w in r["warnings"]))


class ParsingTest(unittest.TestCase):
    def test_parse_id_list(self):
        self.assertEqual(parse_id_list(""), [])
        self.assertEqual(parse_id_list(' "S2-1, S3-2," '), ["S2-1", "S3-2"])

    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8", newline="")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_read_tsv_with_bom_crlf_and_empty_rows(self):
        path = self._write("﻿source1_entity_id\tmatched_entity_ids\r\nS1-1\tS2-1,S3-2\r\nS1-2\t\r\n\r\n")
        warnings = []
        self.assertEqual(read_id_list_tsv(path, warnings, "t"), {"S1-1": {"S2-1", "S3-2"}, "S1-2": set()})
        self.assertEqual(warnings, [])

    def test_read_tsv_flags_problems(self):
        path = self._write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-1\nS1-1\tS3-2\n")
        warnings = []
        mapping = read_id_list_tsv(path, warnings, "t")
        self.assertEqual(mapping, {"S1-1": {"S2-1", "S3-2"}})
        self.assertEqual(len(warnings), 2)

    def test_cli_end_to_end(self):
        gt = self._write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
        pr = self._write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n")
        self.assertEqual(main(["-p", pr, "-g", gt, "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
