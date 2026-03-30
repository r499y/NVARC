import unittest
import numpy as np

from arc_decoder import ArcDecoder, SelectionConfig, score_kgmon
from pseudo_loo_reranker import rerank_candidates
from structural_constraints import infer_train_invariants, check_candidate_constraints


class DummyDataset:
    def __init__(self):
        self.queries = {
            "task_0": {
                "train": [
                    {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                    {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                    {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                ],
                "test": [
                    {"input": [[0, 0], [0, 0]]},
                ],
            }
        }


class PseudoRerankTests(unittest.TestCase):
    def test_rerank_penalizes_structural_issues(self):
        candidates = [
            {"solution": np.array([[12, 12], [12, 12]]), "base_score": 10.0},
            {"solution": np.array([[1, 1], [1, 1]]), "base_score": 1.0},
        ]
        train_pairs = DummyDataset().queries["task_0"]["train"]
        test_input = DummyDataset().queries["task_0"]["test"][0]["input"]

        scored, confidence = rerank_candidates(candidates, train_pairs, test_input)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertTrue(np.array_equal(scored[0].solution, np.array([[1, 1], [1, 1]])))

    def test_constraint_checker_rejects_fixed_shape_violation_with_reason(self):
        train_pairs = DummyDataset().queries["task_0"]["train"]
        invariants = infer_train_invariants(train_pairs)

        logs = []
        result = check_candidate_constraints(
            candidate=np.array([[1, 1, 1], [1, 1, 1]]),
            test_input=np.array([[0, 0], [0, 0]]),
            invariants=invariants,
            logger=logs.append,
        )

        self.assertFalse(result.is_valid)
        self.assertIn("violates_fixed_output_shape", result.rejection_reasons)
        self.assertTrue(any("constraint-reject" in line for line in logs))

    def test_decoder_flag_off_keeps_existing_base_order(self):
        decoder = ArcDecoder(DummyDataset(), n_guesses=2, selection_config=SelectionConfig(enable_pseudo_test_rerank=False))
        decoder.decoded_results = {
            "task_0": {
                "a": {"solution": np.array([[2, 2], [2, 2]]), "beam_score": 0.0, "score_aug": [0.0]},
                "b": {"solution": np.array([[1, 1], [1, 1]]), "beam_score": 1.0, "score_aug": [1.0]},
            }
        }

        ranked = decoder.run_selection_algo(score_kgmon)
        self.assertTrue(np.array_equal(ranked["task_0"][0], np.array([[2, 2], [2, 2]])))

    def test_decoder_flag_on_returns_confidence_metadata(self):
        decoder = ArcDecoder(DummyDataset(), n_guesses=2, selection_config=SelectionConfig(enable_pseudo_test_rerank=True))
        decoder.decoded_results = {
            "task_0": {
                "a": {"solution": np.array([[12, 12], [12, 12]]), "beam_score": 0.0, "score_aug": [0.0]},
                "b": {"solution": np.array([[1, 1], [1, 1]]), "beam_score": 1.0, "score_aug": [1.0]},
            }
        }

        meta = decoder.run_selection_algo_with_metadata(score_kgmon)
        self.assertIn("confidence", meta["task_0"])
        self.assertIn("ranked_candidates", meta["task_0"])
        self.assertGreaterEqual(meta["task_0"]["confidence"], 0.0)
        self.assertLessEqual(meta["task_0"]["confidence"], 1.0)
        self.assertTrue(np.array_equal(meta["task_0"]["ranked_candidates"][0], np.array([[1, 1], [1, 1]])))


if __name__ == "__main__":
    unittest.main()
