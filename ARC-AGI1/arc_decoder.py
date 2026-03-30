import os
import bz2
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

from pseudo_loo_reranker import rerank_candidates


def hashable(guess):
    return tuple(map(tuple, guess))


def score_sum(guesses, getter):
    scores = _group_scores(guesses, getter)
    return [x["solution"] for x in scores]


def _group_scores(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)

    grouped = []
    for grouped_guesses, solution in scores.values():
        grouped.append(
            {
                "base_score": float(getter(grouped_guesses)),
                "solution": solution,
                "group_size": len(grouped_guesses),
                "raw_guesses": grouped_guesses,
            }
        )
    grouped = sorted(grouped, key=(lambda x: x["base_score"]), reverse=True)
    return grouped


def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline - g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score


def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]

algorithm_to_getter = {
    score_full_probmul_3.__name__: getter_full_probmul_3,
    score_kgmon.__name__: getter_kgmon,
}


@dataclass
class SelectionConfig:
    enable_pseudo_test_rerank: bool = False

    @classmethod
    def from_env(cls):
        return cls(enable_pseudo_test_rerank=os.getenv("NVARC_ENABLE_PSEUDO_LOO_RERANK", "0") == "1")


class ArcDecoder:
    """ARC decoding and candidate selection.

    Pseudo-test leave-one-out reranking is feature flagged and OFF by default.
    When enabled, grouped candidate solutions are reranked by:
      1) existing base score from selection algorithm
      2) cheap structural checks
      3) pseudo-test leave-one-out consistency on train pairs

    `run_selection_algo` preserves historical return type: dict[key] -> list[np.ndarray].
    `run_selection_algo_with_metadata` exposes confidence and debug hooks for future
    diversity-aware attempt-2 routing.
    """

    def __init__(self, dataset, n_guesses, selection_config: SelectionConfig | None = None):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}
        self.selection_config = selection_config or SelectionConfig.from_env()

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def _task_context(self, basekey: str):
        # Keys in this notebook pipeline are usually "<puzzle_id>_<test_idx>"
        puzzle_id, task_idx = basekey.rsplit("_", 1)
        task_idx = int(task_idx)

        if basekey in self.dataset.queries:
            query = self.dataset.queries[basekey]
            test_input = query["test"][0]["input"]
            train_pairs = query["train"]
            return train_pairs, test_input

        query = self.dataset.queries[puzzle_id]
        test_input = query["test"][task_idx]["input"]
        train_pairs = query["train"]
        return train_pairs, test_input

    def run_selection_algo_with_metadata(self, selection_algorithm=score_kgmon, enable_pseudo_test_rerank: bool | None = None):
        if enable_pseudo_test_rerank is None:
            enable_pseudo_test_rerank = self.selection_config.enable_pseudo_test_rerank

        getter = algorithm_to_getter.get(selection_algorithm.__name__)
        if getter is None:
            raise ValueError(f"Unsupported selection algorithm: {selection_algorithm.__name__}")

        output = {}
        for bk, v in self.decoded_results.items():
            grouped = _group_scores({k: g for k, g in v.items()}, getter)

            if enable_pseudo_test_rerank:
                train_pairs, test_input = self._task_context(bk)
                reranked, confidence = rerank_candidates(grouped, train_pairs=train_pairs, test_input=test_input)

                ranked_candidates = [x.solution for x in reranked]
                output[bk] = {
                    "ranked_candidates": ranked_candidates,
                    "confidence": confidence,
                    "candidate_debug": [
                        {
                            "base_score": x.base_score,
                            "structural_score": x.structural_score,
                            "pseudo_loo_score": x.pseudo_loo_score,
                            "final_score": x.final_score,
                            "issues": x.structural_issues,
                            # Hook for future diversity-aware attempt_2 logic
                            "diversity_key": f"{x.solution.shape[0]}x{x.solution.shape[1]}:{len(np.unique(x.solution))}",
                        }
                        for x in reranked
                    ],
                }
            else:
                ranked_candidates = [x["solution"] for x in grouped]
                output[bk] = {
                    "ranked_candidates": ranked_candidates,
                    "confidence": 1.0 if len(ranked_candidates) <= 1 else 0.5,
                    "candidate_debug": [],
                }

        return output

    def run_selection_algo(self, selection_algorithm=score_kgmon, enable_pseudo_test_rerank: bool | None = None):
        metadata = self.run_selection_algo_with_metadata(
            selection_algorithm=selection_algorithm,
            enable_pseudo_test_rerank=enable_pseudo_test_rerank,
        )
        return {k: v["ranked_candidates"] for k, v in metadata.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
        print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1 / num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
