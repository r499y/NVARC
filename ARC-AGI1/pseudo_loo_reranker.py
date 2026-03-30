"""Pseudo-test leave-one-out reranking utilities for ARC candidate selection.

Scoring combines three terms:
1) base model score (existing ranking signal)
2) structural constraint checker score + rejection reasons
3) pseudo-test leave-one-out consistency against train-pair profiles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from structural_constraints import check_candidate_constraints, infer_train_invariants


Grid = np.ndarray


@dataclass(frozen=True)
class RerankWeights:
    base: float = 0.25
    structural: float = 0.25
    pseudo_loo: float = 0.50


@dataclass
class CandidateScore:
    solution: Grid
    base_score: float
    structural_score: float
    pseudo_loo_score: float
    final_score: float
    structural_issues: list[str]


def _to_array(grid: Any) -> Grid:
    return np.asarray(grid, dtype=np.int16)


def _extract_io(pair: Any) -> tuple[Grid, Grid] | None:
    if isinstance(pair, dict) and "input" in pair and "output" in pair:
        return _to_array(pair["input"]), _to_array(pair["output"])
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return _to_array(pair[0]), _to_array(pair[1])
    return None


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _profile_from_pairs(pairs: list[tuple[Grid, Grid]]) -> dict[str, float]:
    shape_h = []
    shape_w = []
    out_palette = []
    same_shape_change = []

    for inp, out in pairs:
        shape_h.append(_safe_ratio(out.shape[0], inp.shape[0]))
        shape_w.append(_safe_ratio(out.shape[1], inp.shape[1]))
        out_palette.append(float(len(np.unique(out))))

        if inp.shape == out.shape:
            same_shape_change.append(float(np.mean(inp != out)))

    return {
        "ratio_h": float(np.mean(shape_h)) if shape_h else 1.0,
        "ratio_w": float(np.mean(shape_w)) if shape_w else 1.0,
        "palette": float(np.mean(out_palette)) if out_palette else 1.0,
        "change": float(np.mean(same_shape_change)) if same_shape_change else 0.5,
    }


def _candidate_to_profile(candidate: Grid, test_input: Grid) -> dict[str, float]:
    same_shape_change = float(np.mean(candidate != test_input)) if candidate.shape == test_input.shape else 1.0
    return {
        "ratio_h": _safe_ratio(candidate.shape[0], test_input.shape[0]),
        "ratio_w": _safe_ratio(candidate.shape[1], test_input.shape[1]),
        "palette": float(len(np.unique(candidate))),
        "change": same_shape_change,
    }


def pseudo_loo_consistency(candidate: Grid, train_pairs: list[Any], test_input: Any) -> float:
    if len(train_pairs) < 2:
        return 0.5

    converted: list[tuple[Grid, Grid]] = []
    for pair in train_pairs:
        io_pair = _extract_io(pair)
        if io_pair is not None:
            converted.append(io_pair)

    if len(converted) < 2:
        return 0.5

    test_input_arr = _to_array(test_input)
    cand_profile = _candidate_to_profile(candidate, test_input_arr)

    fold_scores: list[float] = []
    for holdout_idx in range(len(converted)):
        kept = [p for i, p in enumerate(converted) if i != holdout_idx]
        proto = _profile_from_pairs(kept)

        diff_h = abs(cand_profile["ratio_h"] - proto["ratio_h"])
        diff_w = abs(cand_profile["ratio_w"] - proto["ratio_w"])
        diff_palette = abs(cand_profile["palette"] - proto["palette"]) / 10.0
        diff_change = abs(cand_profile["change"] - proto["change"])

        sim = 1.0 - min(1.0, 0.35 * diff_h + 0.35 * diff_w + 0.15 * diff_palette + 0.15 * diff_change)
        fold_scores.append(sim)

    return float(np.mean(fold_scores)) if fold_scores else 0.5


def rerank_candidates(
    candidates: list[dict[str, Any]],
    train_pairs: list[Any],
    test_input: Any,
    weights: RerankWeights | None = None,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[CandidateScore], float]:
    if weights is None:
        weights = RerankWeights()

    if not candidates:
        return [], 0.0

    invariants = infer_train_invariants(train_pairs)

    base_scores = np.asarray([float(c["base_score"]) for c in candidates], dtype=np.float32)
    base_min = float(np.min(base_scores))
    base_max = float(np.max(base_scores))
    base_span = max(base_max - base_min, 1e-6)

    scored: list[CandidateScore] = []
    for c in candidates:
        solution = _to_array(c["solution"])
        base_norm = (float(c["base_score"]) - base_min) / base_span

        constraint_result = check_candidate_constraints(solution, test_input, invariants, logger=logger)

        if constraint_result.is_valid:
            loo_score = pseudo_loo_consistency(solution, train_pairs, test_input)
        else:
            loo_score = 0.0

        final_score = (
            weights.base * base_norm
            + weights.structural * constraint_result.score
            + weights.pseudo_loo * loo_score
        )
        scored.append(
            CandidateScore(
                solution=solution,
                base_score=float(c["base_score"]),
                structural_score=constraint_result.score,
                pseudo_loo_score=loo_score,
                final_score=float(final_score),
                structural_issues=constraint_result.rejection_reasons + constraint_result.warnings,
            )
        )

    scored.sort(key=lambda x: x.final_score, reverse=True)

    if len(scored) == 1:
        confidence = 1.0
    else:
        margin = scored[0].final_score - scored[1].final_score
        confidence = float(max(0.0, min(1.0, 0.5 + margin)))

    return scored, confidence
