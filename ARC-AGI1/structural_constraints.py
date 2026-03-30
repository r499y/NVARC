"""Deterministic structural constraint checks for ARC candidate outputs.

This module is intentionally lightweight and reusable:
- infer cheap invariants from train examples,
- validate malformed/impossible candidates early,
- provide rejection reasons for logging/debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Grid = np.ndarray
LoggerFn = Callable[[str], None]


@dataclass(frozen=True)
class StructuralInvariants:
    has_train_data: bool
    fixed_output_shape: tuple[int, int] | None
    all_io_same_shape: bool
    fixed_output_palette: set[int] | None
    output_palette_union: set[int]
    min_palette_size: int
    max_palette_size: int


@dataclass
class ConstraintCheckResult:
    is_valid: bool
    score: float
    rejection_reasons: list[str]
    warnings: list[str]


def _to_array(grid: Any) -> Grid:
    return np.asarray(grid, dtype=np.int16)


def _extract_io(pair: Any) -> tuple[Grid, Grid] | None:
    if isinstance(pair, dict) and "input" in pair and "output" in pair:
        return _to_array(pair["input"]), _to_array(pair["output"])
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return _to_array(pair[0]), _to_array(pair[1])
    return None


def infer_train_invariants(train_pairs: list[Any]) -> StructuralInvariants:
    converted: list[tuple[Grid, Grid]] = []
    for pair in train_pairs:
        io_pair = _extract_io(pair)
        if io_pair is not None:
            converted.append(io_pair)

    if not converted:
        return StructuralInvariants(
            has_train_data=False,
            fixed_output_shape=None,
            all_io_same_shape=False,
            fixed_output_palette=None,
            output_palette_union=set(),
            min_palette_size=0,
            max_palette_size=10,
        )

    output_shapes = [out.shape for _, out in converted]
    fixed_output_shape = output_shapes[0] if len(set(output_shapes)) == 1 else None
    all_io_same_shape = all(inp.shape == out.shape for inp, out in converted)

    output_palettes = [set(np.unique(out).tolist()) for _, out in converted]
    fixed_output_palette = output_palettes[0] if len({tuple(sorted(p)) for p in output_palettes}) == 1 else None

    palette_sizes = [len(p) for p in output_palettes]
    output_palette_union: set[int] = set().union(*output_palettes)

    return StructuralInvariants(
        has_train_data=True,
        fixed_output_shape=fixed_output_shape,
        all_io_same_shape=all_io_same_shape,
        fixed_output_palette=fixed_output_palette,
        output_palette_union=output_palette_union,
        min_palette_size=min(palette_sizes),
        max_palette_size=max(palette_sizes),
    )


def check_candidate_constraints(
    candidate: Any,
    test_input: Any,
    invariants: StructuralInvariants,
    logger: LoggerFn | None = None,
) -> ConstraintCheckResult:
    reasons: list[str] = []
    warnings: list[str] = []

    grid = np.asarray(candidate)
    if grid.ndim != 2:
        reasons.append("not_2d")
    else:
        h, w = grid.shape
        if h < 1 or w < 1:
            reasons.append("empty_axis")
        if h > 30 or w > 30:
            reasons.append("too_large")

    if not np.issubdtype(grid.dtype, np.integer):
        reasons.append("non_integer_dtype")

    try:
        unique_vals = set(np.unique(grid).tolist())
    except Exception:
        unique_vals = set()
        reasons.append("invalid_values")

    if unique_vals and (min(unique_vals) < 0 or max(unique_vals) > 9):
        reasons.append("out_of_palette")

    test_input_grid = np.asarray(test_input)
    if grid.ndim == 2 and invariants.fixed_output_shape is not None and tuple(grid.shape) != invariants.fixed_output_shape:
        reasons.append("violates_fixed_output_shape")

    if grid.ndim == 2 and invariants.all_io_same_shape and test_input_grid.ndim == 2:
        if tuple(grid.shape) != tuple(test_input_grid.shape):
            reasons.append("violates_io_same_shape_invariant")

    if invariants.fixed_output_palette is not None and unique_vals and not unique_vals.issubset(invariants.fixed_output_palette):
        reasons.append("violates_fixed_output_palette")

    if invariants.output_palette_union and unique_vals and unique_vals.isdisjoint(invariants.output_palette_union):
        reasons.append("disjoint_from_train_output_palette")

    if unique_vals:
        palette_size = len(unique_vals)
        if palette_size < invariants.min_palette_size or palette_size > invariants.max_palette_size:
            warnings.append("palette_size_outside_train_range")

    score = 1.0
    if reasons:
        score = 0.0
    elif warnings:
        score = max(0.0, 1.0 - 0.15 * len(warnings))

    if logger is not None:
        for reason in reasons:
            logger(f"[constraint-reject] {reason}")
        for warning in warnings:
            logger(f"[constraint-warn] {warning}")

    return ConstraintCheckResult(
        is_valid=len(reasons) == 0,
        score=score,
        rejection_reasons=reasons,
        warnings=warnings,
    )
