# SPDX-License-Identifier: MIT
"""Region traversal and egglog scheduling, loaded only when optimisation is requested."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from time import perf_counter
from typing import Any, Literal

from egglog import EGraph, Ruleset, TreeCostModel, i64
from egglog.egraph import UnstableCombinedRuleset

from ..algorithms.rewriting import complete_indices, zero_like
from ..algorithms.traversal import map_values, memoized
from ..nodes import AbstractExpression, IndexSum, Integer, Product, Sum
from .costs import loop_cost
from .encoding import decode, encode
from .optimise import OptimisationOptions, OptimisationReport, OptimisationResult, RegionReport
from .properties import is_defined
from .rules import algebra, constant_cleanup, defined, independent, indexed_rules
from .structural import simplify


def region_size(expression: AbstractExpression, limit: int) -> int:
    """Count unique region nodes, stopping at limit + 1 for oversized regions."""
    pending, seen = [expression], set[AbstractExpression]()
    while pending and len(seen) <= limit:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if isinstance(node, Sum | Product):
            pending.extend((node.first, node.second))
        elif isinstance(node, IndexSum) and len(node.indices) == 1:
            pending.append(node.expression)
    return len(seen)


def search(
    graph: EGraph, rules: Ruleset | UnstableCombinedRuleset, iterations: int
) -> tuple[Literal["saturated", "iteration_limit"], int]:
    """Run bounded algebraic search, returning why and when it stopped."""
    for step in range(1, iterations + 1):
        if not graph.run(rules * 1).updated:
            return "saturated", step
    return "iteration_limit", iterations


def run_region(
    original: AbstractExpression,
    options: OptimisationOptions,
    known_defined: Callable[[AbstractExpression], bool],
    cost_model: TreeCostModel[Any] | None,
) -> tuple[AbstractExpression, RegionReport]:
    """Search one region with a private e-graph and return its result and report."""
    size = region_size(original, options.max_region_nodes)
    if size > options.max_region_nodes:
        return original, RegionReport(size, "region_limit", 0, False, 0, 0, 0, 0, 0)
    selected = [
        rules
        for name, rules in (("arithmetic", algebra), ("indexed", indexed_rules))
        if name in options.rule_groups
    ]
    rules = selected[0] if len(selected) == 1 else selected[0] | selected[1]
    extraction_cost = cost_model or (loop_cost if options.cost == "loops" else None)
    start = perf_counter()
    region = encode(original, encode_indices="indexed" in options.rule_groups)
    encoded_at = perf_counter()
    graph = EGraph()
    graph.register(
        region.expression,
        *(defined(term) for node, term in region.terms if known_defined(node)),
        *(
            independent(i64(k), i64(j))
            for k, atom in enumerate(region.originals)
            for j, index in enumerate(region.indices)
            if index not in dict(atom.free_indices)
        ),
    )
    stop, rounds = search(graph, rules, options.iterations)
    graph.run(constant_cleanup.saturate())
    searched_at = perf_counter()
    # egglog 14 leaves generic arguments unspecified in this overload.
    extracted = graph.extract(region.expression, cost_model=extraction_cost)  # pyright: ignore[reportUnknownMemberType]
    extracted_at = perf_counter()
    candidate = decode(extracted, region)
    if isinstance(candidate, Integer) and candidate.value == 0:
        candidate = zero_like(original, original.argument_spaces)
    candidate = complete_indices(original, candidate)
    return candidate, RegionReport(
        size,
        stop,
        rounds,
        candidate != original,
        encoded_at - start,
        searched_at - encoded_at,
        extracted_at - searched_at,
        perf_counter() - extracted_at,
        region.alpha_merges,
    )


def run(
    root: AbstractExpression,
    options: OptimisationOptions,
    cost_model: TreeCostModel[Any] | None,
) -> OptimisationResult:
    """Optimise maximal supported regions, caching traversal within this call."""
    start = perf_counter()
    reports: list[RegionReport] = []
    known_defined = memoized(is_defined)

    @cache
    def region_operands(node: AbstractExpression) -> AbstractExpression:
        return (
            map_values(node, region_operands)
            if isinstance(node, Sum | Product | IndexSum) and not node.argument_spaces
            else visit(node)
        )

    @cache
    def visit(node: AbstractExpression) -> AbstractExpression:
        if (
            not options.preserve_floating_point_evaluation
            and isinstance(node, Sum | Product | IndexSum)
            and not node.value_shape
            and not node.argument_spaces
        ):
            result, report = run_region(region_operands(node), options, known_defined, cost_model)
            reports.append(report)
        else:
            result = map_values(node, visit)
        replacement = simplify(
            result, known_defined, preserve_evaluation=options.preserve_floating_point_evaluation
        )
        return result if replacement == result else visit(replacement)

    result = visit(root)
    return OptimisationResult(result, OptimisationReport(tuple(reports), perf_counter() - start))
