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
from ..nodes import (
    Absolute,
    AbstractExpression,
    Conjugate,
    ImaginaryPart,
    IndexSum,
    Integer,
    Logarithm,
    Power,
    Product,
    RealPart,
    Sum,
)
from ..nodes.structure import expression_operands
from .costs import loop_cost
from .encoding import decode, encode
from .optimise import OptimisationOptions, OptimisationReport, OptimisationResult, RegionReport
from .properties import is_defined
from .rules import (
    algebra,
    constant_cleanup,
    defined,
    independent,
    indexed_rules,
    is_real,
    unary_rules,
)
from .structural import simplify


def in_region(node: AbstractExpression, preserve_evaluation: bool) -> bool:
    """Safe unary operations are always supported; other scalar operations are optional."""
    return isinstance(node, Absolute | Conjugate | RealPart | ImaginaryPart) or (
        not preserve_evaluation
        and isinstance(node, Sum | Product | IndexSum | Power | Logarithm)
        and not node.value_shape
        and not node.argument_spaces
    )


def region_size(expression: AbstractExpression, limit: int, preserve_evaluation: bool) -> int:
    """Count unique region nodes, stopping at limit + 1 for oversized regions."""
    pending, seen = [expression], set[AbstractExpression]()
    while pending and len(seen) <= limit:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if in_region(node, preserve_evaluation) and (
            not isinstance(node, IndexSum) or len(node.indices) == 1
        ):
            pending.extend(expression_operands(node))
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
    preserve = options.preserve_floating_point_evaluation
    size = region_size(original, options.max_region_nodes, preserve)
    if size > options.max_region_nodes:
        return original, RegionReport(size, "region_limit", 0, False, 0, 0, 0, 0, 0)
    rules: Ruleset | UnstableCombinedRuleset = unary_rules
    if not preserve:
        rules = rules | constant_cleanup
        for name, group in (("arithmetic", algebra), ("indexed", indexed_rules)):
            if name in options.rule_groups:
                rules = rules | group
    extraction_cost = cost_model or (loop_cost if options.cost == "loops" else None)
    start = perf_counter()
    region = encode(
        original,
        encode_indices="indexed" in options.rule_groups,
        preserve_evaluation=preserve,
    )
    encoded_at = perf_counter()
    graph = EGraph()
    graph.register(
        region.expression,
        *(defined(term) for node, term in region.terms if known_defined(node)),
        *(is_real(term) for node, term in region.terms if node.scalar_type == "real"),
        *(
            independent(i64(k), i64(j))
            for k, atom in enumerate(region.originals)
            for j, index in enumerate(region.indices)
            if index not in dict(atom.free_indices)
        ),
    )
    stop, rounds = search(graph, rules, options.iterations)
    if not preserve:
        graph.run(constant_cleanup.saturate())
    searched_at = perf_counter()
    # egglog 14 leaves generic arguments unspecified in this overload.
    extracted = graph.extract(region.expression, cost_model=extraction_cost)  # pyright: ignore[reportUnknownMemberType]
    extracted_at = perf_counter()
    candidate = decode(extracted, region)
    if isinstance(candidate, Integer) and candidate.value == 0:
        candidate = zero_like(original, original.argument_spaces)
    # Scalar algebra can identify a complex expression with a real value. Retain
    # the original real result type without assuming every representative is real.
    if original.scalar_type == "real" and candidate.scalar_type == "complex":
        candidate = RealPart(candidate)
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
    preserve = options.preserve_floating_point_evaluation
    reports: list[RegionReport] = []
    known_defined = memoized(is_defined)

    @cache
    def region_operands(node: AbstractExpression) -> AbstractExpression:
        return map_values(node, region_operands) if in_region(node, preserve) else visit(node)

    @cache
    def visit(node: AbstractExpression) -> AbstractExpression:
        if in_region(node, preserve):
            result, report = run_region(region_operands(node), options, known_defined, cost_model)
            reports.append(report)
        else:
            result = map_values(node, visit)
        replacement = simplify(result, known_defined, preserve_evaluation=preserve)
        return result if replacement == result else visit(replacement)

    result = visit(root)
    return OptimisationResult(result, OptimisationReport(tuple(reports), perf_counter() - start))
