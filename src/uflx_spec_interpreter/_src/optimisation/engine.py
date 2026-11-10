# SPDX-License-Identifier: MIT
"""Region traversal and egglog scheduling, loaded only when optimisation is requested."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

from egglog import EGraph, Ruleset, TreeCostModel, i64
from egglog.egraph import UnstableCombinedRuleset

from ..algorithms.rewriting import complete_indices, zero_like
from ..algorithms.traversal import map_values
from ..nodes import AbstractExpression, IndexSum, Integer, Product, Sum
from .costs import loop_cost
from .encoding import Codec
from .optimise import OptimisationOptions, OptimisationReport, OptimisationResult, RegionReport
from .properties import definedness
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


def run(
    root: AbstractExpression,
    options: OptimisationOptions,
    cost_model: TreeCostModel[Any] | None,
) -> OptimisationResult:
    """Optimise maximal supported scalar regions and reconstruct their surroundings."""
    reports: list[RegionReport] = []
    memo: dict[AbstractExpression, AbstractExpression] = {}
    prepared: dict[AbstractExpression, AbstractExpression] = {}
    start = perf_counter()
    known_defined = definedness()
    selected = [
        rules
        for name, rules in (("arithmetic", algebra), ("indexed", indexed_rules))
        if name in options.rule_groups
    ]
    search: Ruleset | UnstableCombinedRuleset = selected[0]
    for rules in selected[1:]:
        search = search | rules
    extraction_cost = cost_model or (loop_cost if options.cost == "loops" else None)

    def run_region(original: AbstractExpression) -> AbstractExpression:
        size = region_size(original, options.max_region_nodes)
        if size > options.max_region_nodes:
            reports.append(RegionReport(size, "region_limit", 0, False, 0, 0, 0, 0, 0))
            return original
        codec = Codec(encode_indices="indexed" in options.rule_groups)
        t = perf_counter()
        encoded = codec.encode(original)
        encode_seconds = perf_counter() - t
        t = perf_counter()
        graph = EGraph()
        graph.register(
            encoded,
            *(defined(term) for node, term in codec.memo.items() if known_defined(node)),
            *(
                independent(i64(k), i64(j))
                for k, atom in enumerate(codec.originals)
                for j, index in enumerate(codec.index_nodes)
                if index not in dict(atom.free_indices)
            ),
        )
        stop: Literal["saturated", "iteration_limit"] = "iteration_limit"
        rounds = 0
        for _ in range(options.iterations):
            rounds += 1
            if not graph.run(search * 1).updated:
                stop = "saturated"
                break
        # Always finish arithmetic exposed by search; this phase does not expand terms.
        graph.run(constant_cleanup.saturate())
        search_seconds = perf_counter() - t
        t = perf_counter()
        # egglog 14 leaves generic arguments unspecified in this overload.
        extracted = graph.extract(encoded, cost_model=extraction_cost)  # pyright: ignore[reportUnknownMemberType]
        extract_seconds = perf_counter() - t
        t = perf_counter()
        candidate = codec.decode(extracted)
        if isinstance(candidate, Integer) and candidate.value == 0:
            candidate = zero_like(original, original.argument_spaces)
        candidate = complete_indices(original, candidate)
        decode_seconds = perf_counter() - t
        reports.append(
            RegionReport(
                size,
                stop,
                rounds,
                candidate != original,
                encode_seconds,
                search_seconds,
                extract_seconds,
                decode_seconds,
                codec.alpha_merges,
            )
        )
        return candidate

    def region_operands(node: AbstractExpression) -> AbstractExpression:
        if node not in prepared:
            prepared[node] = (
                map_values(node, region_operands)
                if isinstance(node, Sum | Product | IndexSum) and not node.argument_spaces
                else visit(node)
            )
        return prepared[node]

    def visit(node: AbstractExpression) -> AbstractExpression:
        if node in memo:
            return memo[node]
        if (
            not options.preserve_floating_point
            and isinstance(node, Sum | Product | IndexSum)
            and not node.value_shape
            and not node.argument_spaces
        ):
            result = run_region(region_operands(node))
        else:
            result = map_values(node, visit)
        replacement = simplify(result, known_defined)
        if replacement != result and (
            not options.preserve_floating_point or replacement.scalar_type == result.scalar_type
        ):
            replacement = complete_indices(result, replacement)
            if replacement != result:
                result = visit(replacement)
        memo[node] = result
        return result

    result = visit(root)
    return OptimisationResult(result, OptimisationReport(tuple(reports), perf_counter() - start))
