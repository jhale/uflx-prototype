# SPDX-License-Identifier: MIT
"""Optional entry point and immutable policy/report objects for optimisation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..nodes import AbstractExpression

if TYPE_CHECKING:
    from egglog import TreeCostModel


@dataclass(frozen=True)
class OptimisationOptions:
    """Select algebraic search and its budgets; these are not wall-clock limits.

    preserve_floating_point_evaluation defaults to False, permitting mathematical rewrites
    that alter floating-point evaluation. True retains arithmetic operations,
    grouping and reduction order, using structural and safe unary simplifications.
    """

    rule_groups: tuple[Literal["arithmetic", "indexed", "herbie"], ...] = ("arithmetic", "indexed")
    preserve_floating_point_evaluation: bool = False
    iterations: int = 3
    max_region_nodes: int = 60
    cost: Literal["tree", "loops"] = "loops"

    def __post_init__(self) -> None:
        if type(self.preserve_floating_point_evaluation) is not bool:
            raise ValueError("preserve_floating_point_evaluation must be a bool")
        if not self.rule_groups or set(self.rule_groups) - {"arithmetic", "indexed", "herbie"}:
            raise ValueError("select arithmetic, indexed and/or herbie rule groups")
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(self.max_region_nodes) is not int or self.max_region_nodes < 1:
            raise ValueError("max_region_nodes must be a positive integer")
        if self.cost not in ("tree", "loops"):
            raise ValueError("cost must be tree or loops")


@dataclass(frozen=True)
class RegionReport:
    """Status of exploratory search before the non-expanding finishing phase."""

    nodes: int
    search_stop: Literal["saturated", "iteration_limit", "region_limit"]
    iterations: int
    changed: bool
    encode_seconds: float
    search_seconds: float
    extract_seconds: float
    decode_seconds: float
    alpha_merges: int


@dataclass(frozen=True)
class OptimisationReport:
    """Per-region outcomes and whole-invocation elapsed time."""

    regions: tuple[RegionReport, ...]
    total_seconds: float


@dataclass(frozen=True)
class OptimisationResult:
    """The reconstructed expression and its optimisation report."""

    expression: AbstractExpression
    report: OptimisationReport


def optimise(
    expression: AbstractExpression,
    options: OptimisationOptions | None = None,
    *,
    cost_model: TreeCostModel[Any] | None = None,
) -> OptimisationResult:
    """Optimise supported scalar regions, retaining other UFLx operations.

    With preserve_floating_point_evaluation=False, uses exact real/complex identities. True retains
    arithmetic operations and reduction order while simplifying structure and safe unary operations.
    Opaque atoms share modulo alpha-equivalence; cross-invocation caching and
    general binder rewriting are not implemented.
    Traversal is recursive; deep graphs are outside the initial supported scope.
    """
    if sys.version_info < (3, 12):
        raise ImportError("UFLx optimisation requires Python 3.12 or newer")
    try:
        from .engine import run
    except ModuleNotFoundError as error:
        if error.name == "egglog":
            raise ImportError(
                "Install uflx-spec-interpreter[optimisation] to enable optimisation"
            ) from error
        raise
    return run(expression, options or OptimisationOptions(), cost_model)
