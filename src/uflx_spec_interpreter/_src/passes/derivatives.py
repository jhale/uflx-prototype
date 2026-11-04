from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from ..algorithms.rewriting import _rewrite
from ..algorithms.traversal import _everywhere
from ..nodes import DirectionalDerivative
from ..rules.differentiation import _deriv
from .actions import apply_actions

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def apply_derivatives(f: AbstractExpression) -> AbstractExpression:
    # Derivative evaluation does no optional algebraic collection.
    differential = cache(_deriv)

    def evaluate(e: AbstractExpression) -> AbstractExpression:
        match e:
            case DirectionalDerivative(g, x, a, h):
                return differential(g, x, a, h)
            case _:
                return e

    # Applying a newly exposed action can reveal another derivative.
    seen: set[AbstractExpression] = set()
    while f not in seen:
        seen.add(f)
        f = apply_actions(_rewrite(f, evaluate))
        if _derivatives_complete(f):
            return f
    raise ValueError("derivative rewriting made no progress", f)


def _derivatives_complete(e: AbstractExpression) -> bool:
    return _everywhere(e, lambda n: not isinstance(n, DirectionalDerivative))
