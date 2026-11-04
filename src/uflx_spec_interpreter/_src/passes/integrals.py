from __future__ import annotations

from typing import TYPE_CHECKING

from ..algorithms.rewriting import _rewrite
from ..algorithms.traversal import _everywhere
from ..nodes import (
    Integral,
    Mesh,
)
from ..rules.integration import _mesh_integral

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def expand_integrals(f: AbstractExpression) -> AbstractExpression:
    # Inner integrals first; integrand indices remain free.
    return _rewrite(f, _mesh_integral)


def _integrals_expanded(e: AbstractExpression) -> bool:
    def holds(n: AbstractExpression) -> bool:
        match n:
            case Integral(_, Mesh()):
                return False
            case _:
                return True

    return _everywhere(e, holds)
