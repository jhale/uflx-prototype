from __future__ import annotations

from typing import TYPE_CHECKING

from ..algorithms.rewriting import rewrite
from ..algorithms.traversal import everywhere
from ..nodes import (
    Integral,
    Mesh,
)
from ..rules.integration import mesh_integral

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def expand_integrals(f: AbstractExpression) -> AbstractExpression:
    # Inner integrals first; integrand indices remain free.
    return rewrite(f, mesh_integral)


def integrals_expanded(e: AbstractExpression) -> bool:
    def holds(n: AbstractExpression) -> bool:
        match n:
            case Integral(_, Mesh()):
                return False
            case _:
                return True

    return everywhere(e, holds)
