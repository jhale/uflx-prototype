"""Shared applicability of the two Piola differential identities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..nodes import Curl, Divergence, PushForward, TransformationLaw

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


def piola_law(expression: AbstractExpression) -> TransformationLaw | None:
    """Return the output transformation law, or None when no identity applies.

    This inspects the differential and mapped field only. A pull-back must
    separately establish that its map selects a cell of the matching mesh.
    """
    match expression:
        case Divergence(PushForward(_, d, TransformationLaw(("contravariant",), 1)), p) if p == d:
            return TransformationLaw((), 1)
        case Curl(PushForward(_, d, TransformationLaw(("covariant",), 0)), p) if (
            p == d and d.dimension == d.ambient_coordinate_domain.dimension
        ):
            return TransformationLaw(("contravariant",) * len(expression.value_shape), 1)
        case _:
            return None
