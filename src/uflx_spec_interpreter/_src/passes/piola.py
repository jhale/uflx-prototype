"""Shared applicability of the two Piola differential identities."""

from ..nodes import Curl, Divergence, PushForward, TransformationLaw


def piola_law(expression):
    """Return the output transformation law, or None when no identity applies.

    This inspects the differential and mapped field only. A pull-back must
    separately establish that its map selects a cell of the matching mesh.
    """
    match expression:
        case Divergence(PushForward(_, D, TransformationLaw(("contravariant",), 1)), P) if P == D:
            return TransformationLaw((), 1)
        case Curl(PushForward(_, D, TransformationLaw(("covariant",), 0)), P) if (
            P == D and D.dimension == D.ambient_coordinate_domain.dimension
        ):
            return TransformationLaw(("contravariant",) * len(expression.value_shape), 1)
    return None
