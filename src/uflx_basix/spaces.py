from uflx.nodes import ScalarType

from .nodes import BasixSpace


def space(
    label,
    domain,
    family,
    degree,
    discontinuous=False,
    scalar_type=ScalarType.real,
    *,
    shape: tuple[int, ...] = (),
):
    """A Basix space on a topology or reference cell, optionally with a block shape.

    Blocked DOFs are interleaved: scalar DOF first, then flattened component.
    Intrinsically vector-valued elements keep their Basix value shape.
    """
    if any(n <= 0 for n in shape):
        raise ValueError("block shape extents must be positive")
    V = BasixSpace(label, domain, family.value, degree, int(discontinuous), scalar_type, shape)
    if shape and any(E.value_shape for E in V.elements):
        raise ValueError("blocked spaces require a scalar base element")
    return V
