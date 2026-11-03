from uflx_spec_interpreter import ScalarType

from .basix_nodes import BasixSpace


def basix_space(
    label,
    domain,
    family,
    degree,
    discontinuous=False,
    scalar_type=ScalarType.real,
    *,
    shape: tuple[int, ...] = (),
):
    """A Basix space, optionally with independent scalar copies in a value shape.

    Blocked DOFs are interleaved: scalar DOF first, then flattened component.
    Intrinsically vector-valued elements keep their Basix value shape.
    """
    if any(n <= 0 for n in shape):
        raise ValueError("block shape extents must be positive")
    V = BasixSpace(label, domain, family.value, degree, int(discontinuous), scalar_type, shape)
    if shape and any(E.value_shape for E, _ in V.elements):
        raise ValueError("blocked spaces require a scalar base element")
    return V
