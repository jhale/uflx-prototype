from basix.finite_element import FiniteElement

from uflx.nodes import AbstractTopology, ScalarType

from ._elements import element_data
from .nodes import BasixCell, BasixSpace


def space(
    domain: BasixCell | AbstractTopology,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> BasixSpace:
    """Attach a native Basix element to a UFLx space.

    The domain is a Basix reference cell or a UFLx topology. On a topology, one
    element applies to every slot, or supply a tuple with one element per slot.
    A block shape interleaves copies of a scalar element at each DOF.
    """
    if isinstance(element, tuple):
        data = tuple(element_data(e) for e in element)
    else:
        data = (element_data(element),)
        if isinstance(domain, AbstractTopology):
            data *= len(domain.cell_types)
    return BasixSpace(domain, data, scalar_type, tuple(shape))
