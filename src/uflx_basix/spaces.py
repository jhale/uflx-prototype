from basix.finite_element import FiniteElement

from uflx.nodes import AbstractTopology, ScalarType

from ._elements import element_data
from .nodes import BasixCell, BasixSpace


def space(
    label: str,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    domain: BasixCell | AbstractTopology | None = None,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> BasixSpace:
    """Attach a native Basix element to a UFLx space.

    The default domain is the element's reference cell. On a topology, one
    element applies to every slot, or supply a tuple with one element per slot.
    A block shape interleaves copies of a scalar element at each DOF.
    """
    if isinstance(element, tuple):
        data = tuple(element_data(e) for e in element)
    else:
        data = (element_data(element),)
        if isinstance(domain, AbstractTopology):
            data *= len(domain.cell_types)
    if domain is None:
        if len(data) != 1:
            raise ValueError("multiple elements require a topology")
        domain = BasixCell(data[0].cell)
    return BasixSpace(label, domain, data, scalar_type, tuple(shape))
