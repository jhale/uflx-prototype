from basix.finite_element import FiniteElement

from uflx.nodes import AbstractTopology, CellMap, ScalarType

from ._elements import element_data
from .nodes import BasixCell, BasixSpace, RealScalarSpace


def real_scalar_space() -> RealScalarSpace:
    """Return the space of spatially constant real scalars."""
    return RealScalarSpace()


def local_space(
    domain: BasixCell | AbstractTopology | CellMap,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> BasixSpace:
    """Construct a cell-local finite element space from native Basix elements.

    A physical cell map binds the space to that piece in reference coordinates.
    A Basix reference cell supplies its shape alone. On a topology, one
    element applies to every slot, or supply a tuple with one element per slot.
    Each cell has its own coefficients. An assembler can identify shared DOFs
    across cells. A block shape interleaves copies of a scalar element at each DOF.
    """
    if isinstance(element, tuple):
        data = tuple(element_data(e) for e in element)
    else:
        data = (element_data(element),)
        if isinstance(domain, AbstractTopology):
            data *= len(domain.cell_types)
    return BasixSpace(domain, data, scalar_type, tuple(shape))
