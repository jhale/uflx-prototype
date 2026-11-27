# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from basix.finite_element import FiniteElement

from uflx.nodes import AbstractTopology, CellMap, MappedSpace, Mesh, ScalarType

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


def function_space(
    mesh: Mesh,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> MappedSpace:
    """Declare a physical field space using the Basix element's value mapping.

    The solver supplies coefficients in the local element's DOF order.
    This constructor supports value DOFs. Geometry-dependent reconstruction
    for physical derivative DOFs is currently unsupported.
    """
    reference = local_space(mesh.topology, element, shape=shape, scalar_type=scalar_type)
    if any(e.interpolation_nderivs for e in reference.elements):
        raise NotImplementedError("physical spaces with derivative DOFs are not implemented")
    return MappedSpace(reference, mesh, reference.law)
