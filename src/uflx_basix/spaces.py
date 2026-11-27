# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from typing import overload

from basix.finite_element import FiniteElement

from uflx.nodes import AbstractTopology, MappedSpace, Mesh, ScalarType

from ._elements import element_data
from .nodes import BasixCell, BasixSpace, RealScalarSpace


def real_scalar_space() -> RealScalarSpace:
    """Return the space of spatially constant real scalars."""
    return RealScalarSpace()


@overload
def function_space(
    domain: BasixCell | AbstractTopology,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> BasixSpace: ...


@overload
def function_space(
    domain: Mesh,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> MappedSpace: ...


def function_space(
    domain: BasixCell | AbstractTopology | Mesh,
    element: FiniteElement | tuple[FiniteElement, ...],
    *,
    shape: tuple[int, ...] = (),
    scalar_type=ScalarType.real,
) -> BasixSpace | MappedSpace:
    """Declare a space on a reference cell, a topology, or a physical mesh.

    A single element applies to every topology slot, or supply one per slot.
    A block shape interleaves copies of a scalar element at each DOF.
    On a mesh the space uses the element's value mapping and reference DOF
    layout. To bind a reference space to a mesh cell, use V.on_cell(mesh[i]).
    """
    support = domain.topology if isinstance(domain, Mesh) else domain
    if not isinstance(support, (BasixCell, AbstractTopology)):
        raise TypeError(
            "expected a reference cell, topology, or mesh; use V.on_cell for a cell map"
        )
    if isinstance(element, tuple):
        data = tuple(element_data(e) for e in element)
    else:
        data = (element_data(element),)
        if isinstance(support, AbstractTopology):
            data *= len(support.cell_types)
    reference = BasixSpace(support, data, scalar_type, tuple(shape))
    if isinstance(domain, Mesh):
        if any(e.interpolation_nderivs for e in reference.elements):
            raise NotImplementedError("physical spaces with derivative DOFs are not implemented")
        return MappedSpace(reference, domain, reference.law)
    return reference
