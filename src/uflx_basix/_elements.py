# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Immutable construction data for native Basix elements in a UFLx graph."""

from functools import cached_property
from typing import final

import basix
import numpy as np

from uflx.nodes import Base


@final
class BuiltinElement(Base):
    identifier = "uflx_basix.BuiltinElement"
    family: int
    cell: int
    degree: int
    lagrange_variant: int
    dpc_variant: int
    discontinuous: bool
    dof_ordering: tuple
    dtype: str

    @cached_property
    def native(self):
        return basix.create_element(
            basix.ElementFamily(self.family),
            basix.CellType(self.cell),
            self.degree,
            basix.LagrangeVariant(self.lagrange_variant),
            basix.DPCVariant(self.dpc_variant),
            self.discontinuous,
            list(self.dof_ordering),
            self.dtype,
        )


@final
class CustomElement(Base):
    identifier = "uflx_basix.CustomElement"
    cell: int
    value_shape: tuple
    wcoeffs: tuple
    x: tuple
    M: tuple
    interpolation_nderivs: int
    map_type: int
    sobolev_space: int
    discontinuous: bool
    embedded_subdegree: int
    embedded_superdegree: int
    polyset_type: int
    dtype: str

    @cached_property
    def native(self):
        def array(data):
            shape, values = data
            return np.frombuffer(bytes.fromhex(values), dtype=self.dtype).reshape(shape)

        return basix.create_custom_element(
            basix.CellType(self.cell),
            self.value_shape,
            array(self.wcoeffs),
            [[array(a) for a in entities] for entities in self.x],
            [[array(a) for a in entities] for entities in self.M],
            self.interpolation_nderivs,
            basix.MapType(self.map_type),
            basix.SobolevSpace(self.sobolev_space),
            self.discontinuous,
            self.embedded_subdegree,
            self.embedded_superdegree,
            basix.PolysetType(self.polyset_type),
            dtype=np.dtype(self.dtype),
        )


def element_data(element):
    """Retain the supplied object and enough data to reconstruct it independently."""
    if not isinstance(element, basix.finite_element.FiniteElement):
        raise TypeError("expected a Basix FiniteElement")
    dtype = np.dtype(element.dtype).str
    if element.family == basix.ElementFamily.custom:

        def pack(array):
            return (array.shape, array.tobytes().hex())

        data = CustomElement(
            element.cell_type.value,
            tuple(element.value_shape),
            pack(element.wcoeffs),
            tuple(tuple(pack(a) for a in entities) for entities in element.x),
            tuple(tuple(pack(a) for a in entities) for entities in element.M),
            element.interpolation_nderivs,
            element.map_type.value,
            element.sobolev_space.value,
            element.discontinuous,
            element.embedded_subdegree,
            element.embedded_superdegree,
            element.polyset_type.value,
            dtype,
        )
    else:
        data = BuiltinElement(
            element.family.value,
            element.cell_type.value,
            element.degree,
            element.lagrange_variant.value,
            element.dpc_variant.value,
            element.discontinuous,
            tuple(element.dof_ordering),
            dtype,
        )
    # Equivalent elements share the same interned node and its cached native object.
    vars(data).setdefault("native", element)
    return data
