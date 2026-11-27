# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix

from .nodes import BasixCell, BasixTopology


def cell(cell_type: basix.CellType) -> BasixCell:
    return BasixCell(cell_type.value)


def create_topology(*cell_types: basix.CellType) -> BasixTopology:
    """Build one local cell slot per Basix cell type, in the order supplied."""
    return BasixTopology(tuple(cell(cell_type) for cell_type in cell_types))


# Keep the original factory name available to existing callers.
topology = create_topology
