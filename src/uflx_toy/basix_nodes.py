from __future__ import annotations

from typing import final

import basix
import numpy as np

from uflx_spec_interpreter import tensor
from uflx_spec_interpreter._src.nodes import AbstractReferenceCell, Float, Zero


@final
class BasixCell(AbstractReferenceCell):
    identifier = "toy.BasixCell"
    # A reference cell of Basix, whose vertices and facet normals Basix tabulates.
    cell_type: basix.CellType
    opaque = ("cell_type",)

    @property
    def dimension(self):
        return len(basix.cell.topology(self.cell_type)) - 1

    @property
    def num_vertices(self):
        return len(basix.cell.topology(self.cell_type)[0])

    @property
    def vertices(self):
        if self.dimension == 0:
            return Zero((1, 0), (), ())
        return table(basix.cell.geometry(self.cell_type))

    @property
    def num_facets(self):
        return len(basix.cell.topology(self.cell_type)[self.dimension - 1])

    def local_facet(self, local_vertices):
        # The number of the facet of the cell with these local vertices, in the order of Basix.
        facets = basix.cell.topology(self.cell_type)[self.dimension - 1]
        return next(f for f, v in enumerate(facets) if set(v) == set(local_vertices))

    @property
    def facet_normals(self):
        # The outward normal of each facet, a constant table with no support.
        return table(basix.cell.facet_outward_normals(self.cell_type))


def table(values):
    return tensor(tuple(tensor(tuple(Float(float(x)) for x in row)) for row in np.asarray(values)))
